import json
import shutil
import sys
import textwrap
from pathlib import Path

import pytest

from flybot.live.execution import WalletLedger
from flybot.live.perp_execution import DryRunPerpExecutor, NodePerpExecutor, PerpExecutionError, PerpMirror
from flybot.live.settings import Settings, SettingsError
from flybot.perp import PerpCosts, PerpPaperEpisode, rows_to_frame
from tests.test_perp import rows

PAPER = PerpCosts(taker_bps=0.0, slip_bps=0.0, gas_per_action=0.0, stake=1000.0, leverage=2.0)
BID, ASK = 99.98, 100.02


class FakeAccount:
    """A perps account that fills every taker order in full and lets quotes rest."""

    dry = False

    def __init__(self, position=0.0, resting=None, readable=True):
        self.pos, self.calls, self._n, self.readable = position, [], 0, readable
        self.resting = {o: ("buy", 1.0, 1.0) for o in (resting or [])}
        self.eq = 100.0

    def markets(self):
        return ["BTC/USDso:USDso"]

    def mark(self, symbol):
        return (BID + ASK) / 2

    def book(self, symbol):
        return BID, ASK

    def place(self, symbol, side, price, qty, kind, lift_to=0.0):
        self._n += 1
        oid = f"o{self._n}"
        self.calls.append(("place", side, round(price, 6), round(qty, 9), kind))
        if kind == "market":
            self.pos += qty if side == "long" else -qty
            return {"sent": True, "order_id": None, "filled": qty}
        self.resting[oid] = (side, price, qty)
        return {"sent": True, "order_id": oid, "filled": 0.0}

    def cancel(self, symbol, order_id):
        self.calls.append(("cancel", order_id))
        return self.resting.pop(order_id, None) is not None

    def open_orders(self, symbol):
        return [{"id": k, "side": v[0], "price": v[1], "remaining": v[2]} for k, v in self.resting.items()]

    def position(self, symbol):
        return self.pos if self.readable else None

    def close(self, symbol):
        self.calls.append(("close",))
        was, self.pos = self.pos, 0.0
        return {"sent": was != 0.0, "filled": abs(was), "left": 0.0}

    def equity(self):
        return self.eq


def settings(**kw):
    return Settings(**{"live_stake_usdso": 20.0, "perp_leverage": 2.0, "requote_bps": 5.0, **kw})


def mirror(ex, **kw):
    return PerpMirror(ex, "BTC/USDso:USDso", PAPER.stake, PAPER.leverage, settings(**kw), lambda k, r: None)


def frame(n=80, **kw):
    return rows_to_frame(rows(n, start=100.0, **kw))


def episode(arm, df, trend_dir=0, **params):
    ep = PerpPaperEpisode(arm, PAPER, params)
    ep.start(df.iloc[0], trend_dir)
    return ep


def places(ex, kind=None):
    return [c for c in ex.calls if c[0] == "place" and (kind is None or c[4] == kind)]


# ---------- scale ----------
def test_live_size_is_the_live_margin_times_leverage_against_the_paper_one():
    m = mirror(FakeAccount())
    assert m.scale == pytest.approx(20.0 * 2.0 / (1000.0 * 2.0))          # 40 USDso notional for 2000 of paper
    assert mirror(FakeAccount(), perp_leverage=4.0).scale == pytest.approx(0.04)


# ---------- taker arms: the paper position is the target ----------
def test_a_trend_entry_is_one_taker_order_at_the_touch_and_the_finish_closes_it():
    ex, m, df = FakeAccount(), mirror(FakeAccount()), frame()
    m.executor = ex
    ep = episode("p_trend", df, trend_dir=1)
    m.on_start(ep)
    assert places(ex) == [("place", "long", ASK, pytest.approx(ep.pos * m.scale), "market")]
    assert ex.pos == pytest.approx(ep.pos * m.scale)
    m.on_row(ep)                                                       # nothing moved: nothing sent
    assert len(places(ex)) == 1
    ep.finish(df.iloc[-1])
    m.finish(ep)
    assert ("close",) in ex.calls and ex.pos == 0.0


def test_a_guard_flatten_in_the_paper_episode_closes_the_real_position():
    ex, df = FakeAccount(), frame()
    m = mirror(ex)
    ep = episode("p_trend", df, trend_dir=-1)
    m.on_start(ep)
    assert ex.pos < 0
    ep.pos, ep.guarded = 0.0, True                                     # what the paper guard does
    m.on_row(ep)
    assert ex.pos == 0.0 and ("close",) in ex.calls


def test_a_paying_carry_is_kept_open_at_the_end_and_not_re_entered_by_the_next_episode():
    ex, df = FakeAccount(), frame(funding=0.0005)                      # positive funding: carry is short
    m = mirror(ex)
    ep = episode("p_carry", df)
    m.on_start(ep)
    held = ex.pos
    assert held < 0
    ep.finish(df.iloc[-1])
    assert ep.keep_open
    m.finish(ep)
    assert ex.pos == held and ("close",) not in ex.calls               # still paying: nothing closed
    nxt = PerpPaperEpisode("p_carry", PAPER, {}, inherit_pos=ep.pos)
    nxt.start(df.iloc[0])
    m.on_start(nxt)
    assert len(places(ex, "market")) == 1                              # the same position, no second entry


def test_a_position_left_on_chain_is_closed_when_the_new_episode_does_not_own_it():
    ex, df = FakeAccount(position=0.5), frame()                        # a restart: the fly forgot a carry
    m = mirror(ex)
    m.on_start(episode("flat", df))
    assert ex.pos == 0.0 and ("close",) in ex.calls


def test_a_restart_cancels_only_the_orders_this_mirror_placed(tmp_path):
    store = tmp_path / "open.json"
    ex, df = FakeAccount(), frame()
    m = mirror(ex)
    m.store = store
    m.on_start(episode("p_maker", df))                                 # two quotes, remembered on disk
    mine = set(json.loads(store.read_text())["orders"])
    assert mine == set(ex.resting)
    ex.resting["hand"] = ("buy", 90.0, 1.0)                            # the owner's own order in the same market
    again = mirror(ex)                                                 # the process restarted: a fresh mirror
    again.store = store
    again.on_start(episode("flat", df))
    assert set(ex.resting) == {"hand"}                                 # ours cancelled, the owner's left alone
    assert json.loads(store.read_text())["orders"] == []
    third = mirror(FakeAccount(resting=["x"]))                         # no store at all: nothing is touched
    third.on_start(episode("flat", df))
    assert third.executor.resting == {"x": ("buy", 1.0, 1.0)}


def test_a_close_that_could_not_be_sent_is_tried_again_on_the_next_row():
    class Outage(FakeAccount):
        down = True

        def close(self, symbol):
            if self.down:
                self.calls.append(("close",))
                return {"sent": False, "filled": 0.0, "left": self.pos, "reason": "market is un-priceable"}
            return super().close(symbol)

    ex, df = Outage(), frame()
    m = mirror(ex)
    ep = episode("p_trend", df, trend_dir=1)
    m.on_start(ep)
    ep.pos, ep.guarded = 0.0, True
    m.on_row(ep)
    assert ex.pos > 0 and m.last_target != 0.0                         # the close failed and is still owed
    ex.down = False
    m.on_row(ep)
    assert ex.pos == 0.0 and [c for c in ex.calls if c == ("close",)].__len__() == 2


def test_an_unreadable_position_sends_nothing_and_is_retried_on_the_next_row():
    ex, df = FakeAccount(readable=False), frame()
    m = mirror(ex)
    ep = episode("p_trend", df, trend_dir=1)
    m.on_start(ep)
    assert places(ex) == [] and m.last_target is None
    ex.readable = True
    m.on_row(ep)
    assert len(places(ex, "market")) == 1


def test_dust_differences_are_left_alone():
    ex, df = FakeAccount(), frame()
    m = mirror(ex)
    ep = episode("p_trend", df, trend_dir=1)
    m.on_start(ep)
    ex.pos -= 0.001                                                    # 0.1 USDso short of the target
    m.last_target = None
    m.on_row(ep)
    assert len(places(ex, "market")) == 1


def test_an_entry_is_never_larger_than_the_live_stake_allows():
    ex, df = FakeAccount(), frame()
    m = mirror(ex)
    ep = episode("p_trend", df, trend_dir=1)
    ep.pos = ep.pos * 10                                               # a paper episode gone wrong
    m.on_start(ep)
    cap = 20.0 * 2.0 * 1.05 / m._mark(ep)
    assert places(ex)[0][3] == pytest.approx(cap)


# ---------- maker: quotes are mirrored, fills are not chased ----------
def test_maker_quotes_rest_post_only_at_the_paper_prices_and_follow_big_moves_only():
    ex, df = FakeAccount(), frame()
    m = mirror(ex)
    ep = episode("p_maker", df, spread_bps=20, skew_bps=10, clip_frac=0.25)
    m.on_start(ep)
    q = places(ex, "post_only")
    assert [c[1] for c in q] == ["long", "short"]
    assert q[0][2] == pytest.approx(ep.bids[0], abs=1e-6) and q[1][2] == pytest.approx(ep.asks[0], abs=1e-6)
    assert q[0][3] == pytest.approx(ep.clip * m.scale)
    ep.bids = [ep.bids[0] * (1 + 0.0002)]                              # 2 bps: inside the requote tolerance
    m.on_row(ep)
    assert len(places(ex, "post_only")) == 2
    ep.bids = [ep.bids[0] * (1 + 0.002)]                               # 20 bps: cancel and replace the bid only
    m.on_row(ep)
    assert len(places(ex, "post_only")) == 3 and [c for c in ex.calls if c[0] == "cancel"] == [("cancel", "o1")]
    ex.pos = 0.1                                                       # a real fill came in on the way
    ep.finish(df.iloc[-1])
    m.finish(ep)
    assert ex.resting == {} and ("close",) in ex.calls and ex.pos == 0.0


def test_a_maker_starts_flat_even_when_the_account_is_not():
    ex, df = FakeAccount(position=-0.3), frame()
    m = mirror(ex)
    m.on_start(episode("p_maker", df))
    assert ex.pos == 0.0 and ("close",) in ex.calls and len(ex.resting) == 2


def test_a_guarded_maker_pulls_its_quotes_and_closes():
    ex, df = FakeAccount(), frame()
    m = mirror(ex)
    ep = episode("p_maker", df)
    m.on_start(ep)
    ex.pos = 0.2                                                       # a real fill came in
    ep.guarded = True
    m.on_row(ep)
    assert ex.resting == {} and ex.pos == 0.0


def test_a_quote_the_pool_refused_is_tried_again_on_the_next_row():
    class Refuses(FakeAccount):
        def place(self, symbol, side, price, qty, kind, lift_to=0.0):
            if kind == "post_only" and self._n == 0:
                self._n += 1
                self.calls.append(("place", side, round(price, 6), round(qty, 9), kind))
                return {"sent": False, "filled": 0.0, "no_fill": True, "reason": "post only would cross"}
            return super().place(symbol, side, price, qty, kind, lift_to)

    ex, df = Refuses(), frame()
    m = mirror(ex)
    ep = episode("p_maker", df)
    m.on_start(ep)
    assert m.quotes["buy"] is None and m.quotes["sell"] is not None
    m.on_row(ep)
    assert m.quotes["buy"] is not None


# ---------- dry run ----------
def test_the_dry_run_executor_logs_and_reads_flat():
    rows_ = []
    ex = DryRunPerpExecutor(lambda k, r: rows_.append((k, r)))
    r = ex.place("BTC/USDso:USDso", "long", 100.0, 0.1, "market")
    assert not r["sent"] and r["dry_run"] and ex.position("BTC/USDso:USDso") == 0.0 and ex.equity() is None
    assert rows_[0][1]["mode"] == "dry" and rows_[0][1]["action"] == "place"


# ---------- the ledger sees the perps account ----------
def test_the_ledger_counts_the_perps_equity_once_beside_the_spot_wallet():
    perps = FakeAccount()
    led = WalletLedger(None, perps=perps, clock=lambda: 1_790_000_000.0)
    assert led.day_pnl is None
    assert led.update_perps() == 0.0
    perps.eq = 88.0
    assert led.update_perps() == -12.0
    assert led.update_perps() == -12.0                                 # read again: not counted twice


# ---------- settings ----------
def test_perp_settings_are_validated():
    s = Settings.from_env({"NETWORK": "hideki", "OWNER_ADDRESS": "0x" + "ab" * 20, "PERP_LEVERAGE": "3", "PERP_MARKETS": "all",
                           "DRY_RUN": "false", "PRIVATE_KEY": "0xabc"})
    assert s.owner_address == "0x" + "ab" * 20 and s.perp_leverage == 3.0 and s.perp_live and s.markets == ()
    assert not Settings.from_env({"NETWORK": "hideki", "PERP_MARKETS": "all"}).perp_live     # a dry run sends nothing
    assert not Settings.from_env({"DRY_RUN": "false", "PRIVATE_KEY": "0xabc"}).perp_live   # no perp markets
    with pytest.raises(SettingsError, match="OWNER_ADDRESS"):
        Settings.from_env({"OWNER_ADDRESS": "0x1234"})
    with pytest.raises(SettingsError, match="PERP_LEVERAGE"):
        Settings.from_env({"PERP_LEVERAGE": "25"})


# ---------- the sidecar client ----------
FAKE_SIDECAR = textwrap.dedent("""
    import { createInterface } from "node:readline";
    const rl = createInterface({ input: process.stdin });
    const out = (o) => process.stdout.write(JSON.stringify(o) + "\\n");
    rl.on("line", (line) => {
      const r = JSON.parse(line);
      if (r.op === "init") out({ id: r.id, ok: true, address: "0xfeed", markets: [{ symbol: "BTC/USDso:USDso" }], linked: {} });
      else if (r.op === "slow") setTimeout(() => out({ id: r.id, ok: true }), 5000);
      else if (r.op === "die") process.exit(3);
      else if (r.op === "place") out({ id: r.id, ok: true, sent: true, order_id: "42", filled: 0, qty: r.qty, price: r.price });
      else if (r.op === "position") out({ id: r.id, ok: true, position: { side: "short", contracts: 0.25, entry: 1, mark: 1 } });
      else if (r.op === "shutdown") { out({ id: r.id, ok: true }); process.exit(0); }
      else out({ id: r.id, ok: false, error: "unknown op " + r.op });
    });
""")

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def fake_sidecar(tmp_path: Path) -> Path:
    (tmp_path / "node_modules").mkdir()
    script = tmp_path / "executor.mjs"
    script.write_text(FAKE_SIDECAR)
    return script


@needs_node
def test_the_client_starts_the_sidecar_and_reads_its_replies(tmp_path):
    rows_ = []
    ex = NodePerpExecutor(Settings(owner_address="0x" + "cd" * 20), lambda k, r: rows_.append(r), script=fake_sidecar(tmp_path))
    try:
        assert ex.address == "0xfeed" and ex.markets() == ["BTC/USDso:USDso"] and ex.argv[-2:] == ["--owner", "0x" + "cd" * 20]
        r = ex.place("BTC/USDso:USDso", "long", 100.0, 0.1, "market")
        assert r["order_id"] == "42" and rows_[-1]["action"] == "place" and rows_[-1]["ok"]
        assert ex.position("BTC/USDso:USDso") == -0.25
        with pytest.raises(PerpExecutionError, match="unknown op"):
            ex._call("nothing")
    finally:
        ex.stop()
    assert not ex.alive()


@needs_node
def test_a_dead_or_silent_sidecar_is_replaced_and_the_failure_is_logged(tmp_path):
    rows_ = []
    ex = NodePerpExecutor(Settings(), lambda k, r: rows_.append(r), script=fake_sidecar(tmp_path))
    try:
        with pytest.raises(PerpExecutionError, match="exited"):
            ex._call("die")
        assert not ex.alive()
        assert ex.position("BTC/USDso:USDso") == -0.25                 # a fresh sidecar answered
        assert ex.alive()
        with pytest.raises(PerpExecutionError, match="timed out"):
            ex._call("slow", timeout=0.5)
        r = ex.place("BTC/USDso:USDso", "long", 100.0, 0.1, "market")
        assert r["order_id"] == "42"
        assert any(x.get("action") == "sidecar" for x in rows_)
    finally:
        ex.stop()


def test_the_client_refuses_to_start_without_the_script_or_its_modules(tmp_path):
    with pytest.raises(PerpExecutionError, match="missing"):
        NodePerpExecutor(Settings(), lambda k, r: None, script=tmp_path / "executor.mjs")
    (tmp_path / "executor.mjs").write_text("")
    with pytest.raises(PerpExecutionError, match="node_modules"):
        NodePerpExecutor(Settings(), lambda k, r: None, script=tmp_path / "executor.mjs")


# ---------- the fly keeps going when the mirror fails ----------
MB_DIR = Path(__file__).resolve().parents[1] / "data" / "mb"


@pytest.mark.skipif(not (MB_DIR / "edges.parquet").exists(), reason="run scripts/extract_mb.py first")
def test_a_mirror_failure_is_logged_and_the_paper_fly_keeps_trading(tmp_path):
    from flybot.live.perp_live import PerpFly
    from flybot.mb.graph import load_graph

    class Broken:
        def on_start(self, ep):
            raise RuntimeError("sidecar down")

        def on_row(self, ep):
            raise RuntimeError("sidecar down")

        def finish(self, ep):
            raise RuntimeError("sidecar down")

    fly = PerpFly("BTC/USDso:USDso", load_graph(MB_DIR), tmp_path, PerpCosts(stake=1000.0), {}, seed=0, episode_min=30)
    fly.mirror = Broken()
    done = [out for r in rows(200) if (out := fly.on_row(r))]
    assert done                                                        # episodes still finish
    errors = [json.loads(l) for l in (fly.dir / "errors.jsonl").read_text().splitlines()]
    assert {e["where"] for e in errors} >= {"mirror.on_start", "mirror.on_row", "mirror.finish"}
