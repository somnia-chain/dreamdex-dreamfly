import json
import shutil
import textwrap
from pathlib import Path

import pytest

from flybot.ec import ECCosts
from flybot.live.ec_execution import (CLAIM_EVERY_S, GIVE_UP_AFTER_S, MAX_TRIES, DryRunECExecutor, ECExecutionError,
                                      ECMirror, NodeECExecutor)
from flybot.live.execution import WalletLedger
from flybot.live.settings import Settings

MB_DIR = Path(__file__).resolve().parents[1] / "data" / "mb"
needs_mb = pytest.mark.skipif(not (MB_DIR / "edges.parquet").exists(), reason="run scripts/extract_mb.py first")
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


class Window:
    def __init__(self, market_id="0x7c21", symbol="BTC-0-29SEP26-1500/USDso"):
        self.market_id, self.symbol = market_id, symbol


class FakeVenue:
    """A venue that fills a buy fully at the cap (or not at all), settles on command and pays when asked."""

    dry = False

    def __init__(self, fill=True):
        self.fill, self.calls, self.settled, self.paid, self.usdso = fill, [], {}, set(), 1000.0

    def market(self, market_id):
        return {"tradable": True}

    def buy(self, market_id, side, cap, notional, ttl_s):
        self.calls.append(("buy", market_id, side, cap, notional))
        if not self.fill:
            return {"sent": True, "filled": 0.0, "cost": 0.0, "no_fill": True}
        units = notional / cap
        self.usdso -= notional
        return {"sent": True, "filled": units, "cost": notional, "avg_price": cap, "tx": "0xab"}

    def position(self, market_id):
        return {"up": 0.0, "down": 0.0}

    def claim(self, market_id):
        self.calls.append(("claim", market_id))
        state = self.settled.get(market_id)
        if state is None:
            return {"claimed": False, "redeemed": [], "reason": "not settled yet (status 2)"}
        if state == "lost":
            return {"claimed": False, "redeemed": [], "reason": "nothing to claim"}
        self.paid.add(market_id)
        self.usdso += state
        return {"claimed": True, "redeemed": [{"side": "up", "amount": 1.0}], "tx": "0xcd"}

    def sweep(self):
        self.calls.append(("sweep",))
        return {"claimed": 0, "markets": 0}

    def wallet(self):
        return {"address": "0xme", "somi": 1.0, "usdso": self.usdso}

    def wallet_usdso(self):
        return self.usdso


def mirror(venue, tmp_path, **kw):
    rows = []
    s = Settings(**{"live_stake_usdso": 10.0, **kw})
    m = ECMirror(venue, s, lambda k, r: rows.append((k, r)), store=tmp_path / "ec_bets.json")
    m.rows = rows
    return m


# ---------- one bet per window ----------
def test_one_filled_bet_per_window_then_nothing_more(tmp_path):
    v = FakeVenue()
    m = mirror(v, tmp_path)
    w = Window()
    m.bet(w, "up", 0.95, 20.0)
    m.bet(w, "up", 0.95, 18.0)          # the paper order is still pending on the next sample
    m.bet(w, "up", 0.96, 15.0)
    buys = [c for c in v.calls if c[0] == "buy"]
    assert len(buys) == 1 and buys[0][2:] == ("up", 0.95, 10.0)
    b = m.bets[w.market_id]
    assert b["filled"] == pytest.approx(10 / 0.95) and b["cost"] == 10.0 and b["price"] == pytest.approx(0.95)
    assert m.open_cost() == 10.0


def test_a_no_fill_is_retried_a_few_times_then_left(tmp_path):
    v = FakeVenue(fill=False)
    m = mirror(v, tmp_path)
    w = Window()
    for left in (20.0, 18.0, 15.0, 12.0, 10.0):
        m.bet(w, "down", 0.9, left)
    assert len([c for c in v.calls if c[0] == "buy"]) == MAX_TRIES
    assert m.bets[w.market_id]["filled"] == 0.0 and m.open_cost() == 0.0


# ---------- settle and claim ----------
def test_a_settled_bet_is_claimed_and_a_lost_one_is_let_go(tmp_path):
    v = FakeVenue()
    m = mirror(v, tmp_path)
    won, lost = Window("0x01", "A"), Window("0x02", "B")
    m.bet(won, "up", 0.9, 20.0)
    m.bet(lost, "down", 0.8, 20.0)
    v.settled["0x01"] = 11.0                       # the venue pays the winner
    m.settle(won, {"truth": "up", "voided": False})
    assert "0x01" in v.paid and "0x01" not in m.unclaimed and m.open_cost() == 10.0
    v.settled["0x02"] = "lost"
    m.settle(lost, {"truth": "up", "voided": False})
    assert "0x02" not in m.unclaimed and m.open_cost() == 0.0 and not m.bets


def test_an_unsettled_window_is_claimed_later_from_the_tape_loop(tmp_path):
    v = FakeVenue()
    t = [1000.0]
    m = mirror(v, tmp_path)
    m.clock = lambda: t[0]
    w = Window()
    m.bet(w, "up", 0.9, 20.0)
    m.settle(w, {"truth": "up", "voided": False})   # the venue has not finalised yet
    assert w.market_id in m.unclaimed
    m.claim_due()                                   # too soon: nothing happens
    assert len([c for c in v.calls if c[0] == "claim"]) == 1
    v.settled[w.market_id] = 11.0
    t[0] += CLAIM_EVERY_S + 1
    m.claim_due()
    assert w.market_id in v.paid and not m.unclaimed


def test_a_window_that_never_becomes_claimable_is_dropped_with_a_log_row(tmp_path):
    v = FakeVenue()
    t = [1000.0]
    m = mirror(v, tmp_path)
    m.clock = lambda: t[0]
    w = Window()
    m.bet(w, "up", 0.9, 20.0)
    m.settle(w, {"truth": None, "voided": False})
    t[0] += GIVE_UP_AFTER_S + CLAIM_EVERY_S + 1
    m.claim_due()
    assert not m.unclaimed and any(r.get("action") == "claim_given_up" for _, r in m.rows)


def test_bets_and_claims_survive_a_restart_and_the_restart_sweeps(tmp_path):
    v = FakeVenue()
    m = mirror(v, tmp_path)
    a, b = Window("0x01", "A"), Window("0x02", "B")
    m.bet(a, "up", 0.9, 20.0)
    m.bet(b, "up", 0.9, 20.0)
    m.settle(b, {"truth": "up", "voided": False})   # not claimable yet
    again = mirror(v, tmp_path)
    assert set(again.bets) == {"0x01"} and set(again.unclaimed) == {"0x02"} and again.open_cost() == 20.0
    again.on_start()
    assert ("sweep",) in v.calls
    assert json.loads((tmp_path / "ec_bets.json").read_text())["unclaimed"]["0x02"]["cost"] == 10.0


# ---------- the wallet ledger ----------
def test_the_ledger_counts_open_bets_at_cost_and_reads_usdso_through_the_ec_sidecar(tmp_path):
    v = FakeVenue()
    m = mirror(v, tmp_path)
    led = WalletLedger(None, ec_mirror=m)
    led.update_ec()
    assert led.day_pnl == 0.0
    w = Window()
    m.bet(w, "up", 0.9, 20.0)                        # 10 USDso leaves the wallet and sits in the bet
    led.update_ec()
    assert led.day_pnl == pytest.approx(0.0)
    v.settled[w.market_id] = "lost"
    m.settle(w, {"truth": "down", "voided": False})
    led.update_ec()
    assert led.day_pnl == pytest.approx(-10.0)
    v.usdso += 11.0                                  # a win paid back by a later claim shows as profit
    led.update_ec()
    assert led.day_pnl == pytest.approx(1.0)


# ---------- the fly drives the mirror ----------
@needs_mb
def test_the_paper_order_triggers_the_real_bet_and_the_settle_hands_it_on(tmp_path):
    from flybot.live.ec_live import ECFly
    from flybot.mb.graph import load_graph
    from tests.test_ec_live import MID, book

    ec = ECFly(load_graph(MB_DIR), tmp_path, ECCosts(stake=20.0), {}, seed=0)
    ec.scaler.n, ec.scaler.scale = 25, 1.0
    v = FakeVenue()
    ec.mirror = ECMirror(v, Settings(live_stake_usdso=5.0), ec.log, store=tmp_path / "EC" / "ec_bets.json")
    ec.on_row(book(24, up_ask=0.93, prob_up=0.97, z=2.0))
    w = ec.windows[MID]
    w.arm = "ec_sharp"
    ec.on_row(book(21, up_ask=0.93, prob_up=0.97, z=2.0))        # the paper order goes out: so does the real one
    assert [c for c in v.calls if c[0] == "buy"] == [("buy", MID, "up", 0.95, 5.0)]
    ec.on_row(book(18, up_ask=0.94, prob_up=0.97, z=2.0))        # paper fill; the real bet is already placed
    assert len([c for c in v.calls if c[0] == "buy"]) == 1
    v.settled[MID] = 5.5
    ec.on_row({"type": "ec_resolution", "marketId": MID, "symbol": w.symbol, "truth": "up", "voided": False,
               "wall": "2026-09-29T15:00:32.000Z"})
    assert MID in v.paid and not ec.mirror.unclaimed


@needs_mb
def test_the_guard_uses_the_wallet_when_live_and_a_broken_mirror_does_not_stop_the_fly(tmp_path):
    from flybot.live.ec_live import ECFly
    from flybot.mb.graph import load_graph
    from tests.test_ec_live import MID, book

    ec = ECFly(load_graph(MB_DIR), tmp_path, ECCosts(stake=20.0), {}, seed=0, daily_loss_stop=50.0)
    ec.day_loss = lambda: -60.0
    ec.on_row(book(24, up_ask=0.93, prob_up=0.97, z=2.0))
    assert ec.windows[MID].arm == "flat" and ec.windows[MID].decided_at["guard_flat"]

    class Broken:
        dry = False

        def bet(self, *a):
            raise RuntimeError("venue down")

    ec2 = ECFly(load_graph(MB_DIR), tmp_path / "b", ECCosts(stake=20.0), {}, seed=0)
    ec2.mirror = Broken()
    ec2.on_row(book(24, up_ask=0.93, prob_up=0.97, z=2.0))
    ec2.windows[MID].arm = "ec_sharp"
    ec2.on_row(book(21, up_ask=0.93, prob_up=0.97, z=2.0))
    assert ec2.windows[MID].pending == ("up", 0.95)
    assert any(json.loads(l)["where"] == "mirror.bet" for l in (tmp_path / "b" / "EC" / "errors.jsonl").read_text().splitlines())


# ---------- dry run, settings ----------
def test_dry_run_logs_the_bet_and_holds_nothing(tmp_path):
    rows = []
    ex = DryRunECExecutor(lambda k, r: rows.append(r))
    m = ECMirror(ex, Settings(), lambda k, r: None, store=tmp_path / "s.json")
    m.bet(Window(), "up", 0.9, 20.0)
    assert rows[-1]["mode"] == "dry" and rows[-1]["action"] == "buy" and m.open_cost() == 0.0
    m.settle(Window(), {"truth": "up"})
    assert not m.unclaimed


def test_event_contract_bets_are_live_on_mainnet_only():
    live = {"DRY_RUN": "false", "PRIVATE_KEY": "0x1", "EC_ENABLED": "true"}
    assert Settings.from_env({**live, "NETWORK": "mainnet", "MAINNET_CONFIRM": "yes"}).ec_live
    assert not Settings.from_env({**live, "NETWORK": "testnet"}).ec_live          # the windows are on mainnet
    assert not Settings.from_env({"NETWORK": "mainnet", "EC_ENABLED": "true"}).ec_live    # dry run
    assert not Settings.from_env({**live, "NETWORK": "mainnet", "MAINNET_CONFIRM": "yes", "EC_ENABLED": "false"}).ec_live


# ---------- the sidecar client ----------
FAKE = textwrap.dedent("""
    import { createInterface } from "node:readline";
    const rl = createInterface({ input: process.stdin });
    const out = (o) => process.stdout.write(JSON.stringify(o) + "\\n");
    rl.on("line", (line) => {
      const r = JSON.parse(line);
      if (r.op === "init") out({ id: r.id, ok: true, address: "0xfeed", markets: 6, collateral: "0xusd" });
      else if (r.op === "buy") out({ id: r.id, ok: true, sent: true, filled: r.notional / r.cap, cost: r.notional, avg_price: r.cap, tx: "0x1" });
      else if (r.op === "claim") out({ id: r.id, ok: true, claimed: true, redeemed: [{ side: "up", amount: 2 }], tx: "0x2" });
      else if (r.op === "wallet") out({ id: r.id, ok: true, address: "0xfeed", somi: 1, usdso: 42.5 });
      else if (r.op === "shutdown") { out({ id: r.id, ok: true }); process.exit(0); }
      else out({ id: r.id, ok: false, error: "unknown op " + r.op });
    });
""")


@needs_node
def test_the_ec_client_talks_to_its_sidecar(tmp_path):
    (tmp_path / "node_modules").mkdir()
    script = tmp_path / "ec_executor.mjs"
    script.write_text(FAKE)
    rows = []
    ex = NodeECExecutor(Settings(network="mainnet"), lambda k, r: rows.append(r), script=script)
    try:
        assert ex.address == "0xfeed" and ex.argv[-2:] == ["--network", "mainnet"]
        r = ex.buy("0x7c21", "up", 0.95, 10.0, 20.0)
        assert r["filled"] == pytest.approx(10 / 0.95) and rows[-1]["action"] == "buy" and rows[-1]["ok"]
        assert ex.claim("0x7c21")["claimed"] and ex.wallet_usdso() == 42.5
        with pytest.raises(ECExecutionError, match="unknown op"):
            ex._call("nothing")
        assert ex.market("0x7c21") is None and rows[-1]["ok"] is False      # a refused read is logged, not raised
    finally:
        ex.stop()
    assert not ex.alive()
