import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flybot.arms import ARM_NAMES, Costs
from flybot.live.paper import PaperEpisode
from flybot.live.runner import MarketFly, Runner
from flybot.market import LONG
from flybot.mb.graph import load_graph

MB_DIR = Path(__file__).resolve().parents[1] / "data" / "mb"
pytestmark = pytest.mark.skipif(not (MB_DIR / "edges.parquet").exists(), reason="run scripts/extract_mb.py first")


def candles(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    df = pd.DataFrame({"open": o, "close": c, "high": np.maximum(o, c) * 1.0005, "low": np.minimum(o, c) * 0.9995,
                       "volume": rng.lognormal(3, 0.5, n)})
    df["timestamp"] = 1_790_000_000_000 + np.arange(n) * 60_000
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def test_market_fly_episode_roundtrip_and_persistence(tmp_path):
    g = load_graph(MB_DIR)
    mf = MarketFly("SOMI:USDso", g, tmp_path, Costs(stake=100.0), {"maker": {"spread_bps": 5}}, seed=0, episode_min=5)
    mf.candles = candles(LONG + 40)
    t0 = int(mf.candles["timestamp"].iloc[-1])
    arm = mf.decide(t0, 99.99, 100.01, daily_stop_hit=False)
    assert arm in ARM_NAMES and mf.episode is not None
    assert (tmp_path / "SOMI_USDso" / "decisions.jsonl").exists()
    d = mf.last_decision
    assert len(d["scores"]) == len(ARM_NAMES) and 0 < d["kc_frac"] < 0.2

    mf.episode.on_touch(t0 + 1000, 99.90, 99.92)
    mf.episode.on_touch(t0 + 2000, 100.10, 100.12)
    row = mf.finish(t0 + 5 * 60_000, 100.0, 100.02)
    assert row["arm"] == arm and mf.episode is None and mf.n_episodes == 1
    assert (tmp_path / "SOMI_USDso" / "episodes.jsonl").exists()
    assert (tmp_path / "SOMI_USDso" / "w_kc_mbon.npy").exists()

    # reload into a fresh object: weights, scaler and counters come back
    mf2 = MarketFly("SOMI:USDso", g, tmp_path, Costs(stake=100.0), {}, seed=0, episode_min=5)
    assert mf2.n_episodes == 1
    assert np.allclose(mf2.fly.m.w_kc_mbon, mf.fly.m.w_kc_mbon)
    assert mf2.scaler.n == mf.scaler.n
    snap = mf2.snapshot()
    assert snap["symbol"] == "SOMI:USDso" and snap["live"] is None


def test_daily_stop_forces_flat(tmp_path):
    g = load_graph(MB_DIR)
    mf = MarketFly("WETH:USDso", g, tmp_path, Costs(), {}, seed=1, episode_min=5)
    mf.candles = candles(LONG + 10, seed=3)
    t0 = int(mf.candles["timestamp"].iloc[-1])
    assert mf.decide(t0, 99.9, 100.1, daily_stop_hit=True) == "flat"
    assert mf.last_decision["guard_flat"] is True


def test_runner_mirrors_episodes_into_a_dry_run_order_log(tmp_path):
    from flybot.live.execution import DryRunExecutor
    from flybot.live.runner import Runner
    from flybot.live.settings import Settings

    r = Runner("testnet", ["SOMI:USDso"], tmp_path, 5, {"SOMI:USDso": {"maker": {"spread_bps": 5}}},
               Costs(stake=100.0), 50.0, 0, settings=Settings())
    m = r.mirrors["SOMI:USDso"]
    assert isinstance(m.executor, DryRunExecutor)
    f = r.flies["SOMI:USDso"]
    f.candles = candles(LONG + 40)
    t0 = int(f.candles["timestamp"].iloc[-1])
    f.fly.explore = 0.0
    arm = f.decide(t0, 99.99, 100.01, daily_stop_hit=False)
    if arm == "flat":  # the untouched brain may pick flat; force a maker episode for the check
        from flybot.live.paper import PaperEpisode
        f.episode = PaperEpisode("maker", f.costs, {"spread_bps": 5})
        f.episode.start(t0, 99.99, 100.01)
    m.on_start(f.episode, 99.99, 100.01)
    m.finish(f.episode, 99.99, 100.01)
    log = (tmp_path / "SOMI_USDso" / "orders.jsonl").read_text().splitlines()
    assert log and '"mode": "dry"' in log[0]


def test_runner_without_settings_has_no_mirrors(tmp_path):
    from flybot.live.runner import Runner
    r = Runner("testnet", ["SOMI:USDso"], tmp_path, 5, {}, Costs(stake=100.0), 50.0, 0)
    assert r.mirrors == {}


def test_paper_flies_have_no_daily_loss_stop(tmp_path):
    import math

    from flybot.live.runner import Runner
    from flybot.live.settings import Settings

    tapes = tmp_path / "tape"
    tapes.mkdir()
    r = Runner("testnet", ["SOMI:USDso"], tmp_path / "s", 5, {}, Costs(stake=100.0), Settings().daily_loss_stop(), 0,
               perp_tape=tapes, perp_symbols=["BTC/USDso:USDso"], ec_tape=tapes, settings=Settings())
    assert r.daily_loss_stop == math.inf
    assert r.ec.daily_loss_stop == math.inf
    assert all(p.daily_loss_stop == math.inf for p in r.perps.values())
    # a fly that is far down for the day is still allowed to decide
    ec = r.ec
    ec.day_pnl = -500.0
    assert not (ec.day_pnl <= -ec.daily_loss_stop)
    assert r.ledger is None                                     # a dry run has no wallet to watch


def test_event_contracts_follow_the_network_and_testnet_bets_stay_out_of_the_usdso_ledger(tmp_path, monkeypatch):
    from flybot.live import runner as runner_mod
    from flybot.live.ec_execution import DryRunECExecutor
    from flybot.live.runner import Runner
    from flybot.live.settings import Settings

    class Signer(DryRunECExecutor):
        dry = False                                              # stands in for the sidecar: nothing is started

    monkeypatch.setattr(runner_mod, "build_ec_executor", lambda s, log: Signer(log))
    tapes = tmp_path / "tape"
    tapes.mkdir()
    live = {"DRY_RUN": "false", "PRIVATE_KEY": "0x1", "EC_ENABLED": "true", "MARKETS": "none"}
    s = Settings.from_env({**live, "NETWORK": "testnet"})
    r = Runner("testnet", [], tmp_path / "t", 5, {}, Costs(stake=100.0), 50.0, 0, ec_tape=tapes, settings=s)
    assert r.flies == {} and r.ec_tail.prefix == "testnet_ec_"
    assert r.ledger is not None and r.ledger.ec_mirror is None   # tUSDC bets are not counted as USDso
    r.ec.day_pnl = -7.0
    assert r.ec.day_loss() == -7.0                               # the testnet EC stop watches the fly itself
    s = Settings.from_env({**live, "NETWORK": "mainnet", "MAINNET_CONFIRM": "yes"})
    r = Runner("mainnet", [], tmp_path / "m", 5, {}, Costs(stake=100.0), 50.0, 0, ec_tape=tapes, settings=s)
    assert r.ec_tail.prefix == "mainnet_ec_" and r.ledger.ec_mirror is r.ec.mirror


def test_hideki_spot_reads_books_and_candles_from_the_sdk_sidecar(tmp_path, monkeypatch):
    import asyncio

    from flybot.live import runner as runner_mod
    from flybot.live.runner import Runner
    from flybot.live.settings import Settings
    from flybot.live.spot_sdk import candles_frame

    class Feed:
        dry = True

        def __init__(self, settings, log):
            self.calls = []

        def book(self, symbol, depth=5):
            return {"symbol": symbol, "bids": [{"price": 0.18, "quantity": 5}], "asks": [{"price": 0.19, "quantity": 7}]}

        def candles(self, symbol, limit=300):
            self.calls.append((symbol, limit))
            return candles_frame([{"timestamp": 1_790_000_000_000, "open": 0.18, "high": 0.19, "low": 0.18,
                                   "close": 0.185, "volume": 3.0}])

        def stop(self):
            pass

    monkeypatch.setattr(runner_mod, "NodeSpotExecutor", Feed)
    s = Settings.from_env({"NETWORK": "hideki", "MARKETS": "STT:USDso"})
    r = Runner("hideki", ["STT:USDso"], tmp_path, 60, {}, Costs(stake=100.0), 50.0, 0, settings=s)
    assert isinstance(r.spot_feed, Feed) and r.mirrors["STT:USDso"].executor.dry   # dry run: orders are only logged

    async def one_poll():
        task = asyncio.create_task(r.poll_books())
        await asyncio.sleep(0.05)
        r.stop.set()
        await task
        await r._minute_for(r.flies["STT:USDso"], "STT:USDso", "", 1_790_000_600_000)

    asyncio.run(one_poll())
    assert r.touch["STT:USDso"] == (0.18, 0.19)
    f = r.flies["STT:USDso"]
    assert r.spot_feed.calls == [("STT:USDso", 1000)] and len(f.candles) >= LONG + 2   # the quiet minutes are filled


def test_with_live_orders_the_daily_stop_watches_the_wallet_not_the_paper(tmp_path):
    from flybot.live.execution import WalletLedger
    from flybot.live.runner import Runner
    from flybot.live.settings import Settings

    class Wallet:
        dry = False
        quote, base = 1000.0, 0.0

        def quote_balance(self, s):
            return self.quote

        def base_balance(self, s):
            return self.base

    r = Runner("testnet", ["SOMI:USDso"], tmp_path / "s", 5, {}, Costs(stake=100.0), 50.0, 0, settings=Settings())
    f = r.flies["SOMI:USDso"]
    w = Wallet()
    f.day_pnl = -80.0                                           # paper says a bad day
    assert r.day_loss(f) == -80.0                               # no wallet read yet: paper is all there is
    r.ledger = WalletLedger(w, clock=lambda: 1_790_000_000.0)
    r.ledger.update("SOMI:USDso", 0.99, 1.01)
    assert r.day_loss(f) == 0.0                                 # the wallet is level: no stop, whatever the paper says
    w.quote = 940.0
    r.ledger.update("SOMI:USDso", 0.99, 1.01)
    assert r.day_loss(f) == -60.0 and r.day_loss(f) <= -r.daily_loss_stop   # the wallet is down 60: stop


def test_concurrent_episodes_share_one_brain_and_cover_different_strategies(tmp_path):
    g = load_graph(MB_DIR)
    mf = MarketFly("WETH:USDso", g, tmp_path, Costs(stake=100.0), {}, seed=3, episode_min=60, slots=3)
    mf.candles = candles(LONG + 80, seed=5)
    mf.fly.explore = 0.6                                           # push the slots apart so the check is not luck
    t0 = int(mf.candles["timestamp"].iloc[-1])
    arms = [mf.decide(t0 + i * 20 * 60_000, 99.9, 100.1, False, i) for i in range(3)]
    assert len(mf.running()) == 3 and len(set(arms)) == 3          # no two slots run the same strategy
    assert mf.episode is mf.eps[0] and mf.last_start == t0 + 40 * 60_000
    rows = [mf.finish(t0 + 3_600_000 + i * 60_000, 100.0, 100.2, i) for i in (2, 0, 1)]
    assert [r["slot"] for r in rows] == [2, 0, 1] and not mf.running()
    assert mf.n_episodes == 3                                       # every slot's result reached the same fly
    assert mf.snapshot()["parallel"] == 0


def test_the_runner_staggers_slots_through_the_episode(tmp_path, monkeypatch):
    import asyncio

    import flybot.live.runner as rn

    r = rn.Runner("testnet", ["WETH:USDso"], tmp_path, 60, {}, Costs(stake=100.0), float("inf"), 0, slots=3)
    df = candles(LONG + 200, seed=2)
    monkeypatch.setattr(rn, "fetch_candles", lambda base, sym, n: df)
    r.touch["WETH:USDso"] = (99.9, 100.1)
    f = r.flies["WETH:USDso"]
    t = int(df["timestamp"].iloc[-1])
    started = []
    for minute in range(0, 61):
        asyncio.run(r._minute_for(f, "WETH:USDso", "x", t + minute * 60_000))
        started.append(len(f.running()))
    # one at once at the start, then a new slot every 20 minutes, three running by minute 40
    assert started[0] == 1 and started[19] == 1 and started[20] == 2 and started[40] == 3


def test_a_quiet_market_still_starts_rounds(tmp_path, monkeypatch):
    """The candle API returns only minutes that traded: 40 traded minutes must still give the fly a full series."""
    import asyncio

    import flybot.live.runner as rn

    r = rn.Runner("testnet", ["WETH:USDso"], tmp_path, 60, {}, Costs(stake=100.0), float("inf"), 0, slots=1)
    df = candles(LONG + 200, seed=3).iloc[::25].reset_index(drop=True)     # one traded minute in 25
    assert len(df) < LONG + 2
    monkeypatch.setattr(rn, "fetch_candles", lambda base, sym, n: df)
    r.touch["WETH:USDso"] = (99.9, 100.1)
    f = r.flies["WETH:USDso"]
    asyncio.run(r._minute_for(f, "WETH:USDso", "x", int(df["timestamp"].iloc[-1]) + 60_000))
    assert len(f.candles) >= LONG + 2 and len(f.running()) == 1


STABLE = {"arms": ["maker"], "slots": 1, "slip_bps": 2.0,
          "maker": {"mode": "inside", "tick": 0.0001, "improve_ticks": 1, "trade_fills": True, "stake": 250.0, "clip_frac": 1.0}}


def test_a_market_only_ever_picks_its_allowed_arms(tmp_path):
    g = load_graph(MB_DIR)
    mf = MarketFly("USDC.e:USDso", g, tmp_path, Costs(), STABLE, seed=0, episode_min=5)
    assert {ARM_NAMES[i] for i in mf.disabled} == {"grid", "trend"}      # flat is always allowed
    mf.candles = candles(LONG + 40)
    t0 = int(mf.candles["timestamp"].iloc[-1])
    seen = set()
    for n in range(60):
        seen.add(mf.decide(t0 + n, 0.9998, 1.0002, daily_stop_hit=False))
        mf.eps[0] = None
    assert seen <= {"maker", "flat"} and seen


def test_slots_are_capped_by_the_allowed_arms_and_other_markets_are_unchanged(tmp_path):
    g = load_graph(MB_DIR)
    assert MarketFly("USDC.e:USDso", g, tmp_path, Costs(), {"arms": ["maker"]}, slots=3).slots == 2     # maker and flat
    assert MarketFly("USDC.e:USDso", g, tmp_path, Costs(), STABLE, slots=3).slots == 1                   # its own "slots"
    plain = MarketFly("SOMI:USDso", g, tmp_path, Costs(), {}, slots=3)
    assert plain.slots == 3 and not plain.disabled


def test_a_market_can_run_its_own_episode_length(tmp_path):
    g = load_graph(MB_DIR)
    assert MarketFly("USDC.e:USDso", g, tmp_path, Costs(), {"episode_min": 240}, episode_min=60).episode_min == 240
    assert MarketFly("SOMI:USDso", g, tmp_path, Costs(), {}, episode_min=60).episode_min == 60


def test_a_market_can_carry_its_own_measured_cost(tmp_path):
    g = load_graph(MB_DIR)
    base = Costs()
    mf = MarketFly("USDC.e:USDso", g, tmp_path, base, STABLE)
    assert mf.costs.slip_bps == 2.0 and base.slip_bps == 3.0 and mf.costs.stake == base.stake
    assert MarketFly("SOMI:USDso", g, tmp_path, base, {}).costs.slip_bps == 3.0


def test_trades_reach_the_running_episodes_of_their_market(tmp_path):
    params = {"USDC.e:USDso": STABLE}
    r = Runner("mainnet", ["USDC.e:USDso", "SOMI:USDso"], tmp_path, 60, params, Costs(), math.inf, 0)
    assert r.trade_syms == ["USDC.e:USDso"]                       # only the market that fills from prints subscribes
    f = r.flies["USDC.e:USDso"]
    f.eps[0] = PaperEpisode("maker", f.costs, STABLE["maker"])
    f.eps[0].start(0, 0.9998, 1.0002)
    sell = {"symbol": "USDC.e:USDso", "price": "0.9998", "quantity": "8.8", "side": "sell", "timestamp": 5}
    r.on_trades({"channel": "trades", "type": "snapshot", "trades": [sell]})
    assert not f.eps[0].fills                                      # the snapshot on subscribing is history
    r.on_trades({"channel": "trades", "type": "update", "trades": [sell]})
    assert len(f.eps[0].fills) == 1 and f.eps[0].inv == pytest.approx(8.8)
    live = {"channel": "trades", "type": "update", "symbol": "USDC.e:USDso",          # the real message: symbol on the message, one trade
            "trade": {"id": "1791023332406-0", "price": "0.9998", "quantity": "2", "side": "sell", "timestamp": 6}}
    r.on_trades(live)
    assert len(f.eps[0].fills) == 2 and f.eps[0].inv == pytest.approx(10.8)
    r.on_trades({**live, "symbol": "SOMI:USDso"})                  # another market, no episode running
    r.on_trades({"channel": "trades", "type": "update", "trades": [{"symbol": "NOPE:USDso", "price": "1", "quantity": "1", "side": "buy"}]})
    r.on_trades({"channel": "trades", "type": "update", "trades": [{"symbol": "USDC.e:USDso", "price": None}]})
    assert len(f.eps[0].fills) == 2


def test_densify_fills_the_quiet_minutes_with_the_last_close():
    from flybot.live.runner import densify
    end = 1_790_000_000_000 // 60_000 * 60_000
    sparse = pd.DataFrame({"timestamp": [end - 50 * 60_000, end - 10 * 60_000], "open": [1.0001, 0.9999], "high": [1.0002, 1.0000],
                           "low": [1.0000, 0.9998], "close": [1.0002, 0.9999], "volume": [5.0, 3.0]})
    d = densify(sparse, end + 20_000, 60, fallback=1.0)
    assert len(d) == 60 and d["timestamp"].iloc[-1] == end and d["timestamp"].is_monotonic_increasing
    assert d["close"].iloc[-1] == 0.9999 and d["close"].iloc[-30] == 1.0002          # the last known close is carried through
    quiet = d[d["volume"] == 0]
    assert len(quiet) == 58 and (quiet["high"] == quiet["low"]).sum() >= 50 and not d.isna().any().any()
    real = d[d["volume"] > 0]
    assert list(real["close"]) == [1.0002, 0.9999] and list(real["low"]) == [1.0000, 0.9998]       # real candles keep their range
    assert d["close"].iloc[0] == 1.0002                                                                # before the first candle: its close


def test_densify_with_no_candles_uses_the_fallback_price():
    from flybot.live.runner import densify
    d = densify(pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]), 1_790_000_000_000, 5, fallback=0.9999)
    assert len(d) == 5 and set(d["close"]) == {0.9999} and (d["volume"] == 0).all()


def test_perp_symbols_all_makes_one_fly_per_market_on_the_tape(tmp_path):
    import json

    from flybot.live.settings import Settings
    from tests.test_perp import rows

    tapes = tmp_path / "tape"
    tapes.mkdir()
    lines = []
    for sym in ("BTC/USDso:USDso", "ZRO/USDso:USDso", "PUMP/USDso:USDso"):
        for r in rows(3):
            lines.append(json.dumps({**r, "symbol": sym}))
    (tapes / "hideki_perp_20261018.jsonl").write_text("\n".join(lines) + "\n")
    r = Runner("testnet", [], tmp_path / "s", 5, {}, Costs(stake=100.0), Settings().daily_loss_stop(), 0,
               perp_tape=tapes, perp_symbols=["all"], settings=Settings(), perp_prefix="hideki_perp_")
    assert r.perps == {}
    r.feed_tapes()
    assert sorted(r.perps) == ["BTC/USDso:USDso", "PUMP/USDso:USDso", "ZRO/USDso:USDso"]
    assert len({p.fly.rng.bit_generator.state["state"]["state"] for p in r.perps.values()}) == 3   # their own seeds
    named = Runner("testnet", [], tmp_path / "s2", 5, {}, Costs(stake=100.0), Settings().daily_loss_stop(), 0,
                   perp_tape=tapes, perp_symbols=["BTC/USDso:USDso"], settings=Settings(), perp_prefix="hideki_perp_")
    named.feed_tapes()
    assert list(named.perps) == ["BTC/USDso:USDso"]
    # in a dry run every perp fly gets a mirror that only logs, no sidecar, and no daily stop
    assert r.perp_exec is not None and r.perp_exec.dry and named.perp_exec.dry
    assert all(p.mirror is not None and p.mirror.executor is r.perp_exec for p in r.perps.values())
    assert all(p.daily_loss_stop == math.inf for p in list(r.perps.values()) + list(named.perps.values()))
    assert r.ledger is None


def test_perp_flies_on_the_delta_brain_start_from_the_bundled_perps_brain(tmp_path, monkeypatch):
    import json

    import flybot.live.runner as runner_mod
    from flybot.live.perp_live import PerpFly
    from flybot.live.settings import Settings
    from flybot.perp import PerpCosts
    from tests.test_perp import rows

    trained = PerpFly("_brain", load_graph(Path(runner_mod.ROOT) / "data" / "mb"), tmp_path / "src", PerpCosts(), {},
                      brain="delta")
    trained.fly.w[:] = 0.25
    trained.scaler.n, trained.scaler.scale = 500, 0.5
    trained.save()
    monkeypatch.setattr(runner_mod, "PERP_WARM", trained.dir)
    tapes = tmp_path / "tape"
    tapes.mkdir()
    (tapes / "hideki_perp_20261112.jsonl").write_text(
        "\n".join(json.dumps({**r, "symbol": s}) for s in ("BTC/USDso:USDso", "ETH/USDso:USDso") for r in rows(3)) + "\n")
    old = tmp_path / "s" / "ETH_USDso_USDso"      # a market that ran on the other brain keeps its books
    old.mkdir(parents=True)
    (old / "state.json").write_text(json.dumps({"scaler": {"n": 3, "scale": 1.0}, "equity": 42.0, "day": "", "day_pnl": 0.0}))

    def make(state, brain):
        r = Runner("testnet", [], state, 5, {}, Costs(stake=100.0), Settings().daily_loss_stop(), 0, perp_tape=tapes,
                   perp_symbols=["all"], settings=Settings(), perp_prefix="hideki_perp_", brain=brain)
        r.feed_tapes()
        return r.perps

    p = make(tmp_path / "s", "delta")
    btc, eth = p["BTC/USDso:USDso"], p["ETH/USDso:USDso"]
    assert btc.brain == "delta" and (btc.fly.w == 0.25).all() and btc.scaler.n == 500
    assert (eth.fly.w == 0.25).all() and eth.equity == 42.0 and eth.scaler.n == 3
    btc.fly.w[:] = 0.75                           # learned since: a restart must not overwrite it
    btc.save()
    assert (make(tmp_path / "s", "delta")["BTC/USDso:USDso"].fly.w == 0.75).all()
    mb = make(tmp_path / "s_mb", "mb")["BTC/USDso:USDso"]
    assert mb.brain == "mb" and not (mb.dir / "w_delta.npy").exists()
    off = Runner("testnet", [], tmp_path / "s3", 5, {}, Costs(stake=100.0), Settings().daily_loss_stop(), 0,
                 perp_tape=tapes, perp_symbols=["all"], settings=None, perp_prefix="hideki_perp_")
    off.feed_tapes()
    assert off.perp_exec is None and all(p.mirror is None for p in off.perps.values())
