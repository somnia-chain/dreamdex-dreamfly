from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flybot.arms import Costs
from flybot.live.bench import HOUR_MS, ArmBench
from flybot.live.paper import Fill
from flybot.live.runner import MarketFly
from flybot.market import LONG
from flybot.mb.graph import load_graph

MB_DIR = Path(__file__).resolve().parents[1] / "data" / "mb"
needs_mb = pytest.mark.skipif(not (MB_DIR / "edges.parquet").exists(), reason="run scripts/extract_mb.py first")
LOSS = -0.5       # three of these lose 1.5 on a 1000 stake, past the 0.1 percent line


def bench(**kw):
    return ArmBench(stake=1000.0, **kw)


# ---------- the rule ----------
def test_three_traded_losses_in_a_row_bench_the_arm():
    b = bench()
    assert b.record("maker", LOSS, True, 0) is None
    assert b.record("maker", LOSS, True, 1) is None
    event = b.record("maker", LOSS, True, 2)
    assert event["arm"] == "maker" and event["hours"] == 12.0
    assert b.blocked(3) == {"maker"} and b.blocked(2 + 12 * HOUR_MS) == set()


def test_a_win_or_a_small_loss_keeps_the_arm_in_play():
    b = bench()
    for pnl in (LOSS, LOSS, 0.2, LOSS, LOSS):
        assert b.record("grid", pnl, True, 0) is None          # the win breaks the run
    assert b.blocked(1) == set()
    tiny = bench()
    for _ in range(5):
        assert tiny.record("grid", -0.01, True, 0) is None     # losses, but together under 0.1 percent of the stake
    assert tiny.blocked(1) == set()


def test_episodes_without_a_fill_and_flat_are_ignored():
    b = bench()
    for _ in range(6):
        assert b.record("maker", -0.2, False, 0) is None       # gas only, nothing traded
        assert b.record("flat", -9.0, True, 0) is None
    assert b.blocked(1) == set() and "maker" not in b.recent and "flat" not in b.recent


def test_a_lost_trial_doubles_the_bench_and_a_clean_one_clears_it():
    b = bench(hours=12.0, max_hours=168.0)
    for i in range(3):
        b.record("maker", LOSS, True, i)
    back = 2 + 12 * HOUR_MS
    assert b.blocked(back - 1) == {"maker"} and b.blocked(back) == set()
    b.note_start("maker", back)                                  # the bench ran out: this episode is the trial
    assert "maker" in b.trial and "maker" not in b.until
    again = b.record("maker", -3.0, True, back + 1)
    assert again["hours"] == 24.0 and again["strikes"] == 1 and b.blocked(back + 2) == {"maker"}

    b.until["maker"] = 0
    b.note_start("maker", back + 100)
    assert b.record("maker", 0.4, True, back + 200) is None      # the trial did not lose: clean again
    assert b.strikes == {} and b.blocked(back + 300) == set()


def test_the_bench_is_capped():
    b = bench(hours=100.0, max_hours=168.0)
    b.strikes["maker"] = 3
    assert b._bench("maker", 0, "x")["hours"] == 168.0


def test_it_survives_a_restart_and_can_be_switched_off():
    b = bench()
    for i in range(3):
        b.record("maker", LOSS, True, i)
    b.record("grid", LOSS, True, 5)
    b2 = bench()
    b2.load(b.to_dict())
    assert b2.blocked(3) == {"maker"} and b2.recent == {"grid": [LOSS]}
    assert ArmBench.from_params(False, 1000.0) is None
    assert ArmBench.from_params(None, 1000.0).after == 3
    assert ArmBench.from_params({"after": 2, "hours": 1}, 500.0).stake == 500.0


# ---------- inside the fly ----------
def candles(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    df = pd.DataFrame({"open": o, "close": c, "high": np.maximum(o, c) * 1.0005, "low": np.minimum(o, c) * 0.9995,
                       "volume": rng.lognormal(3, 0.5, n)})
    df["timestamp"] = 1_790_000_000_000 + np.arange(n) * 60_000
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


@needs_mb
def test_a_benched_arm_is_never_chosen_and_comes_back_after_its_bench(tmp_path):
    g = load_graph(MB_DIR)
    mf = MarketFly("WETH:USDso", g, tmp_path, Costs(stake=1000.0), {}, seed=0, episode_min=5)
    mf.candles = candles(LONG + 40)
    t0 = int(mf.candles["timestamp"].iloc[-1])
    mf.bench.until["maker"] = t0 + 12 * HOUR_MS
    seen = set()
    for n in range(80):
        seen.add(mf.decide(t0 + n, 99.99, 100.01, daily_stop_hit=False))
        mf.eps[0] = None
    assert "maker" not in seen and seen
    assert mf.snapshot()["benched"] == {"maker": t0 + 12 * HOUR_MS}

    later = t0 + 13 * HOUR_MS
    back = set()
    for n in range(80):
        back.add(mf.decide(later + n, 99.99, 100.01, daily_stop_hit=False))
        mf.eps[0] = None
    assert "maker" in back and "maker" in mf.bench.trial        # offered again: the pick is its trial


@needs_mb
def test_a_market_with_every_arm_benched_goes_flat(tmp_path):
    g = load_graph(MB_DIR)
    mf = MarketFly("WETH:USDso", g, tmp_path, Costs(stake=1000.0), {}, seed=0, episode_min=5)
    mf.candles = candles(LONG + 40)
    t0 = int(mf.candles["timestamp"].iloc[-1])
    for a in ("maker", "grid", "trend"):
        mf.bench.until[a] = t0 + HOUR_MS
    assert {mf.decide(t0 + n, 99.99, 100.01, daily_stop_hit=False) for n in range(20)} == {"flat"}


@needs_mb
def test_losing_episodes_bench_the_arm_and_the_bench_is_saved(tmp_path):
    g = load_graph(MB_DIR)
    params = {"arms": ["maker"]}
    mf = MarketFly("WETH:USDso", g, tmp_path, Costs(stake=1000.0), params, seed=0, episode_min=5)
    mf.candles = candles(LONG + 40)
    t0 = int(mf.candles["timestamp"].iloc[-1])
    n_maker = 0
    for i in range(60):
        ts = t0 + i * 60_000
        arm = mf.decide(ts, 99.99, 100.01, daily_stop_hit=False)
        ep = mf.eps[0]
        if arm == "maker":
            n_maker += 1
            ep.fills.append(Fill(ts, "buy", 100.0, 1.0, False))
            ep.finish = lambda *a: -2.0
        mf.finish(ts + 1000, 100.0, 100.02)
        if mf.bench.blocked(ts + 1000):
            break
    assert mf.bench.blocked(ts + 1000) == {"maker"} and n_maker == 3
    assert (tmp_path / "WETH_USDso" / "bench.jsonl").exists()

    again = MarketFly("WETH:USDso", g, tmp_path, Costs(stake=1000.0), params, seed=0, episode_min=5)
    assert again.bench.blocked(ts + 1000) == {"maker"}                 # remembered across a restart
    off = MarketFly("SOMI:USDso", g, tmp_path, Costs(stake=1000.0), {"bench": False}, seed=0, episode_min=5)
    assert off.bench is None and off.benched(0) == frozenset() and off.snapshot()["benched"] == {}
