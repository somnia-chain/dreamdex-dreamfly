from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flybot.arms import ARM_NAMES, Costs
from flybot.live.runner import MarketFly
from flybot.market import LONG
from flybot.mb.graph import load_graph

MB_DIR = Path(__file__).resolve().parents[1] / "data" / "mb"
needs_mb = pytest.mark.skipif(not (MB_DIR / "edges.parquet").exists(), reason="run scripts/extract_mb.py first")


def candles(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    df = pd.DataFrame({"open": o, "close": c, "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999,
                       "volume": rng.lognormal(3, 0.5, n)})
    df["timestamp"] = 1_790_000_000_000 + np.arange(n) * 60_000
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def run_one(tmp_path, params, brain="delta"):
    mf = MarketFly("WETH:USDso", load_graph(MB_DIR), tmp_path, Costs(stake=1000.0), params, seed=0, episode_min=30, brain=brain)
    mf.candles = candles(LONG + 60)
    mf.scaler.n, mf.scaler.scale = 100, 1e-4          # past the reward scaler's warm-up, so rewards are not zero
    start = int(mf.candles["timestamp"].iloc[LONG + 10])
    arm = mf.decide(start, 99.99, 100.01, daily_stop_hit=False)
    before = mf.fly.w.copy() if brain == "delta" else None
    row = mf.finish(int(mf.candles["timestamp"].iloc[-1]), 100.0, 100.02)
    return mf, arm, row, before


@needs_mb
def test_the_strategies_that_did_not_run_are_taught_from_the_same_window(tmp_path):
    mf, arm, row, before = run_one(tmp_path, {})
    assert row["taught"] == [a for a in ARM_NAMES if a not in (arm, "flat")]
    changed = [a for a in ARM_NAMES if a != "flat" and not np.allclose(mf.fly.w[ARM_NAMES.index(a)], before[ARM_NAMES.index(a)])]
    assert set(changed) == set(row["taught"]) | ({arm} - {"flat"})   # each taught pool moved, and so did the one that ran
    assert np.allclose(mf.fly.w[ARM_NAMES.index("flat")], 0.0)       # flat is worth zero and never taught


@needs_mb
def test_teaching_follows_the_market_arm_list_and_can_be_switched_off(tmp_path):
    _, arm, row, _ = run_one(tmp_path, {"arms": ["grid", "trend"]})
    assert set(row["taught"]) <= {"grid", "trend"} and "maker" not in row["taught"]
    _, _, off, _ = run_one(tmp_path / "off", {"teach_all": False})
    assert off["taught"] == []


@needs_mb
def test_the_original_brain_is_taught_too(tmp_path):
    _, arm, row, _ = run_one(tmp_path, {}, brain="mb")
    assert row["taught"] == [a for a in ARM_NAMES if a not in (arm, "flat")]
