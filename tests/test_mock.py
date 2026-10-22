from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flybot import selectors as S
from flybot.arms import ARM_NAMES
from flybot.market import FEATURES
from flybot.mb.graph import load_graph
from flybot.mb.model import MBModel
from flybot.mock import REGIMES, mock_candles
from flybot.pretrain import counterfactual

MB_DIR = Path(__file__).resolve().parents[1] / "data" / "mb"


def test_the_mock_market_is_repeatable_and_switches_through_every_regime():
    a, ra = mock_candles(20, 3)
    b, rb = mock_candles(20, 3)
    assert len(a) == 20 * 24 * 60 and a["close"].equals(b["close"]) and (ra == rb).all()
    assert set(ra) == set(REGIMES)
    assert (a["high"] >= a[["open", "close"]].max(axis=1)).all() and (a["low"] <= a[["open", "close"]].min(axis=1)).all()
    assert not a["close"].equals(mock_candles(20, 4)[0]["close"])
    changes = [i for i in range(1, len(ra)) if ra[i] != ra[i - 1]]
    assert len(changes) >= 3 and all(ra[i] != ra[i - 1] for i in changes)


@pytest.mark.skipif(not (MB_DIR / "edges.parquet").exists(), reason="run scripts/extract_mb.py first")
def test_counterfactual_pretraining_teaches_every_strategy_but_flat():
    g = load_graph(MB_DIR)
    fly = S.Fly(MBModel(g, seed=0), seed=0)
    w0 = fly.m.w_kc_mbon.copy()
    df, _ = mock_candles(8, 1)
    n = counterfactual(fly, df, 60)
    assert n > 100 and not np.allclose(fly.m.w_kc_mbon, w0)
    moved = [k for k, p in enumerate(fly.pools) if not np.allclose(fly.m.w_kc_mbon[p], w0[p])]
    assert ARM_NAMES.index("flat") not in moved and len(moved) >= 2


def _state(rng, vol, trend):
    row = {f: 0.0 for f in FEATURES}
    row.update(vol_short=vol * (1 + 0.05 * rng.normal()), vol_long=vol * 2, trend=trend + 0.05 * rng.normal(), vol_ratio=0.5 + vol)
    return pd.Series(row)


@pytest.mark.skipif(not (MB_DIR / "edges.parquet").exists(), reason="run scripts/extract_mb.py first")
def test_delta_fly_learns_which_strategy_pays_in_which_state_and_never_teaches_flat():
    g = load_graph(MB_DIR)
    rng = np.random.default_rng(0)
    fly = S.DeltaFly(MBModel(g, seed=0), seed=0, explore=0.0, margin=0.01)
    trend_i, grid_i, flat_i = ARM_NAMES.index("trend"), ARM_NAMES.index("grid"), ARM_NAMES.index("flat")
    calm, wild = (0.001, 2.0), (0.01, 0.0)             # a trending state and a choppy one
    for n in range(400):
        for vol, tr in (calm, wild):
            f = _state(rng, vol, tr)
            kc, _ = fly.m.sniff(fly.enc.encode(f))
            s = fly.scores(kc)
            for k, good in ((trend_i, (vol, tr) == calm), (grid_i, (vol, tr) == wild)):
                fly.last = (k, kc, float(s[k]))
                fly.update(1.0 if good else -1.0)
    assert not fly.w[flat_i].any()
    picks = {name: [ARM_NAMES[fly.choose(_state(rng, *st))] for _ in range(20)] for name, st in (("calm", calm), ("wild", wild))}
    assert picks["calm"].count("trend") >= 18 and picks["wild"].count("grid") >= 18
    only = fly.choose(_state(rng, *calm), exclude={trend_i, grid_i})
    assert ARM_NAMES[only] in ("maker", "flat")          # excluded strategies are not offered


@pytest.mark.skipif(not (MB_DIR / "edges.parquet").exists(), reason="run scripts/extract_mb.py first")
def test_a_live_delta_fly_decides_trains_saves_and_reloads(tmp_path):
    from flybot.arms import Costs
    from flybot.live.runner import MarketFly
    from flybot.market import LONG

    g = load_graph(MB_DIR)
    df, _ = mock_candles(6, 2)
    mf = MarketFly("WETH:USDso", g, tmp_path, Costs(stake=1000.0), {}, seed=0, episode_min=60, brain="delta")
    assert mf.fly.__class__.__name__ == "DeltaFly" and mf.snapshot()["brain"] == "delta"
    n = counterfactual(mf.fly, df, 60, mf.costs, mf.params, scaler=mf.scaler)
    assert n > 50 and mf.weights_moved() > 0 and mf.scaler.n > 0
    mf.candles = df.iloc[: LONG + 60]
    t0 = int(mf.candles["timestamp"].iloc[-1])
    arm = mf.decide(t0, 99.99, 100.01, daily_stop_hit=False)
    assert arm in ARM_NAMES
    mf.finish(t0 + 60_000, 100.0, 100.02)
    assert (tmp_path / "WETH_USDso" / "w_delta.npy").exists() and not (tmp_path / "WETH_USDso" / "w_kc_mbon.npy").exists()
    again = MarketFly("WETH:USDso", g, tmp_path, Costs(stake=1000.0), {}, seed=0, episode_min=60, brain="delta")
    assert np.allclose(again.fly.w, mf.fly.w) and again.n_episodes == 1
    only = MarketFly("USDC.e:USDso", g, tmp_path, Costs(), {"arms": ["maker"]}, seed=0, brain="delta")
    counterfactual(only.fly, df, 60, only.costs, only.params)
    assert not only.fly.w[ARM_NAMES.index("grid")].any() and only.fly.w[ARM_NAMES.index("maker")].any()
