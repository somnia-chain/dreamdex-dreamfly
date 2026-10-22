import numpy as np
import pandas as pd

from flybot import arms, market, sim


def candles(closes, spread=0.001, start="2026-09-01"):
    c = np.asarray(closes, dtype=float)
    o = np.concatenate([[c[0]], c[:-1]])
    df = pd.DataFrame({
        "open": o, "close": c,
        "high": np.maximum(o, c) * (1 + spread), "low": np.minimum(o, c) * (1 - spread),
        "volume": np.ones_like(c),
    })
    df["timestamp"] = (pd.Timestamp(start, tz="UTC").value // 10**6) + np.arange(len(c)) * 60_000
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def test_flat_arm_never_trades():
    df = candles([100, 101, 102])
    r = arms.flat(df, df, arms.Costs(), {})
    assert r.pnl == 0.0 and r.trades == 0


def test_trend_arm_profits_on_a_continuing_move():
    hist = candles(np.linspace(100, 110, 61))
    ep = candles(np.linspace(110, 120, 10))
    r = arms.trend(ep, hist, arms.Costs(), {"lookback": 60, "threshold": 0.01})
    assert r.trades == 2
    assert r.pnl > 0

    ep_rev = candles(np.linspace(110, 100, 10))
    assert arms.trend(ep_rev, hist, arms.Costs(), {"lookback": 60, "threshold": 0.01}).pnl < 0


def test_trend_arm_sits_out_below_threshold():
    hist = candles([100.0] * 61)
    ep = candles(np.linspace(100, 101, 10))
    assert arms.trend(ep, hist, arms.Costs(), {}).trades == 0


def test_maker_earns_spread_in_a_flat_choppy_market():
    ep = candles([100, 100.5, 100, 100.5, 100] * 6, spread=0.004)
    r = arms.maker(ep, ep, arms.Costs(), {"spread_bps": 20})
    assert r.trades > 0
    assert r.pnl > 0


def test_grid_fills_levels_and_flattens():
    ep = candles([100, 99, 98, 99, 100, 101, 102, 101, 100], spread=0.0)
    r = arms.grid(ep, ep, arms.Costs(), {"levels": 2, "step_bps": 50})
    assert r.trades > 0


def test_reward_is_vol_scaled_and_clipped():
    ep = candles(np.linspace(100, 101, 15))
    c = arms.Costs()
    assert sim.reward_of(0.0, ep, c) == 0.0
    assert sim.reward_of(1e9, ep, c) == 3.0
    assert sim.reward_of(-1e9, ep, c) == -3.0
    assert sim.reward_of(1.0, ep, c) > 0


def test_reward_scaler_normalises_to_unit_scale_and_clips():
    rng = np.random.default_rng(0)
    sc = sim.RewardScaler(warm=20)
    xs = rng.normal(0, 0.04, 500)  # tiny raw rewards, like real episodes
    out = np.array([sc(x) for x in xs])
    assert np.all(out[:20] == 0.0)
    assert 0.5 < np.abs(out[20:]).mean() < 1.5
    assert sc(100.0) == 3.0 and sc(-100.0) == -3.0


def test_reward_ewma_smooths_per_arm():
    """With reward_ewma on, the delivered reward is a per-arm running average, so two episodes of the
    same arm with opposite raw rewards deliver a smaller second reward than the raw one."""
    from flybot import selectors as S

    df = candles(np.cumsum(np.random.default_rng(1).normal(0, 0.002, 2000)) + 100)
    seen: list[float] = []

    class Spy(S.Single):
        def update(self, r):
            seen.append(r)

    sim.run(df, Spy("trend"), 30, adaptive_reward=False, reward_ewma=0.5)
    raw = [e.reward for e in sim.run(df, S.Single("trend"), 30, adaptive_reward=False).episodes]
    assert len(seen) == len(raw)
    assert np.abs(seen).mean() <= np.abs(raw).mean() + 1e-9


def test_features_use_no_lookahead():
    df = candles(np.cumsum(np.random.default_rng(0).normal(0, 0.001, 400)) + 100)
    f_all = market.feature_table(df)
    f_cut = market.feature_table(df.iloc[:300].reset_index(drop=True))
    pd.testing.assert_frame_equal(f_all.iloc[:300], f_cut)


def test_odour_encoder_is_disjoint_and_bounded():
    enc = market.OdourEncoder(n_pn=100, per_feature=8, seed=1)
    all_slots = np.concatenate(list(enc.slots.values()))
    assert len(set(all_slots)) == len(all_slots)
    feats = {k: 0.3 for k in market.FEATURES}
    for _ in range(30):
        pn = enc.encode(feats)
    assert pn.min() >= 0 and pn.max() <= 1.0 and (pn > 0).sum() > 0
