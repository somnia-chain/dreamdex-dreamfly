import numpy as np
import pandas as pd
import pytest

from flybot.perp import PERP_FEATURES, PerpCosts, PerpPaperEpisode, perp_feature_table, rows_to_frame, trend_direction

C = PerpCosts(taker_bps=0.0, slip_bps=0.0, gas_per_action=0.0, stake=1000.0, leverage=2.0)


def rows(n: int, start: float = 100.0, drift: float = 0.0, funding: float = 0.0, spread: float = 0.02, seed: int = 0):
    rng = np.random.default_rng(seed)
    out, m = [], start
    for i in range(n):
        m *= np.exp(drift + rng.normal(0, 0.0005))
        out.append({"type": "perp", "symbol": "BTC/USDso:USDso", "wall": pd.Timestamp("2026-09-29T00:00:00Z") + pd.Timedelta(minutes=i),
                    "mark": m, "row_mark": m, "index": m * 1.0001, "funding_8h": funding, "oi": 10 + i % 3,
                    "best_bid": m - spread, "best_ask": m + spread})
    for r in out:
        r["wall"] = r["wall"].isoformat()
    return out


def test_frame_and_features_no_lookahead():
    df = rows_to_frame(rows(80))
    f = perp_feature_table(df)
    assert list(f.columns) == list(PERP_FEATURES)
    assert f["vol_short"].iloc[:4].isna().all() and np.isfinite(f["vol_short"].iloc[-1])
    assert np.isfinite(f["basis"].iloc[0]) and abs(f["basis"].iloc[0] + 1.0) < 0.01  # mark 1 bp under index
    # a change in the future must not change the past
    f2 = perp_feature_table(rows_to_frame(rows(80)[:60]))
    assert np.allclose(f.iloc[:60].fillna(0).to_numpy(), f2.fillna(0).to_numpy())


def test_stale_mark_falls_back_to_row_mark():
    r = rows(3)
    r[1]["mark"] = None
    df = rows_to_frame(r)
    assert len(df) == 3 and np.isfinite(df["mark"]).all()


def test_maker_fills_when_touch_crosses_and_earns_spread():
    df = rows_to_frame(rows(3))
    ep = PerpPaperEpisode("p_maker", C, {"spread_bps": 20, "skew_bps": 0, "clip_frac": 0.5})
    ep.start(df.iloc[0])
    assert len(ep.bids) == 1 and len(ep.asks) == 1
    low = df.iloc[1].copy()
    low["mark"], low["best_bid"], low["best_ask"] = 99.75, 99.7, 99.75  # ask down to our bid near 99.8: we buy
    ep.on_row(low)
    assert ep.pos > 0 and len(ep.fills) == 1 and not ep.fills[0].taker
    high = df.iloc[2].copy()
    high["mark"], high["best_bid"], high["best_ask"] = 100.6, 100.6, 100.65  # bid up through our ask: we sell
    ep.on_row(high)
    assert abs(ep.pos) < 1e-9 and len(ep.fills) == 2
    assert ep.finish(high) > 0


def test_trend_uses_leverage_and_flattens():
    df = rows_to_frame(rows(3))
    ep = PerpPaperEpisode("p_trend", C, {})
    ep.start(df.iloc[0], trend_dir=1)
    mid = (df.iloc[0]["best_bid"] + df.iloc[0]["best_ask"]) / 2
    assert ep.pos > 0 and abs(ep.pos * mid - C.stake * C.leverage) < 1e-6  # notional = margin x leverage
    up = df.iloc[2].copy()
    up["mark"], up["best_bid"], up["best_ask"] = 101.0, 100.98, 101.02
    pnl = ep.finish(up)
    assert pnl > 0 and abs(ep.pos) < 1e-9 and ep.done


def test_carry_sits_short_when_funding_positive_and_collects():
    r = rows(130, funding=0.001)  # 10 bps per 8h, well above the default floor
    df = rows_to_frame(r)
    ep = PerpPaperEpisode("p_carry", C, {})
    ep.start(df.iloc[0])
    assert ep.pos < 0
    for i in range(1, 125):
        ep.on_row(df.iloc[i])
    assert ep.funding_paid < 0  # short receives when funding is positive
    assert abs(ep.funding_paid) > 0.0


def test_carry_skips_tiny_funding():
    df = rows_to_frame(rows(3, funding=0.00001))
    ep = PerpPaperEpisode("p_carry", C, {})
    ep.start(df.iloc[0])
    assert ep.pos == 0.0 and not ep.fills


def test_carry_enters_above_min_apr_and_sits_out_below_it():
    low = rows_to_frame(rows(3, funding=0.00004))           # about 4% APR, under the 10% entry floor
    ep = PerpPaperEpisode("p_carry", C, {})
    ep.start(low.iloc[0])
    assert ep.pos == 0.0
    paying = rows_to_frame(rows(3, funding=0.000188))       # a real perp test network rate, about 20% APR
    ep = PerpPaperEpisode("p_carry", C, {})
    ep.start(paying.iloc[0])
    assert ep.pos < 0                                       # positive funding pays shorts


def test_carry_is_held_across_episodes_while_it_pays_and_closed_when_the_rate_drops():
    df = rows_to_frame(rows(5, funding=0.001))
    ep = PerpPaperEpisode("p_carry", C, {})
    ep.start(df.iloc[0])
    pnl = ep.finish(df.iloc[1])
    assert ep.keep_open and ep.pos < 0 and ep.done          # still paying: not flattened
    assert pnl == pytest.approx(ep.cash + ep.pos * float(df.iloc[1]["mark"]) - ep.funding_paid)
    nxt = PerpPaperEpisode("p_carry", C, {}, inherit_pos=ep.pos)
    nxt.start(df.iloc[2])
    assert nxt.pos == ep.pos and not nxt.fills and nxt.pnl() == pytest.approx(0.0)   # taken over at the mark, no entry cost
    dropped = df.iloc[3].copy()
    dropped["funding_8h"] = 0.00001                         # about 1% APR, under the 3% exit floor
    nxt.on_row(dropped)
    assert nxt.pos == 0.0 and nxt.fills and nxt.fills[-1].side == "buy"
    other = PerpPaperEpisode("p_trend", C, {}, inherit_pos=ep.pos)
    other.start(df.iloc[2])
    assert other.pos == 0.0 and other.fills[0].side == "buy"  # a different strategy closes the inherited carry first


def test_guard_flattens_on_large_loss_and_stays_out():
    df = rows_to_frame(rows(3))
    ep = PerpPaperEpisode("p_trend", C, {})
    ep.start(df.iloc[0], trend_dir=1)
    crash = df.iloc[1].copy()
    crash["mark"], crash["best_bid"], crash["best_ask"] = 70.0, 69.9, 70.1  # -30% on 2x = -60% of margin
    ep.on_row(crash)
    assert ep.guarded and abs(ep.pos) < 1e-9
    rebound = df.iloc[2].copy()
    rebound["mark"], rebound["best_bid"], rebound["best_ask"] = 100.0, 99.9, 100.1
    ep.on_row(rebound)
    assert abs(ep.pos) < 1e-9 and ep.finish(rebound) < -0.5 * C.stake


def test_one_sided_book_falls_back_to_mark():
    r = rows(2)
    r[0]["best_bid"] = None
    df = rows_to_frame(r)
    ep = PerpPaperEpisode("p_trend", C, {})
    ep.start(df.iloc[0], trend_dir=-1)
    assert ep.fills[0].price == df.iloc[0]["mark"]


def test_trend_direction_threshold():
    up = rows_to_frame(rows(40, drift=0.001))
    assert trend_direction(up, {"lookback": 15, "threshold": 0.002}) == 1
    flat = rows_to_frame(rows(40))
    assert trend_direction(flat, {"lookback": 15, "threshold": 0.05}) == 0
