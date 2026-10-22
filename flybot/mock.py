"""A mock market that switches between regimes, each built so a different strategy should do best.

Real markets give a fly few clean examples of "this is when a maker wins, this is when a grid wins". A mock market
can give as many as wanted, with the answer known. It is made from four regimes that last hours and then switch,
the way a real market changes character:

  trend   steady drift one way, low noise            -> trend should win
  range   price pulled back to an anchor (OU)        -> grid should win
  bounce  small fast swings around a mid price       -> maker should win (both its quotes fill, it earns the spread)
  chop    large random swings, no structure          -> flat should win (everything else loses)

The mock is true by construction: it can teach and show that the fly tells these regimes apart, but it cannot say
how much real markets look like it. Pretraining on it must be checked on real held-out data (scripts/mock_regimes.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REGIMES = ("trend", "range", "bounce", "chop")


@dataclass(frozen=True)
class MockSpec:
    mean_hours: float = 8.0         # average length of a regime before it switches
    min_hours: float = 3.0
    trend_drift: float = 0.00006    # per minute, sign drawn per segment
    trend_noise: float = 0.0002
    range_pull: float = 0.08        # OU pull back to the anchor per minute
    range_noise: float = 0.002
    bounce_pull: float = 0.4
    bounce_noise: float = 0.001
    bounce_drift: float = 0.00004   # how fast the mid price drifts under the bounce, per minute
    chop_noise: float = 0.0025
    chop_drift: float = 0.0015
    chop_flip_min: float = 4.0      # average minutes between drift flips
    start_price: float = 100.0
    start: str = "2026-07-01"


def _pulled(rng: np.random.Generator, m: int, pull: float, noise: float) -> np.ndarray:
    """Log distance from an anchor that is pulled back each minute (an Ornstein-Uhlenbeck walk)."""
    x, out = 0.0, np.empty(m)
    shocks = rng.normal(0, noise, m)
    for i in range(m):
        x += -pull * x + shocks[i]
        out[i] = x
    return out


def mock_candles(days: int, seed: int, spec: MockSpec = MockSpec()) -> tuple[pd.DataFrame, np.ndarray]:
    """One-minute candles for `days` days plus the regime of every minute."""
    rng = np.random.default_rng(seed)
    n = days * 24 * 60
    close = np.empty(n)
    regime = np.empty(n, dtype=object)
    p, t, last = spec.start_price, 0, None
    while t < n:
        r = REGIMES[rng.integers(len(REGIMES))]
        while r == last:                                   # always a real switch
            r = REGIMES[rng.integers(len(REGIMES))]
        last = r
        m = min(int(max(spec.min_hours, rng.exponential(spec.mean_hours)) * 60), n - t)
        if r == "trend":
            steps = rng.choice([-1, 1]) * spec.trend_drift + rng.normal(0, spec.trend_noise, m)
            seg = p * np.exp(np.cumsum(steps))
        elif r == "range":
            seg = p * np.exp(_pulled(rng, m, spec.range_pull, spec.range_noise))
        elif r == "bounce":
            # fast swings around a mid price that itself drifts: a maker re-quotes around the last price and keeps up,
            # a grid pinned to the episode's opening price is left behind
            mid = np.cumsum(np.full(m, rng.choice([-1, 1]) * spec.bounce_drift))
            seg = p * np.exp(mid + _pulled(rng, m, spec.bounce_pull, spec.bounce_noise))
        else:
            # whipsaw: strong drift that flips direction every few minutes, so a follower buys tops
            flips = np.cumsum(rng.random(m) < 1 / spec.chop_flip_min) % 2 * 2 - 1
            seg = p * np.exp(np.cumsum(flips * spec.chop_drift + rng.normal(0, spec.chop_noise, m)))
        close[t : t + m], regime[t : t + m] = seg, r
        p, t = float(seg[-1]), t + m
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(rng.normal(0, 0.0004, n))
    df = pd.DataFrame({"open": open_, "close": close, "high": np.maximum(open_, close) * (1 + wick),
                       "low": np.minimum(open_, close) * (1 - wick), "volume": rng.lognormal(3, 0.5, n)})
    df["timestamp"] = (pd.Timestamp(spec.start, tz="UTC").value // 10**6) + np.arange(n) * 60_000
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df, regime
