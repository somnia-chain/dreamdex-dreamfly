"""Candle loading, market features, and the feature -> projection-neuron ("odour") encoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDLES = ROOT / "data" / "candles"

FEATURES = ("vol_short", "vol_long", "trend", "range", "vol_ratio", "volume_z", "hour_sin", "hour_cos")
SHORT, LONG = 15, 120  # minutes


def load_candles(symbol: str, interval: str = "1m") -> pd.DataFrame:
    df = pd.read_parquet(CANDLES / f"{symbol.replace(':', '_')}_{interval}.parquet")
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per candle, using only that candle and earlier ones. First LONG rows are NaN."""
    c = df["close"]
    r = np.log(c).diff()
    out = pd.DataFrame(index=df.index)
    out["vol_short"] = r.rolling(SHORT).std() * np.sqrt(SHORT)
    out["vol_long"] = r.rolling(LONG).std() * np.sqrt(LONG)
    out["trend"] = (c / c.shift(LONG) - 1.0) / (out["vol_long"] + 1e-9)
    out["range"] = ((df["high"] - df["low"]) / c).rolling(SHORT).mean()
    out["vol_ratio"] = out["vol_short"] / (out["vol_long"] / np.sqrt(LONG / SHORT) + 1e-9)
    v = np.log1p(df["volume"])
    out["volume_z"] = (v - v.rolling(LONG).mean()) / (v.rolling(LONG).std() + 1e-9)
    hour = df["ts"].dt.hour + df["ts"].dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out[list(FEATURES)]


class OdourEncoder:
    """Maps a feature vector to PN rates. Each feature owns a disjoint set of `per_feature` PNs with
    Gaussian tuning curves spread over the feature's robust range, so nearby values share PNs.
    The scaling uses running quantiles from data seen so far (no lookahead)."""

    def __init__(self, n_pn: int, per_feature: int = 8, seed: int = 0, warm: int = 500,
                 features: tuple = FEATURES):
        self.n_pn = n_pn
        self.per_feature = per_feature
        self.features = tuple(features)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_pn)
        self.slots = {f: perm[i * per_feature : (i + 1) * per_feature] for i, f in enumerate(self.features)}
        assert len(self.features) * per_feature <= n_pn
        self.centres = np.linspace(0.0, 1.0, per_feature)
        self.width = 1.0 / per_feature
        self.history: dict[str, list[float]] = {f: [] for f in self.features}
        self.warm = warm

    def _scale(self, f: str, x: float, record: bool = True) -> float:
        h = self.history[f]
        if record:
            h.append(x)
        if len(h) < 20:
            return 0.5
        lo, hi = np.quantile(h[-5000:], [0.05, 0.95])
        return float(np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0))

    def encode(self, feats: pd.Series | dict, record: bool = True) -> np.ndarray:
        """record=False: a look without adding the values to the running range (a sniff that is not a
        decision must not tilt the scaling toward whatever cadence the sniffs come at)."""
        pn = np.zeros(self.n_pn)
        for f in self.features:
            x = float(feats[f])
            if not np.isfinite(x):
                continue
            u = self._scale(f, x, record)
            pn[self.slots[f]] = np.exp(-0.5 * ((self.centres - u) / self.width) ** 2)
        return pn
