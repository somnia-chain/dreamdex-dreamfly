"""Event-contract (Up/Down binary) arms on recorded resolved windows.

Each window is one episode: the fly sees the window's state at the last recorded read (~20 s before
expiry) and picks an arm. Arms are last-seconds takers; the fill is simulated at the recorded book mid
plus a haircut (there is no depth in this data), so results are a ceiling, not an estimate.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EC_FEATURES = ("fav_prob", "z_abs", "sigma_bps", "is_5m", "is_1h", "is_btc", "hour_sin", "hour_cos")
EC_ARMS = ("ec_late", "ec_sharp", "flat")


@dataclass(frozen=True)
class ECCosts:
    stake: float = 20.0  # USDso per window
    haircut: float = 0.02  # probability points paid above the recorded mid (spread + adverse move)


def load_windows(db: Path | str) -> pd.DataFrame:
    c = sqlite3.connect(str(db))
    w = pd.read_sql(
        "select symbol, at, asset, interval, truth, market_prob, z_score, sigma_bps, disputed "
        "from windows where truth is not null and market_prob is not null order by at", c)
    w["ts"] = pd.to_datetime(w["at"], unit="ms", utc=True)
    w["fav_up"] = (w["market_prob"] >= 0.5).astype(int)
    w["fav_prob"] = np.where(w["fav_up"] == 1, w["market_prob"], 1 - w["market_prob"])
    w["z_abs"] = w["z_score"].abs()
    w["is_5m"] = (w["interval"] == "5m").astype(float)
    w["is_1h"] = (w["interval"] == "1h").astype(float)
    w["is_btc"] = (w["asset"] == "BTC").astype(float)
    hour = w["ts"].dt.hour + w["ts"].dt.minute / 60
    w["hour_sin"], w["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    return w.reset_index(drop=True)


def _buy_favourite(row: pd.Series, c: ECCosts) -> float:
    """Pay fav_prob + haircut per share for the favourite; win pays 1."""
    price = min(row["fav_prob"] + c.haircut, 0.999)
    shares = c.stake / price
    won = int(row["truth"]) == int(row["fav_up"])
    return shares * (1.0 - price) if won else -c.stake


def ec_late(row: pd.Series, c: ECCosts, p: dict) -> float:
    """Oracle-lead late taker with the recorded filters: favourite >= fav_min and |z| >= z_min."""
    if row["fav_prob"] >= p.get("fav_min", 0.95) and row["z_abs"] >= p.get("z_min", 1.6):
        return _buy_favourite(row, c)
    return 0.0


def ec_sharp(row: pd.Series, c: ECCosts, p: dict) -> float:
    """Sharpened book, 5m only: buy the favourite when logit(book) x gain clears the threshold."""
    if row["interval"] != "5m":
        return 0.0
    pr = float(np.clip(row["fav_prob"], 1e-4, 1 - 1e-4))
    sharp = 1 / (1 + np.exp(-p.get("gain", 1.3) * np.log(pr / (1 - pr))))
    return _buy_favourite(row, c) if sharp >= p.get("threshold", 0.9) else 0.0


def flat(row: pd.Series, c: ECCosts, p: dict) -> float:
    return 0.0


EC_ARM_FN = {"ec_late": ec_late, "ec_sharp": ec_sharp, "flat": flat}


def run_ec(w: pd.DataFrame, selector, costs: ECCosts | None = None, params: dict | None = None,
           adaptive_reward: bool = True) -> pd.DataFrame:
    """adaptive_reward: P&L over stake goes through the same running scale as spot and perps (zero-reward
    windows do not shrink it), so a typical taken window lands near +/-1 and the pool scores can clear the
    decision margin. Without it EC rewards stay tiny and the fly rarely exploits."""
    from .sim import RewardScaler
    costs = costs or ECCosts()
    params = params or {}
    scale = RewardScaler() if adaptive_reward else (lambda x: float(np.clip(x, -3, 3)))
    rows = []
    for _, r in w.iterrows():
        k = selector.choose(r)
        arm = EC_ARMS[k]
        pnl = EC_ARM_FN[arm](r, costs, params.get(arm, {}))
        rew = scale(pnl / costs.stake)
        selector.update(rew)
        rows.append({"ts": r["ts"], "interval": r["interval"], "asset": r["asset"], "arm": arm, "pnl": pnl, "reward": rew})
    return pd.DataFrame(rows)
