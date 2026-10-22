"""Episode loop: at each decision the selector sees the market features, picks an arm, the arm runs
for one episode, and the risk-adjusted P&L comes back as the reward."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .arms import ARM_NAMES, ARMS, Costs
from .market import LONG, feature_table


@dataclass
class EpisodeLog:
    start: pd.Timestamp
    arm: str
    pnl: float
    reward: float
    trades: int


@dataclass
class RunResult:
    episodes: list[EpisodeLog] = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([e.__dict__ for e in self.episodes])


def reward_of(pnl: float, ep: pd.DataFrame, costs: Costs, clip: float = 3.0) -> float:
    """P&L divided by the stake's realised move over the episode (a vol unit), clipped."""
    r = np.log(ep["close"]).diff().dropna()
    vol = float(r.std() * np.sqrt(len(r))) if len(r) > 1 else 0.0
    unit = costs.stake * max(vol, 1e-4)
    return float(np.clip(pnl / unit, -clip, clip))


class RewardScaler:
    """Adaptive gain on the reward, like a dopamine system that adapts to the size of recent rewards:
    divides by a running mean of |reward| so a typical episode lands near ±1, then clips. Uses only
    past episodes (no lookahead)."""

    def __init__(self, alpha: float = 0.02, warm: int = 20, clip: float = 3.0, floor: float = 1e-6,
                 skip_zero: bool = True, eps: float = 0.0):
        """skip_zero: episodes with exactly zero reward (the flat arm, an arm that never traded) do not
        shrink the scale, so a long flat stretch cannot turn the next real reward into a saturating jolt.
        eps: likewise for |reward| below eps (an arm that only paid gas and never filled)."""
        self.alpha, self.warm, self.clip, self.floor, self.skip_zero, self.eps = alpha, warm, clip, floor, skip_zero, eps
        self.n, self.scale = 0, 0.0

    def __call__(self, x: float) -> float:
        out = float(np.clip(x / self.scale, -self.clip, self.clip)) if self.n >= self.warm and self.scale > self.floor else 0.0
        if (self.skip_zero and x == 0.0) or abs(x) < self.eps:
            return out
        if self.n < self.warm:
            self.scale += (abs(x) - self.scale) / (self.n + 1)
        else:
            self.scale += self.alpha * (abs(x) - self.scale)
        self.n += 1
        return out


def run(df: pd.DataFrame, selector, episode_min: int, costs: Costs | None = None, params: dict | None = None,
        warmup_min: int = LONG, adaptive_reward: bool = True, reward_ewma: float = 0.0) -> RunResult:
    """reward_ewma > 0: the reward delivered is a per-arm exponential average of that arm's scaled
    rewards (alpha = reward_ewma) including the current one, so the teaching signal carries the arm's
    recent expected value rather than one noisy draw."""
    costs = costs or Costs()
    params = params or {}
    feats = feature_table(df)
    out = RunResult()
    scale = RewardScaler() if adaptive_reward else (lambda x: x)
    ewma: dict[str, float] = {}
    i = warmup_min
    while i + episode_min <= len(df):
        f = feats.iloc[i]
        arm_idx = selector.choose(f)
        arm = ARM_NAMES[arm_idx]
        ep = df.iloc[i : i + episode_min]
        hist = df.iloc[max(0, i - LONG) : i]
        res = ARMS[arm](ep, hist, costs, params.get(arm, {}))
        rew = scale(reward_of(res.pnl, ep, costs, clip=np.inf))
        if reward_ewma > 0:
            ewma[arm] = rew if arm not in ewma else ewma[arm] + reward_ewma * (rew - ewma[arm])
            rew = ewma[arm]
        selector.update(rew)
        out.episodes.append(EpisodeLog(df["ts"].iloc[i], arm, res.pnl, rew, res.trades))
        i += episode_min
    return out


def summarise(res: RunResult, skip_first: int = 0) -> dict:
    d = res.frame().iloc[skip_first:]
    if d.empty:
        return {"pnl": 0.0, "sharpe": 0.0, "max_dd": 0.0, "episodes": 0, "switches": 0}
    eq = d["pnl"].cumsum()
    dd = float((eq - eq.cummax()).min())
    sharpe = float(d["pnl"].mean() / (d["pnl"].std() + 1e-9) * np.sqrt(len(d)))
    switches = int((d["arm"] != d["arm"].shift()).sum() - 1)
    mix = d["arm"].value_counts(normalize=True).round(2).to_dict()
    return {"pnl": float(d["pnl"].sum()), "sharpe": sharpe, "max_dd": dd, "episodes": len(d),
            "switches": switches, "mix": mix}
