"""Teach a fly from history where every strategy's outcome is known.

Live, the fly only learns from the strategy it picked, and it often picks flat, so it hears little. In history
(candles, or the mock market) every strategy's result for every episode can be computed, so each episode teaches all
of them: the market is sniffed once and each strategy's pool is taught its own result, as if it had been chosen.
Afterwards the fly runs live as usual, still learning from what it picks.
"""

from __future__ import annotations

import pandas as pd

from .arms import ARM_NAMES, ARMS, Costs
from .market import LONG, feature_table
from .sim import RewardScaler, reward_of


def counterfactual(fly, df: pd.DataFrame, episode_min: int, costs: Costs | None = None, params: dict | None = None,
                   passes: int = 1, scaler: RewardScaler | None = None) -> int:
    """Run `fly` over `df` teaching every strategy on every episode. Returns the number of episodes used.

    `params` is the market's arm settings; its "arms" list, when present, limits which strategies are taught.
    Pass the live fly's `scaler` to leave it in the state it would have after living through this history."""
    costs, params = costs or Costs(), params or {}
    allowed = params.get("arms")
    feats = feature_table(df)
    scale = scaler or RewardScaler()
    n = 0
    for _ in range(passes):
        i = LONG
        while i + episode_min <= len(df):
            f = feats.iloc[i]
            ep, hist = df.iloc[i : i + episode_min], df.iloc[max(0, i - LONG) : i]
            kc, _ = fly.m.sniff(fly.enc.encode(f))
            s = fly.scores(kc)
            for k, arm in enumerate(ARM_NAMES):
                if arm == "flat" or (allowed and arm not in allowed):
                    continue                      # flat never trades, so there is nothing to teach it
                res = ARMS[arm](ep, hist, costs, params.get(arm, {}))
                fly.last = (k, kc, float(s[k]))
                fly.update(scale(reward_of(res.pnl, ep, costs, clip=float("inf"))))
            n += 1
            i += episode_min
    return n
