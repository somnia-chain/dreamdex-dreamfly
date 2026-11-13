"""Strategy arms. Each runs for one episode on a slice of 1-minute candles, starts flat, ends flat.

Fill model is candle-level: a resting quote is filled when the candle's low/high crosses it. This
flatters makers (no queue, no adverse selection inside the candle); results are relative, not absolute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    """Defaults follow the live pools: maker and taker fees are 0 (read from getPoolParams, 2026-09-27);
    a taker crosses a ~4 bps spread (median from a recorded top-of-book tape), so ~2-3 bps slippage."""

    taker_bps: float = 0.0
    maker_bps: float = 0.0
    slip_bps: float = 3.0
    stake: float = 1000.0  # quote units of notional per arm
    # Gas per order action in quote units. Mainnet SOMI:USDso, 2026-09-27: ~600k gas per placed order,
    # ~1.06M per amend, 6 gwei, SOMI ~0.22 USDso -> 0.0008-0.0014 USDso per action.
    gas_per_action: float = 0.001


@dataclass
class Result:
    pnl: float
    trades: int
    actions: int = 0  # on-chain order actions (place / amend / cancel) charged gas


def _taker_cost(notional: float, c: Costs) -> float:
    return abs(notional) * (c.taker_bps + c.slip_bps) / 1e4


def _charge_gas(r: Result, c: Costs) -> Result:
    r.pnl -= r.actions * c.gas_per_action
    return r


def flat(ep: pd.DataFrame, hist: pd.DataFrame, c: Costs, p: dict) -> Result:
    return Result(0.0, 0, 0)


def trend(ep: pd.DataFrame, hist: pd.DataFrame, c: Costs, p: dict) -> Result:
    """Enter on the sign of the trailing return if it clears a threshold; hold to the episode end."""
    look = int(p.get("lookback", 60))
    thr = float(p.get("threshold", 0.002))
    if len(hist) < look:
        return Result(0.0, 0, 0)
    ret = hist["close"].iloc[-1] / hist["close"].iloc[-look] - 1.0
    if abs(ret) < thr:
        return Result(0.0, 0, 0)
    d = np.sign(ret)
    entry, exit_ = float(ep["open"].iloc[0]), float(ep["close"].iloc[-1])
    pnl = c.stake * d * (exit_ / entry - 1.0) - 2 * _taker_cost(c.stake, c)
    return _charge_gas(Result(pnl, 2, 2), c)


def _run_quotes(ep: pd.DataFrame, c: Costs, quotes_fn, max_inv_units: float) -> Result:
    """Shared engine for maker and grid: quotes_fn(prev_close, inv) -> (bids, asks) as price lists;
    each level trades `clip` base units when crossed. Every candle in which the quote set changes is
    one amend per level (gas); the final flatten is one taker order."""
    cash, inv, trades, actions = 0.0, 0.0, 0, 0
    prev = float(ep["open"].iloc[0])
    last_quotes: tuple = ()
    for _, k in ep.iterrows():
        bids, asks, clip = quotes_fn(prev, inv)
        q = (tuple(round(b, 10) for b in bids), tuple(round(a, 10) for a in asks))
        if q != last_quotes:
            actions += len(bids) + len(asks)
            last_quotes = q
        for px in bids:
            if k["low"] <= px and inv + clip <= max_inv_units:
                cash -= px * clip * (1 + c.maker_bps / 1e4)
                inv += clip
                trades += 1
        for px in asks:
            if k["high"] >= px and inv - clip >= -max_inv_units:
                cash += px * clip * (1 - c.maker_bps / 1e4)
                inv -= clip
                trades += 1
        prev = float(k["close"])
    close = float(ep["close"].iloc[-1])
    actions += len(last_quotes[0]) + len(last_quotes[1]) if last_quotes else 0  # cancel at episode end
    if inv != 0.0:
        cash += inv * close - _taker_cost(inv * close, c)
        trades += 1
        actions += 1
    return _charge_gas(Result(cash, trades, actions), c)


def maker(ep: pd.DataFrame, hist: pd.DataFrame, c: Costs, p: dict) -> Result:
    """Two-sided quotes at ±spread around the last close, skewed against inventory."""
    spread = float(p.get("spread_bps", 20.0)) / 1e4
    skew = float(p.get("skew_bps", 10.0)) / 1e4
    ref = float(ep["open"].iloc[0])
    clip = c.stake * float(p.get("clip_frac", 0.25)) / ref
    max_inv = c.stake / ref

    def quotes(prev: float, inv: float):
        shift = -skew * (inv / max_inv)  # long inventory -> quote lower
        mid = prev * (1 + shift)
        return [mid * (1 - spread)], [mid * (1 + spread)], clip

    return _run_quotes(ep, c, quotes, max_inv)


def grid(ep: pd.DataFrame, hist: pd.DataFrame, c: Costs, p: dict) -> Result:
    """Ladder of n levels every `step_bps` around the episode-open anchor; each level fills once per side
    until re-armed by a fill on the other side."""
    n = int(p.get("levels", 3))
    step = float(p.get("step_bps", 30.0)) / 1e4
    anchor = float(ep["open"].iloc[0])
    clip = c.stake * float(p.get("clip_frac", 0.2)) / anchor
    max_inv = c.stake / anchor
    bids = [anchor * (1 - step * k) for k in range(1, n + 1)]
    asks = [anchor * (1 + step * k) for k in range(1, n + 1)]
    state = {"bids": list(bids), "asks": list(asks)}

    def quotes(prev: float, inv: float):
        # re-arm: any bid level below the last price is live, any ask above it is live
        live_b = [b for b in state["bids"] if b < prev]
        live_a = [a for a in state["asks"] if a > prev]
        return live_b, live_a, clip

    return _run_quotes(ep, c, quotes, max_inv)


ARMS = {"maker": maker, "grid": grid, "trend": trend, "flat": flat}
ARM_NAMES = tuple(ARMS)
