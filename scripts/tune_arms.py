"""Walk-forward tuning of the arms: pick each arm's parameters per market on a train window, then
report every arm's P&L on the following test window with tuned vs default parameters.

Writes data/results/arm_params.json (per market, per arm) for scripts/backtest.py --params.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybot.arms import ARMS, Costs  # noqa: E402
from flybot.market import LONG, load_candles  # noqa: E402

GRIDS = {
    "maker": {"spread_bps": [10, 20, 40, 80, 160], "skew_bps": [5, 20], "clip_frac": [0.25]},
    "grid": {"step_bps": [20, 40, 80, 160, 320], "levels": [2, 4], "clip_frac": [0.2]},
    "trend": {"lookback": [15, 30, 60, 120], "threshold": [0.001, 0.003, 0.006, 0.012]},
}


def combos(grid: dict) -> list[dict]:
    return [dict(zip(grid, v)) for v in itertools.product(*grid.values())]


def arm_pnls(df: pd.DataFrame, arm: str, p: dict, episode: int, costs: Costs) -> np.ndarray:
    out = []
    i = LONG
    while i + episode <= len(df):
        out.append(ARMS[arm](df.iloc[i : i + episode], df.iloc[max(0, i - LONG) : i], costs, p).pnl)
        i += episode
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=["WETH:USDso", "WBTC:USDso", "SOMI:USDso"])
    ap.add_argument("--episode", type=int, default=60)
    ap.add_argument("--split", default="2026-08-03", help="train before this UTC date, test from it")
    ap.add_argument("--metric", default="sharpe", choices=["sharpe", "pnl"])
    ap.add_argument("--out", default="arm_params.json")
    a = ap.parse_args()
    costs = Costs()
    out = ROOT / "data" / "results"
    params: dict[str, dict[str, dict]] = {}
    rows = []
    split = pd.Timestamp(a.split, tz="UTC")
    for sym in a.symbols:
        df = load_candles(sym)
        train = df[df["ts"] < split].reset_index(drop=True)
        test = df[df["ts"] >= split].reset_index(drop=True)
        params[sym] = {}
        for arm, grid in GRIDS.items():
            best, best_score = None, -np.inf
            for p in combos(grid):
                x = arm_pnls(train, arm, p, a.episode, costs)
                score = x.sum() if a.metric == "pnl" else x.mean() / (x.std() + 1e-9) * np.sqrt(len(x))
                if score > best_score:
                    best, best_score = p, score
            params[sym][arm] = best
            default = arm_pnls(test, arm, {}, a.episode, costs).sum()
            tuned = arm_pnls(test, arm, best, a.episode, costs).sum()
            train_tuned = arm_pnls(train, arm, best, a.episode, costs).sum()
            rows.append({"symbol": sym, "arm": arm, "params": best, "train_pnl_tuned": round(train_tuned),
                         "test_pnl_default": round(default), "test_pnl_tuned": round(tuned)})
            print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    print(f"\n== walk-forward: train < {a.split}, test >= {a.split}, episode {a.episode} min, metric {a.metric} ==")
    print(t.to_string(index=False))
    print("\ntest P&L summed over markets: default %.0f, tuned %.0f" % (t.test_pnl_default.sum(), t.test_pnl_tuned.sum()))
    (out / a.out).write_text(json.dumps(params, indent=2))
    t.to_csv(out / a.out.replace(".json", "_tuning.csv"), index=False)


if __name__ == "__main__":
    main()
