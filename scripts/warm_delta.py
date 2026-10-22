"""Replay recorded candles into the delta-rule fly, one brain per spot market, so it wakes up knowing which
strategy paid in which market state instead of starting blank.

Every episode of the history teaches every strategy its own result (flybot/pretrain.py). Writes the weights, the
reward scaler and the encoder history to <out>/<market>/, which the kit copies into a fresh state directory.

  .venv/bin/python scripts/warm_delta.py --out config/warm_delta
  .venv/bin/python scripts/warm_delta.py --out state/warm_delta --since 2026-06-01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybot.arms import Costs  # noqa: E402
from flybot.live.runner import MarketFly  # noqa: E402
from flybot.market import load_candles  # noqa: E402
from flybot.mb.graph import load_graph  # noqa: E402
from flybot.pretrain import counterfactual  # noqa: E402

SPOT = ["SOMI:USDso", "WETH:USDso", "WBTC:USDso"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "config" / "warm_delta"))
    ap.add_argument("--params", default=str(ROOT / "config" / "arm_params.json"))
    ap.add_argument("--markets", nargs="*", default=SPOT)
    ap.add_argument("--episode-min", type=int, default=60)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--since", default=None, help="only replay candles from this date (UTC)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out, params = Path(a.out), json.loads(Path(a.params).read_text())
    graph = load_graph(ROOT / "data" / "mb")
    for sym in a.markets:
        df = load_candles(sym)
        if a.since:
            df = df[df["ts"] >= pd.Timestamp(a.since, tz="UTC")].reset_index(drop=True)
        mf = MarketFly(sym, graph, out, Costs(), params.get(sym, {}), a.seed, a.episode_min, brain="delta")
        n = counterfactual(mf.fly, df, a.episode_min, mf.costs, mf.params, a.passes, scaler=mf.scaler)
        mf.save()          # equity and episode counters stay at zero: replayed history is not paper-live results
        print(json.dumps({"market": sym, "episodes": n, "weights_moved": mf.weights_moved(), "scaler": round(mf.scaler.scale, 5)}), flush=True)


if __name__ == "__main__":
    main()
