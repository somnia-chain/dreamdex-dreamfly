"""Small hyperparameter grid for the fly on the synthetic regime test (tuning seeds), then a report on
fresh seeds for the best config vs the bandit. Keeps tuning and reporting seeds disjoint."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybot import selectors as S  # noqa: E402
from flybot.arms import Costs  # noqa: E402
from flybot.mb.graph import load_graph  # noqa: E402
from flybot.mb.model import MBModel  # noqa: E402
from flybot.sim import run  # noqa: E402
from scripts.synthetic_regimes import synth  # noqa: E402

GRID = {
    "explore": [0.05, 0.1],
    "eta": [0.3, 1.0],
    "recover": [0.001, 0.005],
    "prediction_error": [False, True],
}
TUNE_SEEDS, TEST_SEEDS = (0, 1), (10, 11, 12)
EPISODE, DAYS = 60, 40
COSTS = Costs(taker_bps=5.0, slip_bps=0.0)


def score(cfg: dict, seeds, graph) -> float:
    total = 0.0
    for seed in seeds:
        df, _ = synth(DAYS, 12, seed)
        fly = S.Fly(MBModel(graph, seed=seed), seed=seed, **cfg)
        res = run(df, fly, EPISODE, costs=COSTS).frame()
        total += res["pnl"].iloc[7 * 24 :].sum()
    return total / len(seeds)


def main() -> None:
    graph = load_graph(ROOT / "data" / "mb")
    rows = []
    for values in itertools.product(*GRID.values()):
        cfg = dict(zip(GRID.keys(), values))
        rows.append({**cfg, "pnl_tune": score(cfg, TUNE_SEEDS, graph)})
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows).sort_values("pnl_tune", ascending=False)
    print("\n== tuning grid (mean P&L over tuning seeds) ==")
    print(t.to_string(index=False))
    best = {k: t.iloc[0][k] for k in GRID}
    best["prediction_error"] = bool(best["prediction_error"])

    print("\n== fresh seeds: best fly vs default fly vs bandit ==")
    out = {"fly_best": 0.0, "fly_default": 0.0, "bandit": 0.0}
    for seed in TEST_SEEDS:
        df, _ = synth(DAYS, 12, seed)
        for name, sel in {
            "fly_best": S.Fly(MBModel(graph, seed=seed), seed=seed, **best),
            "fly_default": S.Fly(MBModel(graph, seed=seed), seed=seed, prediction_error=False),
            "bandit": S.Bandit(seed),
        }.items():
            out[name] += run(df, sel, EPISODE, costs=COSTS).frame()["pnl"].iloc[7 * 24 :].sum() / len(TEST_SEEDS)
    print({k: round(v) for k, v in out.items()}, "best cfg:", best)
    t.to_csv(ROOT / "data" / "results" / "fly_tune.csv", index=False)


if __name__ == "__main__":
    main()
