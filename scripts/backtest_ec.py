"""Event-contract arms: the fly vs baselines on recorded resolved windows, walk-forward by date.

Fills are simulated at the recorded book mid + haircut (no depth in the data): a ceiling, not an estimate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybot import selectors as S  # noqa: E402
from flybot.ec import EC_ARMS, EC_FEATURES, ECCosts, load_windows, run_ec  # noqa: E402
from flybot.mb.graph import load_graph  # noqa: E402
from flybot.mb.model import MBModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "ec" / "windows.db"))
    ap.add_argument("--split", default="2026-09-24", help="score only windows from this UTC date")
    ap.add_argument("--haircut", type=float, default=0.02)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    w = load_windows(a.db)
    split = pd.Timestamp(a.split, tz="UTC")
    costs = ECCosts(haircut=a.haircut)
    graph = load_graph(ROOT / "data" / "mb")
    print(f"windows: {len(w)} {w['ts'].iloc[0]:%m-%d} -> {w['ts'].iloc[-1]:%m-%d}; test from {a.split}; haircut {a.haircut}")
    print("favourite hit-rate by fav_prob bucket (whole set):")
    w["bucket"] = pd.cut(w["fav_prob"], [0.5, 0.7, 0.9, 0.95, 0.98, 1.0])
    print(w.groupby("bucket", observed=True).apply(lambda d: pd.Series({"n": len(d), "hit": (d["truth"] == d["fav_up"]).mean().round(3)})).to_string())

    rows = []
    out = ROOT / "data" / "results"
    for seed in range(a.seeds):
        sels = {f"only_{x}": S.Single(x, arms=EC_ARMS) for x in EC_ARMS}
        sels["random"] = S.Random(seed, arms=EC_ARMS)
        sels["bandit"] = S.Bandit(seed, arms=EC_ARMS, features=EC_FEATURES)
        sels["fly"] = S.Fly(MBModel(graph, seed=seed), seed=seed, arms=EC_ARMS, features=EC_FEATURES,
                            condition_features=("fav_prob", "z_abs"), condition_bins=2)
        for name, sel in sels.items():
            if seed > 0 and name.startswith("only_"):
                continue
            res = run_ec(w, sel, costs)
            test = res[res["ts"] >= split]
            rows.append({"seed": seed, "selector": name, "pnl": test["pnl"].sum(), "trades": int((test["arm"] != "flat").sum()),
                         "mix": test["arm"].value_counts(normalize=True).round(2).to_dict()})
            if name == "fly":
                res.to_csv(out / f"ec_fly_s{seed}.csv", index=False)
    t = pd.DataFrame(rows)
    t.to_csv(out / "ec_results.csv", index=False)
    agg = t.groupby("selector")[["pnl", "trades"]].mean().round(1).sort_values("pnl", ascending=False)
    print(f"\n== EC held-out (>= {a.split}), stake {costs.stake}, mean over seeds ==")
    print(agg.to_string())
    print("fly mix seed 0:", t[(t.selector == "fly") & (t.seed == 0)].iloc[0]["mix"])


if __name__ == "__main__":
    main()
