"""Phase 2: the fly vs baselines on recorded DreamDEX spot candles, sweeping episode length."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybot import selectors as S  # noqa: E402
from flybot.arms import ARM_NAMES, Costs  # noqa: E402
from flybot.market import load_candles  # noqa: E402
from flybot.mb.graph import load_graph  # noqa: E402
from flybot.mb.model import MBModel  # noqa: E402
from flybot.sim import run, summarise  # noqa: E402

WARMUP_DAYS = 7


def selectors_for(seed: int, graph, directed: int = 0) -> dict:
    out = {f"only_{a}": S.Single(a) for a in ARM_NAMES}
    out["random"] = S.Random(seed)
    out["rule"] = S.Rule()
    out["bandit"] = S.Bandit(seed)
    out["fly"] = S.Fly(MBModel(graph, seed=seed), seed=seed)
    if directed:
        out["fly_directed"] = S.Fly(MBModel(graph, seed=seed), seed=seed, directed=directed, explore=0.02)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=["WETH:USDso", "WBTC:USDso", "SOMI:USDso"])
    ap.add_argument("--episodes", nargs="*", type=int, default=[5, 15, 60])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--days", type=int, default=0, help="limit to the last N days (0 = all)")
    ap.add_argument("--taker-bps", type=float, default=Costs.taker_bps)
    ap.add_argument("--maker-bps", type=float, default=Costs.maker_bps)
    ap.add_argument("--slip-bps", type=float, default=Costs.slip_bps)
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--params", default="", help="json of per-market arm params (from tune_arms.py)")
    ap.add_argument("--since", default="", help="report P&L only from this UTC date (warm-up still applies)")
    ap.add_argument("--reward-ewma", type=float, default=0.0, help="per-arm EWMA alpha on the reward (0 = off)")
    ap.add_argument("--fly-directed", type=int, default=0, help="add a fly that tries each arm N times per condition")
    a = ap.parse_args()
    all_params = json.loads(Path(a.params).read_text()) if a.params else {}
    costs = Costs(taker_bps=a.taker_bps, maker_bps=a.maker_bps, slip_bps=a.slip_bps)
    out = ROOT / "data" / "results"
    out.mkdir(parents=True, exist_ok=True)
    graph = load_graph(ROOT / "data" / "mb")

    rows = []
    for sym in a.symbols:
        df = load_candles(sym)
        if a.days:
            df = df[df["ts"] >= df["ts"].iloc[-1] - pd.Timedelta(days=a.days)].reset_index(drop=True)
        print(f"\n{sym}: {len(df)} candles {df['ts'].iloc[0]:%m-%d} -> {df['ts'].iloc[-1]:%m-%d}", flush=True)
        for ep in a.episodes:
            skip = WARMUP_DAYS * 24 * 60 // ep
            for seed in range(a.seeds):
                for name, sel in selectors_for(seed, graph, a.fly_directed).items():
                    if seed > 0 and name.startswith(("only_", "rule")):
                        continue  # deterministic
                    res = run(df, sel, ep, costs=costs, params=all_params.get(sym, {}), reward_ewma=a.reward_ewma)
                    if a.since:
                        fr = res.frame()
                        skip = int((fr["start"] < pd.Timestamp(a.since, tz="UTC")).sum())
                    s = summarise(res, skip_first=skip)
                    rows.append({"symbol": sym, "episode": ep, "seed": seed, "selector": name, **s})
                    if name.startswith("fly"):
                        res.frame().to_csv(out / f"{name}_{sym.replace(':', '_')}_{ep}m_s{seed}{a.tag}.csv", index=False)
            part = pd.DataFrame([r for r in rows if r["symbol"] == sym and r["episode"] == ep])
            agg = part.groupby("selector")[["pnl", "sharpe", "max_dd", "switches"]].mean().round(2)
            agg = agg.sort_values("pnl", ascending=False)
            print(f"\n-- {sym} episode {ep} min (mean over seeds, after {WARMUP_DAYS}-day warm-up) --")
            print(agg.to_string(), flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(out / f"results{a.tag}.csv", index=False)
    print("\n== fly vs baselines: mean P&L by episode length, all symbols ==")
    print(table.pivot_table(index="selector", columns="episode", values="pnl", aggfunc="mean").round(1).to_string())


if __name__ == "__main__":
    main()
