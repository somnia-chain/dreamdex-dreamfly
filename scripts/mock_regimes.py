"""Can the fly learn which strategy fits which regime, and does that carry to real markets?

Part 1, mock market (flybot/mock.py, regimes with a known best strategy). On held-out mock days it compares:
  fly_cold    learns online from scratch
  fly_pre     first learns through other mock days, then runs the test days (learning stays on)
  bandit, rule, random, each fixed strategy, and flat
and reports P&L, the worst drop from a peak (the "does it lose tremendously" number), how often it picked the
regime's best strategy, and which strategy it picked in each regime.

Part 2, real markets (--real). The fly pretrained on mock days runs the held-out DreamDEX candles, next to a fresh
fly and the fixed strategies. The mock is true by construction, so this is the check that pretraining on it does
not hurt where the answer is not known.

  .venv/bin/python scripts/mock_regimes.py --seeds 4
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybot import selectors as S  # noqa: E402
from flybot.arms import ARM_NAMES, ARMS, Costs  # noqa: E402
from flybot.market import LONG, load_candles  # noqa: E402
from flybot.mb.graph import load_graph  # noqa: E402
from flybot.mb.model import MBModel  # noqa: E402
from flybot.mock import REGIMES, mock_candles  # noqa: E402
from flybot.pretrain import counterfactual  # noqa: E402
from flybot.sim import run  # noqa: E402

EPISODE = 60


def oracle(df: pd.DataFrame, regime: np.ndarray, costs: Costs, params: dict) -> pd.DataFrame:
    """Every strategy's P&L on every episode, and the best one (flat unless something earns more than nothing)."""
    rows, i = [], LONG
    while i + EPISODE <= len(df):
        ep, hist = df.iloc[i : i + EPISODE], df.iloc[max(0, i - LONG) : i]
        pnls = {a: ARMS[a](ep, hist, costs, params.get(a, {})).pnl for a in ARM_NAMES}
        top = max(pnls, key=pnls.get)
        rows.append({"regime": regime[i], **pnls, "best": top if pnls[top] > 0 else "flat"})
        i += EPISODE
    return pd.DataFrame(rows)


def drawdown(pnl: np.ndarray) -> float:
    cum = np.cumsum(pnl)
    return float((cum - np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]).min()) if len(cum) else 0.0


def job(a: tuple) -> dict:
    seed, days_train, days_test, fly_kw, run_kw, delta_kw = a
    graph = load_graph(ROOT / "data" / "mb")
    costs = Costs()
    train, _ = mock_candles(days_train, 100 + seed)
    test, regime = mock_candles(days_test, 200 + seed)
    orc = oracle(test, regime, costs, {})

    def fly() -> S.Fly:
        return S.Fly(MBModel(graph, seed=seed), seed=seed, **fly_kw)

    pre = fly()
    run(train, pre, EPISODE, costs, **run_kw)
    cf = fly()
    counterfactual(cf, train, EPISODE, costs)
    dl = S.DeltaFly(MBModel(graph, seed=seed), seed=seed, **delta_kw)
    counterfactual(dl, train, EPISODE, costs)
    sels = {"fly_cold": fly(), "fly_pre": pre, "fly_cf": cf, "delta_cf": dl,
            "delta_cold": S.DeltaFly(MBModel(graph, seed=seed), seed=seed, **delta_kw),
            "bandit": S.Bandit(seed), "rule": S.Rule(), "random": S.Random(seed)}
    sels.update({f"only_{a}": S.Single(a) for a in ARM_NAMES if a != "flat"})
    out = {"seed": seed, "oracle_pnl": float(orc[list(ARM_NAMES)].max(axis=1).clip(lower=0).sum()), "sel": {}}
    for name, sel in sels.items():
        d = run(test, sel, EPISODE, costs, **(run_kw if name.startswith(('fly', 'delta')) else {})).frame()
        d["regime"], d["best"] = orc["regime"].to_numpy()[: len(d)], orc["best"].to_numpy()[: len(d)]
        mix = pd.crosstab(d["regime"], d["arm"], normalize="index").round(2)
        out["sel"][name] = {"pnl": float(d["pnl"].sum()), "dd": drawdown(d["pnl"].to_numpy()),
                            "hit": float((d["arm"] == d["best"]).mean()),
                            "flat": float((d["arm"] == "flat").mean()), "mix": mix.to_dict(orient="index")}
    return out


def real_job(a: tuple) -> dict:
    market, seed, params, days_train, fly_kw, delta_kw = a
    graph = load_graph(ROOT / "data" / "mb")
    full = load_candles(market)
    cut = int(len(full) * 0.6)                        # walk forward: learn from the earlier 60%, run the later 40%
    past, df = full.iloc[:cut].reset_index(drop=True), full.iloc[cut - LONG :].reset_index(drop=True)
    p, costs = params.get(market, {}), Costs()
    train, _ = mock_candles(days_train, 100 + seed)

    def fly() -> S.Fly:
        return S.Fly(MBModel(graph, seed=seed), seed=seed, **fly_kw)

    def delta() -> S.DeltaFly:
        return S.DeltaFly(MBModel(graph, seed=seed), seed=seed, **delta_kw)

    pre = fly()
    run(train, pre, EPISODE, costs)
    d_mock, d_real = delta(), delta()
    counterfactual(d_mock, train, EPISODE, costs)
    counterfactual(d_real, past, EPISODE, costs, p)
    out = {"market": market, "seed": seed}
    for name, sel in {"fly_fresh": fly(), "fly_mock_pre": pre, "delta_cold": delta(), "delta_mock_pre": d_mock,
                      "delta_real_pre": d_real, "bandit": S.Bandit(seed),
                      **{a: S.Single(a) for a in ARM_NAMES if a != "flat"}}.items():
        d = run(df, sel, EPISODE, costs, p).frame()
        out[name] = {"pnl": float(d["pnl"].sum()), "dd": drawdown(d["pnl"].to_numpy()), "flat": float((d["arm"] == "flat").mean())}
    return out


def mock_report(res: list[dict]) -> None:
    names = list(res[0]["sel"])
    print(f"mock market, held-out days, mean over {len(res)} seeds (oracle P&L {np.mean([r['oracle_pnl'] for r in res]):.0f})")
    print(f"{'selector':10s} {'pnl':>9s} {'worst drop':>11s} {'best pick':>10s} {'flat share':>11s}")
    for n in sorted(names, key=lambda n: -np.mean([r["sel"][n]["pnl"] for r in res])):
        v = [r["sel"][n] for r in res]
        print(f"{n:10s} {np.mean([x['pnl'] for x in v]):9.0f} {np.mean([x['dd'] for x in v]):11.0f} "
              f"{np.mean([x['hit'] for x in v]):10.2f} {np.mean([x['flat'] for x in v]):11.2f}")
    for n in ("fly_cf", "delta_cold", "delta_cf"):
        tab = pd.DataFrame({r: pd.DataFrame(x["sel"][n]["mix"]).T.reindex(columns=ARM_NAMES).fillna(0).stack() for r, x in enumerate(res)}).mean(axis=1).unstack()
        print(f"\n{n}: share of episodes each strategy was picked, per regime")
        print(tab.reindex(list(REGIMES)).round(2).to_string())


def real_report(res: list[dict]) -> None:
    names = ["fly_fresh", "delta_cold", "delta_mock_pre", "delta_real_pre", "bandit", "maker", "grid", "trend"]
    print("\nreal DreamDEX candles, last 40 percent of the window, mean over seeds (pnl / worst drop)")
    print(f"{'market':12s} " + " ".join(f"{n:>15s}" for n in names))
    tot = {n: [0.0, 0.0] for n in names}
    for m in sorted({r["market"] for r in res}):
        rows = [r for r in res if r["market"] == m]
        cells = []
        for n in names:
            pnl, dd = float(np.mean([r[n]["pnl"] for r in rows])), float(np.mean([r[n]["dd"] for r in rows]))
            tot[n][0] += pnl
            tot[n][1] += dd
            cells.append(f"{pnl:8.0f} /{dd:6.0f}")
        print(f"{m:12s} " + " ".join(f"{c:>15s}" for c in cells))
    print(f"{'total':12s} " + " ".join(f"{tot[n][0]:8.0f} /{tot[n][1]:6.0f}".rjust(15) for n in names))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--train-days", type=int, default=90)
    ap.add_argument("--test-days", type=int, default=45)
    ap.add_argument("--real", action="store_true", help="also run the mock-pretrained fly on real DreamDEX candles")
    ap.add_argument("--fly", default="{}", help="JSON of Fly keyword arguments (margin, explore, directed, ...)")
    ap.add_argument("--run", default="{}", help="JSON of sim.run keyword arguments for the fly (reward_ewma, ...)")
    ap.add_argument("--delta", default="{}", help="JSON of DeltaFly keyword arguments (lr, margin, explore)")
    ap.add_argument("--quiet", action="store_true", help="one summary line for the fly instead of the tables")
    ap.add_argument("--params", default=str(ROOT / "config" / "arm_params.json"))
    a = ap.parse_args()
    fly_kw, run_kw, delta_kw = json.loads(a.fly), json.loads(a.run), json.loads(a.delta)
    out = ROOT / "data" / "mock_results"
    out.mkdir(parents=True, exist_ok=True)
    with Pool(min(a.seeds, 8)) as pool:
        res = pool.map(job, [(s, a.train_days, a.test_days, fly_kw, run_kw, delta_kw) for s in range(a.seeds)])
    (out / "mock.json").write_text(json.dumps(res, indent=1))
    if a.quiet:
        v = {n: [r['sel'][n] for r in res] for n in ('fly_cold', 'fly_pre', 'fly_cf', 'delta_cold', 'delta_cf')}
        print(a.fly, a.run, a.delta, *(f"{n}: pnl {np.mean([x['pnl'] for x in vs]):.0f} drop {np.mean([x['dd'] for x in vs]):.0f} hit {np.mean([x['hit'] for x in vs]):.2f} flat {np.mean([x['flat'] for x in vs]):.2f}" for n, vs in v.items()), sep=' | ')
    else:
        mock_report(res)
    if a.real:
        params = json.loads(Path(a.params).read_text())
        markets = ["SOMI:USDso", "WETH:USDso", "WBTC:USDso"]
        with Pool(min(a.seeds * len(markets), 8)) as pool:
            rres = pool.map(real_job, [(m, s, params, a.train_days, fly_kw, delta_kw) for m in markets for s in range(a.seeds)])
        (out / "real.json").write_text(json.dumps(rres, indent=1))
        real_report(rres)


if __name__ == "__main__":
    main()
