"""Regime-discrimination test on synthetic candles: alternating segments where a known arm should win.

trend regime   : steady drift, low noise           -> trend should win
oscillating    : mean-reverting sinusoid + noise   -> maker / grid should win
noisy          : high-vol random walk              -> flat should win

Reports, per selector, P&L and how often it chose the regime's best arm (measured from the arms' own
P&L on each episode), plus a learning curve for the fly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybot import selectors as S  # noqa: E402
from flybot.arms import ARM_NAMES, ARMS, Costs  # noqa: E402
from flybot.market import LONG  # noqa: E402
from flybot.mb.graph import load_graph  # noqa: E402
from flybot.mb.model import MBModel  # noqa: E402
from flybot.sim import run  # noqa: E402

REGIMES = ("trend", "oscillating", "noisy")


def synth(days: int, seg_hours: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = days * 24 * 60
    seg = seg_hours * 60
    price = np.empty(n)
    regime = np.empty(n, dtype=object)
    p = 100.0
    t = 0
    while t < n:
        r = REGIMES[rng.integers(len(REGIMES))]
        m = min(seg, n - t)
        if r == "trend":
            drift = rng.choice([-1, 1]) * 0.0004
            steps = drift + rng.normal(0, 0.0003, m)
            seg_p = p * np.exp(np.cumsum(steps))
        elif r == "oscillating":
            phase = rng.uniform(0, 2 * np.pi)
            seg_p = p * (1 + 0.006 * np.sin(phase + 2 * np.pi * np.arange(m) / 40)) * np.exp(
                np.cumsum(rng.normal(0, 0.0004, m)))
        else:
            seg_p = p * np.exp(np.cumsum(rng.normal(0, 0.003, m)))
        price[t : t + m] = seg_p
        regime[t : t + m] = r
        p = seg_p[-1]
        t += m
    o = np.concatenate([[price[0]], price[:-1]])
    noise = np.abs(rng.normal(0, 0.0005, n))
    df = pd.DataFrame({
        "open": o, "close": price,
        "high": np.maximum(o, price) * (1 + noise), "low": np.minimum(o, price) * (1 - noise),
        "volume": rng.lognormal(3, 0.5, n),
    })
    df["timestamp"] = (pd.Timestamp("2026-07-01", tz="UTC").value // 10**6) + np.arange(n) * 60_000
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df, regime


def best_arm_per_episode(df: pd.DataFrame, episode: int, costs: Costs) -> pd.DataFrame:
    rows = []
    i = LONG
    while i + episode <= len(df):
        ep, hist = df.iloc[i : i + episode], df.iloc[max(0, i - LONG) : i]
        pnls = {a: ARMS[a](ep, hist, costs, {}).pnl for a in ARM_NAMES}
        rows.append({"start": df["ts"].iloc[i], **pnls, "best": max(pnls, key=pnls.get)})
        i += episode
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--seg-hours", type=int, default=12)
    ap.add_argument("--episode", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--taker-bps", type=float, default=5.0)
    a = ap.parse_args()
    costs = Costs(taker_bps=a.taker_bps, slip_bps=0.0)
    graph = load_graph(ROOT / "data" / "mb")
    out = ROOT / "data" / "results"
    out.mkdir(parents=True, exist_ok=True)

    rows, curves = [], []
    for seed in range(a.seeds):
        df, regime = synth(a.days, a.seg_hours, seed)
        oracle = best_arm_per_episode(df, a.episode, costs)
        oracle["regime"] = [regime[df.index[df["ts"] == s][0]] for s in oracle["start"]]
        skip = 7 * 24 * 60 // a.episode
        if seed == 0:
            print("best arm by regime (share of episodes, oracle):")
            print(pd.crosstab(oracle["regime"], oracle["best"], normalize="index").round(2).to_string())
            print(f"oracle P&L after warm-up: {oracle.iloc[skip:][list(ARM_NAMES)].max(axis=1).sum():.0f}\n")
        sels = {"fly": S.Fly(MBModel(graph, seed=seed), seed=seed), "bandit": S.Bandit(seed),
                "rule": S.Rule(), "random": S.Random(seed)}
        sels.update({f"only_{x}": S.Single(x) for x in ARM_NAMES})
        for name, sel in sels.items():
            res = run(df, sel, a.episode, costs=costs).frame()
            res["best"] = oracle["best"].to_numpy()
            res["regime"] = oracle["regime"].to_numpy()
            test = res.iloc[skip:]
            hit = float((test["arm"] == test["best"]).mean())
            by_regime = test.groupby("regime").apply(lambda d: (d["arm"] == d["best"]).mean()).round(2).to_dict()
            rows.append({"seed": seed, "selector": name, "pnl": test["pnl"].sum(), "hit_rate": hit, **by_regime})
            if name == "fly":
                w = res.assign(hit=(res["arm"] == res["best"]).astype(float))
                curves.append(w.groupby(w.index // 24)["hit"].mean().rename(f"seed{seed}"))
    t = pd.DataFrame(rows)
    t.to_csv(out / "synthetic_results.csv", index=False)
    print(f"episode {a.episode} min, {a.days} days, segments {a.seg_hours} h, taker {a.taker_bps} bps; mean over {a.seeds} seeds")
    print(t.groupby("selector")[["pnl", "hit_rate", *REGIMES]].mean().round(2).sort_values("pnl", ascending=False).to_string())
    lc = pd.concat(curves, axis=1).mean(axis=1)
    print("\nfly hit-rate learning curve (per day):")
    print(lc.round(2).to_string())
    lc.to_csv(out / "synthetic_fly_curve.csv")


if __name__ == "__main__":
    main()
