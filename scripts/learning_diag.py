"""Why does the fly sometimes lock onto a losing arm? Run fly variants on every dataset we have and
score each against full information (every arm run on every episode).

Datasets: spot SOMI/WETH/WBTC (candle arms, tuned params, 60-min episodes), perp BTC/ETH (mark-only
replay of the candles through the perp paper engine), EC (recorded windows).
Variants: base; nocouple (pool scores without the MBON->MBON step); cap (one episode may depress a
synapse by at most 20%); pe (prediction-error dopamine); combinations; eta 0.1.
Per run: total P&L, last-third P&L, regret vs the best fixed arm, share of episodes spent on an arm whose
full-information mean is negative ("loser share"), arm mix.

  .venv/bin/python scripts/learning_diag.py --seeds 0 1 --out data/results/learning_diag.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybot.arms import ARM_NAMES, ARMS, Costs  # noqa: E402
from flybot.ec import EC_ARM_FN, EC_ARMS, EC_FEATURES, ECCosts, load_windows  # noqa: E402
from flybot.market import FEATURES, LONG, feature_table, load_candles  # noqa: E402
from flybot.mb.graph import load_graph  # noqa: E402
from flybot.mb.model import MBModel  # noqa: E402
from flybot.perp import LONG as PERP_LONG  # noqa: E402
from flybot.perp import PERP_ARMS, PERP_FEATURES, PerpCosts, PerpPaperEpisode, perp_feature_table, trend_direction  # noqa: E402
from flybot.selectors import Fly  # noqa: E402
from flybot.sim import RewardScaler, reward_of  # noqa: E402
from scripts.warm_start import perp_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "base": {},
    "nocouple": {"couple": False},
    "cap": {"max_depress": 0.2},
    "pe": {"prediction_error": True},
    "nocouple_cap": {"couple": False, "max_depress": 0.2},
    "nocouple_cap_pe": {"couple": False, "max_depress": 0.2, "prediction_error": True},
    "eta01": {"eta": 0.1},
    "nocouple_eta01": {"couple": False, "eta": 0.1},
    # round two: bidirectional plasticity and a reward scale that ignores zero-reward episodes
    "nz": {"_scaler_nz": True},
    "bidir": {"bidirectional": True},
    "bidir_nz": {"bidirectional": True, "_scaler_nz": True},
    "bidir_nocouple_nz": {"bidirectional": True, "couple": False, "_scaler_nz": True},
    "bidir_cap_nz": {"bidirectional": True, "max_depress": 0.2, "_scaler_nz": True},
    "bidir_eta01_nz": {"bidirectional": True, "eta": 0.1, "_scaler_nz": True},
    "bidir_nocouple_cap_nz": {"bidirectional": True, "couple": False, "max_depress": 0.2, "_scaler_nz": True},
}
ROUND2 = ["base", "nz", "bidir", "bidir_nz", "bidir_nocouple_nz", "bidir_cap_nz", "bidir_eta01_nz", "bidir_nocouple_cap_nz"]
# round three: exploitation on top of the zero-skipping scale
VARIANTS.update({
    "nz_m02": {"_scaler_nz": True, "margin": 0.02},
    "nz_m0": {"_scaler_nz": True, "margin": 0.0},
    "nz_e15": {"_scaler_nz": True, "explore": 0.15},
    "nz_m02_e15": {"_scaler_nz": True, "margin": 0.02, "explore": 0.15},
    "nz_bidir_m02": {"_scaler_nz": True, "bidirectional": True, "margin": 0.02},
    "nz_dir1": {"_scaler_nz": True, "directed": 1, "condition_bins": 2, "condition_features": ("vol_short", "trend", "fav_prob", "z_abs")},
    "nz_m02_dir1": {"_scaler_nz": True, "margin": 0.02, "directed": 1, "condition_bins": 2, "condition_features": ("vol_short", "trend", "fav_prob", "z_abs")},
})
ROUND3 = ["nz", "nz_m02", "nz_m0", "nz_e15", "nz_m02_e15", "nz_bidir_m02", "nz_dir1", "nz_m02_dir1"]
# round four: headroom-normalised pool scores (every pool spans the same range whatever cells it holds)
VARIANTS.update({
    "nz_split": {"_scaler_nz": True, "split_score": True},
    "nz_split_m02": {"_scaler_nz": True, "split_score": True, "margin": 0.02},
    "nz_split_m10": {"_scaler_nz": True, "split_score": True, "margin": 0.10},
    "nz_split_bidir": {"_scaler_nz": True, "split_score": True, "bidirectional": True},
    "nz_split_dir1": {"_scaler_nz": True, "split_score": True, "directed": 1, "condition_bins": 2, "condition_features": ("vol_short", "trend", "fav_prob", "z_abs")},
    "nz_split_nocouple": {"_scaler_nz": True, "split_score": True, "couple": False},
})
ROUND4 = ["nz", "nz_split", "nz_split_m02", "nz_split_m10", "nz_split_bidir", "nz_split_dir1", "nz_split_nocouple"]
# round five: pools dealt to equalise approach and avoid drive at construction
VARIANTS.update({
    "nz_bal": {"_scaler_nz": True, "balanced_pools": True},
    "nz_bal_bidir": {"_scaler_nz": True, "balanced_pools": True, "bidirectional": True},
    "nz_bal_m02": {"_scaler_nz": True, "balanced_pools": True, "margin": 0.02},
    "nz_bal_m03_bidir": {"_scaler_nz": True, "balanced_pools": True, "margin": 0.03, "bidirectional": True},
})
ROUND5 = ["nz_bal", "nz_bal_bidir", "nz_bal_m02", "nz_bal_m03_bidir"]
# round six: slower homeostatic recovery so evidence gathered through exploration persists
VARIANTS.update({
    "nz_r002": {"_scaler_nz": True, "recover": 0.002},
    "nz_r001": {"_scaler_nz": True, "recover": 0.001},
    "nz_r0": {"_scaler_nz": True, "recover": 0.0},
    "nz_bidir_r001": {"_scaler_nz": True, "bidirectional": True, "recover": 0.001},
    "nz_bidir_r0": {"_scaler_nz": True, "bidirectional": True, "recover": 0.0},
    "nz_r001_m03": {"_scaler_nz": True, "recover": 0.001, "margin": 0.03},
})
ROUND6 = ["nz_r002", "nz_r001", "nz_r0", "nz_bidir_r001", "nz_bidir_r0", "nz_r001_m03"]
# round eight: the scale also ignores gas-only rewards (|reward| below eps)
VARIANTS.update({
    "nz_eps003": {"_scaler_nz": True, "_scaler_eps": 0.003},
    "nz_eps01": {"_scaler_nz": True, "_scaler_eps": 0.01},
    "nz_eps03": {"_scaler_nz": True, "_scaler_eps": 0.03},
    "nz_eps1": {"_scaler_nz": True, "_scaler_eps": 0.1},
})
ROUND8 = ["nz", "nz_eps003", "nz_eps01", "nz_eps03", "nz_eps1"]


# ---------- datasets: each yields (features frame, list of episode contexts, arm names, pnl_fn) ----------
def spot_dataset(sym: str, params: dict, episode_min: int):
    df = load_candles(sym)
    feats = feature_table(df)
    costs = Costs()
    idx = list(range(LONG, len(df) - episode_min, episode_min))

    def pnl(i: int, arm: str) -> tuple[float, float]:
        ep = df.iloc[i : i + episode_min]
        res = ARMS[arm](ep, df.iloc[max(0, i - LONG) : i], costs, params.get(sym, {}).get(arm, {}))
        return res.pnl, reward_of(res.pnl, ep, costs, clip=np.inf)

    return feats, idx, ARM_NAMES, FEATURES, pnl, True


def perp_dataset(src: str, params: dict, episode_min: int):
    df = perp_frame(src)
    feats = perp_feature_table(df)
    costs = PerpCosts(leverage=2.0)
    idx = list(range(PERP_LONG, len(df) - episode_min - 1, episode_min))

    def pnl(i: int, arm: str) -> tuple[float, float]:
        ep = PerpPaperEpisode(arm, costs, params.get(arm, {}))
        ep.start(df.iloc[i], trend_direction(df.iloc[: i + 1], params.get("p_trend", {})) if arm == "p_trend" else 0)
        for j in range(i + 1, i + episode_min):
            ep.on_row(df.iloc[j])
        p = ep.finish(df.iloc[i + episode_min])
        win = df.iloc[i : i + episode_min + 1].rename(columns={"mark": "close"})
        return p, reward_of(p, win, costs, clip=np.inf)

    return feats, idx, PERP_ARMS, PERP_FEATURES, pnl, True


def ec_dataset(db: Path, params: dict):
    w = load_windows(db)
    costs = ECCosts()
    idx = list(range(len(w)))

    def pnl(i: int, arm: str) -> tuple[float, float]:
        p = EC_ARM_FN[arm](w.iloc[i], costs, params.get("EC", {}).get(arm, {}))
        return p, float(np.clip(p / costs.stake, -3, 3))

    return w, idx, EC_ARMS, EC_FEATURES, pnl, False


def full_info(idx, arms, pnl_fn) -> pd.DataFrame:
    """P&L of every arm on every episode."""
    table = {a: [] for a in arms}
    for i in idx:
        for a in arms:
            table[a].append(pnl_fn(i, a)[0])
    return pd.DataFrame(table)


def run_fly(graph, seed: int, kw: dict, feats, idx, arms, features, table: pd.DataFrame, use_scaler: bool) -> dict:
    kw = dict(kw)
    nz = bool(kw.pop("_scaler_nz", False))
    eps = float(kw.pop("_scaler_eps", 0.0))
    fly = Fly(MBModel(graph, seed=seed), seed=seed, arms=arms, features=features, **kw)
    scaler = RewardScaler(skip_zero=nz, eps=eps) if use_scaler else (lambda x: x)
    chosen, pnls = [], []
    for n, i in enumerate(idx):
        k = fly.choose(feats.iloc[i])
        arm = arms[k]
        p = float(table[arm].iloc[n])
        # reward as the runner would compute it: recompute the vol-scaled raw reward cheaply from P&L
        raw = p / (table["_unit"].iloc[n]) if "_unit" in table else p / 20.0  # EC: P&L over stake
        fly.update(scaler(raw) if use_scaler else float(np.clip(raw, -3, 3)))
        chosen.append(arm)
        pnls.append(p)
    d = pd.DataFrame({"arm": chosen, "pnl": pnls})
    means = table[list(arms)].mean()
    best_fixed = float(table[list(arms)].sum().max())
    losers = set(a for a in arms if means[a] < 0)
    third = len(d) // 3
    return {"pnl": round(float(d["pnl"].sum()), 2), "last_third": round(float(d["pnl"].iloc[-third:].sum()), 2),
            "best_fixed": round(best_fixed, 2), "regret": round(best_fixed - float(d["pnl"].sum()), 2),
            "loser_share": round(float(d["arm"].isin(losers).mean()), 3),
            "mix": json.dumps(d["arm"].value_counts(normalize=True).round(2).to_dict()),
            "weights_moved": round(float((np.abs(fly.m.w_kc_mbon - fly.m.w_kc_mbon0) > 0.01 * fly.m.w_kc_mbon0).mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--variants", nargs="*", default=ROUND3)
    ap.add_argument("--datasets", nargs="*", default=["perp:WETH:USDso", "perp:WBTC:USDso", "spot:SOMI:USDso", "spot:WETH:USDso", "spot:WBTC:USDso", "ec"])
    ap.add_argument("--params", default=str(ROOT / "data" / "results" / "arm_params.json"))
    ap.add_argument("--ec-db", default=str(ROOT / "data" / "ec" / "windows.db"))
    ap.add_argument("--episode-min", type=int, default=60)
    ap.add_argument("--ec-scaler", action="store_true", help="EC rewards through the RewardScaler like spot and perps")
    ap.add_argument("--out", default=str(ROOT / "data" / "results" / "learning_diag.csv"))
    a = ap.parse_args()
    params = json.loads(Path(a.params).read_text())
    graph = load_graph(ROOT / "data" / "mb")
    rows = []
    for ds in a.datasets:
        t0 = time.time()
        if ds.startswith("spot:"):
            feats, idx, arms, features, pnl_fn, use_scaler = spot_dataset(ds[5:], params, a.episode_min)
        elif ds.startswith("perp:"):
            feats, idx, arms, features, pnl_fn, use_scaler = perp_dataset(ds[5:], params, a.episode_min)
        else:
            feats, idx, arms, features, pnl_fn, use_scaler = ec_dataset(Path(a.ec_db), params)
            use_scaler = a.ec_scaler
        table = full_info(idx, arms, pnl_fn)
        if use_scaler:
            # vol unit per episode so the fly's reward matches the runner's reward_of without re-running arms
            units = []
            for n, i in enumerate(idx):
                p = table[arms[0]].iloc[n]
                _, raw = pnl_fn(i, arms[0])
                units.append(p / raw if raw != 0 else np.nan)
            u = pd.Series(units)
            table["_unit"] = u.fillna(u.median() if u.notna().any() else 1.0).abs()
        means = table[list(arms)].mean().round(3).to_dict()
        print(f"{ds}: {len(idx)} episodes, full-info arm means {means}, best fixed {table[list(arms)].sum().max():.1f} ({time.time() - t0:.0f} s)", flush=True)
        for v in a.variants:
            for s in a.seeds:
                r = run_fly(graph, s, VARIANTS[v], feats, idx, arms, features, table, use_scaler)
                rows.append({"dataset": ds, "variant": v, "seed": s, **r})
                print(f"  {v:16s} s{s} pnl {r['pnl']:9.2f} last3 {r['last_third']:8.2f} regret {r['regret']:8.2f} losers {r['loser_share']:.2f} mix {r['mix']}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    summ = out.groupby(["dataset", "variant"])[["pnl", "last_third", "regret", "loser_share"]].mean().round(2)
    print(summ.to_string(), flush=True)
    print(f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
