"""Warm-start the live flies: replay recorded history through the same learning code the runner uses,
so each fly wakes up with the phase-2 knowledge instead of the untouched wiring.

  spot   1-minute candles (data/candles) -> candle-level arms, 60-min episodes, adaptive reward
  perps  the BTC / ETH / SOL candle history as a mark series (no funding, basis or open-interest history
         exists yet; those features stay unknown and the encoder skips them)
  EC     recorded resolved windows (data/ec/*.db) -> the EC arms, reward = P&L / stake

Writes weights, reward scaler and encoder history to <state>/<name>/ and a warm.jsonl summary. Equity
and episode counters stay at zero: warm-start P&L is history, not paper-live results.

  .venv/bin/python scripts/warm_start.py --state state/warm --params data/results/arm_params.json
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
from flybot.ec import EC_ARM_FN, EC_ARMS, ECCosts, load_windows  # noqa: E402
from flybot.live.ec_live import ECFly  # noqa: E402
from flybot.live.perp_live import PerpFly  # noqa: E402
from flybot.live.runner import MarketFly  # noqa: E402
from flybot.market import LONG, feature_table, load_candles  # noqa: E402
from flybot.mb.graph import load_graph  # noqa: E402
from flybot.perp import LONG as PERP_LONG  # noqa: E402
from flybot.perp import PERP_ARMS, PerpCosts, PerpPaperEpisode, perp_feature_table, trend_direction  # noqa: E402
from flybot.sim import reward_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPOT = ["SOMI:USDso", "WETH:USDso", "WBTC:USDso"]
PERP = {"BTC/USDso:USDso": "WBTC:USDso", "ETH/USDso:USDso": "WETH:USDso", "SOL/USDso:USDso": None}


def report(fly, name: str, rows: list[dict]) -> dict:
    d = pd.DataFrame(rows)
    mix = d["arm"].value_counts(normalize=True).round(2).to_dict() if len(d) else {}
    last = d.tail(max(1, len(d) // 5))
    out = {"name": name, "episodes": len(d), "pnl": round(float(d["pnl"].sum()), 2) if len(d) else 0.0,
           "pnl_last_fifth": round(float(last["pnl"].sum()), 2) if len(d) else 0.0, "mix": mix,
           "mix_last_fifth": last["arm"].value_counts(normalize=True).round(2).to_dict() if len(d) else {},
           "weights_moved": fly.weights_moved(), "scaler": round(fly.scaler.scale, 5)}
    fly.log("warm", out)
    fly.save()
    print(json.dumps(out), flush=True)
    return out


def warm_spot(graph, state: Path, params: dict, seed: int, since: str | None, episode_min: int) -> None:
    for sym in SPOT:
        df = load_candles(sym)
        if since:
            df = df[df["ts"] >= pd.Timestamp(since, tz="UTC")].reset_index(drop=True)
        mf = MarketFly(sym, graph, state, Costs(), params.get(sym, {}), seed, episode_min)
        feats = feature_table(df)
        rows = []
        i = LONG
        while i + episode_min <= len(df):
            f = feats.iloc[i]
            k = mf.fly.choose(f)
            arm = ARM_NAMES[k]
            ep = df.iloc[i : i + episode_min]
            res = ARMS[arm](ep, df.iloc[max(0, i - LONG) : i], mf.costs, mf.params.get(arm, {}))
            rew = mf.scaler(reward_of(res.pnl, ep, mf.costs, clip=np.inf))
            mf.fly.update(rew)
            rows.append({"arm": arm, "pnl": res.pnl})
            i += episode_min
        report(mf, sym, rows)


def perp_frame(sym: str) -> pd.DataFrame:
    """A perp-tape-shaped frame from spot candles: mark = close, book = close +/- 2 bps, funding,
    basis and open interest unknown."""
    df = load_candles(sym)
    out = pd.DataFrame({"timestamp": df["timestamp"], "ts": df["ts"], "mark": df["close"], "index": np.nan,
                        "funding_8h": np.nan, "oi": np.nan,
                        "best_bid": df["close"] * (1 - 2e-4), "best_ask": df["close"] * (1 + 2e-4)})
    return out.reset_index(drop=True)


def warm_perps(graph, state: Path, params: dict, seed: int, since: str | None, episode_min: int, leverage: float) -> None:
    for sym, src in PERP.items():
        if src is None:
            print(f"{sym}: no candle history for this asset, left untrained", flush=True)
            continue
        df = perp_frame(src)
        if since:
            df = df[df["ts"] >= pd.Timestamp(since, tz="UTC")].reset_index(drop=True)
        pf = PerpFly(sym, graph, state, PerpCosts(leverage=leverage), params.get(sym, {}), seed, episode_min)
        feats = perp_feature_table(df)
        rows = []
        i = PERP_LONG
        while i + episode_min <= len(df):
            f = feats.iloc[i]
            k = pf.fly.choose(f)
            arm = PERP_ARMS[k]
            hist = df.iloc[: i + 1]
            ep = PerpPaperEpisode(arm, pf.costs, pf.params.get(arm, {}))
            ep.start(df.iloc[i], trend_direction(hist, pf.params.get("p_trend", {})) if arm == "p_trend" else 0)
            for j in range(i + 1, i + episode_min):
                ep.on_row(df.iloc[j])
            pnl = ep.finish(df.iloc[i + episode_min])
            win = df.iloc[i : i + episode_min + 1].rename(columns={"mark": "close"})
            rew = pf.scaler(reward_of(pnl, win, pf.costs, clip=np.inf))
            pf.fly.update(rew)
            rows.append({"arm": arm, "pnl": pnl})
            i += episode_min
        report(pf, sym, rows)


def warm_ec(graph, state: Path, params: dict, seed: int, db: Path, stake: float) -> None:
    w = load_windows(db)
    ec = ECFly(graph, state, ECCosts(stake=stake), params.get("EC", {}), seed)
    rows = []
    for _, r in w.iterrows():
        k = ec.fly.choose(r)
        arm = EC_ARMS[k]
        pnl = EC_ARM_FN[arm](r, ec.costs, ec.params.get(arm, {}))
        ec.fly.update(ec.scaler(pnl / ec.costs.stake))
        rows.append({"arm": arm, "pnl": pnl})
    report(ec, "EC", rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default=str(ROOT / "state" / "warm"))
    ap.add_argument("--params", default=str(ROOT / "data" / "results" / "arm_params.json"))
    ap.add_argument("--since", default="", help="ISO date; default: the whole candle history")
    ap.add_argument("--episode-min", type=int, default=60)
    ap.add_argument("--perp-leverage", type=float, default=2.0)
    ap.add_argument("--ec-db", default=str(ROOT / "data" / "ec" / "windows.db"))
    ap.add_argument("--ec-stake", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=["spot", "perps", "ec"], choices=["spot", "perps", "ec"])
    a = ap.parse_args()
    state = Path(a.state)
    state.mkdir(parents=True, exist_ok=True)
    params = json.loads(Path(a.params).read_text()) if Path(a.params).exists() else {}
    graph = load_graph(ROOT / "data" / "mb")
    t0 = time.time()
    if "spot" in a.only:
        warm_spot(graph, state, params, a.seed, a.since or None, a.episode_min)
    if "perps" in a.only:
        warm_perps(graph, state, params, a.seed, a.since or None, a.episode_min, a.perp_leverage)
    if "ec" in a.only:
        warm_ec(graph, state, params, a.seed, Path(a.ec_db), a.ec_stake)
    print(f"done in {time.time() - t0:.0f} s -> {state}", flush=True)


if __name__ == "__main__":
    main()
