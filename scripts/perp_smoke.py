"""Perp order smoke test on the Hideki test network with a throwaway wallet. It sends real (test) orders: read this first.

It drives the perp order mirror with one paper episode of a chosen strategy and prints what happened on chain: the
margin account before and after, every order placed or cancelled, and the position. At the end the mirror cancels
what is resting and closes the position with a taker order, the same as the live runner does.

  export PRIVATE_KEY=0x...        # a fresh wallet with Hideki STT for gas, never one you care about
  .venv/bin/python scripts/perp_smoke.py --deposit 50          # once: move 50 USDso from the wallet into the Perps account
  .venv/bin/python scripts/perp_smoke.py --arm p_trend --seconds 60
  .venv/bin/python scripts/perp_smoke.py --arm p_maker --seconds 120

  p_trend  one taker order at the touch, then a taker close at the end (this one really trades)
  p_carry  the side funding pays, held, then closed at the end (the smoke does not keep it open)
  p_maker  two post only quotes around the mark, re-quoted as the paper quotes move, all cancelled at the end

With OWNER_ADDRESS set the key trades for that account (linked bot key) and --deposit is refused: fund the account in
the app. It refuses to run in a dry run and never runs on mainnet (there are no perp markets there). Stake is small
by default (10 USDso margin at 2x). Needs Node 20+ and `npm install` in perp/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybot.live.perp_execution import NodePerpExecutor, PerpExecutionError, PerpMirror  # noqa: E402
from flybot.live.settings import Settings  # noqa: E402
from flybot.perp import PerpCosts, PerpPaperEpisode, rows_to_frame  # noqa: E402

PAPER = PerpCosts(stake=1000.0, leverage=2.0)      # the paper episode's size; the mirror scales every order down to --stake


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default="BTC/USDso:USDso")
    ap.add_argument("--arm", choices=["p_trend", "p_carry", "p_maker"], default="p_trend")
    ap.add_argument("--seconds", type=int, default=60, help="how long the episode runs before it is closed")
    ap.add_argument("--stake", type=float, default=10.0, help="live margin in USDso")
    ap.add_argument("--leverage", type=float, default=2.0)
    ap.add_argument("--deposit", type=float, default=0.0, help="move this much USDso from the wallet into the Perps account and exit")
    a = ap.parse_args()

    env = {**os.environ, "DRY_RUN": "false", "NETWORK": "hideki", "MARKETS": "", "EC_ENABLED": "false", "PERP_MARKETS": a.market,
           "LIVE_STAKE_USDSO": str(a.stake), "PERP_LEVERAGE": str(a.leverage)}
    settings = Settings.from_env(env)
    assert settings.perp_live, "this script only makes sense with a key and live orders"

    rows: list[dict] = []

    def log(kind: str, row: dict) -> None:
        rows.append(row)
        print(f"  {kind}: {json.dumps(row, default=str)}", flush=True)

    ex = NodePerpExecutor(settings, log)
    try:
        wallet = ex.wallet()
        pre = ex.preflight(a.market)
        print(f"signer {wallet and wallet['address']}: {wallet and wallet['stt']:.4f} STT, {wallet and wallet['usdso']:.2f} USDso in the wallet")
        print(f"perps account {ex.address}: {json.dumps(pre)}")
        if a.deposit:
            print(f"depositing {a.deposit} USDso into the MarginBank ...")
            print(ex.deposit(a.market, a.deposit))
            print(json.dumps(ex.preflight(a.market)))
            return
        if not pre or pre.get("unlocked", 0) < a.stake:
            sys.exit(f"the Perps account has {pre and pre.get('unlocked')} USDso unlocked, less than the {a.stake} stake: "
                     "run with --deposit first (own account) or fund the account in the app (linked key)")
        if settings.owner_address and not pre.get("linked"):
            sys.exit("this key is not linked to OWNER_ADDRESS on this market: use Link a bot in the app")

        mark = ex.mark(a.market)
        bid, ask = ex.book(a.market)
        if not mark or bid is None or ask is None:
            sys.exit(f"market not priceable or book empty (mark {mark}, bid {bid}, ask {ask})")
        print(f"{a.market}: mark {mark} bid {bid} ask {ask}")

        # one tape row, built from the live reads, is enough for the paper episode to start
        def row(i: int) -> dict:
            m = ex.mark(a.market) or mark
            b, s = ex.book(a.market)
            return {"type": "perp", "symbol": a.market, "wall": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + i)),
                    "mark": m, "row_mark": m, "index": m, "funding_8h": 0.0005, "oi": 0.0, "best_bid": b or m, "best_ask": s or m}

        df = rows_to_frame([row(0)])
        ep = PerpPaperEpisode(a.arm, PAPER, {"min_apr": 0.0} if a.arm == "p_carry" else {})
        ep.start(df.iloc[0], trend_dir=1)
        mirror = PerpMirror(ex, a.market, PAPER.stake, PAPER.leverage, settings, log)
        print(f"start {a.arm}: paper pos {ep.pos:.6f}, scaled {ep.pos * mirror.scale:.6f}")
        mirror.on_start(ep)
        print(f"position after start: {ex.position(a.market)}")
        t_end = time.time() + a.seconds
        i = 0
        while time.time() < t_end:
            time.sleep(min(15, max(1, t_end - time.time())))
            i += 1
            df = rows_to_frame([row(0), row(60 * i)])
            ep.on_row(df.iloc[-1])
            mirror.on_row(ep)
            print(f"  t+{int(time.time() - (t_end - a.seconds))}s paper pos {ep.pos:.6f} quotes {ep.bids} {ep.asks} | chain pos {ex.position(a.market)} "
                  f"open {len(ex.open_orders(a.market) or [])}")
        ep.finish(df.iloc[-1])
        ep.keep_open = False
        mirror.finish(ep)
        time.sleep(3)
        print(f"position after finish: {ex.position(a.market)}; open orders: {ex.open_orders(a.market)}")
        print(f"perps account after: {json.dumps(ex.preflight(a.market))}")
        bad = [r for r in rows if r.get("ok") is False]
        print(f"{len(rows)} order log rows, {len(bad)} failed")
    except PerpExecutionError as e:
        sys.exit(f"sidecar: {e}")
    finally:
        ex.stop()


if __name__ == "__main__":
    main()
