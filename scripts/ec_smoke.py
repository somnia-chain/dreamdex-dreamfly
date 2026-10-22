"""Event contract bet smoke test on mainnet with a throwaway wallet. It spends real USDso: read this first.

It drives the event contract sidecar the way the live mirror does and prints what happened on chain: the wallet
before and after, the window it picked, the bet, the venue's resolution and the claim.

  export PRIVATE_KEY=0x...        # a fresh wallet with a little SOMI for gas and a few USDso, never one you care about
  .venv/bin/python scripts/ec_smoke.py --dry             # everything but the signing: reads the venue, sizes the bet
  .venv/bin/python scripts/ec_smoke.py                   # one real bet of --stake USDso (default 2), then the claim
  .venv/bin/python scripts/ec_smoke.py --claim-only      # just sweep whatever the venue says is claimable

It waits for the next 5 minute window to come within --band seconds of its close (the fly's action band, 25 s),
buys the book's favourite side at its ask, waits for the resolution and claims. One run takes up to ten minutes.
Needs Node 20+ and `npm install` in perp/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybot.live.ec_execution import ECExecutionError, NodeECExecutor  # noqa: E402
from flybot.live.settings import Settings  # noqa: E402


def log(kind: str, row: dict) -> None:
    print(f"  {kind}: {json.dumps(row, default=str)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stake", type=float, default=2.0, help="USDso to bet")
    ap.add_argument("--band", type=float, default=25.0, help="bet when this many seconds are left in the window")
    ap.add_argument("--interval", default="5m", help="window length to pick: 5m, 15m or 1h")
    ap.add_argument("--dry", action="store_true", help="read the venue and size the bet, sign nothing")
    ap.add_argument("--claim-only", action="store_true", help="sweep claimable winnings and exit")
    a = ap.parse_args()

    env = {**os.environ, "NETWORK": "mainnet", "EC_ENABLED": "true", "DRY_RUN": "true" if a.dry else "false",
           "MAINNET_CONFIRM": "yes", "LIVE_STAKE_USDSO": str(max(1.0, a.stake))}
    settings = Settings.from_env(env)
    ex = NodeECExecutor(settings, log, dry_run=a.dry)
    try:
        w = ex.wallet()
        print(f"signer {w and w['address']}: {w and w['somi']:.4f} SOMI, {w and w['usdso']} USDso" if w else "wallet: no signer (dry)")
        if a.claim_only:
            print(json.dumps(ex.sweep()))
            return
        if not a.dry and (not w or (w.get("usdso") or 0) < a.stake or (w.get("somi") or 0) <= 0):
            sys.exit(f"the wallet needs at least {a.stake} USDso and some SOMI for gas")

        # the next window of the wanted length that still has time before the action band; up to three windows
        # are tried, since a book can be empty or one-sided in the last seconds
        side = cap = left = pick = None
        for attempt in range(3):
            pick = None
            while pick is None:
                wins = [x for x in ex.windows() if x.get("interval") == a.interval and (x.get("seconds_left") or 0) > a.band + 5]
                if wins:
                    pick = wins[0]
                    break
                print("no suitable window yet, waiting 20 s ...", flush=True)
                time.sleep(20)
            print(f"window {pick['symbol']} ({pick['market_id']}): {pick['seconds_left']} s left", flush=True)
            while True:
                m = ex.market(pick["market_id"])
                left = m.get("seconds_left") if m else None
                if left is None or left <= a.band:
                    break
                time.sleep(min(10.0, max(1.0, left - a.band)))
            book = ex.book(pick["market_id"]) or {"up": {"asks": []}, "down": {"asks": []}}
            up_ask = book["up"]["asks"][0][0] if book["up"]["asks"] else None
            down_ask = book["down"]["asks"][0][0] if book["down"]["asks"] else None
            print(f"  book: up ask {up_ask}, down ask {down_ask}, {left} s left", flush=True)
            # the favourite is the side the book prices above one half; a lone longshot ask is not it. The fly's own
            # cap stops at 0.95; this smoke proves the order path, so it pays up to 0.999 for a near-certain win
            offers = [(s, p) for s, p in (("up", up_ask), ("down", down_ask)) if p is not None and 0.5 <= p <= 0.999]
            if offers:
                side, cap = max(offers, key=lambda sp: sp[1])
                break
            print("  no favourite on offer in this window, trying the next one", flush=True)
            time.sleep(max(5.0, float(left or 0) + 5))
        if side is None:
            sys.exit("no window offered its favourite at a sensible price; try again later")
        print(f"  betting {a.stake} USDso on {side} at up to {cap}")
        r = ex.buy(pick["market_id"], side, float(cap), a.stake, max(5.0, float(left or 10)))
        print(f"  result: {json.dumps(r, default=str)}")
        if a.dry or not r.get("filled"):
            print("nothing to settle" if not a.dry else "dry run done")
            return
        print(f"  position: {ex.position(pick['market_id'])}")

        print("waiting for the resolution ...", flush=True)
        t_end = time.time() + 15 * 60
        while time.time() < t_end:
            time.sleep(15)
            m = ex.market(pick["market_id"]) or {}
            if m.get("resolved") or m.get("voided"):
                print(f"  resolved: {json.dumps(m)}")
                break
            print(f"  status {m.get('status')} ...", flush=True)
        for i in range(6):
            c = ex.claim(pick["market_id"])
            print(f"  claim: {json.dumps(c, default=str)}")
            if c.get("claimed") or str(c.get("reason", "")).startswith("nothing to claim"):
                break
            time.sleep(20)
        w2 = ex.wallet()
        print(f"wallet after: {w2 and w2['somi']:.4f} SOMI, {w2 and w2['usdso']} USDso")
    except ECExecutionError as e:
        sys.exit(f"sidecar: {e}")
    finally:
        ex.stop()


if __name__ == "__main__":
    main()
