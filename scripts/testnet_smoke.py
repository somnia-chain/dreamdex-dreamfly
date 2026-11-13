"""Live order smoke test on the TESTNET with a throwaway wallet. It sends real (test) orders, so read this first.

It drives the order mirror with one paper episode of a chosen strategy and prints what happened on chain: every order
placed or cancelled, whether it went through, and the wallet before and after. At the end the mirror cancels what is
still resting and closes any inventory with a taker order, the same as the live runner does.

  export PRIVATE_KEY=0x...        # a fresh wallet funded from the testnet faucet, never one you care about
  .venv/bin/python scripts/testnet_smoke.py --arm maker --seconds 90
  .venv/bin/python scripts/testnet_smoke.py --arm trend --seconds 40 --take-buffer-bps 50

  maker  two post-only quotes around the touch, re-quoted as it moves, all cancelled at the end
  grid   a ladder of post-only quotes, all cancelled at the end
  trend  one taker buy through the touch, then a taker sell at the end (this one really trades)

It refuses to run on mainnet or in a dry run. Stake is small by default (10 USDso). Needs `pip install -e ".[live]"`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybot.arms import Costs  # noqa: E402
from flybot.live.execution import SpotMirror, build_executor  # noqa: E402
from flybot.live.paper import PaperEpisode  # noqa: E402
from flybot.live.settings import Settings  # noqa: E402

PAPER_STAKE = 1000.0       # the paper episode's size; the mirror scales every order down to --stake


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default="SOMI:USDso")
    ap.add_argument("--arm", choices=["maker", "grid", "trend"], default="maker")
    ap.add_argument("--seconds", type=int, default=90, help="how long the episode runs")
    ap.add_argument("--stake", type=float, default=10.0, help="USDso at work, whatever the paper episode does")
    ap.add_argument("--take-buffer-bps", type=float, default=None, help="how far through the touch a taker is priced (default 10)")
    ap.add_argument("--out", default="", help="write the order log here as JSON")
    a = ap.parse_args()

    env = {**os.environ, "NETWORK": "testnet", "DRY_RUN": "false", "LIVE_STAKE_USDSO": str(a.stake), "MARKETS": a.market}
    if a.take_buffer_bps is not None:
        env["TAKE_BUFFER_BPS"] = str(a.take_buffer_bps)
    if os.environ.get("NETWORK", "testnet").lower() != "testnet":
        sys.exit("this script only runs on the testnet")
    if not os.environ.get("PRIVATE_KEY"):
        sys.exit("set PRIVATE_KEY to a throwaway testnet wallet first")
    settings = Settings.from_env(env)
    import dreamdex_core as core
    from dreamdex_core.contract import ERC20_ABI
    from web3 import Web3

    rows: list[dict] = []

    def log(kind: str, row: dict) -> None:
        rows.append(row)
        keep = {k: row[k] for k in ("action", "side", "price", "qty", "kind", "ok", "id", "error") if k in row}
        print("  ", json.dumps(keep, default=str)[:260], flush=True)

    os.environ["NETWORK"] = "testnet"
    executor = build_executor(settings, log)
    ctx = core.create_chain_context()
    pool = core.Pool.load(ctx, a.market)
    quote = ctx.w3.eth.contract(address=Web3.to_checksum_address(pool.params.quote_token), abi=ERC20_ABI)

    def balances() -> dict:
        return {native: round(ctx.w3.eth.get_balance(ctx.address) / 1e18, 4) for native in [ctx.net.native_symbol]} | {
            "quote": round(quote.functions.balanceOf(ctx.address).call() / 10 ** pool.quote_decimals, 4),
            "base_in_wallet": round(pool.wallet_base(), 6), "base_in_vault": round(pool.vault_base(), 6)}

    def touch() -> tuple[float, float]:
        t = pool.top_of_book()
        if t.best_bid is None or t.best_ask is None:
            raise RuntimeError(f"one side of the book is empty (bid {t.best_bid}, ask {t.best_ask})")
        return float(t.best_bid), float(t.best_ask)

    print(f"{a.market} on testnet, {a.arm} for {a.seconds}s, stake {a.stake} USDso, address {ctx.address}")
    print("before", balances(), flush=True)
    mirror = SpotMirror(executor, a.market, PAPER_STAKE, settings, log)
    bb, ba = touch()
    params = {"spread_bps": 40, "skew_bps": 0} if a.arm == "maker" else {"levels": 2, "step_bps": 40} if a.arm == "grid" else {}
    ep = PaperEpisode(a.arm, Costs(stake=PAPER_STAKE), params)
    ep.start(int(time.time() * 1000), bb, ba, 1 if a.arm == "trend" else 0)
    mirror.on_start(ep, bb, ba)
    end = time.time() + a.seconds
    while time.time() < end:
        time.sleep(4)
        try:
            bb, ba = touch()
        except Exception as e:  # noqa: BLE001 - a bad read just skips this look
            print("   could not read the book:", str(e)[:100])
            continue
        ep.on_touch(int(time.time() * 1000), bb, ba)
        mirror.sync(ep, bb, ba)
    bb, ba = touch()
    pnl = ep.finish(int(time.time() * 1000), bb, ba)
    mirror.finish(ep, bb, ba)
    time.sleep(6)
    print("paper pnl", round(pnl, 4), "paper fills", len(ep.fills))
    print("after ", balances())
    failed = [r for r in rows if r.get("ok") is False]
    print(f"orders logged {len(rows)}, ok {sum(1 for r in rows if r.get('ok') is True)}, failed {len(failed)}")
    for r in failed[:5]:
        print("  FAILED", r.get("action"), r.get("side"), str(r.get("error"))[:200])
    if a.out:
        Path(a.out).write_text(json.dumps(rows, default=str, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
