#!/usr/bin/env python3
"""Read-only setup check. Sends no orders and, unless you pass --wallet, never touches your key.

  dreamfly doctor            settings, bundled data, DreamDEX reachability, the markets you asked for
  dreamfly doctor --wallet   also loads the key from PRIVATE_KEY and prints the address and balances
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybot.live.settings import Settings, SettingsError  # noqa: E402

REST = {"mainnet": "https://api.dreamdex.io", "testnet": "https://stg.api.dreamdex.io"}
ok_n = bad_n = 0


def check(ok: bool, what: str, hint: str = "") -> bool:
    global ok_n, bad_n
    ok_n += ok
    bad_n += not ok
    print(("  ok    " if ok else "  FAIL  ") + what + ("" if ok or not hint else f"\n          {hint}"))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wallet", action="store_true", help="load PRIVATE_KEY and show the address and balances (read-only)")
    a = ap.parse_args()

    print("settings")
    try:
        s = Settings.from_env()
    except SettingsError as e:
        check(False, f"settings: {e}")
        sys.exit(1)
    check(True, s.summary())
    print(f"  markets: {', '.join(s.markets)}" + (f"; perps: {', '.join(s.perp_markets)}" if s.perp_markets else "")
          + ("; event contracts on" if s.ec else ""))

    print("bundled data")
    for rel in ("data/mb/edges.parquet", "data/mb/neurons.parquet", "site/data/brain.json", "site/data/somas.json",
                "config/arm_params.json"):
        check((ROOT / rel).exists(), rel, "run scripts/extract_mb.py, build_brain_mesh.py and build_soma_cloud.py, or re-clone")
    folder = "warm_delta" if s.brain == "delta" else "warm"
    warm = [m.replace(":", "_") for m in s.markets if (ROOT / "config" / folder / m.replace(":", "_")).exists()]
    print(f"  brain: {s.brain}, warm-started for: {', '.join(warm) or 'none (the flies start blank)'}")

    print("DreamDEX")
    if s.network in REST:
        try:
            with urllib.request.urlopen(REST[s.network] + "/v0/markets", timeout=15) as r:
                body = json.load(r)
                listed = {m.get("symbol") for m in (body["markets"] if isinstance(body, dict) else body)}
            check(True, f"{s.network} API answers, {len(listed)} markets")
            for m in s.markets:
                check(m in listed, f"market {m} exists on {s.network}", f"listed: {', '.join(sorted(x for x in listed if x))}")
        except Exception as e:  # noqa: BLE001
            check(False, f"{s.network} API not reachable: {str(e)[:100]}")
    else:
        check(True, f"{s.network} is the perp network: no spot pools, the perp markets come from the recorder and the sidecar")

    if s.perp_markets or s.ec:
        print("recorders")
        check(shutil.which("node") is not None, "node is installed", "perps and event contracts need the Node sidecars (Node 20+)")
        check((ROOT / "perp" / "node_modules").exists(), "perp/node_modules exists", "run: cd perp && npm install")
    if s.perp_live:
        print("perp orders")
        check((ROOT / "perp" / "executor.mjs").exists(), "perp/executor.mjs exists")
        mode = f"a bot key linked to {s.owner_address}" if s.owner_address else "the key's own perps account"
        check(True, f"perps trade LIVE on {s.network} through {mode}, {s.live_stake_usdso:g} USDso margin x {s.perp_leverage:g} per market")
        if not s.owner_address:
            check(True, "fund the key's Perps balance first: scripts/perp_smoke.py --deposit <USDso>")
    if s.ec_live:
        print("event contract bets")
        check((ROOT / "perp" / "ec_executor.mjs").exists(), "perp/ec_executor.mjs exists")
        check(True, f"event contracts bet LIVE on mainnet, {s.live_stake_usdso:g} USDso per window; the wallet needs USDso and SOMI for gas")
    elif s.live and s.ec:
        check(True, "event contracts stay paper on the testnet (the windows exist on mainnet only)")

    if s.live:
        print("live orders")
        if s.network in REST:
            try:
                import dreamdex_core  # noqa: F401
                check(True, "dreamdex_core is installed")
            except ImportError:
                check(False, "dreamdex_core is not installed", "pip install 'dreamfly[live]'")
        check(s.has_key, "PRIVATE_KEY is set")

    if a.wallet and s.network not in REST:
        print("wallet (read-only, through the perp sidecar)")
        try:
            from flybot.live.perp_execution import NodePerpExecutor

            ex = NodePerpExecutor(s, lambda kind, row: None)
            try:
                w = ex.wallet()
                check(w is not None, f"signer {w and w['address']}: {w and w['stt']:.4f} STT, {w and w['usdso']:.2f} USDso in the wallet")
                pre = ex.preflight(s.perp_markets[0] if s.perp_markets[0] != "all" else "BTC/USDso:USDso")
                check(pre is not None, f"perps account {ex.address}: {json.dumps(pre)}")
            finally:
                ex.stop()
        except Exception as e:  # noqa: BLE001
            check(False, f"wallet check failed: {str(e)[:120]}", "set PRIVATE_KEY, and run npm install in perp/")
    elif a.wallet:
        print("wallet (read-only)")
        try:
            import os

            os.environ["NETWORK"] = s.network
            from dreamdex_core import Pool, create_chain_context

            ctx = create_chain_context()
            bal = ctx.w3.eth.get_balance(ctx.address) / 1e18
            check(True, f"address {ctx.address}, native balance {bal:.4f} {ctx.net.native_symbol}")
            for m in s.markets:
                try:
                    print(f"          {m}: {Pool.load(ctx, m).wallet_base():.6f} base in the wallet")
                except Exception as e:  # noqa: BLE001
                    check(False, f"{m} pool read failed: {str(e)[:100]}")
        except Exception as e:  # noqa: BLE001
            check(False, f"wallet check failed: {str(e)[:120]}", "set PRIVATE_KEY, and install with pip install 'dreamfly[live]'")

    print(f"\n{ok_n} ok, {bad_n} failed")
    sys.exit(1 if bad_n else 0)


if __name__ == "__main__":
    main()
