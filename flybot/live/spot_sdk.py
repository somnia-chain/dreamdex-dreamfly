"""Spot on networks the DreamDEX web API and the bot kit's spot core do not cover (Hideki).

The Node sidecar `perp/spot_executor.mjs` reads books and 1 minute candles through the DreamDEX markets SDK and,
live, places and cancels orders. `NodeSpotExecutor` is the runner's spot executor there (the same four calls as
`LiveExecutor`) and its market data feed: in a dry run it is started with --dry-run and only reads.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .settings import Settings
from .sidecar import CALL_TIMEOUT_S, ROOT, Log, Sidecar, SidecarError

SDK_SPOT_NETWORKS = ("hideki",)     # spot through the markets SDK; testnet and mainnet use the web API and the kit core
CANDLE_COLUMNS = (("timestamp", "int64"), ("open", float), ("high", float), ("low", float), ("close", float),
                  ("volume", float))


def candles_frame(rows: list[dict]) -> pd.DataFrame:
    """The sidecar's candles in the shape `fetch_candles` returns from the web API."""
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in CANDLE_COLUMNS})
    df = pd.DataFrame(rows)
    for c, t in CANDLE_COLUMNS:
        df[c] = df[c].astype(t)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


class NodeSpotExecutor:
    """Spot orders and market data through the Node sidecar. Orders are refused by the sidecar in a dry run."""

    def __init__(self, settings: Settings, log: Log, root: Path = ROOT, node: str = "node", script: Path | None = None,
                 env: dict | None = None):
        self.settings, self.log = settings, log
        self.dry = settings.dry_run
        self.script = script or root / "perp" / "spot_executor.mjs"
        argv = [node, str(self.script), "--network", settings.network] + (["--dry-run"] if self.dry else [])
        self.info: dict = {}
        self.sidecar = Sidecar(argv, self.script, log, env=env, on_init=self._on_init, name="spot_sidecar")

    def _on_init(self, info: dict) -> None:
        self.info = info
        self.log("orders", {"mode": "dry" if self.dry else "live", "action": "spot_sidecar", "address": info.get("address"),
                            "markets": info.get("markets")})

    def stop(self) -> None:
        self.sidecar.stop()

    # ---- market data ----
    def book(self, symbol: str, depth: int = 5) -> dict | None:
        """Best levels, in the web API's orderbook message shape so the runner's Book takes it as a snapshot."""
        try:
            r = self.sidecar.call("book", symbol=symbol, depth=depth)
        except SidecarError:
            return None
        lv = lambda side: [{"price": p, "quantity": q} for p, q in r.get(side) or []]  # noqa: E731
        return {"symbol": symbol, "bids": lv("bids"), "asks": lv("asks")}

    def candles(self, symbol: str, limit: int = 300) -> pd.DataFrame:
        r = self.sidecar.call("candles", CALL_TIMEOUT_S, symbol=symbol, limit=limit)
        return candles_frame(r.get("candles") or [])

    # ---- the executor protocol ----
    def place(self, symbol: str, is_bid: bool, price: float, qty: float, post_only: bool) -> str | None:
        row = {"mode": "live", "action": "place", "symbol": symbol, "side": "buy" if is_bid else "sell", "price": price,
               "qty": qty, "kind": "post_only" if post_only else "ioc"}
        try:
            r = self.sidecar.call("place", CALL_TIMEOUT_S * 2, symbol=symbol, side=row["side"], price=price, qty=qty,
                                  post_only=post_only)
        except SidecarError as e:
            # a timeout or a dead sidecar after sending may have left the order on chain: say so, do not call it failed
            unknown = "timed out" in str(e) or "exited" in str(e)
            self.log("orders", {**row, "ok": False, "outcome": "unknown" if unknown else "failed", "error": str(e)[:300]})
            return None
        self.log("orders", {**row, "ok": True, "id": r.get("order_id"), "status": r.get("status"), "filled": r.get("filled"),
                            "tx": r.get("tx")})
        return r.get("order_id")

    def cancel(self, symbol: str, order_id: str) -> bool:
        row = {"mode": "live", "action": "cancel", "symbol": symbol, "id": order_id}
        try:
            r = self.sidecar.call("cancel", CALL_TIMEOUT_S * 2, symbol=symbol, order_id=order_id)
        except SidecarError as e:          # usually the order already filled or expired
            self.log("orders", {**row, "ok": False, "error": str(e)[:300]})
            return False
        self.log("orders", {**row, "ok": True})
        return bool(r.get("canceled"))

    def _balances(self, symbol: str) -> dict | None:
        return self.sidecar.safe("balances", None, symbol=symbol)

    def base_balance(self, symbol: str) -> float | None:
        b = self._balances(symbol)
        return None if b is None else float(b.get("base") or 0.0)

    def quote_balance(self, symbol: str) -> float | None:
        b = self._balances(symbol)
        return None if b is None else float(b.get("quote") or 0.0)
