"""Real perp orders that follow the perp paper episodes.

The paper episode (`PerpPaperEpisode`) stays the thing the fly learns from. `PerpMirror` keeps the account's real
position and quotes in step with it:

  p_trend, p_carry   the paper position is the target: the mirror sends a taker order for the difference between
                     what the account holds and what the paper episode holds (scaled), so an entry, a guard
                     flatten and the close at the end are all the same move
  p_maker            the paper quotes are mirrored as post only orders and re-quoted as they move; the paper fills
                     are not chased, the real quotes get their own fills; the position is closed at the end
  flat               nothing is placed; a position left by an earlier episode is closed

Orders are signed by the Node sidecar `perp/executor.mjs` (the kit's markets SDK), one request at a time over
pipes. In a dry run nothing is started and every call is written to the log as what would have happened.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .settings import Settings
from .sidecar import CALL_TIMEOUT_S, ROOT, Log, Sidecar, SidecarError

PerpExecutionError = SidecarError


class PerpExecutor(Protocol):
    dry: bool

    def markets(self) -> list[str]: ...
    def mark(self, symbol: str) -> float | None: ...
    def book(self, symbol: str) -> tuple[float | None, float | None]: ...
    def place(self, symbol: str, side: str, price: float, qty: float, kind: str, lift_to: float = 0.0) -> dict: ...
    def cancel(self, symbol: str, order_id: str) -> bool: ...
    def open_orders(self, symbol: str) -> list[dict] | None: ...
    def position(self, symbol: str) -> float | None: ...
    def close(self, symbol: str) -> dict: ...
    def equity(self) -> float | None: ...


class DryRunPerpExecutor:
    """Sends nothing. Every call is written to the log as what would have happened; the account always reads flat."""

    dry = True

    def __init__(self, log: Log):
        self.log, self._n = log, 0

    def markets(self) -> list[str]:
        return []

    def mark(self, symbol: str) -> float | None:
        return None

    def book(self, symbol: str) -> tuple[float | None, float | None]:
        return None, None

    def place(self, symbol: str, side: str, price: float, qty: float, kind: str, lift_to: float = 0.0) -> dict:
        self._n += 1
        oid = f"dry-{self._n}"
        self.log("orders", {"mode": "dry", "action": "place", "id": oid, "symbol": symbol, "side": side, "price": price,
                            "qty": qty, "kind": kind})
        return {"sent": False, "dry_run": True, "order_id": oid, "filled": 0.0}

    def cancel(self, symbol: str, order_id: str) -> bool:
        self.log("orders", {"mode": "dry", "action": "cancel", "id": order_id, "symbol": symbol})
        return True

    def open_orders(self, symbol: str) -> list[dict] | None:
        return []

    def position(self, symbol: str) -> float | None:
        return 0.0

    def close(self, symbol: str) -> dict:
        self.log("orders", {"mode": "dry", "action": "close", "symbol": symbol})
        return {"sent": False, "dry_run": True, "filled": 0.0, "left": 0.0}

    def equity(self) -> float | None:
        return None


class NodePerpExecutor:
    """The perp order path through the Node sidecar `perp/executor.mjs` (see `sidecar.py` for the plumbing)."""

    dry = False

    def __init__(self, settings: Settings, log: Log, root: Path = ROOT, node: str = "node",
                 script: Path | None = None, env: dict | None = None):
        self.settings, self.log = settings, log
        self.script = script or root / "perp" / "executor.mjs"
        self.argv = [node, str(self.script), "--network", settings.network]
        if settings.owner_address:
            self.argv += ["--owner", settings.owner_address]
        self.address: str | None = None
        self.linked: dict[str, bool] = {}
        self._markets: list[str] = []
        self.sidecar = Sidecar(self.argv, self.script, log, env=env, on_init=self._on_init, name="sidecar")

    def _on_init(self, info: dict) -> None:
        self.address = info.get("address")
        self.linked = dict(info.get("linked") or {})
        self._markets = [m["symbol"] for m in info.get("markets", [])]
        self.log("orders", {"mode": "live", "action": "sidecar", "address": self.address, "signer": info.get("signer"),
                            "trading_key": info.get("trading_key"), "markets": len(self._markets),
                            "linked": sum(1 for v in self.linked.values() if v) if self.linked else None})

    def alive(self) -> bool:
        return self.sidecar.alive()

    def stop(self) -> None:
        self.sidecar.stop()

    def _call(self, op: str, timeout: float = CALL_TIMEOUT_S, **args) -> dict:
        return self.sidecar.call(op, timeout, **args)

    # ---- the executor protocol: a failure is logged and answered with the safe value ----
    def _safe(self, op: str, default, **args):
        return self.sidecar.safe(op, default, **args)

    def markets(self) -> list[str]:
        return list(self._markets)

    def mark(self, symbol: str) -> float | None:
        r = self._safe("mark", None, symbol=symbol)
        return None if r is None else r.get("mark")

    def book(self, symbol: str) -> tuple[float | None, float | None]:
        r = self._safe("book", None, symbol=symbol)
        return (None, None) if r is None else (r.get("bid"), r.get("ask"))

    def place(self, symbol: str, side: str, price: float, qty: float, kind: str, lift_to: float = 0.0) -> dict:
        """lift_to: a notional (USDso) up to which a size below the market minimum may be lifted to that minimum."""
        row = {"mode": "live", "action": "place", "symbol": symbol, "side": side, "price": price, "qty": qty, "kind": kind}
        try:
            r = self._call("place", symbol=symbol, side=side, price=price, qty=qty, kind=kind, lift_to=lift_to,
                           timeout=CALL_TIMEOUT_S * 2)
        except PerpExecutionError as e:
            self.log("orders", {**row, "ok": False, "error": str(e)[:400]})
            return {"sent": False, "filled": 0.0, "reason": str(e)[:400]}
        self.log("orders", {**row, "ok": True, "id": r.get("order_id"), "filled": r.get("filled"), "sent": r.get("sent"),
                            "placed_qty": r.get("qty"), "placed_price": r.get("price"), "tx": r.get("tx"),
                            "reason": r.get("reason"), "limited_by": r.get("limited_by")})
        return r

    def cancel(self, symbol: str, order_id: str) -> bool:
        r = self._safe("cancel", None, symbol=symbol, order_id=str(order_id))
        ok = bool(r and r.get("ok"))
        self.log("orders", {"mode": "live", "action": "cancel", "symbol": symbol, "id": str(order_id), "ok": ok,
                            **({"error": r.get("error")} if r and r.get("error") else {})})
        return ok

    def open_orders(self, symbol: str) -> list[dict] | None:
        r = self._safe("open_orders", None, symbol=symbol)
        return None if r is None else list(r.get("orders") or [])

    def position(self, symbol: str) -> float | None:
        """Signed base size the account holds in this market (long positive), 0 when flat, None when unreadable."""
        r = self._safe("position", None, symbol=symbol)
        if r is None:
            return None
        p = r.get("position")
        if not p:
            return 0.0
        return float(p["contracts"]) * (1.0 if p["side"] == "long" else -1.0)

    def close(self, symbol: str) -> dict:
        row = {"mode": "live", "action": "close", "symbol": symbol}
        try:
            r = self._call("close", symbol=symbol, timeout=CALL_TIMEOUT_S * 3)
        except PerpExecutionError as e:
            self.log("orders", {**row, "ok": False, "error": str(e)[:400]})
            return {"sent": False, "filled": 0.0, "left": None, "reason": str(e)[:400]}
        self.log("orders", {**row, "ok": True, "sent": r.get("sent"), "filled": r.get("filled"), "left": r.get("left"),
                            "tx": r.get("tx"), "reason": r.get("reason")})
        return r

    def equity(self) -> float | None:
        r = self._safe("equity", None)
        return None if r is None else r.get("equity")

    def preflight(self, symbol: str) -> dict | None:
        return self._safe("preflight", None, symbol=symbol)

    def deposit(self, symbol: str, amount: float) -> dict:
        return self._call("deposit", symbol=symbol, amount=amount, timeout=CALL_TIMEOUT_S * 3)

    def wallet(self) -> dict | None:
        return self._safe("wallet", None)


def build_perp_executor(settings: Settings, log: Log) -> PerpExecutor:
    if settings.dry_run:
        return DryRunPerpExecutor(log)
    return NodePerpExecutor(settings, log)


@dataclass
class PerpMirror:
    """Follows one perp market's paper episode and keeps the account's real position and quotes in step with it."""

    executor: PerpExecutor
    symbol: str
    paper_stake: float
    paper_leverage: float
    settings: Settings
    log: Log
    store: Path | None = None           # where the ids of the orders this mirror placed are kept across restarts
    quotes: dict[str, tuple[float, str] | None] = field(default_factory=lambda: {"buy": None, "sell": None})
    last_target: float | None = None    # the paper position (scaled) the account was last synced to
    closing: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    DUST_USDSO = 1.0

    @property
    def scale(self) -> float:
        """Paper base units -> live base units: the live margin and leverage against the paper ones."""
        return (self.settings.live_stake_usdso * self.settings.perp_leverage) / (self.paper_stake * self.paper_leverage)

    def _mark(self, ep) -> float:
        if ep is not None and ep.last_mark:
            return float(ep.last_mark)
        return float(self.executor.mark(self.symbol) or 0.0)

    # ---- the orders this mirror owns, kept on disk so a restart can clean up after itself ----
    def _mine(self) -> set[str]:
        if self.store is None or not self.store.exists():
            return set()
        try:
            return set(json.loads(self.store.read_text()).get("orders", []))
        except (ValueError, OSError):
            return set()

    def _remember(self) -> None:
        if self.store is None:
            return
        ids = [q[1] for q in self.quotes.values() if q is not None]
        self.store.write_text(json.dumps({"orders": ids}))

    # ---- position sync (p_trend, p_carry, flat, and every close) ----
    def _sync_position(self, ep, want: float) -> None:
        """Send a taker order for the difference between what the account holds and `want` (signed base)."""
        have = self.executor.position(self.symbol)
        mark = self._mark(ep)
        if have is None or mark <= 0:
            self.log("orders", {"mode": "live", "action": "sync_skipped", "symbol": self.symbol,
                                "why": "position unreadable" if have is None else "no mark"})
            return
        delta = want - have
        if abs(delta) * mark < self.DUST_USDSO:
            self.last_target = want
            return
        if want == 0.0:   # a full close goes through the pool's own close sizing, which is exact on the lot grid
            r = self.executor.close(self.symbol)
            if r.get("left"):
                self.log("orders", {"mode": "live", "action": "close_left", "symbol": self.symbol, "base": r.get("left")})
            if r.get("sent") or r.get("dry_run") or r.get("reason") == "flat":
                self.last_target = want
            return          # a close that could not be sent is tried again on the next row
        bid, ask = self.executor.book(self.symbol)
        side = "long" if delta > 0 else "short"
        touch = ask if side == "long" else bid
        if touch is None:
            self.log("orders", {"mode": "live", "action": "sync_skipped", "symbol": self.symbol, "why": "empty book"})
            return
        cap = self.settings.live_stake_usdso * self.settings.perp_leverage * 1.05 / mark
        self.executor.place(self.symbol, side, touch, min(abs(delta), cap), "market")
        self.last_target = want

    def _target(self, ep) -> float:
        return float(ep.pos) * self.scale if ep is not None else 0.0

    # ---- quotes (p_maker) ----
    def _requote(self, ep) -> None:
        wanted = {"buy": (ep.bids[0] if ep.bids else None), "sell": (ep.asks[0] if ep.asks else None)}
        qty = float(ep.clip) * self.scale
        tol = self.settings.requote_bps / 1e4
        for side in ("buy", "sell"):
            px = wanted[side]
            resting = self.quotes[side]
            if px is None or ep.guarded or ep.done:
                if resting is not None:
                    self.executor.cancel(self.symbol, resting[1])
                    self.quotes[side] = None
                continue
            if resting is not None and abs(resting[0] - px) <= tol * px:
                continue
            if resting is not None:
                self.executor.cancel(self.symbol, resting[1])
                self.quotes[side] = None
            # a quote under the market minimum (a small stake on BTC) may be lifted to it, within the per-side notional
            r = self.executor.place(self.symbol, "long" if side == "buy" else "short", px, qty, "post_only",
                                    lift_to=self.settings.live_stake_usdso * self.settings.perp_leverage)
            oid = r.get("order_id")
            if oid:
                self.quotes[side] = (px, str(oid))
        self._remember()

    def _cancel_quotes(self) -> None:
        for side in ("buy", "sell"):
            if self.quotes[side] is not None:
                self.executor.cancel(self.symbol, self.quotes[side][1])
                self.quotes[side] = None
        self._remember()

    def _cancel_leftovers(self) -> None:
        """Orders THIS mirror placed in an earlier life of the process and that may still rest. Only its own: on a
        linked account the owner's hand-placed orders in the same market are left alone."""
        mine = self._mine()
        if not mine:
            return
        rows = self.executor.open_orders(self.symbol)
        still = mine if rows is None else mine & {str(o["id"]) for o in rows}
        for oid in sorted(still):
            self.executor.cancel(self.symbol, oid)
        self.quotes = {"buy": None, "sell": None}
        self._remember()

    # ---- episode hooks ----
    def on_start(self, ep) -> None:
        with self._lock:
            self.closing = False
            self.quotes = {"buy": None, "sell": None}
            self._cancel_leftovers()
            if ep.arm == "p_maker":
                self._sync_position(ep, 0.0)     # a maker starts flat; whatever an earlier episode left is closed
                self._requote(ep)
            else:
                self._sync_position(ep, self._target(ep))

    def on_row(self, ep) -> None:
        if ep is None or ep.done or self.closing:
            return
        if not self._lock.acquire(blocking=False):
            return
        try:
            if ep.arm == "p_maker":
                if ep.guarded:
                    self._cancel_quotes()
                    self._sync_position(ep, 0.0)
                else:
                    self._requote(ep)
                return
            want = self._target(ep)
            if self.last_target is None or want != self.last_target:
                self._sync_position(ep, want)
        finally:
            self._lock.release()

    def finish(self, ep) -> None:
        """Cancel the quotes, then close the position unless the paper episode kept a paying carry open."""
        self.closing = True
        with self._lock:
            self._cancel_quotes()
            want = self._target(ep) if (ep is not None and ep.keep_open) else 0.0
            self._sync_position(ep, want)
