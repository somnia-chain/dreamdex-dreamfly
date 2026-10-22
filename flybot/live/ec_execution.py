"""Real event contract bets that follow the EC paper fly.

The paper fly (`ECFly`) stays the thing that learns. Per window it picks a strategy in the action band, then on
each book sample decides whether it would send a buy of one side at a cap price when a seller is visible at or
under it. `ECMirror` turns that into one real bet per window:

  bet       when the paper fly would send its order, one immediate-or-cancel buy of the same side at the same
            cap, sized to LIVE_STAKE_USDSO; tried on at most a few samples until something fills, never resting
  settle    a bet cannot be sold back before expiry (there is no early exit), so the window runs to its end; the
            venue's resolution pays a winning share 1 (less the settlement fee), a void pays both sides 0.5
  claim     winnings are paid only when asked: each settled window is claimed, retried until it works, and a
            sweep at start catches anything a restart left behind

Bets are signed by the Node sidecar `perp/ec_executor.mjs` (the kit's markets SDK) on mainnet, the only network
with event contract windows. In a dry run nothing is started and every call is written to the log as what would
have happened.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .settings import Settings
from .sidecar import CALL_TIMEOUT_S, ROOT, Log, Sidecar, SidecarError

ECExecutionError = SidecarError

MAX_TRIES = 3                  # IOC buys per window while the paper order is pending and nothing has filled
CLAIM_EVERY_S = 120.0          # how often settled windows are claimed
SWEEP_EVERY_S = 6 * 3600.0     # how often the venue's own claimable list is swept (plus once at start)
GIVE_UP_AFTER_S = 3 * 86400.0  # an unclaimable window is dropped from the list after this, with a log row


class ECExecutor(Protocol):
    dry: bool

    def market(self, market_id: str) -> dict | None: ...
    def buy(self, market_id: str, side: str, cap: float, notional: float, ttl_s: float) -> dict: ...
    def position(self, market_id: str) -> dict | None: ...
    def claim(self, market_id: str) -> dict: ...
    def sweep(self) -> dict | None: ...
    def wallet(self) -> dict | None: ...


class DryRunECExecutor:
    """Sends nothing. Every bet is written to the log as what would have happened; nothing is ever held."""

    dry = True

    def __init__(self, log: Log):
        self.log, self._n = log, 0

    def market(self, market_id: str) -> dict | None:
        return None

    def buy(self, market_id: str, side: str, cap: float, notional: float, ttl_s: float) -> dict:
        self._n += 1
        self.log("ec_orders", {"mode": "dry", "action": "buy", "market_id": market_id, "side": side, "cap": cap,
                               "notional": notional})
        return {"sent": False, "dry_run": True, "filled": 0.0, "cost": 0.0}

    def position(self, market_id: str) -> dict | None:
        return {"up": 0.0, "down": 0.0}

    def claim(self, market_id: str) -> dict:
        self.log("ec_orders", {"mode": "dry", "action": "claim", "market_id": market_id})
        return {"claimed": False, "dry_run": True, "redeemed": []}

    def sweep(self) -> dict | None:
        return {"claimed": 0, "markets": 0, "dry_run": True}

    def wallet(self) -> dict | None:
        return None


class NodeECExecutor:
    """The event contract order path through the Node sidecar `perp/ec_executor.mjs`."""

    dry = False

    def __init__(self, settings: Settings, log: Log, root: Path = ROOT, node: str = "node",
                 script: Path | None = None, env: dict | None = None, dry_run: bool = False):
        self.settings, self.log = settings, log
        self.script = script or root / "perp" / "ec_executor.mjs"
        self.argv = [node, str(self.script), "--network", settings.network] + (["--dry-run"] if dry_run else [])
        self.dry = dry_run                 # a sidecar in dry run reads the venue for real and signs nothing
        self.address: str | None = None
        self.info: dict = {}
        self.sidecar = Sidecar(self.argv, self.script, log, env=env, on_init=self._on_init, name="ec_sidecar")

    def _on_init(self, info: dict) -> None:
        self.address, self.info = info.get("address"), info
        self.log("ec_orders", {"mode": "live", "action": "sidecar", "address": self.address, "markets": info.get("markets"),
                               "collateral": info.get("collateral")})

    def alive(self) -> bool:
        return self.sidecar.alive()

    def stop(self) -> None:
        self.sidecar.stop()

    def _call(self, op: str, timeout: float = CALL_TIMEOUT_S, **args) -> dict:
        return self.sidecar.call(op, timeout, **args)

    def _safe(self, op: str, default, **args):
        try:
            return self.sidecar.call(op, **args)
        except SidecarError as e:
            self.log("ec_orders", {"mode": "live", "action": op, **args, "ok": False, "error": str(e)[:400]})
            return default

    # ---- the executor protocol ----
    def windows(self) -> list[dict]:
        r = self._safe("list", None)
        return [] if r is None else list(r.get("windows") or [])

    def book(self, market_id: str) -> dict | None:
        return self._safe("book", None, market_id=market_id)

    def market(self, market_id: str) -> dict | None:
        return self._safe("market", None, market_id=market_id)

    def buy(self, market_id: str, side: str, cap: float, notional: float, ttl_s: float) -> dict:
        row = {"mode": "live", "action": "buy", "market_id": market_id, "side": side, "cap": cap, "notional": notional}
        try:
            r = self._call("buy", market_id=market_id, side=side, cap=cap, notional=notional, ttl_s=ttl_s,
                           timeout=CALL_TIMEOUT_S * 2)
        except SidecarError as e:
            self.log("ec_orders", {**row, "ok": False, "error": str(e)[:400]})
            return {"sent": False, "filled": 0.0, "cost": 0.0, "reason": str(e)[:400]}
        self.log("ec_orders", {**row, "ok": True, "sent": r.get("sent"), "filled": r.get("filled"), "cost": r.get("cost"),
                               "avg_price": r.get("avg_price"), "tx": r.get("tx"), "reason": r.get("reason"),
                               "symbol": r.get("symbol")})
        return r

    def position(self, market_id: str) -> dict | None:
        return self._safe("position", None, market_id=market_id)

    def claim(self, market_id: str) -> dict:
        row = {"mode": "live", "action": "claim", "market_id": market_id}
        try:
            r = self._call("claim", market_id=market_id, timeout=CALL_TIMEOUT_S * 2)
        except SidecarError as e:
            self.log("ec_orders", {**row, "ok": False, "error": str(e)[:400]})
            return {"claimed": False, "redeemed": [], "reason": str(e)[:400], "failed": True}
        self.log("ec_orders", {**row, "ok": True, "claimed": r.get("claimed"), "redeemed": r.get("redeemed"), "tx": r.get("tx"),
                               "reason": r.get("reason"), "voided": r.get("voided")})
        return r

    def sweep(self) -> dict | None:
        r = self._safe("sweep", None, timeout=CALL_TIMEOUT_S * 4)
        if r is not None:
            self.log("ec_orders", {"mode": "live", "action": "sweep", "claimed": r.get("claimed"), "markets": r.get("markets"),
                                   "est_payout": r.get("est_payout"), "txs": r.get("txs")})
        return r

    def wallet(self) -> dict | None:
        return self._safe("wallet", None)

    def wallet_usdso(self) -> float | None:
        w = self.wallet()
        return None if w is None else w.get("usdso")


def build_ec_executor(settings: Settings, log: Log) -> ECExecutor:
    if settings.dry_run:
        return DryRunECExecutor(log)
    return NodeECExecutor(settings, log)


@dataclass
class ECMirror:
    """One real bet per window, following the paper fly's own order, and the claiming afterwards."""

    executor: ECExecutor
    settings: Settings
    log: Log
    store: Path | None = None
    clock: object = time.time
    bets: dict[str, dict] = field(default_factory=dict)        # market_id -> {side, tries, filled, cost, symbol, at}
    unclaimed: dict[str, dict] = field(default_factory=dict)   # market_id -> {symbol, at, cost, filled}
    _last_claim: float = 0.0
    _last_sweep: float = 0.0

    def __post_init__(self) -> None:
        self._load()

    # ---- persistence: a restart must know what it holds, the chain alone does not say which bets were ours ----
    def _load(self) -> None:
        if self.store is None or not self.store.exists():
            return
        try:
            d = json.loads(self.store.read_text())
        except ValueError:
            return
        self.bets = dict(d.get("bets") or {})
        self.unclaimed = dict(d.get("unclaimed") or {})

    def _save(self) -> None:
        if self.store is None:
            return
        self.store.parent.mkdir(parents=True, exist_ok=True)
        self.store.write_text(json.dumps({"bets": self.bets, "unclaimed": self.unclaimed}))

    # ---- hooks called by ECFly ----
    def on_start(self) -> None:
        """A restart: claim whatever the venue says is ours. Bets still open ride on as recorded."""
        self.executor.sweep()
        self._last_sweep = self.clock()

    def bet(self, w, side: str, cap: float, seconds_left: float) -> None:
        """The paper fly would send its order now: send ours, unless this window already has a fill or used its tries."""
        b = self.bets.get(w.market_id)
        if b is None:
            b = self.bets[w.market_id] = {"side": side, "tries": 0, "filled": 0.0, "cost": 0.0, "symbol": w.symbol,
                                          "at": self.clock()}
        if b["filled"] > 0 or b["tries"] >= MAX_TRIES:
            return
        b["tries"] += 1
        self._save()                                  # on disk before the network call
        r = self.executor.buy(w.market_id, side, float(cap), float(self.settings.live_stake_usdso),
                              max(5.0, float(seconds_left)))
        filled, cost = float(r.get("filled") or 0.0), float(r.get("cost") or 0.0)
        if filled > 0:
            b["filled"], b["cost"], b["side"] = filled, cost, side
            b["price"] = cost / filled
        self._save()

    def settle(self, w, row: dict) -> None:
        """The window resolved: a filled bet moves to the claim list. The paper result is the fly's business."""
        b = self.bets.pop(w.market_id, None)
        self._save()
        if b is None or b["filled"] <= 0:
            return
        self.unclaimed[w.market_id] = {"symbol": w.symbol, "at": self.clock(), "cost": b["cost"], "filled": b["filled"],
                                       "side": b["side"], "truth": row.get("truth"), "voided": bool(row.get("voided"))}
        self._save()
        self._claim_one(w.market_id)                  # the venue usually has it settled by the time the tape says so
        self._last_claim = self.clock()

    def claim_due(self) -> None:
        """Called from the runner's tape loop: claim settled windows every CLAIM_EVERY_S, sweep every SWEEP_EVERY_S."""
        now = self.clock()
        if self.unclaimed and now - self._last_claim >= CLAIM_EVERY_S:
            self._last_claim = now
            for mid in sorted(self.unclaimed, key=lambda m: self.unclaimed[m]["at"])[:5]:
                self._claim_one(mid)
        if now - self._last_sweep >= SWEEP_EVERY_S:
            self._last_sweep = now
            self.executor.sweep()

    def _claim_one(self, mid: str) -> None:
        u = self.unclaimed.get(mid)
        if u is None:
            return
        r = self.executor.claim(mid)
        reason = str(r.get("reason") or "")
        if r.get("claimed") or r.get("dry_run") or reason.startswith("nothing to claim"):
            self.unclaimed.pop(mid, None)              # paid, or nothing of ours there (a losing bet)
        elif self.clock() - u["at"] > GIVE_UP_AFTER_S:
            self.log("ec_orders", {"mode": "live", "action": "claim_given_up", "market_id": mid, "symbol": u.get("symbol"),
                                   "reason": reason[:200]})
            self.unclaimed.pop(mid, None)
        self._save()

    # ---- for the wallet ledger ----
    def open_cost(self) -> float:
        """USDso that left the wallet for bets not yet paid back: open windows plus settled ones not yet claimed,
        both at cost. The ledger counts them as still ours until the claim lands."""
        return sum(b["cost"] for b in self.bets.values()) + sum(u["cost"] for u in self.unclaimed.values())
