"""Turning a paper episode's intentions into orders.

The paper episode (paper.py) decides what the chosen strategy wants: resting quotes for the maker and grid, one
taker entry for the trend. It also simulates the fills, and its profit is still what the fly learns from. The
mirror here watches the episode and asks an executor to do the same thing for real:

  DryRunExecutor  writes "would place" and "would cancel" and sends nothing (the default, like DRY_RUN=true in
                  the DreamDEX bot kit)
  LiveExecutor    places real orders through dreamdex_core, the bot kit's Python client, with its guards

Limit of this version: the learning signal comes from the simulated episode even when orders are live, so real
fills and the reward can differ. Orders are spot only.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from .settings import Settings

Log = Callable[[str, dict], None]


class ExecutionError(RuntimeError):
    pass


class Executor(Protocol):
    dry: bool

    def place(self, symbol: str, is_bid: bool, price: float, qty: float, post_only: bool) -> str | None: ...
    def cancel(self, symbol: str, order_id: str) -> bool: ...
    def base_balance(self, symbol: str) -> float | None: ...
    def quote_balance(self, symbol: str) -> float | None: ...


class DryRunExecutor:
    """Sends nothing. Every call is written to the log as what would have happened."""

    dry = True

    def __init__(self, log: Log):
        self.log = log
        self._n = 0

    def place(self, symbol: str, is_bid: bool, price: float, qty: float, post_only: bool) -> str | None:
        self._n += 1
        oid = f"dry-{self._n}"
        self.log("orders", {"mode": "dry", "action": "place", "id": oid, "symbol": symbol,
                            "side": "buy" if is_bid else "sell", "price": price, "qty": qty,
                            "kind": "post_only" if post_only else "ioc"})
        return oid

    def cancel(self, symbol: str, order_id: str) -> bool:
        self.log("orders", {"mode": "dry", "action": "cancel", "id": order_id, "symbol": symbol})
        return True

    def base_balance(self, symbol: str) -> float | None:
        return None

    def quote_balance(self, symbol: str) -> float | None:
        return None


POOL_ERRORS = {
    "0xd48c4403": "ImmediateOrCancelNoFill: nothing to fill within TAKE_BUFFER_BPS of the touch (the book moved before the order landed)",
    "0xfb8f41b2": "ERC20InsufficientAllowance: the pool may not pull enough of the token",
    "0xe450d38c": "ERC20InsufficientBalance: the wallet does not hold enough of the token",
}


def explain(error: str) -> str:
    """Put the name of a known pool error in front of the raw revert text, which only carries its 4 byte selector."""
    for selector, name in POOL_ERRORS.items():
        if selector in error:
            return f"{name} | {error}"
    return error


class LiveExecutor:
    """Real orders on the network in NETWORK, through dreamdex_core. Needs PRIVATE_KEY in the environment; this
    class never reads or stores the key itself, dreamdex_core does."""

    dry = False

    def __init__(self, settings: Settings, log: Log):
        try:
            import dreamdex_core as core
        except ImportError as e:
            raise ExecutionError("live orders need the DreamDEX core client: pip install 'dreamfly[live]'") from e
        self.core, self.settings, self.log = core, settings, log
        os.environ["NETWORK"] = settings.network  # dreamdex_core picks the chain from this
        self.ctx = core.create_chain_context()
        self.pools: dict = {}
        self._tx = threading.Lock()  # one chain write at a time: the core's pools each count nonces on their own

    def _pool(self, symbol: str):
        if symbol not in self.pools:
            self.pools[symbol] = self.core.Pool.load(self.ctx, symbol)
        return self.pools[symbol]

    def _ensure_quote_allowance(self, pool, price: float, qty: float) -> None:
        """Let the pool pull the quote token for a bid.

        The core approves only when the allowance is short, but it sends the approval with a fixed 700,000 gas, and
        the first approval on a fresh wallet needs more than that on Somnia (it runs out of gas, the transaction is
        mined with status 0, the core does not look, and every bid then fails with ERC20InsufficientAllowance).
        So the allowance is made here first, with an estimated gas limit and a check that it really went through."""
        from dreamdex_core.config import NATIVE_SENTINEL
        from dreamdex_core.contract import ERC20_ABI
        from dreamdex_core.quant import to_raw
        from web3 import Web3

        p, ctx = pool.params, self.ctx
        price_raw = to_raw(price, pool.quote_decimals)
        price_raw -= price_raw % p.tick_size
        qty_raw = to_raw(qty, pool.base_decimals)
        qty_raw -= qty_raw % p.lot_size
        token, required, _ = pool._contract.functions.getAutoPullRequirement(ctx.address, True, price_raw, qty_raw, 0).call()
        if token.lower() == NATIVE_SENTINEL.lower() or required <= 0:
            return
        erc = ctx.w3.eth.contract(address=Web3.to_checksum_address(token), abi=ERC20_ABI)
        if erc.functions.allowance(ctx.address, pool.address).call() >= required:
            return
        fn = erc.functions.approve(Web3.to_checksum_address(pool.address), required * 8)
        gas = max(int(fn.estimate_gas({"from": ctx.address}) * 1.5), 1_500_000)
        tx = fn.build_transaction({"from": ctx.address, "nonce": pool.nm.reserve(), "gas": gas,
                                   "chainId": ctx.net.chain_id, **pool.nm.gas_fields()})
        try:
            signed = ctx.account.sign_transaction(tx)
            h = ctx.w3.eth.send_raw_transaction(getattr(signed, "raw_transaction", None) or signed.rawTransaction)
            receipt = ctx.w3.eth.wait_for_transaction_receipt(h, timeout=120)
        except Exception:
            pool.nm.reset()
            raise
        if receipt["status"] != 1:
            raise ExecutionError(f"approving the quote token reverted (gas used {receipt['gasUsed']} of {gas})")

    def place(self, symbol: str, is_bid: bool, price: float, qty: float, post_only: bool) -> str | None:
        order_type = self.core.OrderType.POST_ONLY if post_only else self.core.OrderType.IOC
        row = {"mode": "live", "action": "place", "symbol": symbol, "side": "buy" if is_bid else "sell",
               "price": price, "qty": qty, "kind": "post_only" if post_only else "ioc"}
        try:
            with self._tx:
                if is_bid:
                    self._ensure_quote_allowance(self._pool(symbol), price, qty)
                res = self._pool(symbol).place(is_bid=is_bid, price=price, qty=qty, order_type=order_type,
                                               expire_ms=int(self.settings.quote_ttl_min * 60_000))
        except Exception as e:  # noqa: BLE001 - a rejected order must not stop the run
            self.log("orders", {**row, "ok": False, "error": explain(str(e))[:300]})
            return None
        oid = None if res.order_id is None else str(res.order_id)
        self.log("orders", {**row, "ok": True, "id": oid, "tx": res.tx_hash, "gas": res.gas_used})
        return oid

    def cancel(self, symbol: str, order_id: str) -> bool:
        row = {"mode": "live", "action": "cancel", "symbol": symbol, "id": order_id}
        try:
            with self._tx:
                tx = self._pool(symbol).cancel(int(order_id))
        except Exception as e:  # noqa: BLE001 - usually the order already filled or expired
            self.log("orders", {**row, "ok": False, "error": str(e)[:300]})
            return False
        self.log("orders", {**row, "ok": True, "tx": tx})
        return True

    def base_balance(self, symbol: str) -> float | None:
        try:
            return float(self._pool(symbol).wallet_base())
        except Exception as e:  # noqa: BLE001
            self.log("orders", {"mode": "live", "action": "balance", "symbol": symbol, "ok": False, "error": str(e)[:300]})
            return None

    def quote_balance(self, symbol: str) -> float | None:
        """USDso (the quote token) held in the wallet. The core has no reader for it, so it is read like wallet_base."""
        try:
            from dreamdex_core.config import NATIVE_SENTINEL
            from dreamdex_core.contract import ERC20_ABI
            from dreamdex_core.quant import from_raw
            from web3 import Web3

            pool, w3 = self._pool(symbol), self.ctx.w3
            token = pool.params.quote_token
            if token.lower() == NATIVE_SENTINEL.lower():
                raw = w3.eth.get_balance(self.ctx.address)
            else:
                raw = w3.eth.contract(address=Web3.to_checksum_address(token), abi=ERC20_ABI).functions.balanceOf(self.ctx.address).call()
            return float(from_raw(raw, pool.quote_decimals))
        except Exception as e:  # noqa: BLE001
            self.log("orders", {"mode": "live", "action": "quote_balance", "symbol": symbol, "ok": False, "error": str(e)[:300]})
            return None


@dataclass
class WalletLedger:
    """Today's real profit or loss of the trading wallet, in USDso, for the daily loss stop.

    The wallet is worth its USDso plus every market's base inventory at that market's last mid. The first reading of
    each part on a UTC day is its baseline, so the day's number is what the wallet is worth now minus what it was
    worth when the day's trading began. One ledger is shared by all the market mirrors, since they share one wallet.
    Gas (the native coin) is not counted, and a deposit or withdrawal during the day shows up as profit or loss until
    the next day."""

    executor: Executor | None
    clock: Callable[[], float] = time.time
    perps: object | None = None          # a PerpExecutor: its MarginBank equity (USDso) counts as part of the wallet
    ec_mirror: object | None = None      # an ECMirror: its open bets (at cost) count as part of the wallet
    day: str = ""
    quote0: float | None = None
    quote: float | None = None
    perp0: float | None = None
    perp: float | None = None
    ecv0: float | None = None
    ecv: float | None = None
    base0: dict[str, float] = field(default_factory=dict)
    base: dict[str, float] = field(default_factory=dict)
    mid: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _roll(self) -> None:
        today = time.strftime("%Y-%m-%d", time.gmtime(self.clock()))
        if today != self.day:
            self.day, self.quote0, self.perp0, self.ecv0, self.base0 = today, None, None, None, {}

    def update(self, symbol: str, bb: float, ba: float) -> float | None:
        """Read the wallet for one spot market and return today's P&L (None until something has been read)."""
        q = b = None
        if self.executor is not None:
            q, b = self.executor.quote_balance(symbol), self.executor.base_balance(symbol)
        with self._lock:
            self._roll()
            if q is not None:
                self.quote = q
                if self.quote0 is None:
                    self.quote0 = q
            if b is not None:
                self.base[symbol] = b
                self.base0.setdefault(symbol, b)
            self.mid[symbol] = (bb + ba) / 2
            return self.day_pnl

    def update_perps(self) -> float | None:
        """Read the perps account's equity (margin plus unrealised P&L) and return today's P&L."""
        eq = None if self.perps is None else self.perps.equity()
        with self._lock:
            self._roll()
            if eq is not None:
                self.perp = eq
                if self.perp0 is None:
                    self.perp0 = eq
            return self.day_pnl

    def update_ec(self) -> float | None:
        """Read what the event contract bets hold (open and unclaimed, at cost) and, when no spot executor reads
        the wallet, the wallet's USDso through the event contract sidecar. Returns today's P&L."""
        m = self.ec_mirror
        if m is None:
            return self.day_pnl
        open_cost = m.open_cost()
        q = None
        if self.executor is None and hasattr(m.executor, "wallet_usdso"):
            q = m.executor.wallet_usdso()
        with self._lock:
            self._roll()
            self.ecv = open_cost
            if self.ecv0 is None:
                self.ecv0 = open_cost
            if q is not None:
                self.quote = q
                if self.quote0 is None:
                    self.quote0 = q
            return self.day_pnl

    @property
    def day_pnl(self) -> float | None:
        spot = None if self.quote0 is None or self.quote is None else self.quote - self.quote0
        perp = None if self.perp0 is None or self.perp is None else self.perp - self.perp0
        ec = None if self.ecv0 is None or self.ecv is None else self.ecv - self.ecv0
        if spot is None and perp is None:
            return None
        pnl = (spot or 0.0) + (perp or 0.0) + (ec or 0.0)
        for s, b in self.base.items():
            if s in self.base0 and s in self.mid:
                pnl += (b - self.base0[s]) * self.mid[s]
        return pnl


def build_executor(settings: Settings, log: Log) -> Executor:
    return DryRunExecutor(log) if settings.dry_run else LiveExecutor(settings, log)


@dataclass
class SpotMirror:
    """Follows one market's paper episode and keeps real (or would-be) orders in step with it."""

    executor: Executor
    symbol: str
    paper_stake: float
    settings: Settings
    log: Log
    clock: Callable[[], float] = time.time
    open: dict[str, list[tuple[float, str]]] = field(default_factory=lambda: {"buy": [], "sell": []})
    # orders whose cancel failed, with how often it was tried; they are cancelled again at the next sync or finish
    # (an order that really is gone, filled or expired, stops being retried after CANCEL_TRIES)
    stale: dict[str, int] = field(default_factory=dict)
    # the wallet's base balance when the fly was last flat. The close at every episode end works against this, not
    # against the balance at the episode start, so inventory a missed close left behind is closed next time
    base0: float | None = None
    last_sync: float = 0.0
    closing: bool = False
    ledger: WalletLedger | None = None   # shared across markets: the wallet's real day P&L for the daily stop
    _lock: threading.Lock = field(default_factory=threading.Lock)

    CANCEL_TRIES = 3
    DUST_USDSO = 1.0

    @property
    def scale(self) -> float:
        return self.settings.live_stake_usdso / self.paper_stake

    def _through_touch(self, side: str, bb: float, ba: float, times: float = 1.0) -> float:
        buf = times * self.settings.take_buffer_bps / 1e4
        return ba * (1 + buf) if side == "buy" else bb * (1 - buf)

    def _take(self, side: str, qty: float, bb: float, ba: float, times: float = 1.0, cap: bool = True) -> None:
        px = self._through_touch(side, bb, ba, times)
        limit = self.settings.live_stake_usdso * 1.05
        if cap and qty * px > limit:  # an entry is never larger than the stake, whatever the paper episode did
            qty = limit / px
        if qty <= 0:
            return
        self.executor.place(self.symbol, side == "buy", px, qty, post_only=False)

    def _cancel(self, oid: str) -> None:
        if self.executor.cancel(self.symbol, oid):
            self.stale.pop(oid, None)
            return
        n = self.stale.get(oid, 0) + 1
        if n >= self.CANCEL_TRIES:
            self.stale.pop(oid, None)
        else:
            self.stale[oid] = n

    def _retry_stale(self) -> None:
        for oid in list(self.stale):
            self._cancel(oid)

    def _resting(self) -> int:
        return sum(len(v) for v in self.open.values()) + len(self.stale)

    def on_start(self, ep, bb: float, ba: float) -> None:
        with self._lock:
            self.closing = False
            if self.base0 is None:
                self.base0 = self.executor.base_balance(self.symbol)
            self.open = {"buy": [], "sell": []}
            self._retry_stale()
            if self.ledger is not None:   # the day's baseline is the wallet before the first order, not after
                self.ledger.update(self.symbol, bb, ba)
            if ep.arm == "trend" and ep.fills:
                f = ep.fills[0]
                self._take(f.side, f.qty * self.scale, bb, ba)
        self.sync(ep, bb, ba, force=True)

    def sync(self, ep, bb: float, ba: float, force: bool = False) -> None:
        if ep is None or ep.done or self.closing or ep.arm not in ("maker", "grid"):
            return
        now = self.clock()
        if not force and now - self.last_sync < self.settings.min_gap_s:
            return
        if not self._lock.acquire(blocking=False):
            return
        try:
            if self.closing:
                return
            self.last_sync = now
            self._retry_stale()
            tol = self.settings.requote_bps / 1e4
            qty = ep.clip * self.scale
            wants = {"buy": list(ep.bids), "sell": list(ep.asks)}
            for side in ("buy", "sell"):
                keep = []
                for px, oid in self.open[side]:
                    if any(abs(w - px) <= tol * px for w in wants[side]):
                        keep.append((px, oid))
                    else:
                        self._cancel(oid)
                self.open[side] = keep
                for w in wants[side]:
                    if any(abs(w - px) <= tol * px for px, _ in self.open[side]):
                        continue
                    if self._resting() >= self.settings.max_open_orders:
                        break
                    if qty * w > self.settings.live_stake_usdso * 1.05:
                        continue
                    oid = self.executor.place(self.symbol, side == "buy", w, qty, post_only=True)
                    if oid is not None:
                        self.open[side].append((w, oid))
        finally:
            self._lock.release()

    def finish(self, ep, bb: float, ba: float) -> None:
        """Cancel everything resting, then close whatever the wallet holds beyond the flat baseline.

        The close is a taker order through the touch. If it misses (the touch moved), it is sent once more, further
        through, and anything still left is reported and stays in the baseline, so the next episode end tries again.
        Afterwards the wallet is read into the shared ledger, so the daily stop sees what the day really cost."""
        self.closing = True  # a sync that starts now sends nothing more for this episode
        with self._lock:
            self._flatten(ep, bb, ba)
        if self.ledger is not None:
            self.ledger.update(self.symbol, bb, ba)

    def _flatten(self, ep, bb: float, ba: float) -> None:
        for side in ("buy", "sell"):
            for _px, oid in self.open[side]:
                self._cancel(oid)
        self.open = {"buy": [], "sell": []}
        self._retry_stale()
        mid = (bb + ba) / 2
        if self.base0 is None:  # a dry run, or a wallet that could not be read: close the paper inventory
            net = ep.inv * self.scale if ep is not None else 0.0
            if abs(net) * mid >= self.DUST_USDSO:
                self._take("sell" if net > 0 else "buy", abs(net), bb, ba, cap=False)
            return
        for times in (1.0, 5.0):
            now_base = self.executor.base_balance(self.symbol)
            if now_base is None:
                return
            net = now_base - self.base0
            if abs(net) * mid < self.DUST_USDSO:  # flat (dust below one USDso is left alone): move the baseline
                self.base0 = now_base
                return
            self._take("sell" if net > 0 else "buy", abs(net), bb, ba, times, cap=False)
        now_base = self.executor.base_balance(self.symbol)
        if now_base is not None and abs(now_base - self.base0) * mid >= self.DUST_USDSO:
            self.log("orders", {"mode": "live", "action": "close_left", "symbol": self.symbol,
                                "base": now_base - self.base0, "usdso": (now_base - self.base0) * mid})
