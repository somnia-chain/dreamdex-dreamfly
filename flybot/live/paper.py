"""Book-level paper execution for one episode of one arm.

Feed it the touch (best bid, best ask) every time the book changes. A resting paper bid fills when the
best ask comes down to it (someone sells into our price); a resting ask fills when the best bid rises to
it. Takers pay the opposite touch. No queue position is modelled: a touch that reaches our price fills
our whole clip, which still flatters makers somewhat.

Stable pair mode (opt in through the arm params, off for every other market): quotes sit on the market's price
grid (`tick`), can sit a few ticks inside the touch (`mode: inside`), and fill from real trade prints
(`trade_fills`) with queue position instead of from the touch. A taker sell that prints below our bid reached us
first because we are better priced; a print at or above our price only eats the orders queued ahead of us, and
only its overflow at our own level reaches us. Cancellations ahead of us are ignored, so this is conservative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..arms import Costs


@dataclass
class Fill:
    ts: int
    side: str
    price: float
    qty: float
    taker: bool


@dataclass
class PaperEpisode:
    arm: str
    costs: Costs
    params: dict = field(default_factory=dict)
    cash: float = 0.0
    inv: float = 0.0
    fills: list[Fill] = field(default_factory=list)
    actions: int = 0
    bids: list[float] = field(default_factory=list)
    asks: list[float] = field(default_factory=list)
    clip: float = 0.0
    max_inv: float = 0.0
    ref: float = 0.0
    entered: bool = False
    done: bool = False
    _grid_levels: tuple = ()
    tick: float = 0.0              # price grid; 0 = no rounding
    trade_fills: bool = False      # fill from trade prints and queue position instead of from the touch
    inside: bool = False           # quote relative to the touch instead of around the last price
    improve_ticks: int = 1
    _bid_ahead: float = 0.0        # size queued ahead of our bid / ask, from the live book when the quote was placed
    _ask_ahead: float = 0.0
    _bid_left: float = 0.0         # what is left of the resting clip
    _ask_left: float = 0.0
    _ahead_known: bool = False
    _touch: tuple = ()             # last (best bid, best ask, book) seen, so a quote can be re-placed between book updates

    def start(self, ts: int, best_bid: float, best_ask: float, trend_dir: int = 0) -> None:
        mid = (best_bid + best_ask) / 2
        self.ref = mid
        self.tick = float(self.params.get("tick", 0.0))
        self.trade_fills = bool(self.params.get("trade_fills", False))
        self.inside = self.params.get("mode") == "inside" and self.tick > 0
        self.improve_ticks = int(self.params.get("improve_ticks", 1))
        self.max_inv = float(self.params.get("stake", self.costs.stake)) / mid
        self._touch = (best_bid, best_ask, None)
        if self.arm == "maker":
            self.clip = self.max_inv * float(self.params.get("clip_frac", 0.25))
            if self.inside:
                self._requote_inside(best_bid, best_ask, None)
            else:
                self._requote(mid)
        elif self.arm == "grid":
            self.clip = self.max_inv * float(self.params.get("clip_frac", 0.2))
            n = int(self.params.get("levels", 3))
            step = float(self.params.get("step_bps", 30.0)) / 1e4
            self._grid_levels = (tuple(mid * (1 - step * k) for k in range(1, n + 1)),
                                 tuple(mid * (1 + step * k) for k in range(1, n + 1)))
            self._rearm(mid)
        elif self.arm == "trend" and trend_dir != 0:
            qty = self.max_inv
            if trend_dir > 0:
                self._take(ts, "buy", best_ask, qty)
            else:
                self._take(ts, "sell", best_bid, qty)
            self.entered = True

    def _take(self, ts: int, side: str, price: float, qty: float) -> None:
        cost = price * qty * (self.costs.taker_bps + self.costs.slip_bps) / 1e4
        if side == "buy":
            self.cash -= price * qty + cost
            self.inv += qty
        else:
            self.cash += price * qty - cost
            self.inv -= qty
        self.fills.append(Fill(ts, side, price, qty, True))
        self.actions += 1

    def _make(self, ts: int, side: str, price: float, qty: float) -> None:
        fee = price * qty * self.costs.maker_bps / 1e4
        if side == "buy":
            self.cash -= price * qty + fee
            self.inv += qty
        else:
            self.cash += price * qty - fee
            self.inv -= qty
        self.fills.append(Fill(ts, side, price, qty, False))

    def _snap(self, px: float, up: bool) -> float:
        """Round a price onto the market's grid: bids down, asks up. No grid, no change."""
        if not self.tick:
            return px
        n = px / self.tick
        n = math.ceil(n - 1e-9) if up else math.floor(n + 1e-9)
        return round(n * self.tick, 10)

    def _ahead(self, book, price: float, bid: bool) -> float:
        """Size queued at our price or better, from the live book (our own order is not in it)."""
        if book is None:
            return 0.0
        tol = self.tick / 2 if self.tick else 1e-12
        if bid:
            return float(sum(q for p, q in book.bids.items() if p >= price - tol))
        return float(sum(q for p, q in book.asks.items() if p <= price + tol))

    def _requote_inside(self, bb: float, ba: float, book) -> None:
        """Quote `improve_ticks` inside each side of the touch, or join the touch when there is no room to improve
        on both sides. A side at the inventory cap is not quoted. A quote that has not moved keeps its queue place."""
        t = self.tick
        b, a = bb + self.improve_ticks * t, ba - self.improve_ticks * t
        if a - b < t - 1e-12:
            b, a = bb, ba
        b, a = self._snap(b, False), self._snap(a, True)
        want_b = [b] if self.inv + 1e-12 < self.max_inv else []
        want_a = [a] if self.inv - 1e-12 > -self.max_inv else []
        if want_b != self.bids:
            self.bids = want_b
            if want_b:
                self.actions += 1
                self._bid_left, self._bid_ahead = self.clip, self._ahead(book, want_b[0], True)
        elif want_b and not self._ahead_known and book is not None:
            self._bid_ahead = self._ahead(book, want_b[0], True)
        if want_a != self.asks:
            self.asks = want_a
            if want_a:
                self.actions += 1
                self._ask_left, self._ask_ahead = self.clip, self._ahead(book, want_a[0], False)
        elif want_a and not self._ahead_known and book is not None:
            self._ask_ahead = self._ahead(book, want_a[0], False)
        if book is not None:
            self._ahead_known = True
        self._touch = (bb, ba, book)

    def _requote(self, mid: float) -> None:
        spread = float(self.params.get("spread_bps", 20.0)) / 1e4
        skew = float(self.params.get("skew_bps", 10.0)) / 1e4
        shift = -skew * (self.inv / self.max_inv)
        m = mid * (1 + shift)
        new = ([self._snap(m * (1 - spread), False)], [self._snap(m * (1 + spread), True)])
        # Like the live order mirror, leave resting quotes alone while the wanted price is within `requote_bps` of
        # them. Re-placing on every touch change would charge gas for thousands of orders an hour that a real
        # mirror never sends, and that gas was the whole "loss" of the maker on a fast market.
        tol = float(self.params.get("requote_bps", 5.0)) / 1e4
        if (self.bids and self.asks and abs(new[0][0] / self.bids[0] - 1.0) < tol
                and abs(new[1][0] / self.asks[0] - 1.0) < tol):
            return
        if (new[0], new[1]) != (self.bids, self.asks):
            self.bids, self.asks = new
            self.actions += 2

    def _rearm(self, price: float) -> None:
        lb = [b for b in self._grid_levels[0] if b < price]
        la = [a for a in self._grid_levels[1] if a > price]
        if (lb, la) != (self.bids, self.asks):
            self.actions += abs(len(lb) - len(self.bids)) + abs(len(la) - len(self.asks)) or 1
            self.bids, self.asks = lb, la

    def on_touch(self, ts: int, best_bid: float, best_ask: float, book=None) -> None:
        """Book changed. Fill resting paper quotes the touch has reached, then re-quote. `book` (optional, anything
        with .bids and .asks price to size dicts) lets a quote know the queue ahead of it."""
        if self.done or self.arm in ("flat", "trend"):
            return
        if not self.trade_fills:
            for px in list(self.bids):
                if best_ask <= px and self.inv + self.clip <= self.max_inv + 1e-12:
                    self._make(ts, "buy", px, self.clip)
                    self.bids.remove(px)
            for px in list(self.asks):
                if best_bid >= px and self.inv - self.clip >= -self.max_inv - 1e-12:
                    self._make(ts, "sell", px, self.clip)
                    self.asks.remove(px)
        mid = (best_bid + best_ask) / 2
        if self.arm == "maker":
            if self.inside:
                self._requote_inside(best_bid, best_ask, book)
            else:
                self._requote(mid)
        elif self.arm == "grid":
            self._rearm(mid)

    def on_trade(self, ts: int, price: float, qty: float, side: str) -> None:
        """A real trade printed (side is the taker's). Only for markets with `trade_fills`. A taker sell hits our bid,
        a taker buy our ask. A print on the far side of our price reached us first; a print at or inside our price
        only eats the orders queued ahead of us, and its overflow at our own level reaches us."""
        if self.done or not self.trade_fills or self.arm != "maker" or qty <= 0:
            return
        tol = self.tick / 2 if self.tick else 1e-12
        if side == "sell" and self.bids:
            px, room = self.bids[0], self.max_inv - self.inv
            if price < px - tol:
                f = min(qty, self._bid_left, room)
            else:
                eaten = min(qty, self._bid_ahead)
                self._bid_ahead -= eaten
                f = min(qty - eaten if price <= px + tol else 0.0, self._bid_left, room)
            if f > 1e-12:
                self._make(ts, "buy", px, f)
                self._bid_left -= f
                if self._bid_left <= 1e-9 or room - f <= 1e-9:
                    self.bids = []          # consumed or at the cap; the next minute (or touch change) re-quotes
        elif side == "buy" and self.asks:
            px, room = self.asks[0], self.max_inv + self.inv
            if price > px + tol:
                f = min(qty, self._ask_left, room)
            else:
                eaten = min(qty, self._ask_ahead)
                self._ask_ahead -= eaten
                f = min(qty - eaten if price >= px - tol else 0.0, self._ask_left, room)
            if f > 1e-12:
                self._make(ts, "sell", px, f)
                self._ask_left -= f
                if self._ask_left <= 1e-9 or room - f <= 1e-9:
                    self.asks = []

    def on_minute(self, ts: int, best_bid: float, best_ask: float) -> None:
        """Once a minute the maker re-centres on the current mid even if the touch did not move. In inside mode it
        re-places any quote that a fill used up, against the last touch and book seen."""
        if self.arm == "maker" and not self.done:
            if self.inside:
                book = self._touch[2] if self._touch else None
                self._requote_inside(best_bid, best_ask, book)
            else:
                self._requote((best_bid + best_ask) / 2)

    def finish(self, ts: int, best_bid: float, best_ask: float) -> float:
        """Cancel quotes, flatten at the touch, charge gas. Returns episode P&L in quote units."""
        if self.done:
            return self.pnl()
        self.actions += len(self.bids) + len(self.asks)
        self.bids, self.asks = [], []
        if abs(self.inv) > 1e-12:
            if self.inv > 0:
                self._take(ts, "sell", best_bid, self.inv)
            else:
                self._take(ts, "buy", best_ask, -self.inv)
        self.done = True
        return self.pnl()

    def pnl(self) -> float:
        return self.cash - self.actions * self.costs.gas_per_action
