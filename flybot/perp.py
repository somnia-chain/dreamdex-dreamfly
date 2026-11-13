"""Perpetual-futures arms on the testnet recorder tape (poll cadence, one row per market per minute).

Perps have no websocket feed, so the paper engine is minute-level: each new tape row (mark, index,
funding, open interest, top of book) advances the episode. Arms: two-sided maker around mark, trend
follower on mark, funding carry (sit opposite the funding sign and collect it), flat. A guard runs
inside every arm: if the unrealised loss reaches `guard_frac` of the margin the position is flattened
and the arm stays out for the rest of the episode (a paper stand-in for the liquidation waterfall).

Fills are simulated at the recorded touch (maker) or through it (taker) with no queue position, so
results flatter makers and are relative, not absolute.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PERP_FEATURES = ("vol_short", "vol_long", "trend", "basis", "funding", "oi_z", "hour_sin", "hour_cos")
PERP_ARMS = ("p_maker", "p_trend", "p_carry", "flat")
SHORT, LONG = 15, 60  # rows (minutes)


@dataclass(frozen=True)
class PerpCosts:
    """Testnet pools: fees not yet read from chain, so 0 like the spot pools; a taker crosses the
    recorded spread and pays `slip_bps` on top. `stake` is margin in USDso; notional = stake x leverage.
    Funding accrues on the open notional once per hour at funding_8h / 8 (paid when long and positive)."""

    taker_bps: float = 0.0
    maker_bps: float = 0.0
    slip_bps: float = 3.0
    stake: float = 1000.0
    leverage: float = 2.0
    guard_frac: float = 0.5  # flatten when unrealised loss reaches this fraction of the margin
    gas_per_action: float = 0.001


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    """Recorder rows for one market -> minute frame. A stale mark (null) falls back to the row mark."""
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["wall"], utc=True)
    df["timestamp"] = df["ts"].dt.as_unit("ms").astype("int64")  # epoch ms whatever the parsed unit
    df["mark"] = df["mark"].astype(float).fillna(df["row_mark"].astype(float))
    for c in ("index", "funding_8h", "oi", "best_bid", "best_ask"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["mark"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def perp_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per tape row, using only that row and earlier ones. Short windows because the tape is
    young; a feature is NaN (skipped by the encoder) until its window has enough rows."""
    m = df["mark"]
    r = np.log(m).diff()
    out = pd.DataFrame(index=df.index)
    out["vol_short"] = r.rolling(SHORT, min_periods=5).std() * np.sqrt(SHORT)
    out["vol_long"] = r.rolling(LONG, min_periods=SHORT).std() * np.sqrt(LONG)
    out["trend"] = (m / m.shift(SHORT) - 1.0) / (out["vol_long"] + 1e-9)
    out["basis"] = (m / df["index"] - 1.0) * 1e4  # bps, mark rich (+) or cheap (-) to index
    out["funding"] = df["funding_8h"] * 1e4  # bps per 8h
    oi = df["oi"]
    out["oi_z"] = (oi - oi.rolling(LONG, min_periods=SHORT).mean()) / (oi.rolling(LONG, min_periods=SHORT).std() + 1e-9)
    hour = df["ts"].dt.hour + df["ts"].dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out[list(PERP_FEATURES)]


def trend_direction(df: pd.DataFrame, p: dict) -> int:
    look = int(p.get("lookback", SHORT))
    if len(df) < look + 1:
        return 0
    ret = float(df["mark"].iloc[-1] / df["mark"].iloc[-1 - look] - 1.0)
    return int(np.sign(ret)) if abs(ret) >= float(p.get("threshold", 0.002)) else 0


@dataclass
class PerpFill:
    ts: int
    side: str
    price: float
    qty: float
    taker: bool


@dataclass
class PerpPaperEpisode:
    arm: str
    costs: PerpCosts
    params: dict = field(default_factory=dict)
    cash: float = 0.0
    pos: float = 0.0  # base units, signed
    funding_paid: float = 0.0
    fills: list[PerpFill] = field(default_factory=list)
    actions: int = 0
    bids: list[float] = field(default_factory=list)
    asks: list[float] = field(default_factory=list)
    clip: float = 0.0
    max_pos: float = 0.0
    guarded: bool = False
    done: bool = False
    last_mark: float = 0.0
    inherit_pos: float = 0.0   # a carry position the previous episode left open, taken over at the mark
    keep_open: bool = False    # set at finish when a carry position is worth holding into the next episode
    _last_funding_ts: int = 0

    # Funding carry works like the bot kit's perp-funding: hold the side funding pays while the rate is worth it
    # (`min_apr` to enter, `exit_apr` to keep), and step aside when it drops or flips. A position is held across
    # episodes, so the round trip is paid once, not every hour.
    @staticmethod
    def _apr(row: pd.Series) -> float:
        f = row.get("funding_8h")
        return 0.0 if f is None or not np.isfinite(f) else float(f) * 3 * 365

    def _carry_side(self, row: pd.Series, entering: bool) -> int:
        """+1 long, -1 short, 0 stay out. Positive funding: longs pay shorts, so the paid side is short."""
        apr = self._apr(row)
        floor = float(self.params.get("min_apr" if entering else "exit_apr", 0.10 if entering else 0.03))
        if abs(apr) < floor:
            return 0
        return -1 if apr > 0 else 1

    @staticmethod
    def _touch(row: pd.Series) -> tuple[float, float]:
        """Best bid and ask; a one-sided or empty book falls back to the mark."""
        m = float(row["mark"])
        bb = row.get("best_bid")
        ba = row.get("best_ask")
        bb = m if bb is None or not np.isfinite(bb) else float(bb)
        ba = m if ba is None or not np.isfinite(ba) else float(ba)
        if ba < bb:
            bb, ba = m, m
        return bb, ba

    def start(self, row: pd.Series, trend_dir: int = 0) -> None:
        bb, ba = self._touch(row)
        mid = (bb + ba) / 2
        self.last_mark = float(row["mark"])
        self.max_pos = self.costs.stake * self.costs.leverage / mid
        self._last_funding_ts = int(row["timestamp"])
        if self.inherit_pos:
            # taken over at the mark, so this episode starts at zero and only its own moves count
            self.pos, self.cash = self.inherit_pos, -self.inherit_pos * self.last_mark
            if self.arm != "p_carry" or np.sign(self.pos) != self._carry_side(row, entering=False):
                self._flatten(int(row["timestamp"]), bb, ba)
        if self.arm == "p_maker":
            self.clip = self.max_pos * float(self.params.get("clip_frac", 0.25))
            self._requote(mid)
        elif self.arm == "p_trend" and trend_dir != 0:
            self._take(int(row["timestamp"]), "buy" if trend_dir > 0 else "sell", ba if trend_dir > 0 else bb, self.max_pos)
        elif self.arm == "p_carry" and self.pos == 0.0:
            side = self._carry_side(row, entering=True)
            if side:
                self._take(int(row["timestamp"]), "buy" if side > 0 else "sell", ba if side > 0 else bb, self.max_pos)

    def _take(self, ts: int, side: str, price: float, qty: float) -> None:
        cost = price * qty * (self.costs.taker_bps + self.costs.slip_bps) / 1e4
        if side == "buy":
            self.cash -= price * qty + cost
            self.pos += qty
        else:
            self.cash += price * qty - cost
            self.pos -= qty
        self.fills.append(PerpFill(ts, side, price, qty, True))
        self.actions += 1

    def _make(self, ts: int, side: str, price: float, qty: float) -> None:
        fee = price * qty * self.costs.maker_bps / 1e4
        if side == "buy":
            self.cash -= price * qty + fee
            self.pos += qty
        else:
            self.cash += price * qty - fee
            self.pos -= qty
        self.fills.append(PerpFill(ts, side, price, qty, False))

    def _requote(self, mid: float) -> None:
        spread = float(self.params.get("spread_bps", 20.0)) / 1e4
        skew = float(self.params.get("skew_bps", 10.0)) / 1e4
        m = mid * (1 - skew * (self.pos / self.max_pos))
        new = ([m * (1 - spread)], [m * (1 + spread)])
        if (new[0], new[1]) != (self.bids, self.asks):
            self.bids, self.asks = new
            self.actions += 2

    def unrealised(self, mark: float) -> float:
        return self.cash + self.pos * mark - self.funding_paid

    def _accrue_funding(self, row: pd.Series) -> None:
        ts = int(row["timestamp"])
        f = row.get("funding_8h")
        if f is None or not np.isfinite(f) or self.pos == 0.0:
            self._last_funding_ts = ts
            return
        hours = (ts - self._last_funding_ts) / 3_600_000
        if hours >= 1.0:
            # long pays when funding is positive; short receives
            self.funding_paid += self.pos * float(row["mark"]) * float(f) / 8 * int(hours)
            self._last_funding_ts += int(hours) * 3_600_000

    def _flatten(self, ts: int, bb: float, ba: float) -> None:
        self.actions += len(self.bids) + len(self.asks)
        self.bids, self.asks = [], []
        if abs(self.pos) > 1e-12:
            if self.pos > 0:
                self._take(ts, "sell", bb, self.pos)
            else:
                self._take(ts, "buy", ba, -self.pos)

    def on_row(self, row: pd.Series) -> None:
        """A new tape row: fill resting quotes the touch reached, accrue funding, run the guard, requote."""
        if self.done or self.guarded:
            return
        ts = int(row["timestamp"])
        bb, ba = self._touch(row)
        self.last_mark = float(row["mark"])
        for px in list(self.bids):
            if ba <= px and self.pos + self.clip <= self.max_pos + 1e-12:
                self._make(ts, "buy", px, self.clip)
                self.bids.remove(px)
        for px in list(self.asks):
            if bb >= px and self.pos - self.clip >= -self.max_pos - 1e-12:
                self._make(ts, "sell", px, self.clip)
                self.asks.remove(px)
        self._accrue_funding(row)
        if self.pos != 0.0 and self.unrealised(self.last_mark) <= -self.costs.guard_frac * self.costs.stake:
            self._flatten(ts, bb, ba)
            self.guarded = True
            return
        if self.arm == "p_maker":
            self._requote((bb + ba) / 2)
        elif self.arm == "p_carry" and self.pos != 0.0 and np.sign(self.pos) != self._carry_side(row, entering=False):
            self._flatten(ts, bb, ba)       # the rate fell through the exit, or flipped against this side

    def finish(self, row: pd.Series) -> float:
        if self.done:
            return self.pnl()
        bb, ba = self._touch(row)
        self.last_mark = float(row["mark"])
        self._accrue_funding(row)
        if self.arm == "p_carry" and self.pos != 0.0 and not self.guarded \
                and np.sign(self.pos) == self._carry_side(row, entering=False):
            self.keep_open = True           # still paying: hand the position to the next episode at the mark
        else:
            self._flatten(int(row["timestamp"]), bb, ba)
        self.done = True
        return self.pnl()

    def pnl(self) -> float:
        """Realised plus the open position at the last mark (zero when flat), less funding paid and gas."""
        return self.cash + self.pos * self.last_mark - self.funding_paid - self.actions * self.costs.gas_per_action
