"""One fly for all event-contract windows, fed by the mainnet EC recorder tape.

Per window: at the first book sample inside the action band (default 25 s left) the fly picks an arm.
The arm then looks at every later sample down to `min_s` for a visible seller at or under its cap;
a candidate fill counts only if the next sample (about 2.5 s later, the order's flight time) still
shows an ask at or under the cap, and fills at that later ask. The venue's resolution settles it:
a win pays 1 per share, a loss 0, a void 0.5. Reward is P&L over stake, as in the backtest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..ec import EC_ARMS, EC_FEATURES, ECCosts
from .base import FlyState

RULE = {"act_max_s": 25.0, "min_s": 6.0, "fav_min": 0.95, "z_min": 1.6, "max_ask": 0.95, "min_gap": 0.06,
        "max_discount": 0.08, "gain": 1.3, "threshold": 0.9}


def _best_ask(book: dict | None) -> tuple[float, float] | None:
    asks = (book or {}).get("asks") or []
    if not asks:
        return None
    px, qty = asks[0][0], asks[0][1]
    return (float(px), float(qty)) if px is not None and qty and float(qty) > 0 else None


def book_prob_up(row: dict) -> float:
    """The book's price of Up: mean of the Up ask and one minus the Down ask, whichever exist; falls
    back to bids; NaN when the books are empty."""
    up, down = row.get("up") or {}, row.get("down") or {}
    c = []
    if (a := _best_ask(up)):
        c.append(a[0])
    if (a := _best_ask(down)):
        c.append(1.0 - a[0])
    if not c:
        for side, flip in ((up, False), (down, True)):
            bids = side.get("bids") or []
            if bids and bids[0][0] is not None:
                c.append(1.0 - float(bids[0][0]) if flip else float(bids[0][0]))
    return float(np.mean(c)) if c else float("nan")


def features_of(row: dict) -> pd.Series:
    p = book_prob_up(row)
    z = row.get("z")
    sig = row.get("sigma_bps")
    ts = pd.Timestamp(row["wall"])
    hour = ts.hour + ts.minute / 60
    return pd.Series({
        "fav_prob": max(p, 1 - p) if np.isfinite(p) else np.nan,
        "z_abs": abs(float(z)) if z is not None else np.nan,
        "sigma_bps": float(sig) if sig is not None else np.nan,
        "is_5m": float(row.get("interval") == "5m"), "is_1h": float(row.get("interval") == "1h"),
        "is_btc": float(row.get("asset") == "BTC"),
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
    })[list(EC_FEATURES)]


def plan_take(arm: str, row: dict, p: dict) -> tuple[str, float] | None:
    """What the arm wants on this sample: (side, cap price) or None."""
    r = {**RULE, **p}
    if arm == "ec_late":
        pu = row.get("prob_up")
        z = row.get("z")
        if pu is None or z is None:
            return None
        side = "up" if pu >= 0.5 else "down"
        conf = pu if side == "up" else 1 - pu
        if conf < r["fav_min"] or abs(z) < r["z_min"]:
            return None
        return side, min(r["max_ask"], conf - r["min_gap"])
    if arm == "ec_sharp":
        if row.get("interval") != "5m":
            return None
        p_up = book_prob_up(row)
        if not np.isfinite(p_up):
            return None
        side = "up" if p_up >= 0.5 else "down"
        pr = float(np.clip(max(p_up, 1 - p_up), 1e-4, 1 - 1e-4))
        sharp = 1 / (1 + np.exp(-r["gain"] * np.log(pr / (1 - pr))))
        return (side, r["max_ask"]) if sharp >= r["threshold"] else None
    return None


def seller_ok(row: dict, side: str, cap: float, p: dict) -> tuple[float, float] | None:
    """A visible seller at or under the cap and not suspiciously far under it: (ask, qty)."""
    r = {**RULE, **p}
    a = _best_ask(row.get(side))
    if a is None or a[0] > cap or cap - a[0] > r["max_discount"]:
        return None
    return a


class ECWindow:
    def __init__(self, market_id: str, row: dict):
        self.market_id = market_id
        self.symbol, self.asset, self.interval = row.get("symbol"), row.get("asset"), row.get("interval")
        self.expiry = row.get("expiry")
        self.arm: str | None = None
        self.last = None  # the fly's (pool, kc, score) at decision time
        self.pending: tuple[str, float] | None = None  # (side, cap) seen on the previous sample
        self.side: str | None = None
        self.price = 0.0
        self.units = 0.0
        self.samples = 0
        self.decided_at: dict = {}


class ECFly(FlyState):
    def __init__(self, graph, state_dir: Path, costs: ECCosts, params: dict, seed: int = 0, daily_loss_stop: float = 30.0):
        super().__init__("EC", graph, state_dir, seed, EC_ARMS, EC_FEATURES)
        self.costs, self.params, self.daily_loss_stop = costs, params, daily_loss_stop
        self.mirror = None                       # ECMirror when real (or dry run) bets follow this fly's windows
        self.day_loss = lambda: self.day_pnl     # what the daily stop compares; the runner points it at the wallet when live
        self.windows: dict[str, ECWindow] = {}
        self.last_book: dict = {}
        self.recent: list[dict] = []  # last settled windows, for the page

    def on_row(self, row: dict, kill: bool = False) -> dict | None:
        t = row.get("type")
        mid = row.get("marketId")
        if not mid:
            return None
        if t == "ec_open":
            self.windows.setdefault(mid, ECWindow(mid, row))
        elif t == "ec_book":
            w = self.windows.setdefault(mid, ECWindow(mid, row))
            self._book(w, row, kill)
        elif t == "ec_resolution":
            w = self.windows.pop(mid, None)
            if w is not None and w.arm is not None:
                return self._settle(w, row)
        return None

    def _book(self, w: ECWindow, row: dict, kill: bool) -> None:
        w.samples += 1
        left = float(row.get("secondsLeft") or 0.0)
        ts = int(pd.Timestamp(row["wall"]).timestamp() * 1000)
        if ts - int(self.last_sniff.get("ts", 0)) >= 10_000:
            self.note_sniff(features_of(row), ts)
        self.last_book = {"symbol": w.symbol, "interval": w.interval, "asset": w.asset, "secondsLeft": left,
                          "prob_up_book": book_prob_up(row), "prob_up_model": row.get("prob_up"), "z": row.get("z"),
                          "lead_bps": row.get("lead_bps"), "wall": row.get("wall")}
        if w.arm is None:
            if left <= RULE["act_max_s"]:
                self._decide(w, row, kill)
            return
        if w.side is not None or left < RULE["min_s"]:
            return
        p = self.params.get(w.arm, {})
        want = plan_take(w.arm, row, p)
        if w.pending is not None and want is not None and want[0] == w.pending[0]:
            hit = seller_ok(row, want[0], w.pending[1], p)  # the order priced on the previous sample lands now
            if hit is not None:
                w.side, w.price = want[0], hit[0]
                w.units = min(self.costs.stake / hit[0], hit[1])
                self.log("fills", {"marketId": w.market_id, "symbol": w.symbol, "arm": w.arm, "side": w.side,
                                   "price": w.price, "units": round(w.units, 4), "secondsLeft": left})
                w.pending = None
                return
        w.pending = want if want is not None and seller_ok(row, want[0], want[1], p) is not None else None
        if w.pending is not None:
            self._mirror("bet", w, w.pending[0], w.pending[1], left)     # the paper order goes out now; so does ours

    def _mirror(self, hook: str, *args) -> None:
        """The bet mirror follows the paper fly; its failures are logged, never allowed to stop the fly."""
        if self.mirror is None:
            return
        try:
            getattr(self.mirror, hook)(*args)
        except Exception as e:  # noqa: BLE001
            self.log("errors", {"where": f"mirror.{hook}", "error": str(e)[:1500]})

    def _decide(self, w: ECWindow, row: dict, kill: bool) -> None:
        feats = features_of(row)
        kc, scores = self.sniff(feats)
        k = self.fly.choose(feats)
        arm = EC_ARMS[k]
        guard = kill or self.day_loss() <= -self.daily_loss_stop
        if guard and arm != "flat":
            arm, k = "flat", EC_ARMS.index("flat")
            self.fly.last = (k, self.fly.last[1], float(scores[k]))
        w.arm, w.last = arm, self.fly.last
        w.decided_at = {
            "ts": int(pd.Timestamp(row["wall"]).timestamp() * 1000), "marketId": w.market_id, "symbol": w.symbol,
            "interval": w.interval, "asset": w.asset, "secondsLeft": row.get("secondsLeft"), "arm": arm,
            "scores": [round(float(x), 4) for x in scores],
            "kc_active": [int(i) for i in np.flatnonzero(kc > 0)], "kc_frac": round(float((kc > 0).mean()), 4),
            "features": {f: (None if not np.isfinite(feats[f]) else round(float(feats[f]), 6)) for f in EC_FEATURES},
            "guard_flat": guard,
        }
        self.last_decision = w.decided_at
        self.log("decisions", w.decided_at)

    def _settle(self, w: ECWindow, row: dict) -> dict:
        truth = row.get("truth")
        cost = w.units * w.price
        if w.side is None:
            pnl = 0.0
        elif row.get("voided") or truth is None:
            pnl = w.units * 0.5 - cost
        else:
            pnl = w.units * (1.0 - w.price) if truth == w.side else -cost
        rew = self.scaler(pnl / self.costs.stake)  # same running scale as spot and perps
        self.fly.last = w.last
        self.fly.update(rew)
        ts = int(pd.Timestamp(row["wall"]).timestamp() * 1000)
        self.book(ts, pnl)
        out = {"marketId": w.market_id, "symbol": w.symbol, "interval": w.interval, "asset": w.asset, "end": ts,
               "arm": w.arm, "side": w.side, "price": w.price, "units": round(w.units, 4), "truth": truth,
               "voided": bool(row.get("voided")), "pnl": round(pnl, 6), "reward": round(rew, 4),
               "equity": round(self.equity, 6), "dopamine": "PAM" if rew > 0 else "PPL1" if rew < 0 else "none",
               "weights_moved": self.weights_moved()}
        self.log("episodes", out)
        self.recent = (self.recent + [out])[-8:]
        self.save()
        self._mirror("settle", w, row)
        return out

    def snapshot(self) -> dict:
        open_ = [{"symbol": w.symbol, "interval": w.interval, "arm": w.arm, "side": w.side, "price": w.price,
                  "units": round(w.units, 4), "expiry": w.expiry} for w in self.windows.values() if w.arm is not None]
        return {"symbol": "EC", "kind": "ec", **self.brain_snapshot(), "last": self.last_book, "open": open_,
                "recent": self.recent, "tracked": len(self.windows)}
