"""Paper-live loop: the fly chooses an arm every episode for each market, the arm runs against the
live data (book-level paper fills for spot, minute-level for perps, last-seconds taker for event
contracts), the episode's P&L teaches the fly. With settings (DRY_RUN=false) the spot and perp order
mirrors send real orders that follow the paper episodes; otherwise nothing is ever sent.

Spot: WS order book + REST candles, one fly per market. Perps and event contracts: one fly per perp
market and one for all EC windows, fed by the recorders' tapes (see perp/*.mjs). State is persisted
after every episode to <state>/<name>/ and reloaded on start. Every decision and episode goes to
JSONL; <state>/live.json is the snapshot the page reads.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import signal
import sys
import time
import traceback
import urllib.parse
import urllib.request
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import websockets

from ..arms import ARM_NAMES, ARMS, Costs
from ..ec import ECCosts
from ..market import FEATURES, LONG, feature_table
from ..mb.graph import load_graph
from ..perp import PerpCosts
from ..sim import reward_of
from .base import BRAINS, FlyState
from .bench import ArmBench
from .ec_execution import ECMirror, build_ec_executor
from .ec_live import ECFly
from .execution import SpotMirror, WalletLedger, build_executor
from .perp_execution import PerpMirror, build_perp_executor
from .paper import PaperEpisode
from .perp_live import PerpFly
from .recorder import WS, Book
from .settings import NETWORKS, PERP_TAPE_PREFIX, Settings, SettingsError
from .tape import Tail

REST = {"mainnet": "https://api.dreamdex.io", "testnet": "https://stg.api.dreamdex.io"}
ROOT = Path(__file__).resolve().parents[2]
PERP_SYMBOLS = ["all"]      # every perp market the recorder tape carries; or name them
TAPE_POLL_S = 5


def fetch_candles(base: str, symbol: str, limit: int = 300) -> pd.DataFrame:
    url = f"{base}/v0/markets/{urllib.parse.quote(symbol)}/candles?interval=1m&limit={limit}"
    with urllib.request.urlopen(url, timeout=20) as r:
        rows = json.load(r)["candles"]
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in (("timestamp", "int64"), ("open", float), ("high", float),
                                                               ("low", float), ("close", float), ("volume", float))})
    df = pd.DataFrame(rows)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df["timestamp"] = df["timestamp"].astype("int64")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def densify(df: pd.DataFrame, end_ms: int, minutes: int, fallback: float) -> pd.DataFrame:
    """A full minute-by-minute candle series ending at `end_ms`. A market that trades a few times an hour has no
    candle for most minutes; the price did not move in them, so the last close is carried forward (open, high, low
    and close all equal it, volume zero). Minutes before the first known candle use its close, and with no candles
    at all `fallback` (the mid) is used."""
    end = end_ms // 60_000 * 60_000
    grid = np.arange(end - (minutes - 1) * 60_000, end + 60_000, 60_000, dtype="int64")
    if df is None or df.empty:
        close = pd.Series(fallback, index=grid)
        vol = pd.Series(0.0, index=grid)
    else:
        d = df.drop_duplicates("timestamp").set_index("timestamp")
        both = d.index.union(grid)
        close = d["close"].reindex(both).ffill().bfill().reindex(grid)
        vol = d["volume"].reindex(grid).fillna(0.0)
    open_ = close.shift(1).fillna(close.iloc[0])
    hi, lo = np.maximum(open_, close), np.minimum(open_, close)
    if df is not None and not df.empty:        # real candles keep their own range
        real = df.drop_duplicates("timestamp").set_index("timestamp").reindex(grid)
        open_, hi, lo = real["open"].fillna(open_), real["high"].fillna(hi), real["low"].fillna(lo)
    out = pd.DataFrame({"timestamp": grid, "open": open_.values, "high": np.asarray(hi), "low": np.asarray(lo),
                        "close": close.values, "volume": vol.values})
    out["ts"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
    return out


class MarketFly(FlyState):
    """Everything the fly needs for one spot market.

    `slots` paper episodes can run at once on the market. They start staggered through the episode length, avoid
    the strategies the other slots are running, and every finished episode trains the same weights, so the fly
    gets several times more experience per hour. Slot 0 is the primary one: it is what the order mirror follows
    and what the dashboard shows."""

    def __init__(self, symbol: str, graph, state_dir: Path, costs: Costs, params: dict, seed: int = 0,
                 episode_min: int = 60, slots: int = 1, brain: str = "mb"):
        super().__init__(symbol, graph, state_dir, seed, ARM_NAMES, FEATURES, params.get("brain") or brain)
        # Per-market settings in the params, all optional: "arms" (the strategies this market may use, the rest are
        # never picked), "slots" (concurrent episodes here), "slip_bps" (this market's measured taker cost).
        if "slip_bps" in params:
            costs = replace(costs, slip_bps=float(params["slip_bps"]))
        allowed = [a for a in (params.get("arms") or ARM_NAMES) if a in ARM_NAMES]
        if "flat" not in allowed:
            allowed.append("flat")      # doing nothing is always an option
        self.disabled = frozenset(i for i, a in enumerate(ARM_NAMES) if a not in allowed)
        slots = min(int(params.get("slots", slots)), len(allowed)) if self.disabled else int(params.get("slots", slots))
        self.symbol, self.costs, self.params = symbol, costs, params
        self.episode_min = int(params.get("episode_min", episode_min))     # a quiet market can run longer episodes (less gas)
        self.slots = max(1, int(slots))
        self.eps: list[PaperEpisode | None] = [None] * self.slots
        self.starts: list[int] = [0] * self.slots
        self.lasts: list = [None] * self.slots  # the fly's choice for each slot, so each result trains its own choice
        self.last_start: int = 0
        self.candles: pd.DataFrame | None = None
        # "bench": a strategy that keeps losing here is left out for a while (false switches it off, a dict tunes it)
        self.bench = ArmBench.from_params(params.get("bench"), self.costs.stake)
        if self.bench is not None and (self.dir / "bench.json").exists():
            try:
                self.bench.load(json.loads((self.dir / "bench.json").read_text()))
            except (ValueError, OSError):
                pass        # an unreadable bench file just means a clean slate

    def benched(self, ts: int) -> frozenset:
        """Indexes of the strategies the bench is holding back at time ts."""
        if self.bench is None:
            return frozenset()
        return frozenset(ARM_NAMES.index(a) for a in self.bench.blocked(ts) if a in ARM_NAMES)

    @property
    def episode(self) -> PaperEpisode | None:
        return self.eps[0]

    @episode.setter
    def episode(self, ep: PaperEpisode | None) -> None:
        self.eps[0] = ep

    @property
    def episode_start(self) -> int:
        return self.starts[0]

    @episode_start.setter
    def episode_start(self, ts: int) -> None:
        self.starts[0] = ts

    def running(self) -> list[tuple[int, PaperEpisode]]:
        return [(i, e) for i, e in enumerate(self.eps) if e is not None]

    # ---------- episode lifecycle ----------
    def decide(self, ts: int, best_bid: float, best_ask: float, daily_stop_hit: bool, slot: int = 0) -> str:
        feats = feature_table(self.candles).iloc[-1]
        kc, scores = self.sniff(feats)
        busy = {ARM_NAMES.index(e.arm) for i, e in enumerate(self.eps) if e is not None and i != slot}
        held = self.disabled | self.benched(ts)
        k = self.fly.choose(feats, (busy | held) or None)  # encodes again internally; cheap
        arm = ARM_NAMES[k]
        if (daily_stop_hit and arm != "flat") or k in held:   # the second case only if every other arm was busy or held back
            arm, k = "flat", ARM_NAMES.index("flat")
            self.fly.last = (k, self.fly.last[1], float(scores[k]))
        if self.bench is not None:
            self.bench.note_start(arm, ts)
        trend_dir = 0
        if arm == "trend":
            p = self.params.get("trend", {})
            look = int(p.get("lookback", 60))
            c = self.candles["close"]
            ret = c.iloc[-1] / c.iloc[-look] - 1.0 if len(c) >= look else 0.0
            trend_dir = int(np.sign(ret)) if abs(ret) >= float(p.get("threshold", 0.002)) else 0
        self.lasts[slot] = self.fly.last
        self.eps[slot] = PaperEpisode(arm, self.costs, self.params.get(arm, {}))
        self.eps[slot].start(ts, best_bid, best_ask, trend_dir)
        self.starts[slot] = ts
        self.last_start = max(self.last_start, ts)
        self.last_decision = {
            "slot": slot, "ts": ts, "arm": arm, "scores": [round(float(x), 4) for x in scores],
            "kc_active": [int(i) for i in np.flatnonzero(kc > 0)], "kc_frac": round(float((kc > 0).mean()), 4),
            "features": {f: (None if not np.isfinite(feats[f]) else round(float(feats[f]), 6)) for f in FEATURES},
            "trend_dir": trend_dir, "guard_flat": daily_stop_hit,
        }
        self.log("decisions", self.last_decision)
        return arm

    def finish(self, ts: int, best_bid: float, best_ask: float, slot: int = 0) -> dict:
        ep, start = self.eps[slot], self.starts[slot]
        pnl = ep.finish(ts, best_bid, best_ask)
        win = self.candles[self.candles["timestamp"] >= start]
        if len(win) < 3:
            win = self.candles.tail(self.episode_min)
        raw = reward_of(pnl, win, self.costs, clip=np.inf)
        rew = self.scaler(raw)
        taught: list[str] = []
        if self.lasts[slot] is not None:
            self.fly.last = self.lasts[slot]
        self.fly.update(rew)
        if self.lasts[slot] is not None and self.params.get("teach_all", True):
            taught = self.teach_others(ep.arm, self.lasts[slot][1], start, win)
        self.book(ts, pnl)
        row = {"slot": slot, "start": start, "end": ts, "arm": ep.arm, "pnl": round(pnl, 6), "reward": round(rew, 4),
               "raw_reward": round(raw, 6), "fills": len(ep.fills), "actions": ep.actions, "equity": round(self.equity, 6),
               "dopamine": "PAM" if rew > 0 else "PPL1" if rew < 0 else "none", "weights_moved": self.weights_moved(),
               "taught": taught}
        self.log("episodes", row)
        for f in ep.fills:
            self.log("fills", {"arm": ep.arm, **asdict(f)})
        self.eps[slot] = None
        self.lasts[slot] = None
        if self.bench is not None:
            event = self.bench.record(ep.arm, pnl, len(ep.fills) > 0 or ep.actions > 0, ts)
            if event:
                self.log("bench", {"ts": ts, **event})
            (self.dir / "bench.json").write_text(json.dumps(self.bench.to_dict()))
        self.save()
        return row

    def teach_others(self, ran: str, kc: np.ndarray, start: int, win: pd.DataFrame) -> list[str]:
        """The strategies this episode did not run learn what they would have made on the same candles.

        Only the strategy that ran is taught by its paper result, so a fly that mostly sits in flat or avoids a
        strategy would hear almost nothing about the rest. The candle simulation gives every strategy's result for
        the window that just passed (the same call that replays history in pretrain.counterfactual), and each is
        taught as if it had been chosen, from the same sniffed market state. Flat is worth zero and never taught.
        Set `"teach_all": false` in a market's params to switch this off."""
        hist = self.candles[self.candles["timestamp"] < start].tail(LONG)
        taught = []
        for k, arm in enumerate(ARM_NAMES):
            if arm in (ran, "flat") or k in self.disabled:
                continue
            res = ARMS[arm](win, hist, self.costs, self.params.get(arm, {}))
            self.fly.last = (k, kc, 0.0)
            self.fly.update(self.scaler(reward_of(res.pnl, win, self.costs, clip=np.inf)))
            taught.append(arm)
        return taught

    def snapshot(self) -> dict:
        ep = self.eps[0] or next((e for e in self.eps if e is not None), None)
        until = {} if self.bench is None else {a: t for a, t in self.bench.until.items()}
        return {"symbol": self.symbol, "kind": "spot", "parallel": len(self.running()), "benched": until, **self.brain_snapshot(),
                "live": None if ep is None else {"arm": ep.arm, "inv": ep.inv, "cash": round(ep.cash, 6),
                                                  "bids": ep.bids, "asks": ep.asks, "fills": len(ep.fills)}}


class Runner:
    def __init__(self, network: str, symbols: list[str], state: Path, episode_min: int, params: dict,
                 costs: Costs, daily_loss_stop: float, seed: int, perp_tape: Path | None = None,
                 perp_symbols: list[str] | None = None, perp_costs: PerpCosts | None = None,
                 ec_tape: Path | None = None, ec_costs: ECCosts | None = None, settings: Settings | None = None,
                 slots: int = 1, brain: str = "mb", perp_prefix: str = PERP_TAPE_PREFIX):
        self.network, self.symbols, self.state = network, symbols, state
        self.episode_min, self.daily_loss_stop = episode_min, daily_loss_stop
        graph = load_graph(ROOT / "data" / "mb")
        self.flies = {s: MarketFly(s, graph, state, costs, params.get(s, {}), seed, episode_min, slots, brain) for s in symbols}
        self.books = {s: Book() for s in symbols}
        # markets whose paper fills come from real trade prints (stable pairs): we subscribe to their trades channel
        self.trade_syms = sorted(s for s in symbols if params.get(s, {}).get("maker", {}).get("trade_fills"))
        self.touch: dict[str, tuple[float, float]] = {}
        self.stop = asyncio.Event()
        self.perp_tail = Tail(perp_tape, perp_prefix, from_start=True) if perp_tape else None
        # paper only, so no stop. Each perp fly gets its own seed: with one seed they explored the same arm at the
        # same minute on every market, three copies of one bet
        self.perp_all = bool(perp_tape) and list(perp_symbols or []) == ["all"]   # one fly per market seen on the tape
        self.perp_args = (graph, state, perp_costs or PerpCosts(), params, seed, episode_min)
        self.perps: dict[str, PerpFly] = {}     # built below, once the order executors exist
        self.ec_tail = Tail(ec_tape, "mainnet_ec_", from_start=True) if ec_tape else None
        ec_live = settings is not None and settings.ec_live
        self.ec = ECFly(graph, state, ec_costs or ECCosts(), params.get("EC", {}), seed,
                        daily_loss_stop if ec_live else math.inf) if ec_tape else None
        self.warm = True  # first tape read replays history to rebuild features without trading it
        # order mirror: in a dry run it only logs what it would do, live it places real spot orders
        self.mirrors: dict[str, SpotMirror] = {}
        # with live orders the daily stop watches the wallet (one ledger for all markets), not the paper P&L
        self.ledger: WalletLedger | None = None
        self.settings = settings
        # perp orders: the Node sidecar signs them (live) or they are logged (dry run); only when perps are on
        self.perp_exec = None
        # event contract bets: the sidecar signs them on mainnet (live), elsewhere they are logged (dry run or testnet)
        self.ec_exec = None
        if settings is not None:
            def route(kind: str, row: dict) -> None:
                self.flies.get(row.get("symbol"), next(iter(self.flies.values()))).log(kind, row)
            # no spot markets (a perps-only run on Hideki): no spot executor, the chain has no spot pools there
            executor = build_executor(settings, route) if symbols else None
            if perp_tape:
                self.perp_exec = build_perp_executor(settings, lambda kind, row: self._perp_log(kind, row))
            if self.ec is not None:
                self.ec_exec = build_ec_executor(replace(settings, dry_run=not ec_live), self.ec.log)
                self.ec.mirror = ECMirror(self.ec_exec, settings, self.ec.log, store=self.ec.dir / "ec_bets.json")
                self.ec.day_loss = lambda: self.day_loss(self.ec)
            self.ledger = WalletLedger(executor, perps=self.perp_exec,
                                       ec_mirror=self.ec.mirror if self.ec is not None else None) if settings.live else None
            if self.ec is not None:
                self.ec._mirror("on_start")         # claim what a restart left behind; failures are logged, not raised
                if self.ledger is not None:
                    self.ledger.update_ec()
            self.mirrors = {s: SpotMirror(executor, s, costs.stake, settings, route, ledger=self.ledger) for s in symbols}
        if perp_tape and not self.perp_all:        # named perp markets exist from the start; "all" grows with the tape
            self.perps = {s: self.perp_fly(s, i) for i, s in enumerate(perp_symbols or [])}

    def day_loss(self, f) -> float:
        """What the daily stop compares: the wallet's real day P&L when live orders are on and the wallet has been
        read, otherwise the fly's paper P&L for the day."""
        wallet = self.ledger.day_pnl if self.ledger is not None else None
        return f.day_pnl if wallet is None else wallet

    def _perp_log(self, kind: str, row: dict) -> None:
        f = self.perps.get(row.get("symbol"))
        if f is not None:
            f.log(kind, row)
        else:
            with (self.state / "perp_orders.jsonl").open("a") as fh:
                fh.write(json.dumps({"wall": int(time.time() * 1000), **row}, default=str) + "\n")

    def perp_fly(self, symbol: str, i: int) -> PerpFly:
        graph, state, costs, params, seed, episode_min = self.perp_args
        live = self.perp_exec is not None and not self.perp_exec.dry
        f = PerpFly(symbol, graph, state, costs, params.get(symbol, {}), seed + i, episode_min,
                    self.daily_loss_stop if live else math.inf)     # the stop protects real money only
        self._attach_perp(f)
        return f

    def _attach_perp(self, f: PerpFly) -> None:
        if self.perp_exec is None or f.mirror is not None:
            return
        _, _, costs, _, _, _ = self.perp_args
        f.mirror = PerpMirror(self.perp_exec, f.symbol, costs.stake, costs.leverage, self.settings, f.log,
                              store=f.dir / "perp_open_orders.json")
        f.day_loss = lambda f=f: self.day_loss(f)
        if self.ledger is not None:
            self.ledger.update_perps()          # the day's baseline is the account before this fly's first order

    def kill_requested(self) -> bool:
        return (self.state / "KILL").exists()

    def write_live(self) -> None:
        snap = {"wall": int(time.time() * 1000), "network": self.network, "episode_min": self.episode_min,
                "markets": {s: f.snapshot() for s, f in self.flies.items()},
                "perps": {s: f.snapshot() for s, f in self.perps.items()},
                "ec": None if self.ec is None else self.ec.snapshot(),
                "touch": {s: list(t) for s, t in self.touch.items()}, "kill": self.kill_requested(),
                "wallet_day_pnl": None if self.ledger is None else self.ledger.day_pnl,
                "orders": None if self.settings is None else ("live" if self.settings.live else "dry"),
                "perp_orders": None if self.perp_exec is None else ("dry" if self.perp_exec.dry else "live"),
                "ec_orders": None if self.ec_exec is None else ("dry" if self.ec_exec.dry else "live")}
        tmp = self.state / "live.json.tmp"
        tmp.write_text(json.dumps(snap, default=str))
        tmp.replace(self.state / "live.json")

    def on_trades(self, msg: dict) -> None:
        """A trades channel message: hand each real trade to the running episodes of its market. Only live updates
        count; the snapshot sent on subscribing is history."""
        if msg.get("type") != "update":
            return
        items = msg.get("trades") or ([msg["trade"]] if isinstance(msg.get("trade"), dict) else [msg])
        for t in items:
            f = self.flies.get(t.get("symbol") or msg.get("symbol"))
            if f is None or t.get("price") is None:
                continue
            ts = int(t.get("timestamp") or time.time() * 1000)
            price, qty = float(t["price"]), float(t.get("quantity") or t.get("amount") or 0.0)
            for _i, ep in f.running():
                ep.on_trade(ts, price, qty, str(t.get("side") or ""))

    async def book_task(self) -> None:
        if not self.symbols:      # perps only: no spot book to follow
            return
        backoff = 1.0
        while not self.stop.is_set():
            for b in self.books.values():
                b.__init__()
            try:
                async with websockets.connect(WS[self.network], ping_interval=None, max_size=2**22) as ws:
                    await ws.send(json.dumps({"operation": "subscribe", "channel": "orderbook",
                                              "params": {"symbols": self.symbols, "depth": 5}}))
                    if self.trade_syms:
                        await ws.send(json.dumps({"operation": "subscribe", "channel": "trades",
                                                  "params": {"symbols": self.trade_syms, "limit": 1}}))
                    backoff = 1.0

                    async def pinger() -> None:
                        while True:
                            await asyncio.sleep(30)
                            await ws.send(json.dumps({"operation": "ping"}))

                    pt = asyncio.create_task(pinger())
                    try:
                        while not self.stop.is_set():
                            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=90))
                            if msg.get("type") == "shutdown":
                                break
                            if msg.get("channel") == "trades":
                                self.on_trades(msg)
                                continue
                            if msg.get("channel") != "orderbook" or msg.get("type") not in ("snapshot", "update"):
                                continue
                            sym = msg["symbol"]
                            (self.books[sym].snapshot if msg["type"] == "snapshot" else self.books[sym].update)(msg)
                            b = self.books[sym]
                            if b.bids and b.asks:
                                bb, ba = max(b.bids), min(b.asks)
                                if ba > bb:
                                    self.touch[sym] = (bb, ba)
                                    f = self.flies[sym]
                                    for _i, ep in f.running():
                                        ep.on_touch(int(time.time() * 1000), bb, ba, b)
                                    m = self.mirrors.get(sym)
                                    if m is not None and f.episode is not None:  # only the primary slot places orders
                                        # off the event loop: a live order waits for the chain
                                        fut = asyncio.get_running_loop().run_in_executor(None, m.sync, f.episode, bb, ba)
                                        fut.add_done_callback(lambda d, f=f: self._sync_done(d, f))
                    finally:
                        pt.cancel()
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                print(f"book: {str(e)[:120]}", file=sys.stderr, flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    @staticmethod
    def _sync_done(done: asyncio.Future, f: MarketFly) -> None:
        """A mirror sync that raised would otherwise vanish inside the executor; it is logged like any other error."""
        if done.cancelled():
            return
        e = done.exception()
        if e is not None:
            f.log("errors", {"where": "mirror.sync", "error": repr(e)[:1500]})

    async def minute_task(self) -> None:
        """Once a minute: refresh candles, re-centre makers, run episode boundaries."""
        base = REST.get(self.network, "")      # a perps-only run (Hideki) has no spot API; the loop still writes live.json
        while not self.stop.is_set():
            now = int(time.time() * 1000)
            for sym, f in self.flies.items():
                try:
                    await self._minute_for(f, sym, base, now)
                except Exception:  # noqa: BLE001 - one market's trouble must not stop the loop
                    tb = traceback.format_exc()
                    f.log("errors", {"where": "minute", "error": tb[-1500:]})
                    print(f"{sym} minute error: {tb.strip().splitlines()[-1]}", file=sys.stderr, flush=True)
            try:
                self.write_live()
            except Exception as e:  # noqa: BLE001
                print(f"live.json: {e}", file=sys.stderr, flush=True)
            await asyncio.sleep(60 - (time.time() % 60))

    async def _minute_for(self, f: MarketFly, sym: str, base: str, now: int) -> None:
        need = LONG + f.episode_min + 20
        if f.params.get("dense_candles"):
            # a market that trades a few times an hour: look back as far as the API allows (1000 minutes) for the last
            # close, then fill the quiet minutes
            raw = await asyncio.to_thread(fetch_candles, base, sym, 1000)
            mid = sum(self.touch[sym]) / 2 if sym in self.touch else float(raw["close"].iloc[-1]) if len(raw) else 1.0
            f.candles = densify(raw, now, need, mid)
        else:
            f.candles = await asyncio.to_thread(fetch_candles, base, sym, need)
        if sym not in self.touch or f.candles is None or len(f.candles) < LONG + 2:
            return
        f.note_sniff(feature_table(f.candles).iloc[-1], now)  # the fly smells the market every minute
        bb, ba = self.touch[sym]
        m = self.mirrors.get(sym)
        kill = self.kill_requested()
        for i, ep in f.running():
            ep.on_minute(now, bb, ba)
            if now - f.starts[i] >= f.episode_min * 60_000 or kill:
                if i == 0 and m is not None:
                    await asyncio.to_thread(m.finish, ep, bb, ba)
                row = f.finish(now, bb, ba, i)
                print(f"{sym}[{i}] episode {row['arm']} pnl {row['pnl']:.4f} reward {row['reward']:.3f} equity {row['equity']:.3f}", flush=True)
        stagger = f.episode_min * 60_000 // f.slots
        if not kill and now - f.last_start >= stagger:
            for i in range(f.slots):  # one new episode a minute at most, so the slots end up spread through the hour
                if f.eps[i] is None:
                    stop_hit = self.day_loss(f) <= -self.daily_loss_stop
                    arm = f.decide(now, bb, ba, stop_hit, i)
                    print(f"{sym}[{i}] decide -> {arm}", flush=True)
                    if i == 0 and m is not None:
                        await asyncio.to_thread(m.on_start, f.eps[0], bb, ba)
                    break

    def feed_tapes(self) -> None:
        """New recorder rows -> perp and EC flies. On the first read (a restart) the perp history only
        rebuilds features: no episode is opened on stale rows; EC rows older than the current window
        are skipped."""
        kill = self.kill_requested()
        if self.perp_tail is not None:
            rows = self.perp_tail.read_new()
            latest: dict[str, str] = {}
            for r in rows:
                latest[r.get("symbol", "")] = max(latest.get(r.get("symbol", ""), ""), r.get("wall", ""))
            for r in rows:
                sym = r.get("symbol", "")
                if r.get("type") != "perp" or not sym:
                    continue
                if self.perp_all and sym not in self.perps:
                    self.perps[sym] = self.perp_fly(sym, len(self.perps))
                f = self.perps.get(sym)
                if f is None:
                    continue
                try:
                    stale = self.warm and r.get("wall", "") < latest[r["symbol"]]
                    done = f.on_row(r, kill=kill or stale)
                    if done:
                        print(f"{f.symbol} perp episode {done['arm']} pnl {done['pnl']:.4f} reward {done['reward']:.3f} equity {done['equity']:.3f}", flush=True)
                        if self.ledger is not None:
                            self.ledger.update_perps()
                    elif f.episode is not None and f.episode_start == int(f.frame["timestamp"].iloc[-1]):
                        print(f"{f.symbol} perp decide -> {f.episode.arm}", flush=True)
                except Exception:  # noqa: BLE001
                    tb = traceback.format_exc()
                    f.log("errors", {"where": "perp_row", "error": tb[-1500:]})
                    print(f"{f.symbol} perp error: {tb.strip().splitlines()[-1]}", file=sys.stderr, flush=True)
        if self.ec_tail is not None and self.ec is not None:
            rows = self.ec_tail.read_new()
            now_s = time.time()
            for r in rows:
                if r.get("type") == "ec_book" and self.warm and float(r.get("expiry") or 0) < now_s:
                    continue
                try:
                    done = self.ec.on_row(r, kill=kill)
                    if done:
                        print(f"EC {done['symbol']} {done['arm']} side {done['side']} truth {done['truth']} pnl {done['pnl']:.4f} equity {done['equity']:.3f}", flush=True)
                        if self.ledger is not None:
                            self.ledger.update_ec()
                except Exception:  # noqa: BLE001
                    tb = traceback.format_exc()
                    self.ec.log("errors", {"where": "ec_row", "error": tb[-1500:]})
                    print(f"EC error: {tb.strip().splitlines()[-1]}", file=sys.stderr, flush=True)
            if not self.warm:
                self.ec._mirror("claim_due")
        self.warm = False

    async def tape_task(self) -> None:
        while not self.stop.is_set():
            try:
                await asyncio.to_thread(self.feed_tapes)
            except Exception as e:  # noqa: BLE001
                print(f"tape: {e}", file=sys.stderr, flush=True)
            await asyncio.sleep(TAPE_POLL_S)

    async def run(self) -> None:
        tasks = [asyncio.create_task(self.book_task()), asyncio.create_task(self.minute_task())]
        if self.perp_tail is not None or self.ec_tail is not None:
            tasks.append(asyncio.create_task(self.tape_task()))
        await self.stop.wait()
        for t in tasks:
            t.cancel()
        now = int(time.time() * 1000)
        for sym, f in self.flies.items():
            for i, ep in f.running():
                if sym not in self.touch:
                    break
                if i == 0 and sym in self.mirrors:
                    self.mirrors[sym].finish(ep, *self.touch[sym])
                f.finish(now, *self.touch[sym], i)
        for f in self.perps.values():
            if f.episode is not None and f.frame is not None and not f.frame.empty:
                f.finish(f.frame.iloc[-1])
        if self.perp_exec is not None and hasattr(self.perp_exec, "stop"):
            self.perp_exec.stop()
        if self.ec_exec is not None and hasattr(self.ec_exec, "stop"):
            self.ec_exec.stop()
        self.write_live()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default=os.environ.get("NETWORK") or "mainnet", choices=list(NETWORKS),
                    help="testnet (Shannon) or mainnet for spot and event contracts, hideki for perps")
    ap.add_argument("--symbols", nargs="*", default=None, help="spot markets; default is MARKETS from the environment or all three")
    ap.add_argument("--slots", type=int, default=None, help="paper episodes at once per spot market, sharing one brain; default CONCURRENT_EPISODES or 3")
    ap.add_argument("--no-orders", action="store_true", help="do not mirror episodes into orders at all (no dry-run log, no live orders)")
    ap.add_argument("--state", default=str(ROOT / "state" / "live"))
    ap.add_argument("--episode-min", type=int, default=60)
    ap.add_argument("--params", default="", help="per-market arm params json; default is config/arm_params.json")
    ap.add_argument("--stake", type=float, default=1000.0)
    ap.add_argument("--daily-loss-stop", type=float, default=None,
                    help="force the spot flies flat after this daily loss (USDso). Default: 50 with live orders, none otherwise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--brain", default=None, choices=list(BRAINS), help="learning readout for the spot flies; default BRAIN from the environment or delta")
    ap.add_argument("--perp-tape", default="", help="dir with the perp tape (<prefix>*.jsonl) from perp/recorder.mjs; enables perps")
    ap.add_argument("--perp-prefix", default=PERP_TAPE_PREFIX, help="tape file prefix of the perp network being recorded")
    ap.add_argument("--perp-symbols", nargs="*", default=None, help="perp markets to paper trade, or 'all' for every market on the tape")
    ap.add_argument("--perp-leverage", type=float, default=2.0)
    ap.add_argument("--ec-tape", default="", help="dir with mainnet_ec_*.jsonl from perp/ec_recorder.mjs; enables EC")
    ap.add_argument("--ec-stake", type=float, default=20.0)
    a = ap.parse_args()
    try:
        settings = Settings.from_env({**os.environ, "NETWORK": a.network})
    except SettingsError as e:
        sys.exit(f"settings: {e}")
    a.symbols = a.symbols or list(settings.markets)
    a.perp_symbols = a.perp_symbols or list(settings.perp_markets) or PERP_SYMBOLS
    if settings.live and not a.no_orders:
        # the flags can mix tapes for paper research; real orders keep to one network, the same rule as the settings
        try:
            replace(settings, markets=tuple(a.symbols), perp_markets=tuple(a.perp_symbols) if a.perp_tape else (),
                    ec=bool(a.ec_tape)).validate()
        except SettingsError as e:
            sys.exit(f"settings: {e}")
    params_path = Path(a.params) if a.params else ROOT / "config" / "arm_params.json"
    params = json.loads(params_path.read_text()) if params_path.exists() else {}
    state = Path(a.state)
    state.mkdir(parents=True, exist_ok=True)
    costs = Costs(stake=a.stake)
    r = Runner(a.network, a.symbols, state, a.episode_min, params, costs, settings.daily_loss_stop(a.daily_loss_stop), a.seed,
               perp_tape=Path(a.perp_tape) if a.perp_tape else None, perp_symbols=a.perp_symbols,
               perp_costs=PerpCosts(stake=a.stake, leverage=a.perp_leverage),
               ec_tape=Path(a.ec_tape) if a.ec_tape else None, ec_costs=ECCosts(stake=a.ec_stake),
               settings=None if a.no_orders else settings, slots=a.slots or settings.concurrent_episodes,
               brain=a.brain or settings.brain, perp_prefix=a.perp_prefix)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, r.stop.set)
    extra = (f" perps {a.perp_symbols} from {a.perp_tape}" if a.perp_tape else "") + (f" EC from {a.ec_tape}" if a.ec_tape else "")
    print(f"paper-live {a.network} {a.symbols} episode {a.episode_min} min{extra} -> {state}", flush=True)
    print("orders: " + ("off" if a.no_orders else settings.summary()), flush=True)
    loop.run_until_complete(r.run())


if __name__ == "__main__":
    main()
