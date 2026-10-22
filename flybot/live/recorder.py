"""Record the DreamDEX order book and trades over the public WebSocket into daily JSONL tapes.

Tape rows share the shape of the earlier polling/push tapes so replay code can read either:
  {"symbol", "timestamp", "bids": [[px, qty], ...], "asks": [[px, qty], ...], "wall"}      book snapshot
  {"type": "trade", "symbol", "timestamp", "price", "quantity", "side", "wall"}           executed trade
  {"error": ..., "wall"}                                                                  recorder trouble

Rules from the API (no sequence numbers, no replay): every reconnect is a cold start, the local book
is replaced by the fresh snapshot, never merged. Ping every 30 s.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import websockets

WS = {"mainnet": "wss://api.dreamdex.io/v0/ws/public", "testnet": "wss://stg.api.dreamdex.io/v0/ws/public"}
SYMBOLS = ["SOMI:USDso", "WETH:USDso", "WBTC:USDso"]


class Tape:
    def __init__(self, out_dir: Path, prefix: str):
        self.out_dir, self.prefix = out_dir, prefix
        self.day, self.fh = None, None
        out_dir.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict) -> None:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        if day != self.day:
            if self.fh:
                self.fh.close()
            self.fh = (self.out_dir / f"{self.prefix}_{day}.jsonl").open("a")
            self.day = day
        row["wall"] = int(time.time() * 1000)
        self.fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.fh.flush()

    def close(self) -> None:
        if self.fh:
            self.fh.close()


class Book:
    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.timestamp = 0

    def snapshot(self, msg: dict) -> None:
        self.bids = {float(x["price"]): float(x["quantity"]) for x in msg.get("bids", [])}
        self.asks = {float(x["price"]): float(x["quantity"]) for x in msg.get("asks", [])}
        self.timestamp = int(msg.get("timestamp", 0))

    def update(self, msg: dict) -> None:
        for side, levels in (("bids", self.bids), ("asks", self.asks)):
            for x in msg.get(side, []):
                px, q = float(x["price"]), float(x["quantity"])
                if q <= 0:
                    levels.pop(px, None)
                else:
                    levels[px] = q
        self.timestamp = int(msg.get("timestamp", self.timestamp))

    def row(self, symbol: str, depth: int) -> dict:
        bids = sorted(self.bids.items(), reverse=True)[:depth]
        asks = sorted(self.asks.items())[:depth]
        return {"symbol": symbol, "timestamp": self.timestamp, "bids": [[p, q] for p, q in bids],
                "asks": [[p, q] for p, q in asks]}


async def record(network: str, symbols: list[str], out_dir: Path, depth: int, min_gap_ms: int,
                 stop: asyncio.Event) -> None:
    tape = Tape(out_dir, f"{network}_book")
    backoff = 1.0
    while not stop.is_set():
        books = {s: Book() for s in symbols}  # cold start on every connection
        last_write = {s: 0 for s in symbols}
        pending: dict[str, asyncio.TimerHandle] = {}
        try:
            async with websockets.connect(WS[network], ping_interval=None, max_size=2**22) as ws:
                await ws.send(json.dumps({"operation": "subscribe", "channel": "orderbook",
                                          "params": {"symbols": symbols, "depth": depth}}))
                await ws.send(json.dumps({"operation": "subscribe", "channel": "trades",
                                          "params": {"symbols": symbols, "limit": 1}}))
                tape.write({"type": "connect", "network": network, "symbols": symbols})
                backoff = 1.0

                async def pinger() -> None:
                    while True:
                        await asyncio.sleep(30)
                        await ws.send(json.dumps({"operation": "ping"}))

                ping_task = asyncio.create_task(pinger())
                loop = asyncio.get_running_loop()

                def flush(sym: str) -> None:
                    pending.pop(sym, None)
                    tape.write(books[sym].row(sym, depth))
                    last_write[sym] = int(time.time() * 1000)

                try:
                    while not stop.is_set():
                        raw = await asyncio.wait_for(ws.recv(), timeout=90)
                        msg = json.loads(raw)
                        ch, typ = msg.get("channel"), msg.get("type")
                        if typ == "shutdown":
                            tape.write({"type": "shutdown"})
                            break
                        if typ == "error":
                            tape.write({"error": msg.get("errorName"), "message": msg.get("message")})
                            continue
                        if ch == "orderbook" and typ in ("snapshot", "update"):
                            sym = msg["symbol"]
                            if typ == "snapshot":
                                books[sym].snapshot(msg)
                            else:
                                books[sym].update(msg)
                            # debounce: at most one book row per symbol per min_gap_ms
                            now = int(time.time() * 1000)
                            wait = max(0, min_gap_ms - (now - last_write[sym]))
                            if sym not in pending:
                                pending[sym] = loop.call_later(wait / 1000, flush, sym)
                        elif ch == "trades" and typ in ("update", "snapshot"):
                            items = msg.get("trades") or ([msg["trade"]] if isinstance(msg.get("trade"), dict) else [msg])
                            for t in items:
                                if t.get("price") is None:
                                    continue
                                tape.write({"type": "trade", "symbol": msg.get("symbol", t.get("symbol")),
                                            "timestamp": int(t.get("timestamp", 0)), "price": t.get("price"),
                                            "quantity": t.get("quantity"), "side": t.get("side")})
                finally:
                    ping_task.cancel()
                    for h in pending.values():
                        h.cancel()
        except asyncio.CancelledError:
            break
        except Exception as e:  # noqa: BLE001 - a long recorder must survive network trouble
            tape.write({"error": str(e)[:200]})
        if stop.is_set():
            break
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 60.0)
    tape.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="mainnet", choices=list(WS))
    ap.add_argument("--symbols", nargs="*", default=SYMBOLS)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "data" / "tape"))
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--min-gap-ms", type=int, default=250)
    ap.add_argument("--duration", type=float, default=0, help="seconds; 0 = run until killed")
    a = ap.parse_args()

    stop = asyncio.Event()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    if a.duration > 0:
        loop.call_later(a.duration, stop.set)
    print(f"recording {a.network} {a.symbols} -> {a.out}", file=sys.stderr, flush=True)
    loop.run_until_complete(record(a.network, a.symbols, Path(a.out), a.depth, a.min_gap_ms, stop))


if __name__ == "__main__":
    main()
