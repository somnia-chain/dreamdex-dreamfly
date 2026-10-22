"""Backfill DreamDEX spot candles (public REST, no auth) into data/candles/<symbol>_<interval>.parquet.

Pages backwards with endTime until the API returns nothing. Re-running merges with the existing file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "candles"
BASE = "https://api.dreamdex.io/v0/markets"
SYMBOLS = ("WETH:USDso", "WBTC:USDso", "SOMI:USDso")


def fetch_page(symbol: str, interval: str, end_ms: int | None) -> list[dict]:
    q = {"interval": interval, "limit": 1000}
    if end_ms is not None:
        q["endTime"] = end_ms
    url = f"{BASE}/{urllib.parse.quote(symbol)}/candles?{urllib.parse.urlencode(q)}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)["candles"]
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt + 1}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"gave up on {url}")


def backfill(symbol: str, interval: str, max_pages: int) -> pd.DataFrame:
    rows: list[dict] = []
    end: int | None = None
    for _ in range(max_pages):
        page = fetch_page(symbol, interval, end)
        if not page:
            break
        rows.extend(page)
        oldest = min(int(c["timestamp"]) for c in page)
        if end is not None and oldest >= end:
            break
        end = oldest
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["timestamp"] = df["timestamp"].astype("int64")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--symbols", nargs="*", default=SYMBOLS)
    ap.add_argument("--max-pages", type=int, default=200)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for sym in a.symbols:
        path = OUT / f"{sym.replace(':', '_')}_{a.interval}.parquet"
        df = backfill(sym, a.interval, a.max_pages)
        if path.exists():
            df = pd.concat([pd.read_parquet(path), df])
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        df.to_parquet(path, index=False)
        t0 = pd.to_datetime(df["timestamp"].iloc[0], unit="ms", utc=True)
        t1 = pd.to_datetime(df["timestamp"].iloc[-1], unit="ms", utc=True)
        print(f"{sym} {a.interval}: {len(df)} candles, {t0:%Y-%m-%d %H:%M} -> {t1:%Y-%m-%d %H:%M} UTC -> {path.name}")


if __name__ == "__main__":
    main()
