"""Follow a recorder's daily JSONL files (<dir>/<prefix>YYYYMMDD.jsonl): return new complete rows
each call, roll to the newest day file when one appears, survive restarts by starting at the end of
the current file (or at its start with `from_start`)."""

from __future__ import annotations

import json
from pathlib import Path


class Tail:
    def __init__(self, directory: Path | str, prefix: str, from_start: bool = False):
        self.dir, self.prefix = Path(directory), prefix
        self.path: Path | None = None
        self.offset = 0
        self.buf = b""
        if not from_start:
            p = self._newest()
            if p is not None:
                self.path, self.offset = p, p.stat().st_size

    def _newest(self) -> Path | None:
        files = sorted(self.dir.glob(f"{self.prefix}*.jsonl"))
        return files[-1] if files else None

    def read_new(self) -> list[dict]:
        newest = self._newest()
        rows: list[dict] = []
        if newest is None:
            return rows
        if self.path is None or newest != self.path:
            rows += self._drain()  # finish the old file first
            self.path, self.offset, self.buf = newest, 0, b""
        rows += self._drain()
        return rows

    def _drain(self) -> list[dict]:
        if self.path is None or not self.path.exists():
            return []
        with self.path.open("rb") as fh:
            fh.seek(self.offset)
            chunk = fh.read()
        self.offset += len(chunk)
        data = self.buf + chunk
        lines = data.split(b"\n")
        self.buf = lines.pop()  # possibly incomplete last line
        out = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return out
