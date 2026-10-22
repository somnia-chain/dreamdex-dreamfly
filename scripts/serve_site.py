#!/usr/bin/env python3
"""Serve the static flybot site plus the live state files.

  /live.json              -> <state>/live.json
  /episodes/<name>.jsonl  -> <state>/<name>/episodes.jsonl
  everything else         -> site/

Usage: python scripts/serve_site.py [--state state/live] [--port 8765]
"""
import argparse
import base64
import gzip
import hmac
import http.server
import os
import re
import socketserver
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
NAME_RE = re.compile(r"^/episodes/([A-Za-z0-9_.\-]+)\.jsonl$")
DATA_RE = re.compile(r"^/data/([A-Za-z0-9_.\-]+\.json)$")

# Cache rules. Live state must never be cached; the multi-MB brain data changes only when the build
# scripts run, so it is cached for a day and revalidated by ETag; code and markup revalidate every time
# (a 304 costs nothing and a deploy shows up on the next reload).
NO_STORE = "no-store, no-cache, must-revalidate, max-age=0"
DATA_CACHE = "public, max-age=86400, stale-while-revalidate=604800"
REVALIDATE = "no-cache"

_gz_lock = threading.Lock()
_gz_cache: dict[str, tuple[tuple[int, int], bytes, int]] = {}  # name -> ((mtime_ns, size), gzipped, raw length)


def gzipped(path: Path) -> tuple[bytes, int, str]:
    """Gzip a data file once per version (mtime + size), kept in memory. Returns (body, raw length, etag)."""
    st = path.stat()
    key = (st.st_mtime_ns, st.st_size)
    with _gz_lock:
        hit = _gz_cache.get(path.name)
        if hit and hit[0] == key:
            return hit[1], hit[2], f'"{key[0]:x}-{key[1]:x}"'
    raw = path.read_bytes()
    body = gzip.compress(raw, compresslevel=6, mtime=0)
    with _gz_lock:
        _gz_cache[path.name] = (key, body, len(raw))
    return body, len(raw), f'"{key[0]:x}-{key[1]:x}"'


def make_handler(state: Path, password: str = ""):
    """password: if set, every request needs HTTP Basic auth with it (any user name)."""

    class Handler(http.server.SimpleHTTPRequestHandler):
        def authorised(self) -> bool:
            if not password:
                return True
            head = self.headers.get("Authorization", "")
            if head.startswith("Basic "):
                try:
                    given = base64.b64decode(head[6:]).decode().split(":", 1)[-1]
                    if hmac.compare_digest(given.encode(), password.encode()):
                        return True
                except Exception:  # noqa: BLE001 - anything unreadable is a refusal
                    pass
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="DreamFly"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False

        def do_HEAD(self):
            if self.authorised():
                super().do_HEAD()

        def __init__(self, *a, **kw):
            self._cache = NO_STORE
            super().__init__(*a, directory=str(SITE), **kw)

        def do_GET(self):
            p = self.path.split("?", 1)[0].split("#", 1)[0]
            if p == "/healthz":          # for a platform health check: open, says nothing about the fly
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")
                return
            if not self.authorised():
                return
            m = NAME_RE.match(p)
            if m and ".." not in m.group(1) and not (state / m.group(1) / "episodes.jsonl").exists():
                # before the first episode: an empty log, not a 404 the page would keep reporting
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            d = DATA_RE.match(p)
            if d and (SITE / "data" / d.group(1)).is_file():
                return self.send_data(SITE / "data" / d.group(1))
            self._cache = NO_STORE if (p == "/live.json" or m) else REVALIDATE
            super().do_GET()

        def send_data(self, path: Path) -> None:
            """Brain data: gzip (when accepted), ETag, day-long cache, raw length for progress bars."""
            self._cache = DATA_CACHE
            accept_gz = "gzip" in self.headers.get("Accept-Encoding", "")
            body, raw_len, etag = gzipped(path)
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self.send_header("ETag", etag)
                self.end_headers()
                return
            if not accept_gz:
                body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("ETag", etag)
            self.send_header("Vary", "Accept-Encoding")
            self.send_header("X-Raw-Length", str(raw_len))
            if accept_gz:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def translate_path(self, path):
            p = path.split("?", 1)[0].split("#", 1)[0]
            if p == "/live.json":
                return str(state / "live.json")
            m = NAME_RE.match(p)
            if m and ".." not in m.group(1):
                return str(state / m.group(1) / "episodes.jsonl")
            return super().translate_path(path)

        def end_headers(self):
            self.send_header("Cache-Control", self._cache)
            if self._cache == NO_STORE:
                self.send_header("Pragma", "no-cache")
            super().end_headers()

        def guess_type(self, path):
            if str(path).endswith(".jsonl"):
                return "application/x-ndjson"
            return super().guess_type(path)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", default="state/live", help="live state dir (relative to repo root)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")), help="default: $PORT, else 8765")
    args = ap.parse_args()
    state = Path(args.state)
    if not state.is_absolute():
        state = ROOT / state
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("", args.port), make_handler(state, os.environ.get("DASHBOARD_PASSWORD", ""))) as srv:
        print(f"serving {SITE} on http://localhost:{args.port}  (state: {state})")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
