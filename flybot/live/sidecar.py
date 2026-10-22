"""The Node sidecar client shared by the perp and event contract order paths.

A sidecar is one Node process that signs orders through the DreamDEX markets SDK. The runner talks to it over
pipes in JSON lines: one request per line (`{"id", "op", ...}`), one reply per line with the same id and an
`ok` flag. Requests go one at a time, so the signer's nonces stay in order. A sidecar that dies or stops
answering is killed and started again on the next call, and every failure is written to the order log.

The private key is never read here: the sidecar reads PRIVATE_KEY from the environment it inherits.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Log = Callable[[str, dict], None]

CALL_TIMEOUT_S = 90.0
INIT_TIMEOUT_S = 180.0


class SidecarError(RuntimeError):
    """The sidecar refused, died, or did not answer in time."""


class Sidecar:
    """Owns one Node sidecar process: starts it, sends one request at a time, restarts it when it dies.

    `on_init` gets the reply to the `init` request each time the process starts (first start and restarts)."""

    def __init__(self, argv: list[str], script: Path, log: Log, env: dict | None = None,
                 on_init: Callable[[dict], None] | None = None, name: str = "sidecar"):
        self.argv, self.script, self.log, self.env, self.on_init, self.name = argv, script, log, env, on_init, name
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._n = 0
        self.start()

    # ---- process ----
    def start(self) -> None:
        if not self.script.exists():
            raise SidecarError(f"{self.script} is missing")
        if not (self.script.parent / "node_modules").exists():
            raise SidecarError(f"{self.script.parent.name}/node_modules is missing: run npm install in {self.script.parent}")
        try:
            self._proc = subprocess.Popen(self.argv, cwd=self.script.parent, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, text=True, bufsize=1, env=self.env)
        except OSError as e:
            raise SidecarError(f"cannot start {self.argv[0]}: {e}") from e
        threading.Thread(target=self._drain_stderr, args=(self._proc,), daemon=True).start()
        info = self._request("init", timeout=INIT_TIMEOUT_S)
        if self.on_init is not None:
            self.on_init(info)

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        for line in proc.stderr:
            line = line.rstrip()
            if line:
                self.log("orders", {"mode": "live", "action": f"{self.name}_stderr", "text": line[:500]})

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self) -> None:
        p = self._proc
        if p is None:
            return
        try:
            if p.poll() is None:
                p.stdin.write(json.dumps({"id": 0, "op": "shutdown"}) + "\n")
                p.stdin.flush()
                p.wait(timeout=5)
        except Exception:  # noqa: BLE001
            p.kill()
        self._proc = None

    def _restart(self) -> None:
        try:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.kill()
        finally:
            self._proc = None
        self.start()

    # ---- requests ----
    def call(self, op: str, timeout: float = CALL_TIMEOUT_S, **args) -> dict:
        """One request at a time; a sidecar that died since the last call is started again first."""
        with self._lock:
            if not self.alive():
                self._restart()
            return self._request(op, timeout, **args)

    def safe(self, op: str, default, **args):
        """`call`, with a failure logged and answered with the safe value instead of raised."""
        try:
            return self.call(op, **args)
        except SidecarError as e:
            self.log("orders", {"mode": "live", "action": op, **args, "ok": False, "error": str(e)[:400]})
            return default

    def _request(self, op: str, timeout: float = CALL_TIMEOUT_S, **args) -> dict:
        self._n += 1
        rid = self._n
        p = self._proc
        if p is None:
            raise SidecarError(f"{self.name} is not running ({op})")
        try:
            p.stdin.write(json.dumps({"id": rid, "op": op, **args}) + "\n")
            p.stdin.flush()
        except (OSError, ValueError) as e:
            self._proc = None
            raise SidecarError(f"{self.name} gone while sending {op}: {e}") from e
        reply = self._read_reply(p, rid, time.time() + timeout, op)
        if not reply.get("ok"):
            raise SidecarError(reply.get("error") or f"{op} failed")
        return reply

    def _read_reply(self, p: subprocess.Popen, rid: int, deadline: float, op: str) -> dict:
        """One line per reply; the sidecar answers in order, so anything with another id is logged and skipped."""
        box: dict = {}

        def reader() -> None:
            while True:
                line = p.stdout.readline()
                if not line:
                    box["eof"] = True
                    return
                try:
                    d = json.loads(line)
                except ValueError:
                    self.log("orders", {"mode": "live", "action": f"{self.name}_noise", "text": line[:300]})
                    continue
                if d.get("id") == rid:
                    box["reply"] = d
                    return
                self.log("orders", {"mode": "live", "action": f"{self.name}_stray", "reply": d})

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        t.join(max(0.0, deadline - time.time()))
        if "reply" in box:
            return box["reply"]
        # a timeout or a dead process: the sidecar is killed so the next call starts a fresh one
        try:
            p.kill()
        finally:
            self._proc = None
        raise SidecarError(f"{self.name} {'exited' if box.get('eof') else 'timed out'} during {op}")
