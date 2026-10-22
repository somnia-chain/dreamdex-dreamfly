"""One supervised process for containers: the runner, the dashboard server and, only when perps or event
contracts are enabled, the Node recorders.

  dreamfly start

It reads the same settings as the runner (NETWORK, DRY_RUN, MARKETS, PERP_MARKETS, EC_ENABLED, ...). The
dashboard shows exactly the markets the runner reports, so a container started with MARKETS=WETH:USDso shows
that one fly and nothing else. Children are restarted with a growing pause if they die, and SIGTERM is passed to
all of them so the runner can cancel its orders and save the weights.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .settings import PERP_TAPE_PREFIX, Settings, SettingsError

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Job:
    name: str
    argv: list[str]
    cwd: Path
    restart: bool = True
    proc: subprocess.Popen | None = None
    started: float = 0.0
    pause: float = 3.0
    next_start: float = 0.0
    exits: list[int] = field(default_factory=list)


def state_name(market: str) -> str:
    return market.replace(":", "_").replace("/", "_")


def seed_state(settings: Settings, root: Path, state: Path) -> list[str]:
    """Copy the bundled warm-started brains into the state directory for markets that have none yet."""
    seeded = []
    spot = "warm_delta" if settings.brain == "delta" else "warm"       # the brains replayed for each readout
    weights = "w_delta.npy" if settings.brain == "delta" else "w_kc_mbon.npy"
    jobs = [(state_name(m), spot, weights) for m in settings.markets] + ([("EC", "warm", "w_kc_mbon.npy")] if settings.ec else [])
    for name, folder, w in jobs:
        src, dst = root / "config" / folder / name, state / name
        if src.is_dir() and not (dst / w).exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            seeded.append(name)
    return seeded


def build_jobs(settings: Settings, root: Path, state: Path, port: int, python: str = sys.executable) -> list[Job]:
    tape = root / "data" / "tape"
    runner = [python, "-m", "flybot.live.runner", "--state", str(state), "--network", settings.network,
              "--symbols", *settings.markets, "--brain", settings.brain]
    jobs: list[Job] = []
    if settings.perp_markets:
        runner += ["--perp-tape", str(tape), "--perp-prefix", PERP_TAPE_PREFIX,
                   "--perp-symbols", *settings.perp_markets]
        jobs.append(Job("perp-recorder", ["node", "recorder.mjs", "--network", settings.network, "--out", str(tape)],
                        root / "perp"))
    if settings.ec:
        runner += ["--ec-tape", str(tape)]
        jobs.append(Job("ec-recorder", ["node", "ec_recorder.mjs", "--out", str(tape)], root / "perp"))
    jobs.append(Job("runner", runner, root))
    jobs.append(Job("dashboard", [python, str(root / "scripts" / "serve_site.py"), "--state", str(state), "--port", str(port)], root))
    return jobs


class Supervisor:
    def __init__(self, jobs: list[Job], log=print):
        self.jobs, self.log, self.stopping = jobs, log, False

    def _start(self, j: Job) -> None:
        self.log(f"start {j.name}")
        j.proc = subprocess.Popen(j.argv, cwd=j.cwd)
        j.started = time.time()

    def stop(self, *_: object) -> None:
        self.stopping = True

    def run(self, poll_s: float = 1.0) -> int:
        for j in self.jobs:
            self._start(j)
        try:
            while not self.stopping:
                for j in self.jobs:
                    rc = j.proc.poll() if j.proc else None
                    now = time.time()
                    if rc is None:
                        continue
                    if j.next_start == 0.0:  # just noticed it has exited
                        j.exits.append(rc)
                        if not j.restart:
                            self.log(f"{j.name} exited {rc}")
                            self.stopping = True
                            break
                        j.pause = 3.0 if now - j.started > 120 else min(j.pause * 2, 60.0)
                        j.next_start = now + j.pause
                        self.log(f"{j.name} exited {rc}; restarting in {j.pause:.0f} s")
                    elif now >= j.next_start:
                        j.next_start = 0.0
                        self._start(j)
                time.sleep(poll_s)
        finally:
            self._terminate()
        return 0

    def _terminate(self, grace: float = 120.0) -> None:
        """SIGTERM to every child, then wait: the runner closes every live position first, one chain write at a
        time, which with many perp markets takes more than a few seconds."""
        for j in self.jobs:
            if j.proc and j.proc.poll() is None:
                j.proc.send_signal(signal.SIGTERM)
        end = time.time() + grace
        for j in self.jobs:
            if j.proc:
                try:
                    j.proc.wait(timeout=max(0.1, end - time.time()))
                except subprocess.TimeoutExpired:
                    j.proc.kill()


VOLUME = Path("/data")  # where a hosted container (Railway) mounts its persistent volume


def state_dir(env: dict, root: Path = ROOT, volume: Path = VOLUME) -> Path:
    """Where the flies keep what they learn: STATE_DIR when set; else the mounted volume, so a hosted container
    whose variables were pasted over without STATE_DIR still survives a redeploy; else state/live in the repo."""
    if env.get("STATE_DIR"):
        return Path(env["STATE_DIR"])
    if volume.is_dir():
        return volume / "state"
    return root / "state" / "live"


def main() -> None:
    try:
        settings = Settings.from_env()
    except SettingsError as e:
        sys.exit(f"settings: {e}")
    state = state_dir(os.environ)
    state.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("PORT", "8765"))
    seeded = seed_state(settings, ROOT, state)
    print(f"DreamFly: {settings.summary()}", flush=True)
    print(f"markets: {', '.join(settings.markets)}" + (f"; perps: {', '.join(settings.perp_markets)}" if settings.perp_markets else "")
          + ("; event contracts" if settings.ec else ""), flush=True)
    print(f"warm-started brains copied: {', '.join(seeded) or 'none (already present or not bundled)'}", flush=True)
    print(f"dashboard on port {port}", flush=True)
    sup = Supervisor(build_jobs(settings, ROOT, state, port))
    signal.signal(signal.SIGTERM, sup.stop)
    signal.signal(signal.SIGINT, sup.stop)
    sys.exit(sup.run())


if __name__ == "__main__":
    main()
