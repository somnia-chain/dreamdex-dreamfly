"""The `dreamfly` command: one entry point for running the fly, the dashboard and the checks.

  dreamfly start    everything in one supervised process, for containers
  dreamfly run      the runner: flies, order mirror, live state for the dashboard
  dreamfly site     the dashboard server
  dreamfly doctor   read-only setup check
  dreamfly warm     warm-start the flies from recorded history
  dreamfly extract  build the mushroom body subgraph from the Male CNS files

Everything after the subcommand is passed on, so `dreamfly run --help` shows the runner's flags. The scripts
live in the checkout, so install with `pip install -e .` from a clone.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {"site": "serve_site.py", "doctor": "doctor.py", "warm": "warm_start.py", "extract": "extract_mb.py"}
USAGE = "usage: dreamfly {start,run,site,doctor,warm,extract} [options]   (add --help after a command for its options)"


def main(argv: list[str] | None = None) -> None:
    load_dotenv()  # a .env in the working directory fills in settings; real environment variables win
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__ + "\n" + USAGE)
        return
    cmd, rest = args[0], args[1:]
    if cmd == "start":
        from .live.supervisor import main as start_main
        start_main()
        return
    if cmd == "run":
        from .live.runner import main as run_main
        sys.argv = ["dreamfly run", *rest]
        run_main()
        return
    if cmd not in SCRIPTS:
        sys.exit(f"unknown command {cmd!r}\n{USAGE}")
    script = ROOT / "scripts" / SCRIPTS[cmd]
    if not script.exists():
        sys.exit(f"{script} not found: install from a clone with pip install -e .")
    sys.argv = [f"dreamfly {cmd}", *rest]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
