#!/usr/bin/env bash
# Mirror the recorders' tape directory from the box that runs them, every INTERVAL seconds, so a
# local runner can read perps and event contracts. Usage: scripts/sync_tape.sh user@host [interval]
# The order book tape (*_book_*) is large and only used for research, so it stays on the recorder box.
# Set SYNC_SSH to pass options to ssh, for example "ssh -i ~/.ssh/key.pem".
set -euo pipefail
HOST="${1:?user@host}"
INTERVAL="${2:-30}"
SSH="${SYNC_SSH:-ssh}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$HERE/data/tape"
while true; do
  rsync -aq --append -e "$SSH" --exclude='*_book_*' --include='*.jsonl' --exclude='*' "$HOST:~/flybot/data/tape/" "$HERE/data/tape/" || echo "sync failed $(date -u +%FT%TZ)" >&2
  sleep "$INTERVAL"
done
