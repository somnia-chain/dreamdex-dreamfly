import json
from pathlib import Path

import pytest

from flybot.live.perp_live import PerpFly
from flybot.mb.graph import load_graph
from flybot.perp import PERP_ARMS, PerpCosts
from tests.test_perp import rows

MB_DIR = Path(__file__).resolve().parents[1] / "data" / "mb"
needs_mb = pytest.mark.skipif(not (MB_DIR / "edges.parquet").exists(), reason="run scripts/extract_mb.py first")


def make(tmp_path, params=None):
    return PerpFly("BTC/USDso:USDso", load_graph(MB_DIR), tmp_path, PerpCosts(stake=1000.0), params or {}, seed=0, episode_min=30)


def run(fly, n):
    done = []
    for r in rows(n):
        out = fly.on_row(r)
        if out:
            done.append(out)
    return done


@needs_mb
def test_a_perp_strategy_that_keeps_losing_is_benched_and_not_offered(tmp_path):
    fly = make(tmp_path)
    now = 10**13
    for _ in range(3):
        event = fly.bench.record("p_maker", -5.0, True, now)
    assert event and event["arm"] == "p_maker"
    run(fly, 700)
    for r in (json.loads(l) for l in (fly.dir / "episodes.jsonl").read_text().splitlines()):
        assert r["arm"] != "p_maker"                       # the bench ends in the far future, so it is never picked
    assert json.loads((fly.dir / "bench.json").read_text())["until"]["p_maker"] > now


@needs_mb
def test_the_bench_can_be_switched_off_for_a_perp_market(tmp_path):
    assert make(tmp_path, {"bench": False}).bench is None
    assert "flat" in PERP_ARMS
