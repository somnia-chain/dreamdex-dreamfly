"""Perp flies on one shared brain: each market's result trains its own choice, teach_all teaches the strategies that
did not run, and a market on a shared brain keeps only its own books on disk."""

import json
from pathlib import Path

import numpy as np
import pytest

from flybot.live.base import FlyState
from flybot.live.perp_live import PerpFly
from flybot.mb.graph import load_graph
from flybot.perp import PERP_ARMS, PERP_FEATURES, PerpCosts
from tests.test_perp import rows

MB_DIR = Path(__file__).resolve().parents[1] / "data" / "mb"
pytestmark = pytest.mark.skipif(not (MB_DIR / "edges.parquet").exists(), reason="run scripts/extract_mb.py first")


def two_markets(tmp_path, params):
    g = load_graph(MB_DIR)
    core = FlyState("_brain", g, tmp_path, 0, PERP_ARMS, PERP_FEATURES, brain="delta")
    a, b = (PerpFly(sym, g, tmp_path, PerpCosts(stake=1000.0), params, seed=i, episode_min=30, brain="delta", shared=core)
            for i, sym in enumerate(("hl:BTC", "lighter:BTC")))
    return core, a, b


def feed(fly, rs):
    out = []
    for r in rs:
        r = {**r, "symbol": fly.symbol}
        done = fly.on_row(r)
        if done:
            out.append(done)
    return out


def test_markets_share_one_brain_and_each_result_trains_its_own_choice(tmp_path, monkeypatch):
    core, a, b = two_markets(tmp_path, {"bench": False})
    assert a.fly is b.fly is core.fly and a.scaler is core.scaler
    ra, rb = rows(120, seed=1), rows(120, seed=2)
    feed(a, ra[:61])
    feed(b, rb[:61])
    assert a.choice is not None and b.choice is not None
    taught = []
    monkeypatch.setattr(core.fly, "update", lambda r: taught.append(core.fly.last))
    want = a.choice
    feed(a, ra[61:95])                                         # a's round ends while b's is still running
    assert taught and taught[0] is want                        # trained a's choice, not b's (the brain's last)


def test_teach_all_teaches_every_strategy_that_did_not_run(tmp_path):
    core, a, _ = two_markets(tmp_path, {"bench": False, "teach_all": True})
    core.scaler.n, core.scaler.scale = core.scaler.warm, 1e-4     # past the warm-up, where rewards are still zero
    done = feed(a, rows(200, seed=3))
    assert done
    for row in done:
        assert set(row["taught"]) == set(PERP_ARMS) - {row["arm"], "flat"}
    assert np.abs(core.fly.w).sum() > 0                        # the shared weights moved


def test_a_market_on_a_shared_brain_saves_only_its_books(tmp_path):
    core, a, _ = two_markets(tmp_path, {"bench": False})
    feed(a, rows(200, seed=4))
    meta = json.loads((a.dir / "state.json").read_text())
    assert meta["n_episodes"] == a.n_episodes > 0 and "encoder_history" not in meta
    assert not a.weights_file.exists()
    core.save()
    assert core.weights_file.exists()
