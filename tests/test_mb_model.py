from pathlib import Path

import numpy as np
import pytest

from flybot.mb.conditioning import preference, run_protocol
from flybot.mb.graph import NT_SIGN, load_graph
from flybot.mb.model import MBModel

MB_DIR = Path(__file__).resolve().parents[1] / "data" / "mb"
pytestmark = pytest.mark.skipif(not (MB_DIR / "edges.parquet").exists(), reason="run scripts/extract_mb.py first")


@pytest.fixture(scope="module")
def model() -> MBModel:
    return MBModel(load_graph(MB_DIR))


@pytest.fixture
def odours(model: MBModel):
    rng = np.random.default_rng(7)
    return model.random_odour(rng), model.random_odour(rng)


def test_signs_follow_transmitter(model: MBModel):
    w = model.g.weights("MBON", "MBON")
    nt = model.g.ids("MBON")["nt"].to_numpy()
    for j, t in enumerate(nt):
        col = w[:, j]
        if NT_SIGN[t] < 0:
            assert (col <= 0).all()
        else:
            assert (col >= 0).all()


def test_kc_code_is_sparse(model: MBModel, odours):
    for o in odours:
        kc, _ = model.sniff(o)
        frac = (kc > 0).mean()
        assert 0.02 < frac < 0.15


def test_dopamine_pairing_matches_anatomy(model: MBModel):
    """Reward DANs (PAM) should teach the glutamatergic gamma5/beta'2 MBONs (MBON01-03),
    punishment DANs (PPL1) the GABAergic gamma1pedc MBON (MBON11) and the alpha-lobe MBON14."""
    types = model.g.ids("MBON")["type"].to_numpy()
    pam, ppl1 = model.teacher[:, 0], model.teacher[:, 1]
    for t in ("MBON01", "MBON02", "MBON03"):
        assert (pam[types == t] > 0.9).all()
    for t in ("MBON11", "MBON14"):
        assert (ppl1[types == t] > 0.8).all()


def test_appetitive_and_aversive_learning(model: MBModel, odours):
    a, b = odours
    model.reset()
    base = np.subtract(*preference(model, a, b))
    log = run_protocol(model, a, b, [("A+sugar", 10, 1.0, 0.0, 0.0, 0.0)])
    assert log[-1].approach_a - log[-1].approach_b > base + 0.1

    model.reset()
    log = run_protocol(model, a, b, [("B+shock", 10, 0.0, 0.0, 0.0, 1.0)])
    assert log[-1].approach_a - log[-1].approach_b > base + 0.1
    assert log[-1].approach_b < log[0].approach_b


def test_reversal_and_extinction(model: MBModel, odours):
    a, b = odours
    model.reset()
    log = run_protocol(
        model, a, b, [("A+sugar", 10, 1.0, 0.0, 0.0, 0.0), ("reverse", 20, 0.0, 1.0, 1.0, 0.0)]
    )
    after_first = log[10].approach_a - log[10].approach_b
    after_reverse = log[-1].approach_a - log[-1].approach_b
    assert after_reverse < after_first - 0.2

    model.reset()
    base = np.subtract(*preference(model, a, b))
    log = run_protocol(
        model, a, b, [("A+sugar", 10, 1.0, 0.0, 0.0, 0.0), ("alone", 30, 0.0, 0.0, 0.0, 0.0)], recover_rate=0.1
    )
    learned = log[10].approach_a - log[10].approach_b
    final = log[-1].approach_a - log[-1].approach_b
    assert abs(final - base) < 0.3 * abs(learned - base)
    model.reset()
