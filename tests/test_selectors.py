from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flybot import selectors as S
from flybot.arms import ARM_NAMES
from flybot.market import FEATURES
from flybot.mb.graph import load_graph
from flybot.mb.model import MBModel

MB_DIR = Path(__file__).resolve().parents[1] / "data" / "mb"


def feats(**kw) -> pd.Series:
    base = {k: 0.0 for k in FEATURES}
    base.update({"vol_short": 0.01, "vol_long": 0.02, "vol_ratio": 1.0})
    base.update(kw)
    return pd.Series(base)


def test_rule_routes_by_regime():
    r = S.Rule()
    assert ARM_NAMES[r.choose(feats(trend=3.0))] == "trend"
    assert ARM_NAMES[r.choose(feats(vol_ratio=0.5))] == "maker"
    assert ARM_NAMES[r.choose(feats(vol_ratio=1.0))] == "grid"
    assert ARM_NAMES[r.choose(feats(vol_ratio=2.0))] == "flat"
    assert ARM_NAMES[r.choose(feats(trend=np.nan))] == "flat"


def test_bandit_learns_to_prefer_rewarded_arm():
    b = S.Bandit(seed=0)
    f = feats()
    for _ in range(200):
        k = b.choose(f)
        b.update(1.0 if ARM_NAMES[k] == "grid" else -1.0)
    picks = [ARM_NAMES[b.choose(f)] for _ in range(50)]
    assert picks.count("grid") > 40


needs_mb = pytest.mark.skipif(not (MB_DIR / "edges.parquet").exists(), reason="run scripts/extract_mb.py first")


@needs_mb
def test_fly_default_margin_can_be_cleared_by_learning():
    """Pool scores are relative to the pool's baseline activity, so a few rewarded episodes must move a
    score past the default margin; otherwise the fly is stuck on 'flat' forever."""
    fly = S.Fly(MBModel(load_graph(MB_DIR)), seed=0, explore=0.0)
    f = feats(trend=2.0, vol_ratio=1.2)
    for _ in range(30):
        fly.enc.encode(f)
    target = ARM_NAMES.index("trend")
    pn = fly.enc.encode(f)
    kc, _ = fly.m.sniff(pn)
    for _ in range(5):
        fly.last = (target, kc, 0.0)
        fly.update(2.0)
    s = fly.scores(kc)
    assert s[target] - np.delete(s, target).max() > fly.margin


@needs_mb
def test_directed_exploration_tries_every_arm_first():
    fly = S.Fly(MBModel(load_graph(MB_DIR)), seed=0, explore=0.0, directed=2)
    f = feats(trend=0.5, vol_ratio=1.0)
    for _ in range(30):
        fly.enc.encode(f)
    picks = [fly.choose(f) for _ in range(2 * len(ARM_NAMES))]
    for k in range(len(ARM_NAMES)):
        assert picks.count(k) == 2
    fly.update(0.0)


@needs_mb
def test_fly_pools_are_disjoint_and_learns_a_context():
    fly = S.Fly(MBModel(load_graph(MB_DIR)), seed=0, explore=0.0, margin=0.0)
    all_ = np.concatenate(fly.pools)
    assert len(set(all_)) == len(all_) and len(fly.pools) == len(ARM_NAMES)

    f = feats(trend=2.0, vol_ratio=1.2)
    for _ in range(30):
        fly.enc.encode(f)  # settle the running scaler
    target = ARM_NAMES.index("grid")
    for _ in range(40):
        k = fly.choose(f)
        fly.update(1.0 if k == target else -1.0)
    picks = [fly.choose(f) for _ in range(10)]
    assert picks.count(target) >= 8


def test_choose_can_skip_pools_other_episodes_already_run():
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from flybot.market import FEATURES
    from flybot.mb.graph import load_graph
    from flybot.selectors import Fly
    from flybot.mb.model import MBModel

    g = load_graph(Path(__file__).resolve().parents[1] / "data" / "mb")
    fly = Fly(MBModel(g), seed=0, explore=0.0)
    f = pd.Series({k: 0.5 for k in FEATURES})
    free = fly.choose(f)
    for banned in ({free}, {0, 1, 2}, {0, 1, 2, 3}):
        k = fly.choose(f, banned)
        if len(banned) < fly.n:
            assert k not in banned
    fly2 = Fly(MBModel(g), seed=1, explore=1.0)
    picks = {fly2.choose(f, {0, 1}) for _ in range(60)}
    assert picks <= {2, 3} and len(picks) == 2                   # random exploration also stays out of the banned pools
