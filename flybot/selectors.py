"""Strategy selectors: the fly and its baselines. Interface: choose(features) -> arm index; update(reward).

All selectors take an `arms` tuple (names; one must be "flat") and a `features` tuple, defaulting to the
spot arms and spot features so existing callers are unchanged."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .arms import ARM_NAMES
from .market import FEATURES, OdourEncoder
from .mb.model import MBModel

N_ARMS = len(ARM_NAMES)
FLAT = ARM_NAMES.index("flat")


def _vec(f: pd.Series, features: tuple) -> np.ndarray:
    return np.nan_to_num(np.array([float(f[k]) for k in features]))


class Random:
    def __init__(self, seed: int = 0, arms: tuple = ARM_NAMES):
        self.rng = np.random.default_rng(seed)
        self.n = len(arms)

    def choose(self, f: pd.Series) -> int:
        return int(self.rng.integers(self.n))

    def update(self, r: float) -> None:
        pass


class Single:
    def __init__(self, arm: str, arms: tuple = ARM_NAMES):
        self.k = arms.index(arm)

    def choose(self, f: pd.Series) -> int:
        return self.k

    def update(self, r: float) -> None:
        pass


class Rule:
    """What a quant would write for spot: strong trend -> trend; calm -> maker; choppy -> grid; else flat."""

    def __init__(self, trend_thr: float = 1.5, calm_ratio: float = 0.8, chop_ratio: float = 1.3):
        self.trend_thr, self.calm, self.chop = trend_thr, calm_ratio, chop_ratio

    def choose(self, f: pd.Series) -> int:
        if not np.isfinite(f["trend"]):
            return FLAT
        if abs(f["trend"]) > self.trend_thr:
            return ARM_NAMES.index("trend")
        if f["vol_ratio"] < self.calm:
            return ARM_NAMES.index("maker")
        if f["vol_ratio"] < self.chop:
            return ARM_NAMES.index("grid")
        return FLAT

    def update(self, r: float) -> None:
        pass


class Bandit:
    """Contextual Thompson sampling: Bayesian linear regression of reward on features, one per arm."""

    def __init__(self, seed: int = 0, prior: float = 1.0, noise: float = 1.0, arms: tuple = ARM_NAMES,
                 features: tuple = FEATURES):
        self.features = tuple(features)
        self.n = len(arms)
        d = len(self.features) + 1
        self.rng = np.random.default_rng(seed)
        self.A = [np.eye(d) / prior for _ in range(self.n)]
        self.b = [np.zeros(d) for _ in range(self.n)]
        self.noise = noise
        self.last: tuple[int, np.ndarray] | None = None

    def choose(self, f: pd.Series) -> int:
        x = np.append(_vec(f, self.features), 1.0)
        scores = []
        for k in range(self.n):
            cov = np.linalg.inv(self.A[k])
            mu = cov @ self.b[k]
            w = self.rng.multivariate_normal(mu, self.noise * cov)
            scores.append(float(w @ x))
        k = int(np.argmax(scores))
        self.last = (k, x)
        return k

    def update(self, r: float) -> None:
        if self.last is None:
            return
        k, x = self.last
        self.A[k] += np.outer(x, x)
        self.b[k] += r * x


class Fly:
    """The mushroom body chooses. MBONs are split into one pool per arm; the pool with the highest
    approach score wins unless the margin is below `margin`, in which case the fly sits out. The reward
    of the episode teaches only the chosen pool's compartments, through PAM (reward) or PPL1 (punishment)."""

    def __init__(self, model: MBModel, seed: int = 0, margin: float = 0.05, explore: float = 0.1,
                 eta: float = 0.3, recover: float = 0.005, prediction_error: bool = False, pe_gain: float = 1.0,
                 directed: int = 0, condition_features: tuple = ("vol_ratio", "trend", "vol_short", "range"),
                 condition_bins: int = 3, arms: tuple = ARM_NAMES, features: tuple = FEATURES,
                 couple: bool = True, max_depress: float = 1.0, bidirectional: bool = False,
                 split_score: bool = False, balanced_pools: bool = False):
        """directed > 0: before trusting its scores for a market condition, the fly tries every arm
        `directed` times in that condition (conditions = coarse bins of `condition_features`).
        couple=False scores each pool from its own KC input only (no MBON->MBON cross-talk between pools).
        max_depress < 1 caps how much one episode can depress a synapse. bidirectional: see MBModel.learn."""
        self.m = model
        self.m.eta = eta
        self.m.max_depress = max_depress
        self.m.bidirectional = bidirectional
        self.couple = couple
        self.split_score = split_score
        self.recover = recover
        self.rng = np.random.default_rng(seed)
        self.margin, self.explore = margin, explore
        self.prediction_error, self.pe_gain = prediction_error, pe_gain
        self.directed = directed
        self.condition_features, self.condition_bins = condition_features, condition_bins
        self.trials: dict[tuple, np.ndarray] = {}
        self.arms = tuple(arms)
        self.n = len(self.arms)
        self.flat = self.arms.index("flat")
        self.enc = OdourEncoder(model.g.n("PN"), seed=seed, features=features)
        self.pools = self._make_pools_by_drive(seed) if balanced_pools else self._make_pools(seed)
        self.last: tuple[int, np.ndarray, float] | None = None

    def _make_pools_by_drive(self, seed: int) -> list[np.ndarray]:
        """Pools with equal reward headroom and equal punishment headroom: each MBON's typical baseline
        drive (mean over random odours) is measured, then approach cells and avoid cells are dealt
        separately, largest first, each to the pool with the least of that drive so far. Which pool
        gets which cell is still a draw (the seed shuffles ties)."""
        rng = np.random.default_rng(seed)
        drive = np.zeros(self.m.g.n("MBON"))
        for _ in range(24):
            kc, _ = self.m.sniff(rng.random(self.m.g.n("PN")) * 0.3)
            drive += self.m.mbon_of(kc, self.m.w_kc_mbon0, couple=False)
        pools: list[list[int]] = [[] for _ in range(self.n)]
        for sign in (1, -1):
            cells = np.flatnonzero(self.m.approach_sign == sign)
            cells = cells[np.argsort(-drive[cells] + rng.random(cells.size) * 1e-9)]
            load = np.zeros(self.n)
            for c in cells:
                k = int(np.argmin(load + rng.random(self.n) * 1e-9))
                pools[k].append(int(c))
                load[k] += drive[c]
        return [np.array(sorted(p)) for p in pools]

    def _make_pools(self, seed: int) -> list[np.ndarray]:
        """Balanced pools: MBONs sorted by teacher mix, dealt round-robin so every arm gets a similar
        blend of reward-taught (avoid) and punishment-taught (approach) cells."""
        taught = np.flatnonzero(self.m.approach_sign != 0)
        order = taught[np.argsort(self.m.teacher[taught, 0], kind="stable")]
        rng = np.random.default_rng(seed)
        order = np.concatenate([rng.permutation(chunk) for chunk in np.array_split(order, self.n)])
        return [order[k::self.n] for k in range(self.n)]

    def scores(self, kc: np.ndarray) -> np.ndarray:
        """Per-pool approach score relative to what the untrained wiring would say for this odour, as a
        fraction of the pool's untrained activity. Learning only weakens synapses, so without the
        baseline an untried pool could never win; without the scaling the margin has no fixed meaning."""
        mbon = self.m.mbon_of(kc, couple=self.couple)
        mbon0 = self.m.mbon_of(kc, self.m.w_kc_mbon0, couple=self.couple)
        out = []
        for p in self.pools:
            if self.split_score:
                # headroom-normalised: fraction of the pool's avoid drive that learning removed (reward)
                # minus the fraction of its approach drive removed (punishment). Every pool then spans
                # the same range whatever mix of cells it was dealt.
                a = p[self.m.approach_sign[p] > 0]
                v = p[self.m.approach_sign[p] < 0]
                da = float((mbon0[a] - mbon[a]).sum()) / (float(mbon0[a].sum()) + 1e-9) if a.size else 0.0
                dv = float((mbon0[v] - mbon[v]).sum()) / (float(mbon0[v].sum()) + 1e-9) if v.size else 0.0
                out.append(dv - da)
            else:
                scale = float(np.abs(mbon0[p]).sum()) + 1e-9
                out.append(float(self.m.approach_sign[p] @ (mbon[p] - mbon0[p])) / scale)
        return np.array(out)

    def _condition(self, f: pd.Series) -> tuple:
        """Coarse market condition for novelty bookkeeping: equal-frequency bins of a few features,
        using the encoder's running history (no lookahead)."""
        key = []
        b = self.condition_bins
        for name in self.condition_features:
            if name not in self.enc.history:  # feature set without this one (perps, EC): skip it
                continue
            h = self.enc.history[name]
            x = float(f[name])
            if len(h) < 20 or not np.isfinite(x):
                key.append(b // 2)
                continue
            edges = np.quantile(h[-5000:], np.linspace(0, 1, b + 1)[1:-1])
            key.append(int(np.searchsorted(edges, x)))
        return tuple(key)

    def choose(self, f: pd.Series, exclude: frozenset[int] | set[int] | None = None) -> int:
        """exclude: pools not to pick, used when other episodes on the same market already run them, so the
        parallel episodes cover different strategies. Without it the choice is exactly as before."""
        pn = self.enc.encode(f)
        kc, _ = self.m.sniff(pn)
        s = self.scores(kc)
        k = None
        n = None
        if self.directed > 0:
            n = self.trials.setdefault(self._condition(f), np.zeros(self.n, dtype=int))
            untried = np.flatnonzero(n < self.directed)
            if untried.size:
                k = int(untried[np.argmin(n[untried])])
        allowed = [i for i in range(self.n) if not exclude or i not in exclude]
        if k is None and exclude and 0 < len(allowed) < self.n:
            if self.rng.random() < self.explore:
                k = int(self.rng.choice(allowed))
            else:
                order = sorted(allowed, key=lambda i: -s[i])
                if len(order) == 1 or s[order[0]] - s[order[1]] >= self.margin:
                    k = order[0]
                else:
                    k = self.flat if self.flat in allowed else order[0]
        if k is None:
            if self.rng.random() < self.explore:
                k = int(self.rng.integers(self.n))
            else:
                top = np.argsort(s)[::-1]
                k = int(top[0]) if s[top[0]] - s[top[1]] >= self.margin else self.flat
        if n is not None:
            n[k] += 1
        self.last = (k, kc, float(s[k]))
        return k

    def update(self, r: float) -> None:
        """The episode's reward reaches the chosen pool's dopamine neurons. With prediction error on, the
        dopamine signal is reward minus the pool's own current valuation of this odour (the MBON->DAN
        feedback of Eschbach et al. 2020), so an already-learned choice stops being re-taught."""
        if self.last is None:
            return
        k, kc, expected = self.last
        if self.prediction_error:
            r = r - self.pe_gain * expected
        mask = np.zeros(self.m.g.n("MBON"))
        mask[self.pools[k]] = 1.0
        self.m.learn(kc, reward=max(r, 0.0), punish=max(-r, 0.0), mbon_mask=mask)
        self.m.recover(self.recover)


class DeltaFly:
    """The same fly senses (connectome PN -> KC code, same odour encoder) but each strategy learns what the code is
    worth with a delta rule: dopamine carries the error, reward minus what the strategy already expected for this
    code, and the strategy's output weights move toward it. Nothing is only ever weakened, so a strategy can be
    learned good in one market state and bad in another. Flat is worth zero by definition.

    Interface and `last` match Fly, so counterfactual pretraining and the runner can use either."""

    def __init__(self, model: MBModel, seed: int = 0, margin: float = 0.05, explore: float = 0.1, lr: float = 0.05,
                 arms: tuple = ARM_NAMES, features: tuple = FEATURES):
        self.m = model
        self.margin, self.explore, self.lr = margin, explore, lr
        self.rng = np.random.default_rng(seed)
        self.arms = tuple(arms)
        self.n = len(self.arms)
        self.flat = self.arms.index("flat")
        self.enc = OdourEncoder(model.g.n("PN"), seed=seed, features=features)
        self.w = np.zeros((self.n, model.g.n("KC")))
        self.pools: list[np.ndarray] = [np.array([], dtype=int) for _ in self.arms]   # no MBON pools in this readout
        self.last: tuple[int, np.ndarray, float] | None = None

    @staticmethod
    def _code(kc: np.ndarray) -> np.ndarray:
        x = (kc > 0).astype(float)
        return x / (np.sqrt(x.sum()) + 1e-9)          # unit length, so one step size suits every odour

    def scores(self, kc: np.ndarray) -> np.ndarray:
        s = self.w @ self._code(kc)
        s[self.flat] = 0.0
        return s

    def choose(self, f: pd.Series, exclude: frozenset[int] | set[int] | None = None) -> int:
        kc, _ = self.m.sniff(self.enc.encode(f))
        s = self.scores(kc)
        allowed = [i for i in range(self.n) if not exclude or i not in exclude]
        if not allowed:
            allowed = [self.flat]
        if self.rng.random() < self.explore:
            k = int(self.rng.choice(allowed))
        else:
            order = sorted(allowed, key=lambda i: -s[i])
            if order[0] == self.flat or len(order) == 1:
                k = order[0]
            else:
                runner_up = max(s[order[1]], 0.0) if self.flat in allowed else s[order[1]]
                k = order[0] if s[order[0]] - runner_up >= self.margin else (self.flat if self.flat in allowed else order[0])
        self.last = (k, kc, float(s[k]))
        return k

    def update(self, r: float) -> None:
        if self.last is None:
            return
        k, kc, _ = self.last
        if k == self.flat:
            return
        x = self._code(kc)
        self.w[k] += self.lr * (r - self.w[k] @ x) * x
