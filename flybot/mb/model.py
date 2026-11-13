"""Rate model of the mushroom body with dopamine-gated depression at KC->MBON synapses."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .graph import MBGraph

APPROACH_NT = {"gaba", "acetylcholine"}  # MBONs whose activity drives approach; glutamatergic ones drive avoidance


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


@dataclass
class MBModel:
    g: MBGraph
    target_sparsity: float = 0.07
    apl_iters: int = 50
    eta: float = 0.5
    max_depress: float = 1.0  # cap on one episode's fractional depression of a synapse (1 = no cap)
    bidirectional: bool = False  # reward also restores the punished compartment's synapses, and vice versa
    seed: int = 0
    # filled in __post_init__
    w_pn_kc: np.ndarray = field(init=False)
    w_kc_apl: np.ndarray = field(init=False)
    w_apl_kc: np.ndarray = field(init=False)
    w_kc_mbon0: np.ndarray = field(init=False)
    w_kc_mbon: np.ndarray = field(init=False)
    w_mbon_mbon: np.ndarray = field(init=False)
    apl_gain: float = field(init=False)
    teacher: np.ndarray = field(init=False)  # (n_mbon, 2): weight of PAM / PPL1 teaching for each MBON
    approach_sign: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        g = self.g
        self.w_pn_kc = np.abs(g.weights("PN", "KC"))  # PN drive treated as excitatory (uPNs; vPN sign noted)
        self.w_kc_apl = np.abs(g.weights("KC", "APL"))
        self.w_apl_kc = np.abs(g.weights("APL", "KC"))
        self.w_kc_mbon0 = np.abs(g.weights("KC", "MBON"))  # KCs are cholinergic: all positive
        self.w_kc_mbon = self.w_kc_mbon0.copy()
        self.w_mbon_mbon = g.weights("MBON", "MBON")
        self.teacher = self._derive_teachers()
        self.approach_sign = self._derive_valence()
        self.apl_gain = self._tune_apl_gain()

    # ---------- structure derived from the connectome ----------
    def _derive_teachers(self) -> np.ndarray:
        """For each MBON, how strongly PAM vs PPL1 dopamine teaches it: the direct DAN->MBON synapse
        counts, which in the fly sit inside the MBON's own compartment. Normalised so the two columns sum
        to one per MBON (an MBON with no DAN input is never taught)."""
        out = np.stack(
            [self.g.counts["PAM->MBON"].sum(axis=1), self.g.counts["PPL1->MBON"].sum(axis=1)], axis=1
        ).astype(float)
        tot = out.sum(axis=1, keepdims=True)
        return np.divide(out, tot, out=np.zeros_like(out), where=tot > 0)

    def _derive_valence(self) -> np.ndarray:
        """MBON valence from its dopamine input, opposite in sign (Aso et al. 2014): an MBON taught by
        punishment (PPL1) drives approach, one taught by reward (PAM) drives avoidance. Untaught: 0."""
        pam, ppl1 = self.teacher[:, 0], self.teacher[:, 1]
        return np.where(pam + ppl1 == 0, 0.0, np.where(ppl1 > pam, 1.0, -1.0))

    def valence_by_transmitter(self) -> np.ndarray:
        """Comparison only: +1 for GABA/ACh MBONs, -1 for glutamatergic ones."""
        return np.where(self.g.ids("MBON")["nt"].isin(APPROACH_NT), 1.0, -1.0)

    def _tune_apl_gain(self) -> float:
        rng = np.random.default_rng(self.seed)
        odours = [self.random_odour(rng) for _ in range(30)]
        lo, hi = 0.0, 1.0
        while self._sparsity(odours, hi) > self.target_sparsity:
            hi *= 2
            if hi > 1e6:
                break
        for _ in range(40):
            mid = (lo + hi) / 2
            if self._sparsity(odours, mid) > self.target_sparsity:
                lo = mid
            else:
                hi = mid
        return hi

    def _sparsity(self, odours: list[np.ndarray], gain: float) -> float:
        return float(np.mean([(self._kc(o, gain) > 0).mean() for o in odours]))

    # ---------- dynamics ----------
    def random_odour(self, rng: np.random.Generator, frac: float = 0.15) -> np.ndarray:
        n = self.g.n("PN")
        pn = np.zeros(n)
        active = rng.choice(n, size=max(1, int(round(frac * n))), replace=False)
        pn[active] = rng.uniform(0.5, 1.0, size=active.size)
        return pn

    def _kc(self, pn: np.ndarray, gain: float) -> np.ndarray:
        """KC rates with APL feedback. APL is one neuron, so its rate s is a scalar and the fixed point
        s = w_kc_apl . relu(drive - gain * w_apl_kc * s) is found exactly by bisection (rhs decreases in s)."""
        drive = self.w_pn_kc @ pn
        inhib = self.w_apl_kc[:, 0]

        def kc_of(s: float) -> np.ndarray:
            return relu(drive - gain * inhib * s)

        lo, hi = 0.0, float(self.w_kc_apl[0] @ relu(drive))
        for _ in range(self.apl_iters):
            mid = (lo + hi) / 2
            if float(self.w_kc_apl[0] @ kc_of(mid)) > mid:
                lo = mid
            else:
                hi = mid
        return kc_of(hi)

    def sniff(self, pn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Present an odour. Returns (kc rates, mbon rates)."""
        kc = self._kc(pn, self.apl_gain)
        kc = kc / (kc.max() + 1e-12)
        return kc, self.mbon_of(kc)

    def mbon_of(self, kc: np.ndarray, w: np.ndarray | None = None, couple: bool = True) -> np.ndarray:
        """MBON rates for a KC pattern under the current (or a given) KC->MBON weight matrix.
        couple=False reads each MBON from its KC input alone (no MBON->MBON step)."""
        w = self.w_kc_mbon if w is None else w
        mbon = relu(w @ kc)
        if not couple:
            return mbon
        return relu(mbon + self.w_mbon_mbon @ mbon)  # one step of MBON->MBON interaction

    def approach(self, mbon: np.ndarray) -> float:
        return float(self.approach_sign @ mbon)

    # ---------- learning ----------
    def learn(self, kc: np.ndarray, reward: float = 0.0, punish: float = 0.0,
              mbon_mask: np.ndarray | None = None) -> None:
        """Dopamine-gated depression. reward drives PAM, punish drives PPL1; each depresses the KC->MBON
        synapses of the MBONs it teaches, only for KCs active now. `mbon_mask` restricts the dopamine to
        a subset of compartments (used when only the chosen strategy's compartments are taught)."""
        dan = self.teacher @ np.array([reward, punish])  # (n_mbon,)
        if mbon_mask is not None:
            dan = dan * mbon_mask
        depress = self.eta * np.outer(dan, kc)  # (n_mbon, n_kc)
        # max_depress < 1 bounds one episode's bite so a synapse tracks the run of outcomes, not the last one
        self.w_kc_mbon *= np.clip(1.0 - np.minimum(depress, self.max_depress), 0.0, 1.0)
        if self.bidirectional:
            # the dopamine that depresses one compartment's synapses lets the opposite compartment's recover
            # toward baseline by the same fraction (bidirectional plasticity, Handler et al. 2019), so the
            # weights hold the balance of outcomes rather than "ever rewarded / ever punished"
            other = self.teacher @ np.array([punish, reward])
            if mbon_mask is not None:
                other = other * mbon_mask
            restore = np.clip(np.minimum(self.eta * np.outer(other, kc), self.max_depress), 0.0, 1.0)
            self.w_kc_mbon += restore * (self.w_kc_mbon0 - self.w_kc_mbon)

    def recover(self, rate: float = 0.0) -> None:
        """Slow homeostatic drift of weights back toward their anatomical baseline."""
        if rate > 0:
            self.w_kc_mbon += rate * (self.w_kc_mbon0 - self.w_kc_mbon)

    def reset(self) -> None:
        self.w_kc_mbon = self.w_kc_mbon0.copy()
