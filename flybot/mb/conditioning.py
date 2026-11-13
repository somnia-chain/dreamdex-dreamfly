"""Classic olfactory conditioning protocols on the MB model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import MBModel


@dataclass
class Trial:
    step: int
    phase: str
    approach_a: float
    approach_b: float


def preference(model: MBModel, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    return model.approach(model.sniff(a)[1]), model.approach(model.sniff(b)[1])


def run_protocol(
    model: MBModel,
    a: np.ndarray,
    b: np.ndarray,
    phases: list[tuple[str, int, float, float, float, float]],
    recover_rate: float = 0.0,
) -> list[Trial]:
    """phases: (name, n_trials, reward_a, punish_a, reward_b, punish_b). Each trial presents A then B."""
    log: list[Trial] = []
    step = 0
    pa, pb = preference(model, a, b)
    log.append(Trial(step, "baseline", pa, pb))
    for name, n, ra, pa_, rb, pb_ in phases:
        for _ in range(n):
            kc_a, _ = model.sniff(a)
            model.learn(kc_a, reward=ra, punish=pa_)
            kc_b, _ = model.sniff(b)
            model.learn(kc_b, reward=rb, punish=pb_)
            model.recover(recover_rate)
            step += 1
            pa, pb = preference(model, a, b)
            log.append(Trial(step, name, pa, pb))
    return log


def textbook(model: MBModel, rng: np.random.Generator, n: int = 10) -> dict[str, list[Trial]]:
    a, b = model.random_odour(rng), model.random_odour(rng)
    out: dict[str, list[Trial]] = {}

    model.reset()
    out["appetitive_A"] = run_protocol(model, a, b, [("A+sugar", n, 1.0, 0.0, 0.0, 0.0)])

    model.reset()
    out["aversive_B"] = run_protocol(model, a, b, [("B+shock", n, 0.0, 0.0, 0.0, 1.0)])

    model.reset()
    out["reversal"] = run_protocol(
        model, a, b,
        [("A+sugar", n, 1.0, 0.0, 0.0, 0.0), ("B+sugar,A+shock", 2 * n, 0.0, 1.0, 1.0, 0.0)],
    )

    model.reset()
    out["extinction"] = run_protocol(
        model, a, b,
        [("A+sugar", n, 1.0, 0.0, 0.0, 0.0), ("A alone", 3 * n, 0.0, 0.0, 0.0, 0.0)],
        recover_rate=0.1,
    )
    return out
