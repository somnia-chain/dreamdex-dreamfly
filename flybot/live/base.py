"""State shared by every live fly: one mushroom body, a reward scaler, equity and day counters,
JSONL logs, and save/load of the learned weights to <state>/<name>/."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..mb.model import MBModel
from ..selectors import DeltaFly, Fly
from ..sim import RewardScaler

BRAINS = ("mb", "delta")        # mb: the depression-only mushroom body readout; delta: per-strategy delta-rule readout


class FlyState:
    def __init__(self, name: str, graph, state_dir: Path, seed: int, arms: tuple, features: tuple, brain: str = "mb"):
        if brain not in BRAINS:
            raise ValueError(f"brain {brain!r} is not one of {', '.join(BRAINS)}")
        self.name, self.brain = name, brain
        self.dir = state_dir / name.replace(":", "_").replace("/", "_")
        self.dir.mkdir(parents=True, exist_ok=True)
        cls = DeltaFly if brain == "delta" else Fly
        self.fly = cls(MBModel(graph, seed=seed), seed=seed, arms=arms, features=features)
        self.scaler = RewardScaler()
        self.equity: float = 0.0
        self.day_pnl: float = 0.0
        self.day: str = ""
        self.n_episodes = 0
        self.last_decision: dict = {}
        self.last_sniff: dict = {}
        self.load()

    @property
    def weights_file(self) -> Path:
        return self.dir / ("w_delta.npy" if self.brain == "delta" else "w_kc_mbon.npy")

    def save(self) -> None:
        np.save(self.weights_file, self.fly.w if self.brain == "delta" else self.fly.m.w_kc_mbon)
        meta = {"scaler": {"n": self.scaler.n, "scale": self.scaler.scale}, "equity": self.equity,
                "day": self.day, "day_pnl": self.day_pnl, "n_episodes": self.n_episodes,
                "encoder_history": {k: v[-5000:] for k, v in self.fly.enc.history.items()}}
        (self.dir / "state.json").write_text(json.dumps(meta))

    def load(self) -> None:
        w = self.weights_file
        if w.exists():
            if self.brain == "delta":
                self.fly.w = np.load(w)
            else:
                self.fly.m.w_kc_mbon = np.load(w)
        s = self.dir / "state.json"
        if s.exists():
            meta = json.loads(s.read_text())
            self.scaler.n, self.scaler.scale = meta["scaler"]["n"], meta["scaler"]["scale"]
            self.equity, self.day, self.day_pnl = meta["equity"], meta["day"], meta["day_pnl"]
            self.n_episodes = meta.get("n_episodes", 0)
            for k, v in meta.get("encoder_history", {}).items():
                if k in self.fly.enc.history:
                    self.fly.enc.history[k] = list(v)

    def log(self, name: str, row: dict) -> None:
        row = {"wall": int(time.time() * 1000), **row}
        with (self.dir / f"{name}.jsonl").open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def book(self, ts: int, pnl: float) -> None:
        """Add an episode's P&L to the day and lifetime counters."""
        day = time.strftime("%Y-%m-%d", time.gmtime(ts / 1000))
        if day != self.day:
            self.day, self.day_pnl = day, 0.0
        self.day_pnl += pnl
        self.equity += pnl
        self.n_episodes += 1

    def sniff(self, feats, record: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """KC pattern and per-pool scores for a feature row (does not count as a choice)."""
        kc, _ = self.fly.m.sniff(self.fly.enc.encode(feats, record=record))
        return kc, self.fly.scores(kc)

    def note_sniff(self, feats, ts: int) -> None:
        """A look at the market between decisions, for the page: which cells the current state lights
        and what the pools currently say. Leaves the encoder's range and the weights untouched."""
        kc, scores = self.sniff(feats, record=False)
        self.last_sniff = {"ts": int(ts), "kc_active": [int(i) for i in np.flatnonzero(kc > 0)],
                           "kc_frac": round(float((kc > 0).mean()), 4), "scores": [round(float(x), 4) for x in scores]}

    def weights_moved(self) -> float:
        if self.brain == "delta":
            return round(float((np.abs(self.fly.w) > 1e-3).mean()), 4)
        m = self.fly.m
        return round(float((np.abs(m.w_kc_mbon - m.w_kc_mbon0) > 0.01 * m.w_kc_mbon0).mean()), 4)

    def brain_snapshot(self) -> dict:
        return {"brain": self.brain, "equity": round(self.equity, 4), "day_pnl": round(self.day_pnl, 4), "episodes": self.n_episodes,
                "decision": self.last_decision, "sniff": self.last_sniff, "arms": list(self.fly.arms),
                "pools": [[int(i) for i in p] for p in self.fly.pools],
                "mbon_valence": [int(v) for v in self.fly.m.approach_sign]}
