"""Load the extracted mushroom body and build signed, normalised weight matrices for one hemisphere."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = ("PN", "KC", "MBON", "PAM", "PPL1", "APL", "DPM")

# Sign of a presynaptic neuron's output by predicted transmitter. Glutamate is inhibitory at fly MBONs.
# Dopamine is modulatory: it gates plasticity and is not treated as a synaptic drive.
NT_SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "dopamine": 0.0}


@dataclass
class MBGraph:
    neurons: pd.DataFrame  # one side, columns bodyId, group, type, nt, idx (position within its group)
    counts: dict[str, np.ndarray]  # raw synapse-count matrices, key "PRE->POST", shape (n_post, n_pre)

    def ids(self, group: str) -> pd.DataFrame:
        return self.neurons[self.neurons["group"] == group].sort_values("idx")

    def n(self, group: str) -> int:
        return int((self.neurons["group"] == group).sum())

    def weights(self, pre: str, post: str, normalise: bool = True) -> np.ndarray:
        """Signed matrix (n_post, n_pre). Rows normalised so each target's total input has unit L1 norm."""
        w = self.counts[f"{pre}->{post}"].astype(float)
        sign = self.ids(pre)["nt"].map(NT_SIGN).fillna(1.0).to_numpy()
        w = w * sign[None, :]
        if normalise:
            tot = np.abs(w).sum(axis=1, keepdims=True)
            w = np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)
        return w


def load_graph(mb_dir: Path | str, side: str = "R") -> MBGraph:
    mb_dir = Path(mb_dir)
    neurons = pd.read_parquet(mb_dir / "neurons.parquet")
    edges = pd.read_parquet(mb_dir / "edges.parquet")

    neurons = neurons[neurons["side"] == side].copy()
    neurons = neurons.sort_values(["group", "type", "bodyId"]).reset_index(drop=True)
    neurons["idx"] = neurons.groupby("group").cumcount()

    body_group = neurons.set_index("bodyId")["group"]
    body_idx = neurons.set_index("bodyId")["idx"]
    edges = edges[edges["pre"].isin(body_group.index) & edges["post"].isin(body_group.index)].copy()
    edges["gpre"] = edges["pre"].map(body_group)
    edges["gpost"] = edges["post"].map(body_group)
    edges["ipre"] = edges["pre"].map(body_idx)
    edges["ipost"] = edges["post"].map(body_idx)

    sizes = {g: int((neurons["group"] == g).sum()) for g in GROUPS}
    counts: dict[str, np.ndarray] = {}
    for (gpre, gpost), sub in edges.groupby(["gpre", "gpost"]):
        m = np.zeros((sizes[gpost], sizes[gpre]))
        np.add.at(m, (sub["ipost"].to_numpy(), sub["ipre"].to_numpy()), sub["weight"].to_numpy())
        counts[f"{gpre}->{gpost}"] = m
    for a in GROUPS:
        for b in GROUPS:
            counts.setdefault(f"{a}->{b}", np.zeros((sizes[b], sizes[a])))
    return MBGraph(neurons=neurons, counts=counts)
