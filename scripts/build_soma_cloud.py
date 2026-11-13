"""Export every brain neuron's cell-body position as a dim point cloud for the page (context around the
mushroom body). Coordinates are converted like build_brain_mesh.py and centred on the same point.

Output site/data/somas.json: {"units":"um","n":N,"xyz":[x,y,z,...],"kind":[0|1,...]} with kind 1 = optic
lobe (the two big side lobes), 0 = central brain.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data" / "malecns" / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
BRAIN = ROOT / "site" / "data" / "brain.json"
OUT = ROOT / "site" / "data" / "somas.json"
NM_PER_VOXEL = 8.0
GOOD = {"Reviewed", "Roughly traced", "Prelim Roughly traced", "RT Hard to trace"}


def main() -> None:
    a = pd.read_feather(ANN, columns=["somaLocation", "superclass", "statusLabel"])
    a = a[a["somaLocation"].notna() & a["statusLabel"].astype(str).isin(GOOD)]
    sc = a["superclass"].fillna("")
    # brain only: drop the nerve cord, and sensory / ascending cells whose cell bodies sit outside the brain
    keep = sc.str.startswith(("ol_intrinsic", "cb_intrinsic", "visual_projection", "descending"))
    a = a[keep]
    sc = sc.loc[a.index]
    xyz = np.vstack(a["somaLocation"].to_numpy()).astype(float) * NM_PER_VOXEL / 1000.0
    centre = np.array(json.loads(BRAIN.read_text())["centre"])
    xyz -= centre
    kind = sc.loc[a.index].str.startswith("ol").astype(int).to_numpy()
    OUT.write_text(json.dumps({"units": "um", "n": int(len(xyz)),
                               "xyz": [round(float(v), 1) for v in xyz.ravel()],
                               "kind": [int(k) for k in kind]}, separators=(",", ":")))
    print(f"{len(xyz)} somas ({int(kind.sum())} optic lobe), {OUT.stat().st_size / 1e6:.1f} MB -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
