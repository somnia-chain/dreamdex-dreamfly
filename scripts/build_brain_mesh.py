"""Convert the MB neuron SWC skeletons into one compact JSON for the 3D page.

Output site/data/brain.json:
  {"units": "um", "centre": [x,y,z], "groups": ["PN","KC",...],
   "neurons": [{"id": bodyId, "g": group_index, "i": index_within_group, "seg": [[x,y,z,...], ...]}, ...]}
Each neuron is a list of polylines (root-to-tip paths), downsampled so KCs stay light.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SWC = ROOT / "data" / "skeletons"
OUT = ROOT / "site" / "data" / "brain.json"
NM_PER_VOXEL = 8.0
MAX_PTS = {"KC": 40, "PN": 120, "MBON": 300, "PAM": 150, "PPL1": 300, "APL": 600, "DPM": 400}


def read_swc(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [ln.split() for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    a = np.array([[float(r[0]), float(r[2]), float(r[3]), float(r[4]), float(r[6])] for r in rows])
    return a[:, 1:4] * NM_PER_VOXEL / 1000.0, a[:, [0, 4]].astype(int)


def paths(xyz: np.ndarray, idp: np.ndarray, budget: int) -> list[list[float]]:
    """Split the tree into root-to-tip polylines, keep the longest ones within the point budget."""
    id2row = {int(i): k for k, i in enumerate(idp[:, 0])}
    children: dict[int, list[int]] = {}
    for k, (i, p) in enumerate(idp):
        if p >= 0 and int(p) in id2row:
            children.setdefault(id2row[int(p)], []).append(k)
    tips = [k for k in range(len(idp)) if k not in children]
    parent = {k: (id2row.get(int(p), -1) if p >= 0 else -1) for k, (i, p) in enumerate(idp)}
    used = set()
    out = []
    for t in sorted(tips, key=lambda k: -np.linalg.norm(xyz[k] - xyz[0])):
        seg = []
        k = t
        while k != -1 and k not in used:
            seg.append(k)
            used.add(k)
            k = parent[k]
        if k != -1:
            seg.append(k)  # join to the already-used node
        if len(seg) >= 2:
            out.append(seg[::-1])
    out.sort(key=len, reverse=True)
    result, total = [], 0
    for seg in out:
        n = max(2, min(len(seg), budget // 4))
        idx = np.linspace(0, len(seg) - 1, n).round().astype(int)
        pts = xyz[[seg[i] for i in idx]]
        result.append([round(float(v), 2) for v in pts.ravel()])
        total += n
        if total >= budget:
            break
    return result


def main() -> None:
    neurons = pd.read_parquet(ROOT / "data" / "mb" / "neurons.parquet")
    neurons = neurons[neurons["side"] == "R"].sort_values(["group", "type", "bodyId"]).reset_index(drop=True)
    neurons["idx"] = neurons.groupby("group").cumcount()
    groups = ["PN", "KC", "MBON", "PAM", "PPL1", "APL", "DPM"]
    out, allpts, missing = [], [], 0
    for _, r in neurons.iterrows():
        p = SWC / f"{int(r['bodyId'])}.swc"
        if not p.exists():
            missing += 1
            continue
        xyz, idp = read_swc(p)
        segs = paths(xyz, idp, MAX_PTS[r["group"]])
        out.append({"id": int(r["bodyId"]), "g": groups.index(r["group"]), "i": int(r["idx"]), "seg": segs})
        allpts.append(xyz[:: max(1, len(xyz) // 50)])
    centre = np.vstack(allpts).mean(axis=0)
    for n in out:
        n["seg"] = [[round(v - centre[k % 3], 2) for k, v in enumerate(s)] for s in n["seg"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"units": "um", "centre": [round(float(c), 2) for c in centre], "groups": groups,
                               "neurons": out}, separators=(",", ":")))
    npts = sum(len(s) // 3 for n in out for s in n["seg"])
    print(f"{len(out)} neurons ({missing} missing), {npts} points, {OUT.stat().st_size / 1e6:.1f} MB -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
