"""Extract the mushroom-body subcircuit from Male CNS v1.0.

Reads the public flat-connectome Feather files in data/malecns/ and writes:
  data/mb/neurons.parquet   one row per MB neuron (bodyId, group, type, side, nt)
  data/mb/edges.parquet     synapse-count edges where both ends are MB neurons
  data/mb/counts.md         counts table for the plan / research docs

Groups: PN (uniglomerular AL projection neurons), KC, MBON, PAM, PPL1, APL, DPM.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "malecns"
OUT = ROOT / "data" / "mb"

ANN = RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NT = RAW / "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = RAW / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

# Uniglomerular PN: "<glomerulus>_<tract>PN", tract in {ad, l, lv, v}; multiglomerular ones start with "M_".
UNI_PN = re.compile(r"^(?!M_)[A-Za-z0-9+]+_(ad|l|lv|v)PN\d*$")
GOOD_STATUS = {"Reviewed", "Roughly traced", "Prelim Roughly traced", "RT Hard to trace"}


def label_group(row: pd.Series) -> str | None:
    t = row["type"] or ""
    c = row["class"] or ""
    if c == "Kenyon_Cell" or t.startswith("KC"):
        return "KC"
    if c == "MBON" or t.startswith("MBON"):
        return "MBON"
    if c == "DAN" and t.startswith("PAM"):
        return "PAM"
    if c == "DAN" and t.startswith("PPL1"):
        return "PPL1"
    if t == "APL":
        return "APL"
    if t == "DPM":
        return "DPM"
    if c == "ALPN" and UNI_PN.match(t):
        return "PN"
    return None


def load_neurons() -> pd.DataFrame:
    ann = pd.read_feather(ANN, columns=["bodyId", "type", "class", "somaSide", "statusLabel"])
    ann = ann[ann["statusLabel"].astype(str).isin(GOOD_STATUS)].copy()
    ann["type"] = ann["type"].fillna("")
    ann["class"] = ann["class"].fillna("")
    ann["group"] = ann.apply(label_group, axis=1)
    mb = ann.dropna(subset=["group"]).copy()

    nt = pd.read_feather(NT, columns=["body", "consensus_nt", "predicted_nt"])
    nt = nt.drop_duplicates("body").rename(columns={"body": "bodyId"})
    mb = mb.merge(nt, on="bodyId", how="left")
    mb = mb.rename(columns={"somaSide": "side", "consensus_nt": "nt", "predicted_nt": "nt_raw"})
    return mb[["bodyId", "group", "type", "side", "nt", "nt_raw"]].reset_index(drop=True)


def find_edge_columns(cols: list[str]) -> tuple[str, str, str]:
    pre = next(c for c in cols if re.search(r"pre", c, re.I))
    post = next(c for c in cols if re.search(r"post", c, re.I))
    w = next(c for c in cols if re.search(r"weight|count|syn", c, re.I) and c not in (pre, post))
    return pre, post, w


def load_edges(body_ids: set[int]) -> pd.DataFrame:
    import pyarrow.feather as pf

    tbl = pf.read_table(WEIGHTS)
    cols = tbl.column_names
    pre, post, w = find_edge_columns(cols)
    print(f"weights columns: {cols} -> pre={pre} post={post} weight={w}", file=sys.stderr)
    edges = tbl.select([pre, post, w]).to_pandas()
    edges = edges[edges[pre].isin(body_ids) & edges[post].isin(body_ids)]
    return edges.rename(columns={pre: "pre", post: "post", w: "weight"}).reset_index(drop=True)


def counts_table(neurons: pd.DataFrame, edges: pd.DataFrame) -> str:
    order = ["PN", "KC", "MBON", "PAM", "PPL1", "APL", "DPM"]
    by = neurons.groupby(["group", "side"]).size().unstack(fill_value=0).reindex(order)
    lines = ["| group | L | R | total | top nt |", "|---|---|---|---|---|"]
    for g in order:
        sub = neurons[neurons["group"] == g]
        nt = sub["nt"].value_counts(normalize=True)
        top = f"{nt.index[0]} {nt.iloc[0]:.0%}" if len(nt) else "-"
        lines.append(f"| {g} | {by.loc[g].get('L', 0)} | {by.loc[g].get('R', 0)} | {len(sub)} | {top} |")
    lines.append(f"\nMB neurons: {len(neurons)}   MB-internal edges: {len(edges)}   synapses: {int(edges['weight'].sum())}")
    g = neurons.set_index("bodyId")["group"]
    flow = edges.assign(a=edges["pre"].map(g), b=edges["post"].map(g)).groupby(["a", "b"])["weight"].sum()
    lines.append("\nSynapses by group pair (pre -> post), top 15:")
    for (a, b), n in flow.sort_values(ascending=False).head(15).items():
        lines.append(f"  {a:>4} -> {b:<4} {int(n):>9}")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    neurons = load_neurons()
    edges = load_edges(set(neurons["bodyId"]))
    neurons.to_parquet(OUT / "neurons.parquet", index=False)
    edges.to_parquet(OUT / "edges.parquet", index=False)
    table = counts_table(neurons, edges)
    (OUT / "counts.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
