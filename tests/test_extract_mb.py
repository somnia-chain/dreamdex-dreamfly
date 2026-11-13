import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import extract_mb as m  # noqa: E402


def row(type_: str, class_: str = "") -> pd.Series:
    return pd.Series({"type": type_, "class": class_})


def test_label_group_core_types():
    assert m.label_group(row("KCg-m", "Kenyon_Cell")) == "KC"
    assert m.label_group(row("MBON01", "MBON")) == "MBON"
    assert m.label_group(row("PAM04", "DAN")) == "PAM"
    assert m.label_group(row("PPL101", "DAN")) == "PPL1"
    assert m.label_group(row("APL")) == "APL"
    assert m.label_group(row("DPM")) == "DPM"


def test_uniglomerular_pn_pattern():
    assert m.label_group(row("DA1_lPN", "ALPN")) == "PN"
    assert m.label_group(row("VP1m+VP2_lvPN2", "ALPN")) == "PN"
    assert m.label_group(row("M_adPNm5", "ALPN")) is None  # multiglomerular
    assert m.label_group(row("VP3+_l2PN", "ALPN")) is None  # not a listed tract
    assert m.label_group(row("DA1_lPN", "ALLN")) is None  # wrong class


def test_unrelated_neuron_is_dropped():
    assert m.label_group(row("DNa02", "descending")) is None


def test_find_edge_columns():
    assert m.find_edge_columns(["body_pre", "body_post", "weight"]) == ("body_pre", "body_post", "weight")
