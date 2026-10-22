"""Phase 1: run classic conditioning on the connectome MB model and print the learning curves."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybot.mb.conditioning import textbook  # noqa: E402
from flybot.mb.graph import load_graph  # noqa: E402
from flybot.mb.model import MBModel  # noqa: E402


def main() -> None:
    out = ROOT / "data" / "conditioning"
    out.mkdir(parents=True, exist_ok=True)
    g = load_graph(ROOT / "data" / "mb")
    m = MBModel(g)
    rng = np.random.default_rng(1)

    kc, _ = m.sniff(m.random_odour(rng))
    print(f"neurons: { {k: g.n(k) for k in ('PN', 'KC', 'MBON', 'PAM', 'PPL1', 'APL')} }")
    print(f"APL gain {m.apl_gain:.3g}; KC active fraction on a random odour: {(kc > 0).mean():.1%}")

    mb = g.ids("MBON")[["type", "nt"]].copy()
    mb["PAM"] = m.teacher[:, 0].round(2)
    mb["PPL1"] = m.teacher[:, 1].round(2)
    lab = {1.0: "approach", -1.0: "avoid", 0.0: "-"}
    mb["valence"] = [lab[v] for v in m.approach_sign]
    mb["by_nt"] = [lab[v] for v in m.valence_by_transmitter()]
    mb["agree"] = np.where(mb["valence"] == mb["by_nt"], "", "differs")
    print("\nMBON teachers (direct DAN->MBON synapses) and valence (opposite to teacher, Aso 2014);")
    print("`by_nt` is the transmitter rule (GABA/ACh approach, Glu avoid) for comparison:")
    print(mb.to_string(index=False))
    n_taught = int((mb["valence"] != "-").sum())
    n_agree = int(((mb["valence"] == mb["by_nt"]) & (mb["valence"] != "-")).sum())
    print(f"\ntaught MBONs: {n_taught}/{len(mb)}; valence agrees with transmitter rule for {n_agree}")
    mb.to_csv(out / "mbon_teachers.csv", index=False)

    for name, log in textbook(m, rng).items():
        df = pd.DataFrame([t.__dict__ for t in log])
        df["pref_A_minus_B"] = df["approach_a"] - df["approach_b"]
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"\n== {name} ==")
        last = df.groupby("phase", sort=False).tail(1)
        print(last[["step", "phase", "approach_a", "approach_b", "pref_A_minus_B"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
