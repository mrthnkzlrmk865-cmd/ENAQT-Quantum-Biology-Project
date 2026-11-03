#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_lindblad_v1.py — v1 Lindblad haritası (results/data/eta_scan_data.npz,
enaqt_parameter_scan.py, T_MAX = 30 ps) ile sonsuz-zaman Lindblad değerlerinin
karşılaştırması. Amaç: v1 'enaqt_dual_heatmap.png' de kesmeden etkileniyor mu?

Çıktı: results/tables/validation_lindblad_v1.csv, validation_lindblad_v1.txt
"""
from pathlib import Path
import numpy as np
import pandas as pd
import enaqt_core as ec

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)


def main():
    d = np.load(ROOT / "results" / "data" / "eta_scan_data.npz")
    g, r, E, T = d["gamma_range"], d["r_range"], d["eta_grid"], d["mean_time_grid"]
    rows = []
    for i, rr in enumerate(r):
        for j, gg in enumerate(g):
            e, t = ec.lindblad_metrics(rr, gg)
            rows.append(dict(r_nm=rr, gamma_ps_inv=gg, eta_v1=E[i, j], t_v1=T[i, j], eta_inf=e, t_inf=t))
    df = pd.DataFrame(rows)
    df["rel_t"] = (df.t_v1 - df.t_inf).abs() / df.t_inf
    df.to_csv(TAB / "validation_lindblad_v1.csv", index=False)
    per_r = df.groupby("r_nm").rel_t.max()
    first_bad = per_r[per_r > 0.01].index.min()
    near = df[df.r_nm <= 1.2]
    txt = "\n".join([
        "v1 Lindblad haritası (T_MAX=30 ps) vs sonsuz-zaman Lindblad:",
        f"  r ≤ 1.2 nm'de maks |Δ<t>|/<t> = %{100*near.rel_t.max():.2f} (uygulamalar uyumlu)",
        f"  |Δ<t>|/<t> > %1 olan ilk r = {first_bad:.2f} nm",
        f"  r = {r[-1]:.1f} nm: η v1 {df[df.r_nm == r[-1]].eta_v1.min():.3f}–{df[df.r_nm == r[-1]].eta_v1.max():.3f}, "
        f"sonsuz {df[df.r_nm == r[-1]].eta_inf.min():.3f}–{df[df.r_nm == r[-1]].eta_inf.max():.3f}; "
        f"<t> v1 {df[df.r_nm == r[-1]].t_v1.min():.1f}–{df[df.r_nm == r[-1]].t_v1.max():.1f} ps, "
        f"sonsuz {df[df.r_nm == r[-1]].t_inf.min():.1f}–{df[df.r_nm == r[-1]].t_inf.max():.1f} ps",
    ])
    (TAB / "validation_lindblad_v1.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
