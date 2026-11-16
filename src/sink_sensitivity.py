#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sink_sensitivity.py — Γ_sink duyarlılık testi
=============================================
Soru: "~2 ps" en kısa yakalanma süresi fiziksel bir tahmin mi, yoksa seçilen
Γ_sink = 1 ps^-1 değerinin bir sonucu mu? Ve ENAQT kazancının konumu
(S'nin maksimum olduğu r) Γ_sink'e ne kadar bağlı?

Γ_sink ∈ {0.5, 1, 2} ps^-1 için (diğer her şey kanonik model):
  - global min <t> ve konumu
  - <t>_min - 1/Γ_sink  (sink'e "bekleme" süresi çıkarıldıktan sonra kalan)
  - S_max ve konumu
HEOM derinlik 11, Nk = 1, sonsuz zaman. Izgara v2'den seyrek:
r: 1–5 nm (21 nokta), γ: 0 + log 0.01–40 ps^-1 (15 nokta; γ > 40 HEOM'da güvenilmez).

Çıktı: results/tables/sink_sensitivity.csv, sink_sensitivity_by_r.csv
Çalıştırma: python src/sink_sensitivity.py   (~5–10 dk)
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool

import enaqt_core as ec

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)

SINKS = [0.5, 1.0, 2.0]
R_NM = np.round(np.linspace(1.0, 5.0, 21), 3)
GAMMAS = np.logspace(-2, np.log10(40.0), 15)   # yalnızca HEOM-güvenilir aralık (γ ≤ 40)
DEPTH, NK = 11, 1


def _block(args):
    sink, r = args
    p = ec.CANONICAL.with_(gamma_sink_psinv=sink)
    e0, t0 = ec.lindblad_metrics(r, 0.0, p)
    rows = [dict(gamma_sink=sink, r_nm=r, gamma_ps_inv=0.0, eta=e0, mean_t_ps=t0)]
    for g in GAMMAS:
        e, t = ec.heom_metrics(r, g, p, depth=DEPTH, Nk=NK)
        rows.append(dict(gamma_sink=sink, r_nm=r, gamma_ps_inv=g, eta=e, mean_t_ps=t))
    return rows


def main():
    tasks = [(s, r) for s in SINKS for r in R_NM]
    with Pool(os.cpu_count() or 1) as pool:
        df = pd.DataFrame([row for b in pool.map(_block, tasks) for row in b])

    per_r, summary = [], []
    for sink, gs in df.groupby("gamma_sink"):
        for r, g in gs.groupby("r_nm"):
            t0 = g.loc[g.gamma_ps_inv == 0, "mean_t_ps"].iloc[0]
            k = g.mean_t_ps.idxmin()
            per_r.append(dict(gamma_sink=sink, r_nm=r, t_gamma0_ps=t0, t_min_ps=g.mean_t_ps[k],
                              gamma_opt_ps_inv=g.gamma_ps_inv[k], S=t0 / g.mean_t_ps[k]))
        pr = pd.DataFrame([x for x in per_r if x["gamma_sink"] == sink])
        k = gs.mean_t_ps.idxmin()
        j = pr.S.idxmax()
        summary.append(dict(gamma_sink_ps_inv=sink, one_over_sink_ps=1 / sink,
                            global_min_t_ps=gs.mean_t_ps[k], at_r_nm=gs.r_nm[k],
                            at_gamma_ps_inv=gs.gamma_ps_inv[k],
                            t_min_minus_1_over_sink_ps=gs.mean_t_ps[k] - 1 / sink,
                            S_max=pr.S[j], S_max_at_r_nm=pr.r_nm[j],
                            S_max_gamma_opt=pr.gamma_opt_ps_inv[j]))
    pd.DataFrame(per_r).to_csv(TAB / "sink_sensitivity_by_r.csv", index=False)
    sm = pd.DataFrame(summary)
    sm.to_csv(TAB / "sink_sensitivity.csv", index=False)
    print(sm.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
