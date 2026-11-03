#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lindblad_vs_heom_v2.py — v1'deki Lindblad–HEOM karşılaştırmasının (r = 2 nm)
düzeltilmiş tekrarı.

Farklar: sonsuz-zaman metrikler, HEOM derinlik 11, ve Lindblad eğrisi iki
şekilde çizilir: (i) aynı γ (v1 gibi), (ii) 2γ (HEOM'un Markov limitinde
etkin koherans sönüm hızıyla eşleşecek şekilde; bkz.
lindblad_heom_mapping_check.py). Böylece "optimum kayması"nın ne kadarının
ölçek farkından, ne kadarının banyo hafızasından geldiği ayrılır.

Girdi : results/data/heom_2d_scan_v2.csv (heom_2d_scan_v2.py çıktısı)
Çıktı : results/figures/lindblad_vs_heom_v2.png
        results/tables/lindblad_vs_heom_v2.txt
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import enaqt_core as ec

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"; TAB = ROOT / "results" / "tables"
R_FIXED = 2.0


def main():
    df = pd.read_csv(ROOT / "results" / "data" / "heom_2d_scan_v2.csv")
    sel = df[(np.isclose(df.r_nm, R_FIXED)) & (df.gamma_ps_inv > 0)]
    h = sel[(sel.model == "HEOM") & sel.heom_reliable].sort_values("gamma_ps_inv")   # γ ≤ 40 ps^-1
    l = sel[(sel.model == "Lindblad") & (sel.gamma_ps_inv <= h.gamma_ps_inv.max())].sort_values("gamma_ps_inv")
    g = h.gamma_ps_inv.values
    l2 = np.array([ec.lindblad_metrics(R_FIXED, 2 * x)[1] for x in g])   # Lindblad @ 2γ

    def opt(gs, ts):
        k = int(np.argmin(ts)); return gs[k], ts[k]
    gh, th = opt(g, h.mean_t_ps.values)
    gl, tl = opt(l.gamma_ps_inv.values, l.mean_t_ps.values)
    gl2, tl2 = opt(g, l2)
    txt = "\n".join([
        f"Lindblad vs HEOM (r = {R_FIXED} nm, sonsuz zaman, HEOM derinlik 11):",
        f"  HEOM            : γ* = {gh:.3g} ps^-1, <t>min = {th:.3f} ps",
        f"  Lindblad (γ)    : γ* = {gl:.3g} ps^-1, <t>min = {tl:.3f} ps",
        f"  Lindblad (2γ)   : γ* = {gl2:.3g} ps^-1 (HEOM ekseninde), <t>min = {tl2:.3f} ps",
        f"  γ = {g[-1]:.3g} ps^-1'de (güvenilir HEOM aralığının sonu) <t>: HEOM {h.mean_t_ps.values[-1]:.2f} ps, "
        f"Lindblad(2γ) {l2[-1]:.2f} ps",
        "  (γ > 40 ps^-1 HEOM noktaları sayısal olarak güvenilmez olduğundan karşılaştırmaya alınmadı)",
    ])
    (TAB / "lindblad_vs_heom_v2.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)

    fig, ax = plt.subplots(figsize=(6.4, 4), dpi=200)
    ax.plot(g, h.mean_t_ps, "-o", color="#2a78d6", ms=3, lw=2, label="HEOM (γ_c = 2 rad/ps)")
    ax.plot(l.gamma_ps_inv, l.mean_t_ps, "--s", color="#eb6834", ms=3, lw=2, label="Lindblad, dephasing γ (v1 eşleştirmesi)")
    ax.plot(g, l2, ":^", color="#1baf7a", ms=3, lw=2, label="Lindblad, dephasing 2γ (Markov-eşdeğer)")
    ax.set_xscale("log"); ax.set_xlabel(r"$\gamma$ (ps$^{-1}$)")
    ax.set_ylabel(r"$\langle t\rangle$ (ps), sonsuz zaman")
    ax.set_title(f"Lindblad vs HEOM, r = {R_FIXED} nm (v2)", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "lindblad_vs_heom_v2.png"); plt.close(fig)


if __name__ == "__main__":
    main()
