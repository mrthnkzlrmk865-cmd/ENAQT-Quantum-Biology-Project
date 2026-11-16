#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_against_legacy_csv.py
==============================
Amaç: (1) enaqt_core.py'deki numpy/scipy HEOM uygulamasının, orijinal QuTiP
5.3 taramasını (results/data/heom_2d_scan_results.csv) yeniden üretip
üretmediğini 400 noktanın hepsinde kontrol etmek; (2) orijinal T_MAX = 25 ps
kesmesinin η ve <t>'yi ne kadar bozduğunu, aynı noktalarda sonsuz-zaman
(analitik) değerlerle karşılaştırarak göstermek.

Çıktı: results/tables/validation_vs_legacy_csv.csv
       results/tables/validation_summary.txt
Çalıştırma: python src/validate_against_legacy_csv.py   (~1 dk)
"""
from pathlib import Path
import numpy as np
import pandas as pd

import enaqt_core as ec

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "results" / "data" / "heom_2d_scan_results.csv"
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

T_MAX_PS, N_STEPS = 25.0, 220      # orijinal ayarlar (heom_2d_robustness_scan.py)


def main():
    df = pd.read_csv(LEGACY)
    tlist = np.linspace(0, T_MAX_PS, N_STEPS)
    rows = []
    for _, row in df.iterrows():
        g, r = row.gamma_ps_inv, row.r_nm
        M, nA = ec.heom_generator(r, g, depth=5, Nk=1)
        x0 = ec.heom_initial_vec(nA)
        PA, _ = ec.time_domain_PA(M, x0, tlist)
        eta_tr, t_tr = ec.truncated_trapz_metrics(tlist, PA)
        eta_inf, t_inf = ec.infinite_time_metrics(M, x0)
        rows.append(dict(gamma_ps_inv=g, r_nm=r,
                         eta_legacy_qutip=row.eta, t_legacy_qutip=row.mean_trapping_time_ps,
                         eta_trunc_numpy=eta_tr, t_trunc_numpy=t_tr,
                         eta_infinite=eta_inf, t_infinite=t_inf))
    out = pd.DataFrame(rows)
    out["abs_err_eta"] = (out.eta_trunc_numpy - out.eta_legacy_qutip).abs()
    out["rel_err_t"] = ((out.t_trunc_numpy - out.t_legacy_qutip) / out.t_legacy_qutip).abs()
    out.to_csv(OUT / "validation_vs_legacy_csv.csv", index=False)

    big_r = out[out.r_nm >= 4.0]
    lines = [
        "Doğrulama: numpy HEOM (T_MAX=25 ps, trapez) vs orijinal QuTiP CSV (400 nokta)",
        f"  max |Δη|            = {out.abs_err_eta.max():.2e}",
        f"  max |Δ<t>|/<t>      = {out.rel_err_t.max():.2e}",
        "",
        "Kesme etkisi (aynı model, T_MAX=25 ps vs sonsuz zaman), r >= 4 nm:",
        f"  η (kesik)   aralığı = {big_r.eta_trunc_numpy.min():.3f} – {big_r.eta_trunc_numpy.max():.3f}",
        f"  η (sonsuz)  aralığı = {big_r.eta_infinite.min():.3f} – {big_r.eta_infinite.max():.3f}",
        f"  <t> (kesik) aralığı = {big_r.t_trunc_numpy.min():.2f} – {big_r.t_trunc_numpy.max():.2f} ps",
        f"  <t> (sonsuz)aralığı = {big_r.t_infinite.min():.2f} – {big_r.t_infinite.max():.2f} ps",
        "",
        "r başına en büyük göreli <t> hatası ve η hatası (kesik - sonsuz), tüm γ üzerinden:",
        "  (r=1.0 nm'deki hata zaman adımı (dt=0.114 ps) hızlı salınımları çözemediği için;",
        "   r >= ~2 nm'deki hata T_MAX=25 ps kesmesinden)",
    ]
    out["rel_t_trunc"] = (out.t_trunc_numpy - out.t_infinite) / out.t_infinite
    out["d_eta_trunc"] = out.eta_trunc_numpy - out.eta_infinite
    per_r = out.groupby("r_nm").agg(max_abs_rel_t=("rel_t_trunc", lambda s: s.abs().max()),
                                    max_abs_d_eta=("d_eta_trunc", lambda s: s.abs().max()))
    per_r.to_csv(OUT / "truncation_error_by_r.csv")
    for r, row in per_r.iterrows():
        lines.append(f"  r={r:5.2f} nm  |Δ<t>|/<t> ≤ {row.max_abs_rel_t*100:6.2f} %   |Δη| ≤ {row.max_abs_d_eta:.4f}")
    txt = "\n".join(lines)
    (OUT / "validation_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
