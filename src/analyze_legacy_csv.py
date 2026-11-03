#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_legacy_csv.py
=====================
Orijinal QuTiP taramasından (results/data/heom_2d_scan_results.csv) doğrudan
okunabilen sayılar: global minimum <t>'nin konumu ve her r için ENAQT kazancı

    S(r) = <t>(γ_min) / min_γ <t>(γ, r)

(Bu dosyada γ ızgarası doğrusal olduğundan γ_min = 0.05 ps^-1 "γ→0" yerine
kullanılır.) UYARI: Bu CSV T_MAX = 25 ps'de kesilmiş integrallerden gelir;
r ≳ 2 nm'de değerler sistematik olarak hatalıdır (bkz.
results/tables/truncation_error_by_r.csv). Güncel sonuçlar için
src/heom_2d_scan_v2.py kullanılır.

Çıktı: results/tables/legacy_csv_summary.csv, legacy_csv_global_min.txt
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "results" / "data" / "heom_2d_scan_results.csv")
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

imin = df.mean_trapping_time_ps.idxmin()
gm = df.loc[imin]
txt = (f"Eski CSV global minimum <t> = {gm.mean_trapping_time_ps:.4f} ps "
       f"@ r = {gm.r_nm:.3f} nm, γ = {gm.gamma_ps_inv:.3f} ps^-1 (η = {gm.eta:.4f}); "
       f"CSV satır indeksi {imin} (başlık hariç, 0'dan)")
(OUT / "legacy_csv_global_min.txt").write_text(txt + "\n", encoding="utf-8")
print(txt)

rows = []
for r, g in df.groupby("r_nm"):
    g = g.sort_values("gamma_ps_inv")
    t0 = g.mean_trapping_time_ps.iloc[0]
    k = g.mean_trapping_time_ps.idxmin()
    rows.append(dict(r_nm=r, t_at_gamma_min_ps=t0, t_min_ps=g.mean_trapping_time_ps[k],
                     gamma_opt_ps_inv=g.gamma_ps_inv[k], S=t0 / g.mean_trapping_time_ps[k]))
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "legacy_csv_summary.csv", index=False)
print(summary.round(3).to_string(index=False))
