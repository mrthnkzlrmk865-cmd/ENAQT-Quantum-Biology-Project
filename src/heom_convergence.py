#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
heom_convergence.py — HEOM hiyerarşi yakınsama testi
====================================================
Orijinal dokümanda "max_depth 5 ile <%0.2 sapma" iddiası vardı, ancak bunu
gösteren bir script/tablo yoktu. Bu script iddiayı TEST eder.

Test ızgarası: r ∈ {1.4, 2.0, 3.3, 4.5} nm × γ ∈ {1, 12.6, 40, 100} ps^-1
  (a) Derinlik: 5, 7, 9, 11, 13 (Nk = 1); referans = derinlik 13
  (b) Matsubara: Nk = 0, 1, 2, 3 (derinlik 5); referans = Nk 3
  (c) Kararlılık gösterimi: iki noktada derinlik 15
Metrikler sonsuz-zaman η ve <t> (enaqt_core).

Önemli: Bu rejimde (γ_c = 2 rad/ps yavaş banyo, büyük γ'da λ ≳ γ_c) kesilmiş
HEOM derinlik arttıkça her zaman iyileşmez; yüksek derinlikte fiziksel
olmayan (η > 1 veya < 0) sonuçlar çıkabilir. Bu yüzden 'güvenilir' kriteri:
derinlik 11 ile 13 arasında |Δ<t>|/<t> < %1 VE 0 ≤ η ≤ 1.

Çıktı: results/tables/heom_convergence.csv
       results/tables/heom_convergence_summary.txt
Çalıştırma: python src/heom_convergence.py   (~5–10 dk)
"""
from pathlib import Path
import pandas as pd
import enaqt_core as ec

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)

R_TEST = [1.4, 2.0, 3.3, 4.5]
G_TEST = [1.0, 12.6, 40.0, 100.0]
DEPTHS = [5, 7, 9, 11, 13]
REF_DEPTH = 13
DEPTH15_POINTS = [(3.3, 12.6), (4.5, 50.0)]


def main():
    rows = []
    for r in R_TEST:
        for g in G_TEST:
            for d in DEPTHS:
                e, t = ec.heom_metrics(r, g, depth=d, Nk=1)
                rows.append(dict(test="depth", r_nm=r, gamma_ps_inv=g, depth=d, Nk=1, eta=e, mean_t_ps=t))
            for nk in (0, 1, 2, 3):
                e, t = ec.heom_metrics(r, g, depth=5, Nk=nk)
                rows.append(dict(test="matsubara", r_nm=r, gamma_ps_inv=g, depth=5, Nk=nk, eta=e, mean_t_ps=t))
            print(f"r={r} γ={g} tamam", flush=True)
    for r, g in DEPTH15_POINTS:
        for d in (11, 13, 15):
            e, t = ec.heom_metrics(r, g, depth=d, Nk=1)
            rows.append(dict(test="depth15_demo", r_nm=r, gamma_ps_inv=g, depth=d, Nk=1, eta=e, mean_t_ps=t))
    df = pd.DataFrame(rows)

    dep = df[df.test == "depth"].copy()
    ref = dep[dep.depth == REF_DEPTH].set_index(["r_nm", "gamma_ps_inv"])
    dep["rel_err_t_vs_ref"] = [abs(t / ref.loc[(r, g), "mean_t_ps"] - 1)
                               for r, g, t in zip(dep.r_nm, dep.gamma_ps_inv, dep.mean_t_ps)]
    mat = df[df.test == "matsubara"].copy()
    ref_m = mat[mat.Nk == 3].set_index(["r_nm", "gamma_ps_inv"])
    mat["rel_err_t_vs_ref"] = [abs(t / ref_m.loc[(r, g), "mean_t_ps"] - 1)
                               for r, g, t in zip(mat.r_nm, mat.gamma_ps_inv, mat.mean_t_ps)]
    demo = df[df.test == "depth15_demo"]
    pd.concat([dep, mat, demo]).to_csv(TAB / "heom_convergence.csv", index=False)

    lines = [f"HEOM yakınsama: <t> göreli farkı, derinlik d vs derinlik {REF_DEPTH} (Nk=1), sonsuz zaman.",
             "  r (nm)  γ (ps^-1) |  d=5     d=7     d=9     d=11   | η(d=13)"]
    for (r, g), s in dep.groupby(["r_nm", "gamma_ps_inv"]):
        errs = "  ".join(f"%{100*s[s.depth == d].rel_err_t_vs_ref.iloc[0]:5.2f}" for d in DEPTHS[:-1])
        lines.append(f"  {r:4.1f}   {g:6.1f}    | {errs} | {ref.loc[(r, g), 'eta']:.4f}")
    ok = dep[dep.depth == 11]
    allr = ok.groupby("gamma_ps_inv").rel_err_t_vs_ref.max()
    lines.append("")
    lines.append(f"  Derinlik 11: TÜM test r'lerinde %1 kriterini sağlayan γ değerleri: "
                 f"{[float(g) for g in allr[allr < 0.01].index]}; sağlamayan: {[float(g) for g in allr[allr >= 0.01].index]}")
    lines.append(f"  Derinlik 5 maks hata (γ ≤ 40): %{100*dep[(dep.depth == 5) & (dep.gamma_ps_inv <= 40)].rel_err_t_vs_ref.max():.1f}; "
                 f"(γ = 100): %{100*dep[(dep.depth == 5) & (dep.gamma_ps_inv == 100)].rel_err_t_vs_ref.max():.1f}")
    lines.append("Matsubara (derinlik 5, referans Nk=3):")
    for nk in (0, 1, 2):
        s = mat[mat.Nk == nk]
        lines.append(f"  Nk={nk}: maks %{100*s.rel_err_t_vs_ref.max():.3f}")
    lines.append("Kararlılık gösterimi (derinlik 11 / 13 / 15):")
    for (r, g), s in demo.groupby(["r_nm", "gamma_ps_inv"]):
        lines.append(f"  r={r} nm, γ={g} ps^-1: " + "; ".join(
            f"d={int(d)}: η={e:.4f}, <t>={t:.3f} ps" for d, e, t in zip(s.depth, s.eta, s.mean_t_ps)))
    txt = "\n".join(lines)
    (TAB / "heom_convergence_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
