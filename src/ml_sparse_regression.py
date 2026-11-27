#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml_sparse_regression.py — Faz 2A (opsiyonel): veriden mesafe yasası bulma
========================================================================
Soru: Simülasyon verisinden, hangi rejimde transfer hızının tek terimli bir
r^-n yasasına (Förster: n = 6) uyduğunu, hangisinde uymadığını bir "denklem
bulma" yöntemi ayırt edebilir mi?

Yöntem: SINDy fikri (Brunton, Proctor & Kutz 2016, PNAS 113:3932) — aday
terim kütüphanesinden EN SEYREK modeli seçmek — kuvvet-yasası kütüphanesiyle:
    k(r) ≈ Σ_p ξ_p r^(-p),   p = 0, 1, …, 8
Göreli hata minimize edilir (satırlar k ile bölünür).
İlk deneme olarak SINDy'nin ardışık eşikli en küçük kareler (STLSQ) algoritması
denendi; ancak r^-p sütunları neredeyse doğrusal bağımlı olduğu için STLSQ,
birbirini yok eden büyük katsayılı 9 terimli modeller üretti (bilinen
çoklu-doğrusallık sorunu). Bu nedenle küçük kütüphanede tam alt-küme araması
yapılır: 1, 2, 3 terimli tüm kombinasyonlar denenir; maksimum göreli hatası
TOL'un altında kalan EN AZ terimli model seçilir. PySR bu ortamda kurulamadı.

DÜRÜSTLÜK NOTU: Modelde V ∝ r^-3 girdi olarak verilmiştir. Zayıf kaplin
rejiminde r^-6'nın bulunması bir DOĞRULAMA testidir, yeni fizik keşfi değildir.
İlginç olan, yöntemin tek terimli yasanın NEREDE bozulduğunu gösterip
gösteremediğidir.

Hız tanımı (src/forster_limit.py ile aynı): k_T = k_D η / (1 − η), gevşeme
kanalı kapalı, akseptörde hızlı yakalama (Γ_sink = 10 ps⁻¹), k_D = 0.001 ps⁻¹.

Çıktı: results/ml/sparse/sparse_regression_summary.txt, sparse_regression.png
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import enaqt_core as ec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ml" / "sparse"
OUT.mkdir(parents=True, exist_ok=True)

K_D = 0.001
POWERS = np.arange(0, 9)
TOL = 0.01            # kabul edilen maksimum göreli hata
MAX_TERMS = 3

REGIMES = [
    ("FRET-benzeri: ΔE=1000 cm⁻¹, γ=200 ps⁻¹, r=2–10 nm", 1000.0, 200.0, (2.0, 10.0)),
    ("ΔE=1000 cm⁻¹, γ=2 ps⁻¹ (zayıf dephasing), r=2–10 nm", 1000.0, 2.0, (2.0, 10.0)),
    ("kanonik: ΔE=25 cm⁻¹, γ=1 ps⁻¹, r=3–8 nm", 25.0, 1.0, (3.0, 8.0)),
    ("kanonik: ΔE=25 cm⁻¹, γ=1 ps⁻¹, r=1–8 nm", 25.0, 1.0, (1.0, 8.0)),
]


def rates(dE, gamma, r):
    p = ec.CANONICAL.with_(E_donor_cm1=dE, E_acceptor_cm1=0.0, gamma_relax_psinv=1e-12,
                           gamma_sink_psinv=10.0, gamma_loss_psinv=K_D)
    eta = np.array([ec.lindblad_metrics(x, gamma, p)[0] for x in r])
    return K_D * eta / (1 - eta)


def best_subset(r, k, tol=TOL, max_terms=MAX_TERMS):
    """En az terimli ve maks göreli hatası < tol olan model; yoksa en iyi max_terms modeli."""
    Theta = r[:, None] ** (-POWERS[None, :].astype(float))
    A = Theta / k[:, None]
    best_by_m = {}
    for m in range(1, max_terms + 1):
        for sub in combinations(range(len(POWERS)), m):
            sub = list(sub)
            sol, *_ = np.linalg.lstsq(A[:, sub], np.ones(len(r)), rcond=None)
            xi = np.zeros(len(POWERS)); xi[sub] = sol
            err = np.max(np.abs(Theta @ xi / k - 1))
            if m not in best_by_m or err < best_by_m[m][1]:
                best_by_m[m] = (xi, err)
        if best_by_m[m][1] < tol:
            return best_by_m[m][0], best_by_m[m][0] != 0, best_by_m[m][1], best_by_m
    m = max(best_by_m)
    return best_by_m[m][0], best_by_m[m][0] != 0, best_by_m[m][1], best_by_m


def main():
    lines = [f"Seyrek model seçimi (alt-küme araması, kütüphane r^-p, p = 0…8, ≤ {MAX_TERMS} terim, tolerans %{100*TOL:g}):"]
    fig, axes = plt.subplots(1, len(REGIMES), figsize=(4 * len(REGIMES), 3.6), dpi=200)
    rows = []
    for ax, (title, dE, g, (r0, r1)) in zip(axes, REGIMES):
        r = np.geomspace(r0, r1, 40)
        k = rates(dE, g, r)
        xi, active, maxerr, by_m = best_subset(r, k)
        slope = np.polyfit(np.log(r), np.log(k), 1)[0]
        n_local = -np.gradient(np.log(k), np.log(r))
        terms = " + ".join(f"{xi[p]:.3g}·r^-{p}" for p in POWERS[active])
        lines += [f"  [{title}]",
                  f"    bulunan model: k(r) = {terms}",
                  f"    maks göreli hata %{100*maxerr:.2f}" + ("" if maxerr < TOL else "  (TOLERANS SAĞLANAMADI)")
                  + f"; en iyi tek terimli modelin hatası %{100*by_m[1][1]:.2f}",
                  f"    log-log eğim {slope:.3f}; yerel üs aralığı {n_local.min():.2f}–{n_local.max():.2f}"]
        rows.append(dict(rejim=title, terimler=" ".join(f"r^-{p}" for p in POWERS[active]),
                         maks_goreli_hata=maxerr, tek_terim_hata=by_m[1][1], loglog_egim=slope))
        ax.loglog(r, k, "o", ms=3, color="0.3", label="simülasyon")
        ax.loglog(r, (r[:, None] ** (-POWERS[None, :].astype(float))) @ xi, "-", color="#2a78d6", lw=2,
                  label="bulunan model")
        ax.set_title(title.replace(": ", ":\n", 1), fontsize=8)
        ax.set_xlabel("r (nm)"); ax.grid(alpha=0.25, which="both")
        ax.text(0.03, 0.04, "terimler: " + ", ".join(f"r⁻{p}" for p in POWERS[active]), transform=ax.transAxes, fontsize=8)
    axes[0].set_ylabel("transfer hızı k (ps⁻¹)"); axes[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "sparse_regression.png"); plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "sparse_regression.csv", index=False)
    lines.append("Not: r^-6'nın zayıf kaplin rejiminde bulunması bir doğrulamadır (V ∝ r^-3 modele girdi olarak verildi).")
    txt = "\n".join(lines)
    (OUT / "sparse_regression_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
