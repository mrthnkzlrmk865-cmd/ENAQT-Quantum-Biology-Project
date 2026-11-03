#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forster_limit.py — Model ile kararlı-hal (absolute) FRET arasındaki köprü
=========================================================================
Soru: Aynı Hamiltoniyen + Lindblad modeli, FRET boyalarının yaşadığı
rejimde (büyük enerji farkı ΔE, zayıf kaplin V, güçlü dephasing γ)
klasik Förster davranışına iniyor mu?

Beklenti (Haken–Strobl / "Kubo" sınırı, koherans sönüm hızı Γ ≈ γ):
    k_T(r) = 2 V(r)^2 Γ / (Γ^2 + ΔE^2)  ∝  V^2  ∝  r^-6
    E(r)   = k_T / (k_T + k_D) = 1 / (1 + (r/R0)^6),   k_T(R0) = k_D

Kurulum (AÇIKLAYICI parametreler — belirli bir boya çiftinin ölçülmüş
değerleri DEĞİLDİR):
    ΔE = 1000 cm^-1, γ = 200 ps^-1, V(r) = 20 cm^-1 (2 nm / r)^3 (kanonik yasa)
    donör kaybı k_D = 0.001 ps^-1 (τ_D ≈ 1 ns), akseptörde hızlı yakalama
    Γ_sink = 10 ps^-1 (akseptöre ulaşan uyarılma "sayılır"), gevşeme kanalı kapalı.
Bu durumda sink verimi η, FRET verimi E'ye karşılık gelir.

Ayrıca karşılaştırma için kanonik ENAQT rejimi (ΔE = 25 cm^-1, γ = 1 ps^-1)
için etkin üs n(r) = -d ln k / d ln r hesaplanır; Förster'de n = 6.

Çıktı: results/figures/forster_limit_check.png
       results/tables/forster_limit.csv, forster_limit_summary.txt
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import enaqt_core as ec

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)

K_D = 0.001
FRET = ec.CANONICAL.with_(E_donor_cm1=1000.0, E_acceptor_cm1=0.0, gamma_relax_psinv=1e-12,
                          gamma_sink_psinv=10.0, gamma_loss_psinv=K_D)
GAMMA_FRET = 200.0
CANON = ec.CANONICAL.with_(gamma_relax_psinv=1e-12, gamma_sink_psinv=10.0, gamma_loss_psinv=K_D)
GAMMA_CANON = 1.0
R = np.linspace(1.0, 10.0, 46)


def k_eff_from_eta(eta):
    # E = k_T / (k_T + k_D)  ->  k_T = k_D E / (1 - E)
    return K_D * eta / (1.0 - eta)


def main():
    rows = []
    for r in R:
        eta, _ = ec.lindblad_metrics(r, GAMMA_FRET, FRET)
        V = ec.coupling_cm1(r, FRET) * ec.CM1_TO_RADPS
        dE = (FRET.E_donor_cm1 - FRET.E_acceptor_cm1) * ec.CM1_TO_RADPS
        Gam = GAMMA_FRET + 0.5 * FRET.gamma_sink_psinv + FRET.gamma_loss_psinv
        k_hs = 2 * V ** 2 * Gam / (Gam ** 2 + dE ** 2)
        eta_c, _ = ec.lindblad_metrics(r, GAMMA_CANON, CANON)
        rows.append(dict(r_nm=r, eta_model=eta, k_model=k_eff_from_eta(eta), k_haken_strobl=k_hs,
                         eta_canonical=eta_c, k_canonical=k_eff_from_eta(eta_c)))
    df = pd.DataFrame(rows)
    lnr = np.log(df.r_nm)
    df["n_eff_fret"] = -np.gradient(np.log(df.k_model), lnr)
    df["n_eff_canonical"] = -np.gradient(np.log(df.k_canonical), lnr)

    # R0: k_model(R0) = k_D, k ∝ r^-6 uydurmasından
    fit = df.r_nm >= 2.0          # zayıf-kaplin koşulunun sağlandığı bölge
    slope, icpt = np.polyfit(lnr[fit], np.log(df.k_model[fit]), 1)
    R0 = np.exp((icpt - np.log(K_D)) / -slope)
    df["E_forster_fit"] = 1.0 / (1.0 + (df.r_nm / R0) ** 6)
    df.to_csv(TAB / "forster_limit.csv", index=False)

    txt = "\n".join([
        "Förster limiti kontrolü (Lindblad, açıklayıcı FRET-benzeri parametreler):",
        f"  log k – log r uydurma eğimi = {slope:.3f}  (Förster: -6)",
        f"  R0 (k_T = k_D) = {R0:.2f} nm",
        f"  (uydurma r = 2–10 nm üzerinde)",
        f"  r ≥ 2 nm: maks |η_model - 1/(1+(r/R0)^6)| = {np.max(np.abs(df.eta_model - df.E_forster_fit)[fit]):.4f}",
        f"  r ≥ 2 nm: maks |k_model/k_HakenStrobl - 1| = {np.max(np.abs(df.k_model / df.k_haken_strobl - 1)[fit]):.4f}",
        f"  r < 2 nm: maks |k_model/k_HakenStrobl - 1| = {np.max(np.abs(df.k_model / df.k_haken_strobl - 1)[~fit]):.3f}"
        " (k_T artık Γ_sink ile kıyaslanabilir -> geri transfer; nokta-dipol de burada geçersiz)",
        "  Kanonik ENAQT rejiminde (ΔE=25 cm^-1, γ=1 ps^-1) etkin üs n(r) (Förster: 6):",
        *[f"    r = {rr:.1f} nm: n = {df.n_eff_canonical[(df.r_nm - rr).abs().idxmin()]:.2f}"
          for rr in (1.0, 2.0, 3.0, 4.0, 6.0)],
        "  -> kısa mesafede kanonik rejim Förster yasasından sapar; uzak mesafede (zayıf kaplin) ona yaklaşır.",
        "  Uydurma (eğim, R0) yalnızca FRET-benzeri rejime aittir.",
    ])
    (TAB / "forster_limit_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
    a.plot(df.r_nm, df.E_forster_fit, "-", color="#2a78d6", lw=2,
           label=f"Förster: 1/(1+(r/R₀)⁶), R₀ = {R0:.2f} nm")
    a.plot(df.r_nm, df.eta_model, "o", color="#eb6834", ms=4, label="model (Lindblad) verimi η")
    a.set_xlabel("r (nm)"); a.set_ylabel("transfer verimi E"); a.grid(alpha=0.25)
    a.set_title("FRET-benzeri rejimde model → Förster eğrisi", fontsize=10)
    a.legend(fontsize=8)
    b.plot(df.r_nm, df.n_eff_fret, "-", color="#2a78d6", lw=2, label="FRET-benzeri rejim (ΔE=1000 cm⁻¹, γ=200 ps⁻¹)")
    b.plot(df.r_nm, df.n_eff_canonical, "--", color="#eb6834", lw=2, label="kanonik ENAQT rejimi (ΔE=25 cm⁻¹, γ=1 ps⁻¹)")
    b.axhline(6, color="0.6", lw=1, label="Förster üssü (6)")
    b.set_xlabel("r (nm)"); b.set_ylabel(r"etkin üs $n=-d\ln k/d\ln r$"); b.grid(alpha=0.25)
    b.set_title("Etkin mesafe üssü (kısa r’de Förster’den sapma)", fontsize=10)
    b.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "forster_limit_check.png"); plt.close(fig)


if __name__ == "__main__":
    main()
