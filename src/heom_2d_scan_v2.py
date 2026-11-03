#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
heom_2d_scan_v2.py — Düzeltilmiş 2D (γ, r) taraması
===================================================
Orijinal taramaya (src/heom_2d_robustness_scan.py) göre değişiklikler:
  1. η ve <t> SONSUZ ZAMANA kadar analitik hesaplanır (T_MAX kesmesi yok,
     zaman adımı hatası yok) — bkz. enaqt_core.infinite_time_metrics.
  2. γ ızgarası LOGARİTMİKTİR (0.01–100 ps^-1, 41 nokta) + ayrıca γ = 0.
  3. Her r için ENAQT kazancı tanımlanır:
         S(r) = <t>(γ = 0) / min_γ <t>(γ, r)
     "En hızlı nokta" (global min <t>) ile "gürültünün en çok yardım ettiği
     nokta" (max S) ayrı ayrı raporlanır.
  4. Aynı ızgara Lindblad modeli için de hesaplanır (karşılaştırma).

Model ve banyo ayarları orijinal ile aynıdır (kanonik dimer, Γ_sink = 1 ps^-1,
γ_c = 2 rad/ps, Nk = 1). Hiyerarşi derinliği ise 5 yerine 11'dir, çünkü
5 büyük r / büyük γ'da yakınsamamıştır (bkz. src/heom_convergence.py).

Çıktılar:
  results/data/heom_2d_scan_v2.csv               (tüm noktalar, iki model)
  results/tables/enaqt_gain_by_r_v2.csv          (r başına <t>(0), <t>min, γ*, S)
  results/tables/scan_v2_headline.txt            (README'deki sayıların kaynağı)
  results/figures/heom_v2_mean_time_heatmap.png
  results/figures/enaqt_gain_S_vs_r.png
Çalıştırma: python src/heom_2d_scan_v2.py   (~15–30 dk; 2 çekirdek)
            python src/heom_2d_scan_v2.py --reuse  (yalnızca tablo/grafik)
"""
import os
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import enaqt_core as ec

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
DAT = ROOT / "results" / "data"; DAT.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)

GAMMAS = np.logspace(-2, 2, 41)           # ps^-1
R_NM = np.round(np.linspace(1.0, 5.0, 41), 3)
C_HEOM, C_LIND = "#2a78d6", "#eb6834"     # kategorik slot 1 ve 2
# Hiyerarşi derinliği: orijinal üretim ayarı (5) büyük r / büyük γ'da
# yakınsamamıştır. 11, γ ≤ 40 ps^-1 test noktalarında derinlik 13'e göre
# <%1 sapma verir (bkz. results/tables/heom_convergence_summary.txt).
HEOM_DEPTH, HEOM_NK = 11, 1
# γ > 40 ps^-1'de (λ ≳ γ_c) kesilmiş hiyerarşi yakınsamaz ve derinlik
# arttıkça kararsızlaşabilir (η > 1 gibi fiziksel olmayan değerler). Bu
# noktalar CSV'de 'heom_reliable = False' ile işaretlenir ve S(r), γ*(r),
# global minimum hesaplarına KATILMAZ.
HEOM_RELIABLE_GAMMA_MAX = 40.0
N_WORKERS = os.cpu_count() or 1


def _row_block(args):
    r, p, gammas = args
    rows = []
    # γ = 0: banyo yok -> iki model aynı (yalnızca H + Markovian kanallar)
    eta0, t0 = ec.lindblad_metrics(r, 0.0, p)
    for model in ("HEOM", "Lindblad"):
        rows.append(dict(model=model, r_nm=r, gamma_ps_inv=0.0, eta=eta0, mean_t_ps=t0))
    for g in gammas:
        e, t = ec.heom_metrics(r, g, p, depth=HEOM_DEPTH, Nk=HEOM_NK)
        rows.append(dict(model="HEOM", r_nm=r, gamma_ps_inv=g, eta=e, mean_t_ps=t))
        e, t = ec.lindblad_metrics(r, g, p)
        rows.append(dict(model="Lindblad", r_nm=r, gamma_ps_inv=g, eta=e, mean_t_ps=t))
    return rows


def scan(p=ec.CANONICAL, gammas=GAMMAS, r_nm=R_NM, workers=N_WORKERS):
    tasks = [(r, p, gammas) for r in r_nm]
    with Pool(workers) as pool:
        blocks = pool.map(_row_block, tasks)
    return pd.DataFrame([row for b in blocks for row in b])


def mark_reliability(df):
    df = df.copy()
    df["heom_reliable"] = (df.model == "Lindblad") | (df.gamma_ps_inv <= HEOM_RELIABLE_GAMMA_MAX)
    return df


def gain_table(df):
    df = df[df.heom_reliable]
    out = []
    for (model, r), g in df.groupby(["model", "r_nm"]):
        t0 = g.loc[g.gamma_ps_inv == 0, "mean_t_ps"].iloc[0]
        k = g.mean_t_ps.idxmin()
        out.append(dict(model=model, r_nm=r, t_gamma0_ps=t0, t_min_ps=g.mean_t_ps[k],
                        gamma_opt_ps_inv=g.gamma_ps_inv[k], eta_at_opt=g.eta[k],
                        S=t0 / g.mean_t_ps[k]))
    return pd.DataFrame(out)


def headline(df, gt):
    hall = df[(df.model == "HEOM") & (df.gamma_ps_inv > 0)]
    h = hall[hall.heom_reliable]
    k = h.mean_t_ps.idxmin()
    gh = gt[gt.model == "HEOM"].reset_index(drop=True)
    j = gh.S.idxmax()
    half = gh[gh.S >= 1 + 0.5 * (gh.S[j] - 1)]
    lines = [
        f"HEOM v2 (sonsuz zaman, log-γ 0.01–100 ps^-1, Γ_sink = 1 ps^-1, depth = {HEOM_DEPTH}, Nk = {HEOM_NK};",
        f"         HEOM için yalnızca γ ≤ {HEOM_RELIABLE_GAMMA_MAX:g} ps^-1 kullanıldı):",
        f"  En hızlı nokta (global min <t>): <t> = {h.mean_t_ps[k]:.3f} ps @ r = {h.r_nm[k]:.2f} nm, "
        f"γ = {h.gamma_ps_inv[k]:.3g} ps^-1, η = {h.eta[k]:.4f}",
        f"  En büyük ENAQT kazancı: S = {gh.S[j]:.2f} @ r = {gh.r_nm[j]:.2f} nm "
        f"(γ* = {gh.gamma_opt_ps_inv[j]:.3g} ps^-1, <t>(0) = {gh.t_gamma0_ps[j]:.2f} ps -> "
        f"<t>min = {gh.t_min_ps[j]:.2f} ps)",
        f"  S'nin yarı-yükseklik aralığı (S-1 ≥ (Smax-1)/2): r = {half.r_nm.min():.2f}–{half.r_nm.max():.2f} nm",
    ]
    for rr in (1.0, 1.4, 2.0, 3.0, 4.0, 5.0):
        row = gh.iloc[(gh.r_nm - rr).abs().idxmin()]
        lines.append(f"  r = {row.r_nm:.2f} nm: S = {row.S:.3f}, <t>min = {row.t_min_ps:.2f} ps, "
                     f"γ* = {row.gamma_opt_ps_inv:.3g} ps^-1, η(opt) = {row.eta_at_opt:.4f}")
    gl = gt[gt.model == "Lindblad"].reset_index(drop=True)
    jl = gl.S.idxmax()
    lines.append(f"  Lindblad karşılaştırma: Smax = {gl.S[jl]:.2f} @ r = {gl.r_nm[jl]:.2f} nm "
                 f"(γ* = {gl.gamma_opt_ps_inv[jl]:.3g} ps^-1)")
    lines.append(f"  η aralığı (HEOM, güvenilir γ): {h.eta.min():.4f} – {h.eta.max():.4f}")
    bad = hall[~hall.heom_reliable]
    lines.append(f"  γ > {HEOM_RELIABLE_GAMMA_MAX:g} ps^-1 (dışlanan) HEOM noktalarında η aralığı: "
                 f"{bad.eta.min():.4f} – {bad.eta.max():.4f}  (η > 1 -> sayısal kararsızlık)")
    return "\n".join(lines)


def plot_heatmap(df, gt):
    h = df[(df.model == "HEOM") & (df.gamma_ps_inv > 0)]
    Z = h.pivot(index="r_nm", columns="gamma_ps_inv", values="mean_t_ps")
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=200)
    pc = ax.pcolormesh(Z.columns, Z.index, Z.values, cmap="Blues_r",
                       norm=LogNorm(Z.values.min(), Z.values.max()), shading="nearest")
    ax.axvspan(HEOM_RELIABLE_GAMMA_MAX, Z.columns.max() * 1.15, facecolor="white", alpha=0.75,
               hatch="///", edgecolor="0.5", lw=0)
    ax.text(64, 4.85, "HEOM\ngüvenilmez\n(γ>40)", ha="center", va="top", fontsize=7, color="0.2")
    cb = fig.colorbar(pc, ax=ax); cb.set_label(r"$\langle t\rangle$ (ps), sonsuz zaman")
    gh = gt[gt.model == "HEOM"]
    gh = gh[gh.S > 1.01]                 # γ*'yi yalnızca gürültü anlamlı yardım ettiğinde çiz
    ax.plot(gh.gamma_opt_ps_inv, gh.r_nm, color="#eb6834", lw=2,
            label=r"$\gamma^*(r)$: en hızlı γ (yalnızca S > 1.01 olan r)")
    hr = h[h.heom_reliable]
    k = hr.mean_t_ps.idxmin()
    ax.plot(hr.gamma_ps_inv[k], hr.r_nm[k], "o", ms=9, mfc="white", mec="black",
            label=f"global min ⟨t⟩ = {hr.mean_t_ps[k]:.2f} ps")
    ax.set_xscale("log"); ax.set_xlim(Z.columns.min() / 1.15, Z.columns.max() * 1.15)
    ax.set_xlabel(r"dephasing hızı $\gamma$ (ps$^{-1}$)  [HEOM: $\lambda=\gamma\gamma_c/2k_BT$]")
    ax.set_ylabel("donör–akseptör mesafesi r (nm)")
    ax.set_title("HEOM ⟨t⟩(γ, r) — kanonik dimer, Γ_sink = 1 ps⁻¹ (v2)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(FIG / "heom_v2_mean_time_heatmap.png"); plt.close(fig)


def plot_gain(gt):
    fig, ax = plt.subplots(figsize=(6.4, 4), dpi=200)
    for model, c, ls, mk in (("HEOM", C_HEOM, "-", "o"), ("Lindblad", C_LIND, "--", "s")):
        g = gt[gt.model == model]
        ax.plot(g.r_nm, g.S, ls, color=c, lw=2, marker=mk, ms=3, label=model)
        j = g.S.idxmax()
        ax.annotate(f"{model}: S_max = {g.S[j]:.2f} @ r = {g.r_nm[j]:.2f} nm",
                    (g.r_nm[j], g.S[j]), xycoords="data",
                    xytext=(1.05, 3.0 if model == "HEOM" else 2.6), textcoords="data",
                    fontsize=8, arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
    ax.axhline(1, color="0.6", lw=1)
    ax.set_xlabel("donör–akseptör mesafesi r (nm)")
    ax.set_ylabel(r"ENAQT kazancı  $S(r)=\langle t\rangle(\gamma=0)\,/\,\min_\gamma\langle t\rangle$")
    ax.set_title("Gürültünün transferi en çok hızlandırdığı mesafe (model içi)", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "enaqt_gain_S_vs_r.png"); plt.close(fig)


def main():
    import sys
    csv = DAT / "heom_2d_scan_v2.csv"
    if "--reuse" in sys.argv and csv.exists():
        df = pd.read_csv(csv)          # yalnızca son işleme/çizim (hesap tekrarlanmaz)
    else:
        df = scan()
    df = mark_reliability(df)
    df.to_csv(csv, index=False)
    gt = gain_table(df)
    gt.to_csv(TAB / "enaqt_gain_by_r_v2.csv", index=False)
    txt = headline(df, gt)
    (TAB / "scan_v2_headline.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    plot_heatmap(df, gt)
    plot_gain(gt)


if __name__ == "__main__":
    main()
