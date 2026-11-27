#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml_delta_learning.py — Faz 2A: Lindblad → HEOM çok-doğruluklu (Δ) öğrenme
=========================================================================
Soru: Ucuz Lindblad hesabını, az sayıda pahalı HEOM hesabıyla "düzelterek"
HEOM doğruluğuna yaklaşabilir miyiz? Kaç HEOM noktası gerekir?

Üç yaklaşım (hepsi GP, aynı çekirdek; hedef ln⟨t⟩):
  (a) Doğrudan:   GP( ln t_HEOM )
  (b) Δ, aynı γ:  ln t_HEOM = ln t_L(γ)  + GP( ln t_HEOM − ln t_L(γ) )
  (c) Δ, 2γ:      ln t_HEOM = ln t_L(2γ) + GP( ln t_HEOM − ln t_L(2γ) )
(c), v2'de bulunan "~2 çarpanı"nı (λ = γγ_c/2k_BT eşleştirmesi HEOM'da ~2γ
koherans sönümü verir) fiziksel ön bilgi olarak kullanır. Lindblad terimi her
test noktasında DOĞRUDAN (kesin) hesaplanır — ucuzdur.

(Perdikaris ve ark. 2017'nin doğrusal olmayan otoregresif GP'si (NARGP)
daha genel bir seçenektir; burada en basit Δ-öğrenme ile başlanmıştır.)

Öğrenme eğrisi: n_HEOM ∈ {25, 50, 100, 200, 400, 800}, 5 rastgele tohum,
sabit 200 noktalık test kümesi.

Ek: kanonik model (ΔE = 25 cm⁻¹) için t_HEOM / t_L(γ) ve t_HEOM / t_L(2γ)
oranlarının (r, γ) haritası — v2 taramasından DOĞRUDAN (emülatör değil).

Çıktı: results/ml/delta/delta_learning_curve.csv, delta_summary.txt,
       delta_learning_curve.png, heom_lindblad_ratio_map.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import enaqt_core as ec
from ml_emulator import feats, make_gp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "ml" / "data"
OUT = ROOT / "results" / "ml" / "delta"
OUT.mkdir(parents=True, exist_ok=True)

N_TRAIN = [25, 50, 100, 200, 400, 800]
SEEDS = [0, 1, 2, 3, 4]
COLORS = {"dogrudan": "#2a78d6", "delta_g": "#eb6834", "delta_2g": "#1baf7a"}
LABELS = {"dogrudan": "(a) doğrudan GP(HEOM)", "delta_g": "(b) Lindblad(γ) + GP(Δ)",
          "delta_2g": "(c) Lindblad(2γ) + GP(Δ)"}


def main():
    H = pd.read_csv(DATA / "heom_lhs.csv")
    H = H[H.heom_physical].reset_index(drop=True)
    X = feats(H)
    yH = np.log(H.mean_t_heom_ps.values)
    base = {"dogrudan": np.zeros(len(H)),
            "delta_g": np.log(H.mean_t_lind_ps.values),
            "delta_2g": np.log(H.mean_t_lind_2g_ps.values)}

    rng = np.random.default_rng(123)
    perm = rng.permutation(len(H))
    te, pool = perm[:200], perm[200:]

    rows = []
    for seed in SEEDS:
        r2 = np.random.default_rng(seed)
        order = r2.permutation(pool)
        for n in [n for n in N_TRAIN if n <= len(pool)]:
            tr = order[:n]
            for name, b in base.items():
                gp = make_gp()
                gp.fit(X[tr], yH[tr] - b[tr])
                pred = b[te] + gp.predict(X[te])
                rel = np.abs(np.exp(pred - yH[te]) - 1)
                rows.append(dict(seed=seed, n_heom=n, method=name,
                                 median_rel_err=np.median(rel), p95_rel_err=np.quantile(rel, 0.95)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "delta_learning_curve.csv", index=False)

    # sıfır eğitim noktası: yalnızca Lindblad (düzeltmesiz)
    zero = {k: np.median(np.abs(np.exp(v[te] - yH[te]) - 1)) for k, v in base.items() if k != "dogrudan"}

    agg = df.groupby(["method", "n_heom"]).median_rel_err.median().unstack(0)
    lines = ["Δ-öğrenme: test kümesinde (200 HEOM noktası) ⟨t⟩ medyan göreli hata (5 tohumun medyanı)",
             f"  Düzeltmesiz Lindblad(γ):  %{100*zero['delta_g']:.1f}    Düzeltmesiz Lindblad(2γ): %{100*zero['delta_2g']:.1f}",
             "  n_HEOM | " + " | ".join(f"{LABELS[m]:28s}" for m in ["dogrudan", "delta_g", "delta_2g"])]
    for n, row in agg.iterrows():
        lines.append(f"  {n:6d} | " + " | ".join(f"%{100*row[m]:<27.2f}" for m in ["dogrudan", "delta_g", "delta_2g"]))
    for m in ["dogrudan", "delta_g", "delta_2g"]:
        ok = agg[m][agg[m] < 0.05]
        lines.append(f"  {LABELS[m]}: %5 hedefine ulaşılan en küçük n_HEOM = {ok.index.min() if len(ok) else 'ulaşılamadı'}")

    # ---------------- oran haritası (v2 doğrudan) ----------------
    v2 = pd.read_csv(ROOT / "results" / "data" / "heom_2d_scan_v2.csv")
    v2 = v2[(v2.gamma_ps_inv > 0) & (v2.gamma_ps_inv <= 40)]
    h = v2[v2.model == "HEOM"].set_index(["r_nm", "gamma_ps_inv"]).mean_t_ps
    l = v2[v2.model == "Lindblad"].set_index(["r_nm", "gamma_ps_inv"]).mean_t_ps
    ratio_g = (h / l).unstack()
    l2 = pd.Series({(r, g): ec.lindblad_metrics(r, 2 * g)[1] for r, g in h.index})
    ratio_2g = (h / l2).unstack()
    lines += ["",
              "HEOM / Lindblad ⟨t⟩ oranı, kanonik model (ΔE = 25 cm⁻¹), r = 1–5 nm, γ = 0.01–40 ps⁻¹ (v2 doğrudan tarama):",
              f"  t_HEOM / t_L(γ):  aralık {np.nanmin(ratio_g.values):.2f} – {np.nanmax(ratio_g.values):.2f}",
              f"  t_HEOM / t_L(2γ): aralık {np.nanmin(ratio_2g.values):.2f} – {np.nanmax(ratio_2g.values):.2f}",
              "  → '~2 çarpanı' tek bir sabit değildir; uyuşmazlık r ve γ'ya bağlıdır (bkz. heom_lindblad_ratio_map.png)."]
    txt = "\n".join(lines)
    (OUT / "delta_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)

    fig, ax = plt.subplots(figsize=(6.4, 4), dpi=200)
    for m in ["dogrudan", "delta_g", "delta_2g"]:
        g = df[df.method == m].groupby("n_heom").median_rel_err
        ax.plot(g.median().index, 100 * g.median().values, "-o", color=COLORS[m], ms=4, lw=2, label=LABELS[m])
        ax.fill_between(g.median().index, 100 * g.min().values, 100 * g.max().values, color=COLORS[m], alpha=0.15, lw=0)
    ax.axhline(5, color="0.5", lw=1, ls="--", label="%5 hedefi")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("eğitimde kullanılan HEOM noktası"); ax.set_ylabel("⟨t⟩ medyan göreli hata (%)")
    ax.set_title("Lindblad → HEOM: fizik ön bilgisi az veriyle ne kadar kazandırır?", fontsize=10)
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "delta_learning_curve.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=200, sharey=True)
    for ax, R, title in [(axes[0], ratio_g, "t_HEOM / t_Lindblad(γ)"), (axes[1], ratio_2g, "t_HEOM / t_Lindblad(2γ)")]:
        Z = np.log2(R.values)
        lim = np.nanmax(np.abs(Z))
        pc = ax.pcolormesh(R.columns, R.index, Z, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), shading="nearest")
        cb = fig.colorbar(pc, ax=ax); cb.set_label("log₂(oran)  (0 = aynı)")
        ax.set_xscale("log"); ax.set_xlabel("γ (ps⁻¹)"); ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("r (nm)")
    fig.suptitle("Kanonik model (ΔE = 25 cm⁻¹): HEOM ile Lindblad nerede ayrışıyor?", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "heom_lindblad_ratio_map.png"); plt.close(fig)


if __name__ == "__main__":
    main()
