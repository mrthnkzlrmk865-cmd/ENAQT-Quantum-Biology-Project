#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml_emulator.py — Faz 2A: Lindblad / HEOM emülatörleri
=====================================================
Soru: Küçük bir ML modeli, (r, ΔE, γ) → (⟨t⟩, η) eşlemesini doğrudan
hesaplamaya yakın doğrulukla ve dürüst belirsizlikle öğrenebilir mi?
(Şablon: Häse, Kreisbeck & Aspuru-Guzik 2017, Chem. Sci. 8:8419.)

Girdi özellikleri: x = [ln r, ΔE/1000, ln γ];  hedefler: ln⟨t⟩ ve logit η.

Modeller (scikit-learn):
  * GP: Matern(ν=5/2, ARD) + beyaz gürültü; tahmin + belirsizlik (σ)
  * MLP: 3 gizli katman × 64 nöron, erken durdurma (belirsizlik yok)

Deneyler:
  A) Lindblad emülatörü: 20 000 noktadan rastgele %20 test; GP 2 000 nokta
     ile, MLP tüm eğitim kümesiyle (GP'nin O(n³) maliyeti nedeniyle).
     NOT: Lindblad zaten nokta başına < 1 ms sürüyor; burada amaç hız değil,
     emülatör tekniğinin doğrulanmasıdır.
  B) HEOM emülatörü (γ ≤ 40 ps⁻¹): 1 000 noktadan 200 test, 800 eğitim.
     HEOM nokta başına ~0.6 s → emülatör hız kazancı burada anlamlıdır.
  C) Uygulama: HEOM-GP ile kanonik modelde (ΔE = 25 cm⁻¹) S(r) haritası;
     v2 doğrudan HEOM taramasıyla (results/tables/enaqt_gain_by_r_v2.csv)
     karşılaştırma.

Başarı ölçütü (proje içi, önceden belirlenen): test kümesinde ⟨t⟩ medyan
göreli hata < %5; GP %90 aralık kapsaması %85–95.

Çıktı: results/ml/emulator/emulator_metrics.csv, emulator_summary.txt,
       emulator_parity.png (parite + S(r) paneli), S_of_r_emulator_vs_direct.csv
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import enaqt_core as ec

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "ml" / "data"
OUT = ROOT / "results" / "ml" / "emulator"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 7
Z90 = 1.6449
EPS = 1e-6


def feats(df):
    return np.column_stack([np.log(df.r_nm), df.dE_cm1 / 1000.0, np.log(df.gamma_ps_inv)])


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-z))


def make_gp():
    k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0, 1.0, 1.0],
                                                  length_scale_bounds=(1e-2, 1e2), nu=2.5) \
        + WhiteKernel(1e-6, (1e-10, 1e-1))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2, random_state=SEED)


def make_mlp():
    return make_pipeline(StandardScaler(),
                         MLPRegressor(hidden_layer_sizes=(64, 64, 64), activation="tanh",
                                      learning_rate_init=3e-3, max_iter=3000, early_stopping=True,
                                      n_iter_no_change=50, random_state=SEED))


def metrics(name, target, y_true, y_pred, y_std=None, n_train=None, fit_s=None, pred_us=None):
    m = dict(model=name, target=target, n_train=n_train, fit_seconds=fit_s, predict_us_per_point=pred_us)
    if target == "ln_t":
        rel = np.abs(np.exp(y_pred - y_true) - 1)
        m.update(median_rel_err=np.median(rel), p95_rel_err=np.quantile(rel, 0.95), max_rel_err=rel.max())
    else:
        ae = np.abs(expit(y_pred) - expit(y_true))
        m.update(median_abs_err=np.median(ae), p95_abs_err=np.quantile(ae, 0.95), max_abs_err=ae.max())
    if y_std is not None:
        m["coverage90"] = np.mean(np.abs(y_true - y_pred) <= Z90 * y_std)
    return m


def run_models(X_tr, Y_tr, X_te, Y_te, tag, gp_n, rng):
    rows, preds = [], {}
    for j, target in enumerate(["ln_t", "logit_eta"]):
        sub = rng.choice(len(X_tr), size=min(gp_n, len(X_tr)), replace=False)
        gp = make_gp()
        t0 = time.time(); gp.fit(X_tr[sub], Y_tr[sub, j]); fs = time.time() - t0
        t0 = time.time(); mu, sd = gp.predict(X_te, return_std=True); pu = 1e6 * (time.time() - t0) / len(X_te)
        rows.append(metrics(f"{tag}-GP", target, Y_te[:, j], mu, sd, len(sub), fs, pu))
        preds[(target, "GP")] = (mu, sd)
        if target == "ln_t":
            preds["gp_model"] = gp

        mlp = make_mlp()
        t0 = time.time(); mlp.fit(X_tr, Y_tr[:, j]); fs = time.time() - t0
        t0 = time.time(); mu = mlp.predict(X_te); pu = 1e6 * (time.time() - t0) / len(X_te)
        rows.append(metrics(f"{tag}-MLP", target, Y_te[:, j], mu, None, len(X_tr), fs, pu))
        preds[(target, "MLP")] = (mu, None)
    return rows, preds


def main():
    rng = np.random.default_rng(SEED)
    lines = []

    # ---------------- A) Lindblad ----------------
    L = pd.read_csv(DATA / "lindblad_lhs.csv")
    X = feats(L)
    Y = np.column_stack([np.log(L.mean_t_ps), logit(L.eta)])
    perm = rng.permutation(len(L)); nte = len(L) // 5
    te, tr = perm[:nte], perm[nte:]
    rowsA, predsA = run_models(X[tr], Y[tr], X[te], Y[te], "Lindblad", 2000, rng)

    t0 = time.time()
    for _, r in L.iloc[:300].iterrows():
        ec.lindblad_metrics(r.r_nm, r.gamma_ps_inv, ec.CANONICAL.with_(E_donor_cm1=r.dE_cm1, E_acceptor_cm1=0.0))
    lind_us = 1e6 * (time.time() - t0) / 300

    # ---------------- B) HEOM ----------------
    H = pd.read_csv(DATA / "heom_lhs.csv")
    H = H[H.heom_physical].reset_index(drop=True)
    XH = feats(H)
    YH = np.column_stack([np.log(H.mean_t_heom_ps), logit(H.eta_heom)])
    permH = rng.permutation(len(H)); teH, trH = permH[:200], permH[200:]
    rowsB, predsB = run_models(XH[trH], YH[trH], XH[teH], YH[teH], "HEOM", len(trH), rng)
    heom_us = 1e6 * H.heom_seconds.median()

    met = pd.DataFrame(rowsA + rowsB)
    met.to_csv(OUT / "emulator_metrics.csv", index=False)

    lines.append("Emülatör sonuçları (test kümesi; ⟨t⟩ için göreli hata, η için mutlak hata)")
    for _, m in met.iterrows():
        if m.target == "ln_t":
            s = (f"  {m.model:13s} ⟨t⟩: medyan %{100*m.median_rel_err:.2f}, %95'lik %{100*m.p95_rel_err:.2f}, "
                 f"maks %{100*m.max_rel_err:.1f}")
        else:
            s = (f"  {m.model:13s} η  : medyan {m.median_abs_err:.2e}, %95'lik {m.p95_abs_err:.2e}, "
                 f"maks {m.max_abs_err:.2e}")
        if not np.isnan(m.get("coverage90", np.nan)):
            s += f"; %90 aralık kapsaması %{100*m.coverage90:.1f}"
        s += f"  (n_eğitim = {int(m.n_train)}, tahmin {m.predict_us_per_point:.0f} µs/nokta)"
        lines.append(s)
    lines.append(f"  Doğrudan hesap süresi: Lindblad ≈ {lind_us:.0f} µs/nokta, HEOM (derinlik 11) ≈ {heom_us/1e3:.0f} ms/nokta (medyan)")

    # ---------------- C) S(r) uygulaması ----------------
    gpH = predsB["gp_model"]
    v2 = pd.read_csv(ROOT / "results" / "tables" / "enaqt_gain_by_r_v2.csv")
    v2 = v2[v2.model == "HEOM"]
    r_grid = v2.r_nm.values
    g_grid = np.logspace(np.log10(0.1), np.log10(40.0), 200)
    rowsC = []
    for r in r_grid:
        Xg = np.column_stack([np.full_like(g_grid, np.log(r)), np.full_like(g_grid, 0.025), np.log(g_grid)])
        mu, sd = gpH.predict(Xg, return_std=True)
        k = np.argmin(mu)
        t0 = ec.lindblad_metrics(r, 0.0, ec.CANONICAL)[1]         # γ = 0: banyo yok, kesin
        rowsC.append(dict(r_nm=r, S_emulator=t0 / np.exp(mu[k]), gamma_opt_emulator=g_grid[k],
                          t_min_emulator_ps=np.exp(mu[k]), t_min_sd_rel=sd[k]))
    C = pd.DataFrame(rowsC).merge(v2[["r_nm", "S", "gamma_opt_ps_inv", "t_min_ps"]], on="r_nm")
    C = C.rename(columns={"S": "S_direct_v2", "gamma_opt_ps_inv": "gamma_opt_direct_v2", "t_min_ps": "t_min_direct_v2_ps"})
    C.to_csv(OUT / "S_of_r_emulator_vs_direct.csv", index=False)
    relS = np.abs(C.S_emulator / C.S_direct_v2 - 1)
    jE, jD = C.S_emulator.idxmax(), C.S_direct_v2.idxmax()
    lines += ["",
              "Uygulama: kanonik model (ΔE = 25 cm⁻¹) için S(r), HEOM-GP emülatörü vs v2 doğrudan HEOM taraması",
              f"  S göreli farkı: medyan %{100*np.median(relS):.2f}, maks %{100*relS.max():.2f} (41 r noktası)",
              f"  S_max: emülatör {C.S_emulator[jE]:.2f} @ r = {C.r_nm[jE]:.2f} nm; doğrudan {C.S_direct_v2[jD]:.2f} @ r = {C.r_nm[jD]:.2f} nm",
              "  (Emülatör ΔE ∈ [0, 1000] cm⁻¹ genelinde eğitildi; ΔE = 25'e özel eğitilmedi.)"]
    txt = "\n".join(lines)
    (OUT / "emulator_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)

    # ---------------- Şekiller ----------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=200)
    for ax, (Xte_y, preds, title) in zip(axes[:2], [(Y[te, 0], predsA, "Lindblad (test)"),
                                                    (YH[teH, 0], predsB, "HEOM γ ≤ 40 (test)")]):
        mu_g, sd_g = preds[("ln_t", "GP")]
        mu_m, _ = preds[("ln_t", "MLP")]
        lim = [np.exp(Xte_y).min() * 0.8, np.exp(Xte_y).max() * 1.2]
        ax.plot(lim, lim, color="0.6", lw=1)
        ax.scatter(np.exp(Xte_y), np.exp(mu_m), s=4, color="#eb6834", alpha=0.5, label="MLP", rasterized=True)
        ax.scatter(np.exp(Xte_y), np.exp(mu_g), s=4, color="#2a78d6", alpha=0.5, label="GP", rasterized=True)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator([2, 5, 10, 20, 50]))
            axis.set_major_formatter(FormatStrFormatter("%g"))
            axis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("⟨t⟩ doğrudan hesap (ps)"); ax.set_ylabel("⟨t⟩ emülatör (ps)")
        ax.set_title(title, fontsize=10); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25)
    ax = axes[2]
    ax.plot(C.r_nm, C.S_direct_v2, "-", color="0.25", lw=2, label="doğrudan HEOM (v2)")
    ax.plot(C.r_nm, C.S_emulator, "o", color="#2a78d6", ms=3.5, label="HEOM-GP emülatörü")
    ax.axhline(1, color="0.7", lw=1)
    ax.set_xlabel("r (nm)"); ax.set_ylabel("ENAQT kazancı S(r)")
    ax.set_title("Uygulama: S(r), ΔE = 25 cm⁻¹", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "emulator_parity.png"); plt.close(fig)


if __name__ == "__main__":
    main()
