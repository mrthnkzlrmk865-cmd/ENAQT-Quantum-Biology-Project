#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fret_calibration_test.py — Faz 2B: Bayesçi FRET modelinin sahte deneylerle testi
==============================================================================
Soru: Model, gerçek değeri BİLDİĞİMİZ sahte deneylerde dürüst belirsizlik
veriyor mu? (Gerçek deneyden önce yapılması gereken kontrol.)

Her sahte deney k = 1..K için:
  1. θ_true öncelden çekilir (R0, δ, σ, d0).
  2. Örnek yapılar (Pro6, Pro12, Pro20; her biri 3 tekrar) için veri üretilir.
  3. Model MCMC ile uydurulur.
  4. Kaydedilir: her parametre için %90 aralığın θ_true'yu kapsayıp kapsamadığı,
     simülasyon-tabanlı kalibrasyon (SBC; Talts ve ark. 2018) sıralaması,
     posterior daralması (posterior sd / öncel sd), ve uydurmaya katılmayan
     bir yapının (Pro15) ortalama E'si ile tek bir yeni ölçümü için %90
     tahmin aralığı kapsaması.

İyi kalibre bir modelde kapsama oranları ≈ %90, SBC sıralamaları ≈ düzgün olur.

UYARI: Yapılar ve R0 = 5.4 nm (Alexa 488/594 için Schuler ve ark. 2005'te
kullanılan değer) yalnızca ÖRNEKTİR. Bu test, modelin KENDİ varsayımları doğruyken
iyi kalibre olup olmadığını gösterir; gerçek yapının modele uyup uymadığını göstermez.

Çıktı: results/fret/calibration/calibration_runs.csv, calibration_summary.txt,
       results/fret/calibration/calibration_sbc.png
Çalıştırma: python src/fret_calibration_test.py          (~15–25 dk, 2 çekirdek)
            python src/fret_calibration_test.py --quick  (K = 8)
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fret_bayes as fb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fret" / "calibration"
OUT.mkdir(parents=True, exist_ok=True)

QUICK = "--quick" in sys.argv
K = 8 if QUICK else 120
R0_LIT = 5.4
CONSTRUCTS = [dict(name=f"Pro{n}", family="poliprolin", n=n) for n in (6, 12, 20)]
HELDOUT = [dict(name="Pro15", family="poliprolin", n=15)]
N_REP = 3
N_STEPS, BURN, WALKERS = 4000, 1000, 32
SBC_L = 99                 # SBC için seyreltilmiş posterior çekiliş sayısı


def model():
    return fb.FretModel(CONSTRUCTS, fb.default_priors(R0_LIT))


def one(k):
    m = model()
    rng = np.random.default_rng(1000 + k)
    th = m.sample_prior(rng, 1)
    tdict = dict(zip(m.free, th[0]))
    idx = np.repeat(np.arange(len(CONSTRUCTS)), N_REP)
    mean = m.construct_means(th)[0] + tdict["delta"]
    E = mean[idx] + tdict["sigma"] * rng.normal(size=idx.size)

    res = fb.fit(m, idx, E, n_walkers=WALKERS, n_steps=N_STEPS, burn=BURN, seed=k)
    S = res.samples
    row = dict(k=k, acc=res.acc, converged=res.converged(), max_rhat=res.rhat.max(),
               min_steps_over_tau=((N_STEPS - BURN) / res.tau).min())
    thin = S[np.linspace(0, len(S) - 1, SBC_L).astype(int)]
    prior_sd = m.sample_prior(np.random.default_rng(0), 20000).std(axis=0)
    for i, name in enumerate(m.free):
        lo, hi = np.quantile(S[:, i], [0.05, 0.95])
        row[f"{name}_true"] = tdict[name]
        row[f"{name}_med"] = np.median(S[:, i])
        row[f"{name}_cover90"] = lo <= tdict[name] <= hi
        row[f"{name}_sbc_rank"] = int((thin[:, i] < tdict[name]).sum())
        row[f"{name}_contraction"] = S[:, i].std() / prior_sd[i]

    # uydurmaya katılmayan yapı için tahmin
    sub = S[rng.choice(len(S), 4000, replace=False)]
    true_ho = fb.predict(m, th, HELDOUT)[0, 0]
    pred_mean = fb.predict(m, sub, HELDOUT)[:, 0]
    pred_obs = fb.predict(m, sub, HELDOUT, rng=rng, with_noise=True)[:, 0]
    new_obs = true_ho + tdict["sigma"] * rng.normal()
    lo, hi = np.quantile(pred_mean, [0.05, 0.95])
    row.update(heldout_true_mean=true_ho, heldout_pred_med=np.median(pred_mean),
               heldout_cover90_mean=lo <= true_ho <= hi,
               heldout_width90=hi - lo)
    lo, hi = np.quantile(pred_obs, [0.05, 0.95])
    row.update(heldout_cover90_obs=lo <= new_obs <= hi)
    return row


def main():
    with Pool(os.cpu_count() or 1) as pool:
        df = pd.DataFrame(pool.map(one, range(K)))
    df.to_csv(OUT / "calibration_runs.csv", index=False)

    names = model().free
    n = len(df)
    se = np.sqrt(0.9 * 0.1 / n)
    lines = [f"Bayesçi FRET modeli — sahte deney kalibrasyon testi (K = {n}; örnek yapılar Pro6/12/20 × {N_REP} tekrar)",
             f"Yakınsayan uydurma: {df.converged.sum()}/{n}; maks R̂ medyanı {df.max_rhat.median():.3f}; "
             f"min (adım/τ) medyanı {df.min_steps_over_tau.median():.0f}",
             f"Beklenen kapsama %90 ± {100*1.96*se:.1f} (K = {n} için %95 bandı)",
             "", "Parametre  | %90 kapsama | daralma (post sd / öncel sd, medyan) | SBC sıralama χ² p-değeri"]
    from scipy.stats import chisquare
    for name in names:
        ranks = df[f"{name}_sbc_rank"].values
        counts = np.histogram(ranks, bins=10, range=(0, SBC_L + 1))[0]
        p = chisquare(counts).pvalue
        lines.append(f"  {name:8s} |   %{100*df[f'{name}_cover90'].mean():5.1f}    |   {df[f'{name}_contraction'].median():.2f}"
                     f"                               |  {p:.2f}")
    lines += ["",
              f"Dışarıda bırakılan yapı (Pro15): ortalama E için %90 aralık kapsaması %{100*df.heldout_cover90_mean.mean():.1f}, "
              f"tek yeni ölçüm için %{100*df.heldout_cover90_obs.mean():.1f}; aralık genişliği medyanı {df.heldout_width90.median():.3f}",
              f"Pro15 ortalama E mutlak hata (posterior medyan): medyan {np.median(np.abs(df.heldout_pred_med - df.heldout_true_mean)):.3f}",
              "",
              "Yorum: daralma ≈ 1 olan parametreyi veri neredeyse hiç öğretmiyor (öncel baskın).",
              "Bu test yalnızca model varsayımları doğruyken kalibrasyonu gösterir; model hatasını (ör. poliprolin",
              "cis-kıvrımları, κ²) test etmez — onu deneyde δ ve dışarıda-bırakma doğrulaması gösterecek."]
    txt = "\n".join(lines)
    (OUT / "calibration_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)

    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 2.8), dpi=200, sharey=True)
    for ax, name in zip(axes, names):
        ax.hist(df[f"{name}_sbc_rank"], bins=10, range=(0, SBC_L + 1), color="#2a78d6", edgecolor="white")
        ax.axhline(n / 10, color="0.4", lw=1, ls="--")
        ax.set_title(f"{name}  (kapsama %{100*df[f'{name}_cover90'].mean():.0f})", fontsize=9)
        ax.set_xlabel("SBC sıralaması")
    axes[0].set_ylabel("sahte deney sayısı")
    fig.suptitle("Simülasyon-tabanlı kalibrasyon: düzgün histogram = dürüst belirsizlik", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "calibration_sbc.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
