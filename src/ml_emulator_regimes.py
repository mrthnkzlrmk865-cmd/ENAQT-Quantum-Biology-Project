#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml_emulator_regimes.py — Faz 2A: emülatör hatası NEREDE? Hedefli örnekleme işe yarar mı?
=====================================================================================
ml_emulator.py'nin genel sonuçları iyi görünse de (medyan hata < %1), en büyük
hatalar rezonansa yakın, zayıf dephasing'li köşede toplanıyor:
    "koherent köşe" = ΔE < 50 cm⁻¹ ve γ < 1 ps⁻¹
Bu köşe, ENAQT açısından en ilginç rejimdir; ⟨t⟩ burada V ve ΔE'ye keskin
biçimde bağlıdır. Ayrıca GP'nin σ tahmini burada aşırı özgüvenli olabilir.

Deney (Lindblad, hedef ln⟨t⟩, aynı GP, aynı EĞİTİM BÜTÇESİ = 2000 nokta):
  (i)  rastgele: ml_dataset.py LHS kümesinden 2000 nokta
  (ii) hedefli : 1500 LHS noktası + 500 yeni koherent-köşe noktası
Değerlendirme: (a) genel test (LHS'den 4000 nokta), (b) yeni, bağımsız 1000
koherent-köşe test noktası. Lindblad ucuz olduğu için köşe noktaları burada
doğrudan hesaplanır.

Çıktı: results/ml/emulator/emulator_regimes.csv, emulator_regimes_summary.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

import enaqt_core as ec
from ml_emulator import feats, make_gp, Z90

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "ml" / "data"
OUT = ROOT / "results" / "ml" / "emulator"
OUT.mkdir(parents=True, exist_ok=True)

CORNER = dict(r=(0.8, 8.0), dE=(0.0, 50.0), g=(0.1, 1.0))


def corner_points(n, seed):
    u = qmc.LatinHypercube(d=3, seed=seed).random(n)
    r = np.exp(np.log(CORNER["r"][0]) + u[:, 0] * np.log(CORNER["r"][1] / CORNER["r"][0]))
    dE = CORNER["dE"][0] + u[:, 1] * (CORNER["dE"][1] - CORNER["dE"][0])
    g = np.exp(np.log(CORNER["g"][0]) + u[:, 2] * np.log(CORNER["g"][1] / CORNER["g"][0]))
    t = [ec.lindblad_metrics(a, c, ec.CANONICAL.with_(E_donor_cm1=b, E_acceptor_cm1=0.0))[1] for a, b, c in zip(r, dE, g)]
    return pd.DataFrame(dict(r_nm=r, dE_cm1=dE, gamma_ps_inv=g, mean_t_ps=t))


def evaluate(gp, df):
    mu, sd = gp.predict(feats(df), return_std=True)
    y = np.log(df.mean_t_ps.values)
    rel = np.abs(np.exp(mu - y) - 1)
    return dict(median_rel_err=np.median(rel), p95_rel_err=np.quantile(rel, 0.95), max_rel_err=rel.max(),
                coverage90=np.mean(np.abs(y - mu) <= Z90 * sd))


def main():
    L = pd.read_csv(DATA / "lindblad_lhs.csv")
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(L))
    test_all = L.iloc[perm[:4000]]
    pool = L.iloc[perm[4000:]]
    in_corner = (test_all.dE_cm1 < 50) & (test_all.gamma_ps_inv < 1)
    corner_test = corner_points(1000, seed=99)
    corner_train = corner_points(500, seed=98)

    rows = []
    for name, train in [("rastgele 2000", pool.sample(2000, random_state=1)),
                        ("hedefli 1500 + 500 köşe", pd.concat([pool.sample(1500, random_state=1), corner_train]))]:
        gp = make_gp().fit(feats(train), np.log(train.mean_t_ps.values))
        for ev_name, ev in [("genel test (4000)", test_all), ("genel testin köşe kısmı", test_all[in_corner]),
                            ("bağımsız köşe testi (1000)", corner_test)]:
            rows.append(dict(egitim=name, degerlendirme=ev_name, n=len(ev), **evaluate(gp, ev)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "emulator_regimes.csv", index=False)
    lines = ["Lindblad-GP emülatörü: rejime göre hata ve hedefli örnekleme (eğitim bütçesi 2000 nokta)",
             f"Koherent köşe: ΔE < 50 cm⁻¹, γ < 1 ps⁻¹ (girdi uzayı hacminin ~%{100*np.mean(in_corner):.0f}'ü)", ""]
    for r in df.itertuples():
        lines.append(f"  [{r.egitim:24s}] {r.degerlendirme:28s} n={r.n:4d}: medyan %{100*r.median_rel_err:.2f}, "
                     f"%95'lik %{100*r.p95_rel_err:.1f}, maks %{100*r.max_rel_err:.0f}, %90 kapsama %{100*r.coverage90:.0f}")
    lines += ["", "Yorum: Genel ortalama hata, ilginç rejimdeki hatayı gizleyebilir. Emülatörün güvenilirliği",
              "rejim bazında raporlanmalı; az bilgilendirici bölgelere değil, zor bölgelere simülasyon harcanmalıdır."]
    txt = "\n".join(lines)
    (OUT / "emulator_regimes_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
