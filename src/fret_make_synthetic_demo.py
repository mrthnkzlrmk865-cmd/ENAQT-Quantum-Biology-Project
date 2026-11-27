#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fret_make_synthetic_demo.py — Ön-kayıt → karşılaştırma zincirini denemek için
SENTETİK (uydurma değil, açıkça yapay) ölçüm dosyası üretir.

BU GERÇEK BİR ÖLÇÜM DEĞİLDİR. Amaç, araçların uçtan uca çalıştığını ve modelin
kendi varsayımlarına UYMAYAN bir "gerçekliği" fark edip edemediğini göstermektir.

Gizli "gerçeklik" (modelde OLMAYAN bir etki bilerek eklendi):
  * R0_gerçek = 5.4 nm × 1.03, d0 = 0.8 nm, boya yayılımı 0.6 nm (model ile aynı aile)
  * n ≥ 10 olan poliprolinlerde moleküllerin %25'i bir cis-prolin kıvrımı
    nedeniyle %65 uzunlukta (Best ve ark. 2007'deki etkinin niteliksel bir
    taklidi; sayılar keyfîdir) → uzun yapılarda E beklenenden YÜKSEK
  * ölçüm gürültüsü σ = 0.02, her yapı 3 tekrar
Çıktı: data/lab/ornek/SENTETIK_olcumler.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

import fret_bayes as fb

ROOT = Path(__file__).resolve().parents[1]
CONS = pd.read_csv(ROOT / "data" / "lab" / "ornek" / "ORNEK_yapilar.csv")
R0_TRUE, D0, SIGMA, KINK_FRAC, KINK_SCALE = 5.4 * 1.03, 0.8, 0.02, 0.25, 0.65


def true_mean_E(n):
    mu = D0 + fb.RISE_PPII_NM * n
    e_straight = fb.mean_efficiency(np.array([mu]), np.array([fb.S_DYE_NM]), np.array([R0_TRUE]))[0]
    if n < 10:
        return e_straight
    e_kink = fb.mean_efficiency(np.array([D0 + KINK_SCALE * fb.RISE_PPII_NM * n]), np.array([fb.S_DYE_NM]),
                                np.array([R0_TRUE]))[0]
    return (1 - KINK_FRAC) * e_straight + KINK_FRAC * e_kink


def main():
    rng = np.random.default_rng(42)
    rows = []
    for c in CONS.itertuples():
        m = true_mean_E(c.n)
        for j in range(1, int(c.n_rep) + 1):
            rows.append(dict(construct=c.name, replicate=j, E=round(m + SIGMA * rng.normal(), 4),
                             notes="SENTETİK - gerçek ölçüm değil"))
    df = pd.DataFrame(rows)
    out = ROOT / "data" / "lab" / "ornek" / "SENTETIK_olcumler.csv"
    df.to_csv(out, index=False)
    print(df.groupby("construct", sort=False).E.mean().round(3).to_string())
    print("gizli gerçek ortalamalar:", {c.name: round(true_mean_E(c.n), 3) for c in CONS.itertuples()})


if __name__ == "__main__":
    main()
