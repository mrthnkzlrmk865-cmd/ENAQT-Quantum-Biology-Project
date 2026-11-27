#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml_dataset.py — Faz 1A: in silico sentetik veri seti
====================================================
ML modellerini (emülatör, Δ-öğrenme) eğitmek için Lindblad ve HEOM
sonuçlarından tekrarlanabilir bir veri seti üretir.

Girdi uzayı (Latin hiperküp örnekleme, sabit tohum):
    r   ∈ [0.8, 8]   nm      (log-düzgün)
    ΔE  ∈ [0, 1000]  cm^-1   (düzgün)          E_D = ΔE, E_A = 0
    γ   ∈ [0.1, 200] ps^-1   (log-düzgün)  — Lindblad
    γ   ∈ [0.1, 40]  ps^-1   (log-düzgün)  — HEOM (v2'de γ > 40 güvenilmez bulundu)
Diğer parametreler v2 kanonik modeldeki gibidir (V0 = 20 cm^-1 @ 2 nm,
Γ_sink = 1, Γ_loss = 0.001, gevşeme 0.02 ps^-1, T = 298 K).

Çıktılar (η ve ⟨t⟩ sonsuz zamana kadar analitik, enaqt_core):
  results/ml/data/lindblad_lhs.csv     (N_LINDBLAD nokta)
  results/ml/data/heom_lhs.csv         (N_HEOM nokta; aynı noktalarda Lindblad γ ve 2γ de var)
  results/ml/data/heom_depth_check.csv (alt kümede derinlik 9 vs 11 vs 13)
  results/ml/data/DATA_CARD.md         (veri kartı: aralıklar, tohum, süre, uyarılar)

Çalıştırma: python src/ml_dataset.py           (~10–20 dk, 2 çekirdek)
            python src/ml_dataset.py --quick   (küçük sürüm, test için)
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

import enaqt_core as ec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ml" / "data"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20261127
QUICK = "--quick" in sys.argv
N_LINDBLAD = 800 if QUICK else 20000
N_HEOM = 40 if QUICK else 1000
N_DEPTH_CHECK = 6 if QUICK else 40
HEOM_DEPTH, HEOM_NK = 11, 1

R_RANGE = (0.8, 8.0)
DE_RANGE = (0.0, 1000.0)
G_RANGE_L = (0.1, 200.0)
G_RANGE_H = (0.1, 40.0)


def lhs(n, g_range, seed):
    u = qmc.LatinHypercube(d=3, seed=seed).random(n)
    r = np.exp(np.log(R_RANGE[0]) + u[:, 0] * np.log(R_RANGE[1] / R_RANGE[0]))
    dE = DE_RANGE[0] + u[:, 1] * (DE_RANGE[1] - DE_RANGE[0])
    g = np.exp(np.log(g_range[0]) + u[:, 2] * np.log(g_range[1] / g_range[0]))
    return r, dE, g


def params_for(dE):
    return ec.CANONICAL.with_(E_donor_cm1=float(dE), E_acceptor_cm1=0.0)


def _lindblad_row(args):
    r, dE, g = args
    p = params_for(dE)
    eta, t = ec.lindblad_metrics(r, g, p)
    eta0, t0 = ec.lindblad_metrics(r, 0.0, p)
    return dict(r_nm=r, dE_cm1=dE, gamma_ps_inv=g, eta=eta, mean_t_ps=t,
                eta_gamma0=eta0, mean_t_gamma0_ps=t0)


def _heom_row(args):
    r, dE, g = args
    p = params_for(dE)
    t_start = time.time()
    eta_h, t_h = ec.heom_metrics(r, g, p, depth=HEOM_DEPTH, Nk=HEOM_NK)
    sec = time.time() - t_start
    eta_l, t_l = ec.lindblad_metrics(r, g, p)
    eta_l2, t_l2 = ec.lindblad_metrics(r, 2 * g, p)
    return dict(r_nm=r, dE_cm1=dE, gamma_ps_inv=g,
                eta_heom=eta_h, mean_t_heom_ps=t_h,
                eta_lind=eta_l, mean_t_lind_ps=t_l,
                eta_lind_2g=eta_l2, mean_t_lind_2g_ps=t_l2,
                heom_seconds=sec)


def _depth_row(args):
    r, dE, g = args
    p = params_for(dE)
    out = dict(r_nm=r, dE_cm1=dE, gamma_ps_inv=g)
    for d in (9, 11, 13):
        e, t = ec.heom_metrics(r, g, p, depth=d, Nk=HEOM_NK)
        out[f"eta_d{d}"], out[f"t_d{d}"] = e, t
    return out


def main():
    workers = os.cpu_count() or 1
    t0 = time.time()

    rL, dEL, gL = lhs(N_LINDBLAD, G_RANGE_L, SEED)
    with Pool(workers) as pool:
        lind = pd.DataFrame(pool.map(_lindblad_row, zip(rL, dEL, gL), chunksize=200))
    lind.to_csv(OUT / "lindblad_lhs.csv", index=False)
    tL = time.time() - t0
    print(f"Lindblad: {len(lind)} nokta, {tL:.1f} s", flush=True)

    rH, dEH, gH = lhs(N_HEOM, G_RANGE_H, SEED + 1)
    with Pool(workers) as pool:
        heom = pd.DataFrame(pool.map(_heom_row, zip(rH, dEH, gH), chunksize=4))
    heom["heom_physical"] = heom.eta_heom.between(0.0, 1.0)
    heom.to_csv(OUT / "heom_lhs.csv", index=False)
    tH = time.time() - t0 - tL
    print(f"HEOM: {len(heom)} nokta, {tH:.1f} s; fiziksel olmayan η: {int((~heom.heom_physical).sum())}", flush=True)

    idx = np.random.default_rng(SEED + 2).choice(N_HEOM, size=N_DEPTH_CHECK, replace=False)
    with Pool(workers) as pool:
        dep = pd.DataFrame(pool.map(_depth_row, zip(rH[idx], dEH[idx], gH[idx]), chunksize=1))
    dep["rel_t_d11_vs_d13"] = (dep.t_d11 / dep.t_d13 - 1).abs()
    dep["rel_t_d9_vs_d13"] = (dep.t_d9 / dep.t_d13 - 1).abs()
    dep.to_csv(OUT / "heom_depth_check.csv", index=False)
    tD = time.time() - t0 - tL - tH

    card = {
        "seed": SEED, "quick": QUICK,
        "n_lindblad": len(lind), "n_heom": len(heom), "n_depth_check": len(dep),
        "r_nm": R_RANGE, "dE_cm1": DE_RANGE,
        "gamma_lindblad_ps_inv": G_RANGE_L, "gamma_heom_ps_inv": G_RANGE_H,
        "heom_depth": HEOM_DEPTH, "heom_Nk": HEOM_NK,
        "seconds": {"lindblad": round(tL, 1), "heom": round(tH, 1), "depth_check": round(tD, 1)},
        "depth_check_max_rel_t_d11_vs_d13": float(dep.rel_t_d11_vs_d13.max()),
        "depth_check_median_rel_t_d11_vs_d13": float(dep.rel_t_d11_vs_d13.median()),
        "heom_nonphysical_eta": int((~heom.heom_physical).sum()),
    }
    (OUT / "data_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    md = f"""# Veri kartı — in silico ML veri seti (Faz 1A)

Üreten: `src/ml_dataset.py` · tohum `{SEED}` · {'HIZLI TEST SÜRÜMÜ' if QUICK else 'tam sürüm'}

| Dosya | Satır | İçerik |
|---|---|---|
| `lindblad_lhs.csv` | {len(lind)} | Lindblad η, ⟨t⟩ ve aynı noktada γ = 0 referansı |
| `heom_lhs.csv` | {len(heom)} | HEOM (derinlik {HEOM_DEPTH}, Nk = {HEOM_NK}) + aynı noktada Lindblad(γ) ve Lindblad(2γ) |
| `heom_depth_check.csv` | {len(dep)} | Rastgele alt kümede derinlik 9 / 11 / 13 karşılaştırması |

**Girdi aralıkları:** r ∈ {R_RANGE} nm (log-düzgün), ΔE ∈ {DE_RANGE} cm⁻¹ (düzgün),
γ ∈ {G_RANGE_L} ps⁻¹ Lindblad / {G_RANGE_H} ps⁻¹ HEOM (log-düzgün). Latin hiperküp.

**Sabit parametreler:** V0 = 20 cm⁻¹ @ r0 = 2 nm (V ∝ r⁻³), Γ_sink = 1 ps⁻¹, Γ_loss = 0.001 ps⁻¹,
gevşeme 0.02 ps⁻¹ (detailed balance), T = 298 K; HEOM: γ_c = 2 rad/ps, λ = γγ_c/(2k_BT).

**Yakınsama kontrolü (alt küme):** derinlik 11 vs 13, ⟨t⟩ göreli farkı medyan
{card['depth_check_median_rel_t_d11_vs_d13']:.2e}, maksimum {card['depth_check_max_rel_t_d11_vs_d13']:.2e}.
Fiziksel olmayan (η ∉ [0, 1]) HEOM noktası: {card['heom_nonphysical_eta']}.

**Uyarılar:**
- Bu bir **oyuncak model** veri setidir; gerçek bir kromofor çiftinin parametreleri değildir.
- HEOM yakınsaması v2'de yalnızca ΔE = 25 cm⁻¹ için test edilmişti; burada geniş ΔE aralığı için
  yalnızca yukarıdaki alt küme kontrolü vardır.
- V ∝ r⁻³ nokta-dipol yaklaşımı r < ~2 nm'de fiziksel olarak güvenilmezdir (sayısal olarak sorun yok).
- ⟨t⟩'nin mutlak değeri Γ_sink seçimine bağlıdır (v2 bulgusu: min ⟨t⟩ ≈ 2/Γ_sink).
- Modeldeki D→A gevşeme kanalı (0.02 ps⁻¹) **r'den bağımsızdır**. Büyük r'de kaplin çok zayıfladığında
  transferi bu kanal belirler: ⟨t⟩ yaklaşık 1/0.02 + 1/Γ_sink ≈ 51 ps'de doyar, η ise ≈ 0.02/(0.02+0.001) ≈ 0.95'in
  altına inmez. Bu, fiziksel bir mesafe yasası değil, modelin bir varsayımıdır.

Süre: Lindblad {tL:.0f} s, HEOM {tH:.0f} s, derinlik kontrolü {tD:.0f} s ({workers} çekirdek).
"""
    (OUT / "DATA_CARD.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
