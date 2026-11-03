#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lindblad_heom_mapping_check.py — γ ↔ λ eşleştirmesinin kontrolü
================================================================
Orijinal kod Lindblad dephasing hızı γ ile HEOM yeniden düzenlenme enerjisi λ'yı
    λ = γ γ_c / (2 k_B T)
ile eşleştirir ve iki modelin aynı x-ekseninde karşılaştırılabileceğini
varsayar. Bu script, D–A koheransının sönüm hızını iki modelde ölçer:

  * Lindblad: sqrt(γ)|D><D| + sqrt(γ)|A><A|  ->  koherans γ hızıyla söner.
  * HEOM, Markov limiti (hızlı banyo, γ_c = 20 rad/ps): beklenen
        Γ_site = Re ∫ C(t) dt = Σ_k c_k^R / ν_k ≈ 2 λ k_B T / γ_c = γ  (her site)
    ve iki bağımsız site banyosu -> toplam ≈ 2γ.

Yani bu eşleştirmeyle HEOM'un γ ekseni, Lindblad'a göre ~2 kat "daha güçlü"
dephasing anlamına gelir. Lindblad–HEOM optimum kaymasının bir kısmı bu
ölçek farkından gelir; kalan kısım banyo hafızasıdır.

Test: V = 0 (transfer yok), sink/loss/gevşeme kapalı, başlangıç |+> = (|D>+|A>)/√2.
|ρ_DA(t)| üstel uydurma ile sönüm hızı ölçülür.

Çıktı: results/tables/mapping_check.txt
"""
from pathlib import Path
import numpy as np
from scipy.linalg import expm

import enaqt_core as ec

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)

P0 = ec.CANONICAL.with_(V0_cm1=0.0, gamma_relax_psinv=1e-12, gamma_sink_psinv=0.0,
                        gamma_loss_psinv=0.0)
IDX_DA = ec.vec_index(ec.D, ec.A)


def decay_rate(M, nA, t_fit=(1.0, 4.0), n=200):
    x0 = np.zeros(M.shape[0], dtype=complex)
    x0[:4] = 0.5                                  # |+><+| (tüm elemanlar 1/2)
    t = np.linspace(0, t_fit[1], n)
    U = expm(M.toarray() * (t[1] - t[0])) if hasattr(M, "toarray") else expm(M * (t[1] - t[0]))
    x, coh = x0.copy(), []
    for _ in t:
        coh.append(abs(x[IDX_DA])); x = U @ x
    coh = np.array(coh)
    m = t >= t_fit[0]
    return -np.polyfit(t[m], np.log(coh[m]), 1)[0]


def main():
    lines = ["γ ↔ λ eşleştirme kontrolü (V=0, D–A koherans sönüm hızı, ps^-1)",
             "γ_c (rad/ps) | γ (ps^-1) | Lindblad | HEOM (derinlik 7)  | HEOM/Lindblad | analitik Σ c_R/ν × 2 sites"]
    T = ec.kT_radps(P0)
    for gc in (20.0, 2.0):
        for g in (0.5, 1.0):
            L = ec.system_liouvillian(1.0, P0, gamma_deph_psinv=g)
            kL = decay_rate(L, 1)
            M, nA = ec.heom_generator(1.0, g, P0, gamma_c_radps=gc, depth=7, Nk=1)
            kH = decay_rate(M, nA)
            lam = g * gc / (2 * T)
            ck, vk = ec.drude_lorentz_exponents(lam, gc, T, 50)
            analytic = 2 * np.sum(ck.real / vk)
            lines.append(f"  {gc:6.1f}     | {g:5.2f}     | {kL:7.3f}  | {kH:9.3f}          | {kH/kL:5.2f}        | {analytic:6.3f}")
    lines.append("Not: γ_c = 2 rad/ps (orijinal ayar) yavaş banyodur; sönüm üstel değildir,")
    lines.append("bu satırlardaki 'hız' yalnızca 1–4 ps aralığındaki etkin eğimdir.")
    txt = "\n".join(lines)
    (TAB / "mapping_check.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
