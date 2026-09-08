"""
=======================================================================
FAZ 2 - ADIM 2.3: HEOM ile Non-Markovian Doğrulama
=======================================================================
Modül: heom_nonmarkovian_validation.py

AMAÇ: Adım 2.2'deki Lindblad (Markovian, banyo hafızasız - tau_c -> 0)
modelinin varsayımını gevşetip, banyonun SONLU bir korelasyon süresine
(tau_c = 1/gamma_c) sahip olduğu gerçekçi durumu HEOM (Hierarchical
Equations of Motion) ile simüle ediyoruz. Soru: banyo hafızası ENAQT
sweet spot'unu KAYDIRIYOR MU, yoksa tutarlılığı koruyarak transferi
DAHA DA MI hızlandırıyor?

-----------------------------------------------------------------------
(1) FİZİKSEL ÇERÇEVE: DRUDE-LORENTZ SPEKTRAL YOĞUNLUĞU
-----------------------------------------------------------------------
    J(omega) = 2 * lambda * gamma_c * omega / (omega^2 + gamma_c^2)

    - lambda   : reorganizasyon enerjisi (banyo-sistem kaplin gücü)
    - gamma_c  : banyo kesim frekansı = 1/tau_c (tau_c: banyo hafıza/
                 korelasyon süresi)
    - gamma_c -> sonsuz (tau_c -> 0)  ==>  Markovian limit (Lindblad'a
      indirgenir, banyonun "hafızası yok", her an bağımsız gürültü)
    - gamma_c sonlu  ==>  Non-Markovian (banyonun hafızası var, geçmiş
      etkileşimler geleceği etkiler)

-----------------------------------------------------------------------
(2) ADİL KARŞILAŞTIRMA İÇİN HAKEN-STROBL EŞLEŞTİRMESİ
-----------------------------------------------------------------------
    Adım 2.2'de taranan fenomenolojik Lindblad dephasing hızı gamma
    ile, mikroskopik HEOM banyo parametreleri (lambda, gamma_c, T)
    arasında YÜKSEK SICAKLIK / HIZLI BANYO limitinde şu klasik
    (Haken-Strobl) ilişki geçerlidir:

        gamma_Lindblad_esdeğer = 2 * lambda * k_B*T / (hbar * gamma_c)

    Bu ilişkiyi TERSİNE çevirip, Adım 2.2'de taranan HER gamma değeri
    için "aynı Markovian limitte eşdeğer" bir lambda hesaplıyoruz:

        lambda(gamma) = gamma * gamma_c / (2 * k_B*T)

    Böylece gamma_c sabit tutulup (banyo hafızası sabit), lambda
    ayarlanarak Lindblad taramasındaki HER NOKTAYA "adil" bir HEOM
    karşılığı üretiyoruz -- iki modeli aynı eksen (gamma) üzerinde
    doğrudan overlay edebiliyoruz.

    NOT: Bu eşleştirme yalnızca gamma_c -> sonsuz limitinde TAM olarak
    örtüşür; gamma_c SONLU olduğu için (bizim asıl test etmek
    istediğimiz şey) iki eğri arasındaki fark, banyo hafızasının NET
    etkisidir.

-----------------------------------------------------------------------
(3) HESAPLAMA PERFORMANSI
-----------------------------------------------------------------------
    Yakınsama testi (bu modülü üretmeden önce yapıldı):
      max_depth=4, Nk=1  -> 1.6-1.8 s/nokta
      max_depth=6, Nk=2  -> 180+ s/nokta (referans/altın standart)
      Fark: <t> değerlerinde <%0.2 sapma (2.738 vs 2.735 ps gibi)
    Üretim ayarı: max_depth=5, Nk=1 (hız/doğruluk dengesi, ~3.6 s/nokta)
=======================================================================
"""

import numpy as np
import qutip as qt
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath
import matplotlib.pyplot as plt
import time

# --- Sabitler (önceki adımlarla tutarlı) ---
C_LIGHT_CM_PER_PS = 2.99792458e-2
CM1_TO_RADPS = 2 * np.pi * C_LIGHT_CM_PER_PS
KB_CM1_PER_K = 0.695034800

DONOR, ACCEPTOR, SINK, LOSS = 0, 1, 2, 3
DIM = 4

# Kanonik ENAQT sistem parametreleri (Adım 2.2 ile BİREBİR aynı)
E_DONOR_CM1 = 25.0
E_ACCEPTOR_CM1 = 0.0
V0_CM1 = 20.0
R0_NM = 2.0
GAMMA_RELAX_PSINV = 0.02
GAMMA_SINK_PSINV = 1.0
GAMMA_LOSS_PSINV = 0.001
T_KELVIN = 298.0

# HEOM banyo parametreleri
GAMMA_C_RADPS = 2.0     # banyo kesim frekansı (tau_c = 0.5 ps -- "orta ölçekli hafıza")
HEOM_MAX_DEPTH = 5
HEOM_NK = 1


def coupling_from_distance(r_nm, V0_cm1=V0_CM1, r0_nm=R0_NM):
    return V0_cm1 * (r0_nm / r_nm) ** 3


def build_liouvillian(r_nm):
    """
    Koherent Hamiltoniyen + SABİT Markovian kanallar (relaxation, sink,
    loss) -> tek bir Liouvillian olarak birleştirilir. Dephasing BURADA
    YOK -- o, HEOM'daki fonon banyosu tarafından (non-Markovian olarak)
    sağlanacak.
    """
    V_cm1 = coupling_from_distance(r_nm)
    H_cm1 = np.zeros((DIM, DIM), dtype=complex)
    H_cm1[DONOR, DONOR] = E_DONOR_CM1
    H_cm1[ACCEPTOR, ACCEPTOR] = E_ACCEPTOR_CM1
    H_cm1[DONOR, ACCEPTOR] = V_cm1
    H_cm1[ACCEPTOR, DONOR] = V_cm1
    H = qt.Qobj(H_cm1 * CM1_TO_RADPS)

    dE_cm1 = E_DONOR_CM1 - E_ACCEPTOR_CM1
    boltz = np.exp(-dE_cm1 / (KB_CM1_PER_K * T_KELVIN))

    ket_don, ket_acc = qt.basis(DIM, DONOR), qt.basis(DIM, ACCEPTOR)
    ket_sink, ket_loss = qt.basis(DIM, SINK), qt.basis(DIM, LOSS)

    c_ops = [
        np.sqrt(GAMMA_RELAX_PSINV) * (ket_acc * ket_don.dag()),
        np.sqrt(GAMMA_RELAX_PSINV * boltz) * (ket_don * ket_acc.dag()),
        np.sqrt(GAMMA_SINK_PSINV) * (ket_sink * ket_acc.dag()),
    ]
    for sk in [ket_don, ket_acc]:
        c_ops.append(np.sqrt(GAMMA_LOSS_PSINV) * (ket_loss * sk.dag()))

    return qt.liouvillian(H, c_ops)


def mean_trapping_time_from_populations(tlist, P_acceptor, gamma_sink=GAMMA_SINK_PSINV):
    """Adım 2.2 ile aynı tanım: <t> = integral(t*flux)dt / integral(flux)dt"""
    trapz = getattr(np, "trapezoid", None) or np.trapz
    flux = gamma_sink * P_acceptor
    eta = trapz(flux, tlist)
    if eta < 1e-9:
        return np.nan, eta
    mean_t = trapz(tlist * flux, tlist) / eta
    return mean_t, eta


# -----------------------------------------------------------------------
# LINDBLAD (MARKOVIAN) REFERANS ÇÖZÜCÜ -- Adım 2.2 mantığının kısaltılmışı
# -----------------------------------------------------------------------
def run_lindblad(gamma_dephasing_psinv, r_nm, t_max_ps=30.0, n_steps=600):
    V_cm1 = coupling_from_distance(r_nm)
    H_cm1 = np.zeros((DIM, DIM), dtype=complex)
    H_cm1[DONOR, DONOR] = E_DONOR_CM1
    H_cm1[ACCEPTOR, ACCEPTOR] = E_ACCEPTOR_CM1
    H_cm1[DONOR, ACCEPTOR] = V_cm1
    H_cm1[ACCEPTOR, DONOR] = V_cm1
    H = qt.Qobj(H_cm1 * CM1_TO_RADPS)

    dE_cm1 = E_DONOR_CM1 - E_ACCEPTOR_CM1
    boltz = np.exp(-dE_cm1 / (KB_CM1_PER_K * T_KELVIN))
    ket_don, ket_acc = qt.basis(DIM, DONOR), qt.basis(DIM, ACCEPTOR)
    ket_sink, ket_loss = qt.basis(DIM, SINK), qt.basis(DIM, LOSS)

    c_ops = []
    for site in [DONOR, ACCEPTOR]:
        proj = qt.Qobj(np.diag([1.0 if k == site else 0.0 for k in range(DIM)]))
        c_ops.append(np.sqrt(gamma_dephasing_psinv) * proj)
    c_ops += [
        np.sqrt(GAMMA_RELAX_PSINV) * (ket_acc * ket_don.dag()),
        np.sqrt(GAMMA_RELAX_PSINV * boltz) * (ket_don * ket_acc.dag()),
        np.sqrt(GAMMA_SINK_PSINV) * (ket_sink * ket_acc.dag()),
    ]
    for sk in [ket_don, ket_acc]:
        c_ops.append(np.sqrt(GAMMA_LOSS_PSINV) * (ket_loss * sk.dag()))

    rho0 = qt.ket2dm(qt.basis(DIM, DONOR))
    tlist = np.linspace(0, t_max_ps, n_steps)
    e_ops = [qt.Qobj(np.diag([1.0 if k == j else 0.0 for k in range(DIM)])) for j in range(DIM)]

    result = qt.mesolve(H, rho0, tlist, c_ops=c_ops, e_ops=e_ops)
    pops = np.array(result.expect)
    mean_t, eta = mean_trapping_time_from_populations(tlist, pops[ACCEPTOR])
    return mean_t, eta


# -----------------------------------------------------------------------
# HEOM (NON-MARKOVIAN) ÇÖZÜCÜ
# -----------------------------------------------------------------------
def run_heom(gamma_dephasing_equiv_psinv, r_nm, gamma_c_radps=GAMMA_C_RADPS,
             t_max_ps=30.0, n_steps=300, max_depth=HEOM_MAX_DEPTH, nk=HEOM_NK):
    """
    gamma_dephasing_equiv_psinv: Lindblad taramasındaki gamma değeri
    (Haken-Strobl eşleştirmesiyle lambda'ya çevrilecek).
    """
    kT_radps = KB_CM1_PER_K * T_KELVIN * CM1_TO_RADPS
    lam_radps = gamma_dephasing_equiv_psinv * gamma_c_radps / (2 * kT_radps)

    L = build_liouvillian(r_nm)

    proj_don = qt.Qobj(np.diag([1., 0, 0, 0]))
    proj_acc = qt.Qobj(np.diag([0, 1., 0, 0]))

    bath_don = DrudeLorentzBath(Q=proj_don, lam=lam_radps, gamma=gamma_c_radps, T=kT_radps, Nk=nk)
    bath_acc = DrudeLorentzBath(Q=proj_acc, lam=lam_radps, gamma=gamma_c_radps, T=kT_radps, Nk=nk)

    solver = HEOMSolver(L, [bath_don, bath_acc], max_depth=max_depth,
                         options={"progress_bar": False})

    rho0 = qt.ket2dm(qt.basis(DIM, DONOR))
    tlist = np.linspace(0, t_max_ps, n_steps)
    e_ops = [qt.Qobj(np.diag([1.0 if k == j else 0.0 for k in range(DIM)])) for j in range(DIM)]

    result = solver.run(rho0, tlist, e_ops=e_ops)
    pops = np.array([np.real(result.expect[j]) for j in range(DIM)])
    mean_t, eta = mean_trapping_time_from_populations(tlist, pops[ACCEPTOR])
    return mean_t, eta


# =========================================================================
# ANA ÇALIŞTIRMA: LINDBLAD vs HEOM KARŞILAŞTIRMA TARAMASI
# =========================================================================
if __name__ == "__main__":

    r_fixed = 2.0  # nm -- Adım 2.2'deki 1D kesitle karşılaştırılabilir referans mesafe
    gamma_range = np.logspace(np.log10(0.02), np.log10(80.0), 14)  # ps^-1

    print("=" * 70)
    print(f"ADIM 2.3: Lindblad vs HEOM Karşılaştırması (r = {r_fixed} nm sabit)")
    print(f"HEOM banyo: gamma_c = {GAMMA_C_RADPS} ps^-1 (tau_c = {1/GAMMA_C_RADPS:.2f} ps), "
          f"max_depth={HEOM_MAX_DEPTH}, Nk={HEOM_NK}")
    print("=" * 70)

    mean_t_lindblad = np.zeros_like(gamma_range)
    eta_lindblad = np.zeros_like(gamma_range)
    mean_t_heom = np.zeros_like(gamma_range)
    eta_heom = np.zeros_like(gamma_range)

    t0 = time.time()
    for i, g in enumerate(gamma_range):
        mt_l, eta_l = run_lindblad(g, r_fixed)
        mean_t_lindblad[i] = mt_l
        eta_lindblad[i] = eta_l

        mt_h, eta_h = run_heom(g, r_fixed)
        mean_t_heom[i] = mt_h
        eta_heom[i] = eta_h

        print(f"  [{i+1:2d}/{len(gamma_range)}] gamma={g:7.3f} ps^-1  |  "
              f"Lindblad <t>={mt_l:.4f} ps  |  HEOM <t>={mt_h:.4f} ps  |  "
              f"fark={mt_h-mt_l:+.4f} ps")

    elapsed = time.time() - t0
    print(f"\n[OK] Tarama tamamlandı: {len(gamma_range)} nokta x 2 model, {elapsed:.1f} s")

    # --- Sweet spot analizi ---
    idx_min_l = np.argmin(mean_t_lindblad)
    idx_min_h = np.argmin(mean_t_heom)
    gamma_opt_l, t_opt_l = gamma_range[idx_min_l], mean_t_lindblad[idx_min_l]
    gamma_opt_h, t_opt_h = gamma_range[idx_min_h], mean_t_heom[idx_min_h]

    print(f"\n>>> LINDBLAD sweet spot: gamma_opt = {gamma_opt_l:.3f} ps^-1, <t>_min = {t_opt_l:.4f} ps")
    print(f">>> HEOM     sweet spot: gamma_opt = {gamma_opt_h:.3f} ps^-1, <t>_min = {t_opt_h:.4f} ps")
    print(f">>> Sweet spot KAYMASI (gamma): {gamma_opt_h - gamma_opt_l:+.3f} ps^-1 "
          f"({100*(gamma_opt_h/gamma_opt_l - 1):+.1f}%)")
    print(f">>> Minimum <t> farkı: {t_opt_h - t_opt_l:+.4f} ps "
          f"({100*(t_opt_h/t_opt_l - 1):+.1f}%)")

    # --- Grafik: overlay karşılaştırma ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].plot(gamma_range, mean_t_lindblad, 'o-', color='steelblue', linewidth=2,
                 markersize=6, label='Lindblad (Markovian, τ_c→0)')
    axes[0].plot(gamma_range, mean_t_heom, 's-', color='crimson', linewidth=2,
                 markersize=6, label=f'HEOM (Non-Markovian, τ_c={1/GAMMA_C_RADPS:.1f} ps)')
    axes[0].plot(gamma_opt_l, t_opt_l, marker='*', color='steelblue', markersize=22,
                 markeredgecolor='black', zorder=5)
    axes[0].plot(gamma_opt_h, t_opt_h, marker='*', color='crimson', markersize=22,
                 markeredgecolor='black', zorder=5)
    axes[0].set_xscale('log')
    axes[0].set_xlabel(r"Dephasing hızı $\gamma$ (ps$^{-1}$, log ölçek)")
    axes[0].set_ylabel(r"Ortalama Yakalanma Süresi $\langle t \rangle$ (ps)")
    axes[0].set_title(f"Lindblad vs HEOM: ENAQT Sweet Spot (r={r_fixed} nm)")
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3, which='both')

    axes[1].plot(gamma_range, eta_lindblad, 'o-', color='steelblue', linewidth=2,
                 markersize=6, label='Lindblad')
    axes[1].plot(gamma_range, eta_heom, 's-', color='crimson', linewidth=2,
                 markersize=6, label='HEOM')
    axes[1].set_xscale('log')
    axes[1].set_xlabel(r"Dephasing hızı $\gamma$ (ps$^{-1}$, log ölçek)")
    axes[1].set_ylabel(r"Kuantum Verimi $\eta$")
    axes[1].set_title(f"Lindblad vs HEOM: Verim (r={r_fixed} nm)")
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig("/home/claude/lindblad_vs_heom_comparison.png", dpi=300)
    plt.close()
    print("\n[OK] Karşılaştırma grafiği kaydedildi: "
          "/home/claude/lindblad_vs_heom_comparison.png")

    np.savez("/home/claude/lindblad_vs_heom_data.npz",
              gamma_range=gamma_range,
              mean_t_lindblad=mean_t_lindblad, eta_lindblad=eta_lindblad,
              mean_t_heom=mean_t_heom, eta_heom=eta_heom)

    print("\n[OK] Adım 2.3 tamamlandı.")
