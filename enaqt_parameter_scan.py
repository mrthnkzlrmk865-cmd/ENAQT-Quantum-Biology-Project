"""
=======================================================================
FAZ 2 - ADIM 2.2: Parametre Taramaları ve Verim Haritası eta(gamma, r)
=======================================================================
Modül: enaqt_parameter_scan.py

Bu modül Adım 2.1'deki temel Hamiltoniyen/Lindblad iskeletini üç
kritik biyofiziksel unsurla genişletir:

  (1) SINK + LOSS (KAYIP) DENGESİ:
      - Sistem Hilbert uzayı 2 site + 1 "Sink" (reaksiyon merkezi,
        yakalanan/kullanılan enerji) + 1 "Loss" (floresan/non-radiatif
        kayıp rezervuarı) durumuna genişletilir -> toplam 4 boyutlu.
      - Gamma_sink: akseptörden Sink'e yakalama oranı (hızlı, ~ps^-1)
      - Gamma_loss: HER siteden Loss'a kaçak oranı (yavaş, ~0.001 ps^-1,
        floresan yaşam süresi ~ns mertebesine karşılık gelir)
      - Kuantum verimi HEM doğrudan Sink popülasyonundan HEM DE
        eta = integral( Gamma_sink * P_acceptor(t) dt ) formülüyle
        iki bağımsız yöntemle hesaplanır ve çapraz kontrol edilir.

  (2) V(r) DİPOL-DİPOL KAPLİN YASASI:
      V(r) = V_0 * (r_0 / r)^3
      NOT (önemli fiziksel ayrım): Bu, elektronik KAPLİN'in (V, coherent
      coupling) mesafeye bağımlılığıdır (~1/r^3, nokta-dipol yaklaşımı).
      Bu, Förster TRANSFER HIZININ (~1/r^6, Fermi'nin Altın Kuralı'ndan
      k ~ |V|^2 ~ 1/r^6) mesafeye bağımlılığından FARKLIDIR. Kompleks/
      açık kuantum sistem modelimizde biz coherent V(r)'yi Hamiltoniyene
      koyuyoruz; 1/r^6 Förster limiti, gamma >> V olduğu (dephasing baskın,
      Markov/Fermi limiti) durumda modelimizden KENDİLİĞİNDEN ortaya
      çıkmalı - bu Faz 4'te bir tutarlılık kontrolü olarak kullanılacak.

  (3) DETAILED BALANCE (TERMAL GERİ BESLEME):
      Relaxation oranları arasında Gamma_up / Gamma_down = exp(-dE / kT)
      ilişkisi korunur (Adım 2.1'den devam), böylece sistem t->inf
      limitinde doğru Boltzmann dağılımına yönelir (termodinamik tutarlılık).

BİRİM SİSTEMİ NOTU:
  Adım 2.1'de enerjiler/kaplinler cm^-1 girilip rad/ps'e çevrilmişti.
  Bu adımda TARANAN parametre olan dephasing oranı gamma DOĞRUDAN ps^-1
  cinsinden tanımlanıyor (kullanıcı talebi ve Faz 3 deneysel sıcaklık/
  viskozite eşleşmesi için daha doğal birim - floresan ömürleri, kinetik
  hızlar hep ps^-1/ns^-1 cinsinden raporlanır). ps^-1, rad/ps ile aynı
  boyuta (ters zaman) sahip olduğundan Lindblad operatörlerinde doğrudan
  kullanılabilir; ekstra dönüşüm gerekmez.
=======================================================================
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
import time

# Adım 2.1'den taşınan sabitler (tutarlılık için aynı isimlerle)
C_LIGHT_CM_PER_PS = 2.99792458e-2
CM1_TO_RADPS = 2 * np.pi * C_LIGHT_CM_PER_PS      # ~0.188365 rad/ps per cm^-1
KB_CM1_PER_K = 0.695034800                         # Boltzmann sabiti [cm^-1/K]

# Durum indeksleri (4 boyutlu genişletilmiş Hilbert uzayı)
DONOR, ACCEPTOR, SINK, LOSS = 0, 1, 2, 3
DIM = 4


# -----------------------------------------------------------------------
# (2) KAPLİN-MESAFE YASASI: V(r) = V0 * (r0/r)^3
# -----------------------------------------------------------------------
def coupling_from_distance(r_nm: float, V0_cm1: float = 100.0,
                            r0_nm: float = 2.0) -> float:
    """
    Dipol-dipol elektronik kaplin, nokta-dipol yaklaşımıyla.

    Parametreler
    ------------
    r_nm   : donor-akseptör mesafesi (nm) -- DNA bp sayısına karşılık gelir
    V0_cm1 : referans mesafede (r0) ölçülen/kalibre edilen kaplin (cm^-1)
    r0_nm  : referans mesafe (nm), Adım 2.1'deki V=100 cm^-1 @ ~2nm kalibrasyonuna denk

    Geçerlilik uyarısı: r < ~1 nm için nokta-dipol yaklaşımı fiziksel
    olarak bozulur (kromoforların uzaysal boyutu mesafeyle kıyaslanabilir
    hale gelir); r >= 1 nm aralığında kaldığımız için yaklaşım geçerlidir.
    """
    if r_nm <= 0:
        raise ValueError("r_nm pozitif olmalı.")
    return V0_cm1 * (r0_nm / r_nm) ** 3


# -----------------------------------------------------------------------
# SİSTEM PARAMETRELERİ
# -----------------------------------------------------------------------
@dataclass
class ENAQTParams:
    E_donor_cm1: float = 18800.0
    E_acceptor_cm1: float = 15800.0
    r_nm: float = 2.0                  # taranacak: 1-5 nm
    V0_cm1: float = 100.0
    r0_nm: float = 2.0

    gamma_dephasing_psinv: float = 1.0   # taranacak: 0.1-10 ps^-1 (HER İKİ SİTE İÇİN AYNI)
    gamma_relax_psinv: float = 0.5       # sabit: donor->acceptor gevşeme hızı (coherent olmayan kısım)
    bath_temperature_K: float = 298.0

    gamma_sink_psinv: float = 1.0        # sabit: akseptör->RC yakalama hızı
    gamma_loss_psinv: float = 0.001      # sabit: floresan/non-radiatif kayıp (~1 ns yaşam süresi)


# -----------------------------------------------------------------------
# ÖNEMLİ TASARIM NOTU (test taraması sonrası eklendi):
#
# Gerçekçi Cy3/Cy5 enerjileriyle (dE ~ 3000 cm^-1) yapılan tek-nokta testinde
# eta(gamma) neredeyse SABİT kaldı (~0.997) ve <t>(gamma) yalnızca ~%5
# değişti. Sebep: Kubo/motional-narrowing formülüne göre (iki seviyeli
# dephasing-aracılı transfer):
#
#       k_transfer(gamma) = 2*V^2*gamma / (gamma^2 + dE^2)
#
# Bu ifade gamma = dE noktasında MAKSİMUMA ulaşır. Gerçek Cy3/Cy5 çifti için
# dE ~ 3000 cm^-1 ~ 565 rad/ps -> optimal gamma ~565 ps^-1 civarında olurdu;
# bu, biyolojik/deneysel olarak anlamlı bir dephasing rejiminin ÇOK ötesinde
# (gerçek ortam dephasing hızları tipik olarak 0.01-100 ps^-1 aralığındadır).
# Bu yüzden gerçek Cy3/Cy5 sistemi, taradığımız gamma aralığında ENAQT
# eğrisinin yalnızca YÜKSELEN (fonon-yardımlı, "motional narrowing") kolunda
# yaşıyor -- bu da gerçek FRET'in neden çoğunlukla İNKOHERENT (Förster tipi)
# olarak tanımlandığını doğrudan açıklayan, kendi başına anlamlı bir bulgu.
#
# ENAQT'nin KLASİK "tepe-çukur" (Rabi -> optimum -> Zeno) davranışını NET
# biçimde izole edip görselleştirmek için literatürdeki standart yaklaşımı
# izliyoruz (Rebentrost, Mohseni, Plenio ve ark. 2009 dimer modeli): enerji
# farkının kaplinle KIYASLANABİLİR büyüklükte olduğu bir "kanonik ENAQT"
# konfigürasyonu kullanıyoruz. Bu, DNA-Cy3/Cy5 deneysel sistemimizden PARAMETRE
# olarak farklıdır ama AYNI Hamiltoniyen/Lindblad çerçevesini kullanır ve
# fiziksel mekanizmayı temiz biçimde ortaya çıkarır. Faz 4'te iki model
# birlikte yorumlanacak: "gerçek sistem hangi rejimde yaşıyor?" sorusu.
# -----------------------------------------------------------------------
CANONICAL_ENAQT_PARAMS = ENAQTParams(
    E_donor_cm1=25.0,        # küçük enerji farkı (dE=25 cm^-1) -> rezonansa yakın
    E_acceptor_cm1=0.0,
    V0_cm1=20.0,              # r0=2nm'de referans kaplin
    r0_nm=2.0,
    gamma_dephasing_psinv=1.0,  # taranacak (2D grid'de override edilir)
    gamma_relax_psinv=0.02,     # KÜÇÜLTÜLDÜ: artık transfer büyük ölçüde
                                  # dephasing-aracılı (coherent+noise) kanaldan
                                  # geçiyor, ayrı bir "gizli" kanal baskın değil
    bath_temperature_K=298.0,
    gamma_sink_psinv=1.0,
    gamma_loss_psinv=0.001,
)


# -----------------------------------------------------------------------
# GENİŞLETİLMİŞ AÇIK KUANTUM SİSTEM MODELİ (Sink + Loss dahil)
# -----------------------------------------------------------------------
class ENAQTSystem:
    """
    4 boyutlu genişletilmiş sistem: |Donor>, |Acceptor>, |Sink(RC)>, |Loss>.
    Sink ve Loss durumları arasında koherent (Hamiltoniyen) etkileşim YOKTUR;
    onlara giden akış tamamen Lindblad (incoherent) operatörleriyle sağlanır.
    Bu, "kullanılan" (Sink) ve "kaybolan" (Loss) enerjiyi ayırt etmemizi sağlar.
    """

    def __init__(self, params: ENAQTParams):
        self.p = params
        self.H = self._build_hamiltonian()

    def _build_hamiltonian(self) -> qt.Qobj:
        p = self.p
        V_cm1 = coupling_from_distance(p.r_nm, p.V0_cm1, p.r0_nm)

        H_cm1 = np.zeros((DIM, DIM), dtype=complex)
        H_cm1[DONOR, DONOR] = p.E_donor_cm1
        H_cm1[ACCEPTOR, ACCEPTOR] = p.E_acceptor_cm1
        H_cm1[DONOR, ACCEPTOR] = V_cm1
        H_cm1[ACCEPTOR, DONOR] = V_cm1
        # SINK ve LOSS: enerji/kaplin yok (yalnızca akış hedefi)

        return qt.Qobj(H_cm1 * CM1_TO_RADPS)

    def _build_collapse_operators(self) -> list:
        p = self.p
        c_ops = []

        # --- (a) Saf dephasing: donor ve akseptör siteleri üzerinde ---
        for site in [DONOR, ACCEPTOR]:
            proj = qt.Qobj(np.diag([1.0 if k == site else 0.0 for k in range(DIM)]))
            c_ops.append(np.sqrt(p.gamma_dephasing_psinv) * proj)

        # --- (b) Relaxation (donor <-> acceptor), detailed balance ile ---
        dE_cm1 = p.E_donor_cm1 - p.E_acceptor_cm1  # >0 ise donor daha yüksek enerjili
        boltz_up_over_down = np.exp(-dE_cm1 / (KB_CM1_PER_K * p.bath_temperature_K))

        ket_acc = qt.basis(DIM, ACCEPTOR)
        ket_don = qt.basis(DIM, DONOR)

        gamma_down = p.gamma_relax_psinv               # donor -> acceptor (enerji aşağı yönlü)
        gamma_up = p.gamma_relax_psinv * boltz_up_over_down  # acceptor -> donor (termal aktivasyon)

        c_ops.append(np.sqrt(gamma_down) * (ket_acc * ket_don.dag()))
        c_ops.append(np.sqrt(gamma_up) * (ket_don * ket_acc.dag()))

        # --- (c) SINK: acceptor -> Sink (yakalama, "kullanılan" enerji) ---
        ket_sink = qt.basis(DIM, SINK)
        c_ops.append(np.sqrt(p.gamma_sink_psinv) * (ket_sink * ket_acc.dag()))

        # --- (d) LOSS: HER site (donor VE acceptor) -> Loss (floresan/kayıp) ---
        ket_loss = qt.basis(DIM, LOSS)
        for site_ket in [ket_don, ket_acc]:
            c_ops.append(np.sqrt(p.gamma_loss_psinv) * (ket_loss * site_ket.dag()))

        return c_ops

    def evolve(self, t_max_ps: float = 25.0, n_steps: int = 400):
        """
        Zaman evrimini çözer ve 4 durumun popülasyonlarını döndürür.
        Başlangıç: uyarılma Donor'da lokalize.
        """
        rho0 = qt.ket2dm(qt.basis(DIM, DONOR))
        tlist = np.linspace(0, t_max_ps, n_steps)
        c_ops = self._build_collapse_operators()

        e_ops = [qt.Qobj(np.diag([1.0 if k == n else 0.0 for k in range(DIM)]))
                 for n in range(DIM)]

        result = qt.mesolve(self.H, rho0, tlist, c_ops=c_ops, e_ops=e_ops)
        populations = np.array(result.expect)  # shape (4, n_steps): [P_donor, P_acc, P_sink, P_loss]

        return tlist, populations

    def compute_efficiency(self, tlist, populations) -> Tuple[float, float]:
        """
        Kuantum verimini İKİ BAĞIMSIZ yöntemle hesaplar:

        Yöntem A (doğrudan): eta_direct = P_sink(t_final)
            -> Genişletilmiş Hilbert uzayında Sink popülasyonu zaten
               kümülatif olarak biriktiği için t_final'daki değeri
               doğrudan toplam yakalanan olasılığı verir.

        Yöntem B (akı integrali, KULLANICI TALEBİ):
            eta_integral = integral_0^t_final [ Gamma_sink * P_acceptor(t) ] dt
            -> Sink'e giren akışın zamana göre integrali. Trapezoid
               kuralıyla nümerik olarak hesaplanır.

        İki yöntem, tanım gereği (dP_sink/dt = Gamma_sink * P_acceptor(t))
        matematiksel olarak birbirine eşit olmalıdır; aralarındaki fark
        yalnızca nümerik integrasyon hatasından kaynaklanır ve bir
        DOĞRULAMA (sanity check) olarak kullanılır.
        """
        P_acceptor_t = populations[ACCEPTOR]
        P_sink_t = populations[SINK]

        eta_direct = P_sink_t[-1]
        trapz_func = getattr(np, "trapezoid", None) or np.trapz
        flux_t = self.p.gamma_sink_psinv * P_acceptor_t
        eta_integral = trapz_func(flux_t, tlist)

        return eta_direct, eta_integral

    def compute_mean_trapping_time(self, tlist, populations) -> float:
        """
        ENAQT literatüründe (Rebentrost, Mohseni, Plenio ve ark. 2009)
        standart olarak kullanılan İKİNCİ metrik: ORTALAMA YAKALANMA SÜRESİ.

            <t> = [ integral t * Gamma_sink * P_acceptor(t) dt ]
                  / [ integral Gamma_sink * P_acceptor(t) dt ]

        GEREKÇE: Bizim sistemimizde Gamma_sink >> Gamma_loss (fizik olarak
        gerçekçi - gerçek fotosentezde kuantum verimi zaten >%95'tir),
        bu yüzden eta(gamma,r) neredeyse her yerde doygunlaşır (~0.997-0.998)
        ve ENAQT'nin asıl imzası olan "ara dekoharansta tepe noktası" eta
        üzerinde görünmez hale gelir (yalnızca 4. ondalıkta iz bırakır).

        Ancak ENAQT'nin fiziksel özü aslında TRANSFER HIZIYLA ilgilidir:
        - gamma -> 0 (tam koherent): sistem donor-akseptör arasında Rabi
          salınımı yapar, popülasyon geri-ileri gider, ortalama yakalanma
          süresi UZAR.
        - gamma çok büyük (aşırı dekoherent): kuantum Zeno benzeri etkiyle
          site'lar arası etkin geçiş bastırılır (lokalize durum), transfer
          yine YAVAŞLAR.
        - ARA gamma: coherent-to-incoherent geçiş bölgesinde en HIZLI
          transfer gerçekleşir -> <t> için bir MİNİMUM (sweet spot).

        Bu yüzden <t>(gamma, r) haritası, eta doygunlaşsa bile ENAQT'nin
        "sweet spot"unu net biçimde ortaya çıkarır.
        """
        P_acceptor_t = populations[ACCEPTOR]
        trapz_func = getattr(np, "trapezoid", None) or np.trapz
        flux_t = self.p.gamma_sink_psinv * P_acceptor_t

        numerator = trapz_func(tlist * flux_t, tlist)
        denominator = trapz_func(flux_t, tlist)

        if denominator < 1e-12:
            return np.nan
        return numerator / denominator


# -----------------------------------------------------------------------
# 2D PARAMETRE TARAMASI: eta(gamma_dephasing, r)
# -----------------------------------------------------------------------
def run_2d_scan(gamma_range_psinv: np.ndarray, r_range_nm: np.ndarray,
                 base_params: ENAQTParams, t_max_ps: float = 30.0,
                 n_steps: int = 300, verbose: bool = True):
    """
    gamma x r ızgarasında hem kuantum verimi eta'yı HEM DE ortalama
    yakalanma süresi <t>'yi hesaplar (aynı trafik/simülasyondan, ekstra
    hesaplama maliyeti olmadan -- <t> zaten eta_integral hesabı sırasında
    üretilen P_acceptor(t) akışından türetiliyor).

    Returns
    -------
    eta_grid : np.ndarray, shape (len(r_range), len(gamma_range))
    mean_time_grid : np.ndarray, shape (len(r_range), len(gamma_range))  [ps]
    max_error_direct_vs_integral : float (iki yöntem arası maksimum fark, doğrulama)
    """
    n_gamma = len(gamma_range_psinv)
    n_r = len(r_range_nm)
    eta_grid = np.zeros((n_r, n_gamma))
    mean_time_grid = np.zeros((n_r, n_gamma))
    max_discrepancy = 0.0

    t0 = time.time()
    for i, r in enumerate(r_range_nm):
        for j, gamma in enumerate(gamma_range_psinv):
            p = ENAQTParams(
                E_donor_cm1=base_params.E_donor_cm1,
                E_acceptor_cm1=base_params.E_acceptor_cm1,
                r_nm=r,
                V0_cm1=base_params.V0_cm1,
                r0_nm=base_params.r0_nm,
                gamma_dephasing_psinv=gamma,
                gamma_relax_psinv=base_params.gamma_relax_psinv,
                bath_temperature_K=base_params.bath_temperature_K,
                gamma_sink_psinv=base_params.gamma_sink_psinv,
                gamma_loss_psinv=base_params.gamma_loss_psinv,
            )
            system = ENAQTSystem(p)
            tlist, pops = system.evolve(t_max_ps=t_max_ps, n_steps=n_steps)
            eta_direct, eta_integral = system.compute_efficiency(tlist, pops)
            mean_t = system.compute_mean_trapping_time(tlist, pops)

            eta_grid[i, j] = eta_direct
            mean_time_grid[i, j] = mean_t
            discrepancy = abs(eta_direct - eta_integral)
            max_discrepancy = max(max_discrepancy, discrepancy)

        if verbose:
            print(f"  r = {r:.2f} nm taraması tamamlandı ({i+1}/{n_r})")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n[OK] 2D tarama tamamlandı: {n_r}x{n_gamma} = {n_r*n_gamma} "
              f"simülasyon, {elapsed:.1f} saniyede.")
        print(f"[DOĞRULAMA] Yöntem A (direkt) vs Yöntem B (integral) "
              f"arası maks. fark: {max_discrepancy:.2e} "
              f"({'GEÇERLİ' if max_discrepancy < 1e-3 else 'KONTROL GEREKİR'})")

    return eta_grid, mean_time_grid, max_discrepancy


# -----------------------------------------------------------------------
# GÖRSELLEŞTİRME: Heatmap + Kontur + Sweet Spot Tespiti
# -----------------------------------------------------------------------
def plot_enaqt_dual_map(gamma_range, r_range, eta_grid, mean_time_grid,
                          save_path="/home/claude/enaqt_dual_heatmap.png"):
    """
    YAN YANA iki 2D harita:
      (a) eta(gamma, r)      -> doygunluğu gösterir (maksimumu işaretler)
      (b) <t>(gamma, r)      -> ENAQT tepe/dip noktasını NET gösterir
                                 (minimumu işaretler -- en hızlı transfer)
    """
    # eta için maksimum, <t> için MİNİMUM nokta aranır (en hızlı transfer = sweet spot)
    i_eta, j_eta = np.unravel_index(np.argmax(eta_grid), eta_grid.shape)
    i_t, j_t = np.unravel_index(np.nanargmin(mean_time_grid), mean_time_grid.shape)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- (a) Verim haritası ---
    im0 = axes[0].pcolormesh(gamma_range, r_range, eta_grid, shading='auto', cmap='viridis')
    axes[0].set_xscale('log')
    axes[0].plot(gamma_range[j_eta], r_range[i_eta], marker='*', color='red',
                 markersize=18, markeredgecolor='white',
                 label=f"max η\n(γ={gamma_range[j_eta]:.2f}, r={r_range[i_eta]:.2f} nm)")
    axes[0].set_xlabel(r"Dephasing hızı $\gamma$ (ps$^{-1}$, log ölçek)")
    axes[0].set_ylabel("Donör-Akseptör mesafesi $r$ (nm)")
    axes[0].set_title(r"(a) Kuantum Verimi $\eta(\gamma, r)$" + "\n(doygunluk - Γ_sink≫Γ_loss etkisi)")
    axes[0].legend(loc='upper right', fontsize=8)
    plt.colorbar(im0, ax=axes[0], label=r"$\eta$")

    # --- (b) Ortalama yakalanma süresi haritası ---
    im1 = axes[1].pcolormesh(gamma_range, r_range, mean_time_grid, shading='auto', cmap='plasma_r')
    axes[1].set_xscale('log')
    axes[1].plot(gamma_range[j_t], r_range[i_t], marker='*', color='cyan',
                 markersize=18, markeredgecolor='black',
                 label=f"min ⟨t⟩ (ENAQT sweet spot)\n(γ={gamma_range[j_t]:.2f}, r={r_range[i_t]:.2f} nm)")
    axes[1].set_xlabel(r"Dephasing hızı $\gamma$ (ps$^{-1}$, log ölçek)")
    axes[1].set_ylabel("Donör-Akseptör mesafesi $r$ (nm)")
    axes[1].set_title(r"(b) Ortalama Yakalanma Süresi $\langle t \rangle(\gamma, r)$" + "\n(ENAQT tepe/dip noktası NET görünür)")
    axes[1].legend(loc='upper right', fontsize=8)
    plt.colorbar(im1, ax=axes[1], label=r"$\langle t \rangle$ (ps)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Çift harita (eta + <t>) kaydedildi: {save_path}")
    print(f"\n>>> eta maksimumu: gamma={gamma_range[j_eta]:.3f} ps^-1, "
          f"r={r_range[i_eta]:.3f} nm, eta_max={eta_grid[i_eta,j_eta]:.5f}")
    print(f">>> <t> MİNİMUMU (ENAQT SWEET SPOT): gamma={gamma_range[j_t]:.3f} ps^-1, "
          f"r={r_range[i_t]:.3f} nm, <t>_min={mean_time_grid[i_t,j_t]:.4f} ps")

    return (gamma_range[j_eta], r_range[i_eta], eta_grid[i_eta, j_eta],
            gamma_range[j_t], r_range[i_t], mean_time_grid[i_t, j_t])


def plot_enaqt_1d_slice_regimes(gamma_range, r_range, mean_time_grid, r_target_nm=2.0,
                                  save_path="/home/claude/enaqt_1d_regime_slice.png"):
    """
    Sabit r = r_target_nm'de <t> vs gamma kesiti (log-x eksen).
    Üç fiziksel rejim açıkça etiketlenir:
      1) KOHERENT RABİ SALINIMLARI (düşük gamma): sistem donor-akseptör
         arasında ileri-geri salınır, ortalama yakalanma süresi UZUNDUR.
      2) OPTİMAL ENAQT (ara gamma, <t> MİNİMUM): dephasing, koherent
         salınımı yönlendirilmiş/yönelimli akışa çevirir -> en hızlı transfer.
      3) KUANTUM ZENO ETKİSİ (yüksek gamma): aşırı sık "ölçüm benzeri"
         dephasing, site'lar arası etkin geçişi bastırır -> transfer
         yeniden YAVAŞLAR.
    """
    idx_r = np.argmin(np.abs(r_range - r_target_nm))
    r_actual = r_range[idx_r]
    t_curve = mean_time_grid[idx_r, :]

    idx_min = np.nanargmin(t_curve)
    gamma_opt = gamma_range[idx_min]
    t_min = t_curve[idx_min]

    plt.figure(figsize=(9, 6))
    plt.plot(gamma_range, t_curve, color='darkviolet', linewidth=2.5, marker='o', markersize=4)
    plt.xscale('log')
    plt.axvline(gamma_opt, color='black', linestyle='--', alpha=0.6)
    plt.plot(gamma_opt, t_min, marker='*', color='red', markersize=20, zorder=5)

    # Rejim bölgelerini renkli alanlarla göster
    plt.axvspan(gamma_range[0], gamma_opt / 5, color='dodgerblue', alpha=0.12)
    plt.axvspan(gamma_opt / 5, gamma_opt * 5, color='limegreen', alpha=0.12)
    plt.axvspan(gamma_opt * 5, gamma_range[-1], color='orangered', alpha=0.12)

    ymax = np.nanmax(t_curve)
    plt.text(gamma_range[0] * 1.3, ymax * 0.97, "① KOHERENT RABİ\nSALINIMLARI\n(yavaş, salınımlı)",
              fontsize=9, color='navy', va='top')
    plt.text(gamma_opt, ymax * 0.75, f"② OPTİMAL ENAQT\n⟨t⟩ min = {t_min:.3f} ps\nγ_opt = {gamma_opt:.2f} ps⁻¹",
              fontsize=9, color='darkgreen', va='top', ha='center',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.text(gamma_range[-1] * 0.7, ymax * 0.97, "③ KUANTUM ZENO\nETKİSİ\n(lokalizasyon, yavaş)",
              fontsize=9, color='darkred', va='top', ha='right')

    plt.xlabel(r"Dephasing hızı $\gamma$ (ps$^{-1}$, log ölçek)")
    plt.ylabel(r"Ortalama Yakalanma Süresi $\langle t \rangle$ (ps)")
    plt.title(f"ENAQT'nin Üç Rejimi: r = {r_actual:.2f} nm'de Kesit")
    plt.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] 1D rejim kesit grafiği kaydedildi: {save_path}")
    print(f">>> r={r_actual:.2f} nm kesitinde optimal gamma = {gamma_opt:.3f} ps^-1, "
          f"<t>_min = {t_min:.4f} ps")


# =========================================================================
# ANA ÇALIŞTIRMA
# =========================================================================
if __name__ == "__main__":

    print("=" * 70)
    print("ADIM 2.2: Tek Nokta Testi - Sink/Loss Dinamiği Doğrulaması")
    print("=" * 70)

    base_params = ENAQTParams()  # varsayılan değerler
    test_system = ENAQTSystem(base_params)
    tlist, pops = test_system.evolve(t_max_ps=25.0, n_steps=400)
    eta_direct, eta_integral = test_system.compute_efficiency(tlist, pops)

    print(f"Varsayılan parametrelerle (r={base_params.r_nm} nm, "
          f"gamma={base_params.gamma_dephasing_psinv} ps^-1):")
    print(f"  eta (direkt Sink popülasyonu)      = {eta_direct:.5f}")
    print(f"  eta (Gamma_sink*P_acc integrali)   = {eta_integral:.5f}")
    print(f"  Fark (doğrulama)                   = {abs(eta_direct-eta_integral):.2e}")
    print(f"  Son durumda: P_donor={pops[DONOR][-1]:.4f}, "
          f"P_acceptor={pops[ACCEPTOR][-1]:.4f}, "
          f"P_sink={pops[SINK][-1]:.4f}, P_loss={pops[LOSS][-1]:.4f}")
    print(f"  Toplam popülasyon (korunum kontrolü): "
          f"{sum(pops[k][-1] for k in range(DIM)):.6f} (1.0 olmalı)")

    # Popülasyon zaman evrimini de görselleştirelim (4 durum)
    plt.figure(figsize=(8, 6))
    labels_4 = ["Donör", "Akseptör", "Sink (RC, kullanılan)", "Loss (kayıp)"]
    colors_4 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for k in range(DIM):
        plt.plot(tlist, pops[k], label=labels_4[k], color=colors_4[k], linewidth=2)
    plt.xlabel("Zaman (ps)")
    plt.ylabel("Popülasyon")
    plt.title("Sink/Loss Dahil Tam Sistem Dinamiği (Tek Nokta Testi)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("/home/claude/single_point_sink_loss_dynamics.png", dpi=300)
    plt.close()
    print("[OK] Tek nokta dinamik grafiği kaydedildi: "
          "/home/claude/single_point_sink_loss_dynamics.png")

    print("\n" + "=" * 70)
    print("ADIM 2.2: 2D PARAMETRE TARAMASI - KANONİK ENAQT MODELİ")
    print("(near-rezonant toy dimer: dE=25 cm^-1, V0=20 cm^-1 @ r0=2nm)")
    print("=" * 70)

    # Logaritmik gamma taraması: Rabi (düşük) -> ENAQT optimum (orta) ->
    # Zeno (yüksek) rejimlerinin HEPSİNİ tek grafikte görebilmek için
    gamma_range = np.logspace(np.log10(0.01), np.log10(100.0), 32)   # ps^-1
    r_range = np.linspace(1.0, 5.0, 26)                                # nm

    # NOT: n_steps=300 ile yapılan ilk denemede, güçlü kaplinli (kısa r)
    # bölgelerde hızlı koherent salınımlar trapezoidal integrasyonu
    # under-sample ediyordu (eta_direct vs eta_integral arası %11 sahte
    # fark -> aliasing hatası, gerçek fizik değil). n_steps=800 ile bu
    # hata <1e-3 seviyesine düşüyor (doğrulandı, bkz. yandaki test).
    eta_grid, mean_time_grid, max_discrepancy = run_2d_scan(
        gamma_range, r_range, CANONICAL_ENAQT_PARAMS,
        t_max_ps=30.0, n_steps=800, verbose=True
    )

    plot_enaqt_dual_map(gamma_range, r_range, eta_grid, mean_time_grid)
    plot_enaqt_1d_slice_regimes(gamma_range, r_range, mean_time_grid, r_target_nm=2.0)

    # Ham veriyi de kaydedelim (Faz 3/4 karşılaştırması için)
    np.savez("/home/claude/eta_scan_data.npz",
              gamma_range=gamma_range, r_range=r_range,
              eta_grid=eta_grid, mean_time_grid=mean_time_grid)
    print("\n[OK] Ham tarama verisi kaydedildi: /home/claude/eta_scan_data.npz")

    print("\n[OK] Adım 2.2 tamamlandı.")
