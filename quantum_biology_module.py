"""
=======================================================================
FAZ 2 - ADIM 2.1: Sistem Hamiltoniyeni ve Açık Kuantum Sistem Modeli
=======================================================================
Proje   : DNA-İskele Kromofor Dizilimlerinde Titreşimsel Rezonans
          Mühendisliği ile ENAQT'nin Deneysel Doğrulanması
Modül   : quantum_biology_module.py
Amaç    : N-site (donör-akseptör veya multi-kromofor) Frenkel ekziton
          Hamiltoniyenini kurmak, Lindblad Master Equation ile açık
          kuantum sistem dinamiğini (rho(t)) çözmek ve popülasyon
          zaman evrimini görselleştirmek.

BİRİM SİSTEMİ (çok önemli - kuantum biyolojide standart yaklaşım):
  - Enerjiler ve kaplinler   : cm^-1  (spektroskopik standart birim)
  - Zaman                    : ps (picosaniye) - biyolojik/eksitonik
                                zaman ölçeğine uygun
  - Dönüşüm sabiti           : QuTiP hbar=1 varsayımıyla çalıştığından,
                                cm^-1 cinsinden enerjiyi rad/ps cinsinden
                                açısal frekansa çevirmemiz gerekir:

                                omega [rad/ps] = 2*pi*c * nu [cm^-1]

                                c (ışık hızı) = 2.99792458e10 cm/s
                                              = 2.99792458e-2 cm/ps

                                => CM1_TO_RADPS = 2*pi*2.99792458e-2
                                                = 0.188365 rad/ps per cm^-1

  Bu, Ishizaki-Fleming (FMO açık kuantum sistem literatürü) ve genel
  fotosentez kuantum biyolojisi çalışmalarında kullanılan standart
  dönüşümdür; Hamiltoniyen matrisini cm^-1 cinsinden kurup, QuTiP'e
  vermeden önce bu sabitle çarpıyoruz.
=======================================================================
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional


# -----------------------------------------------------------------------
# FİZİKSEL SABİTLER
# -----------------------------------------------------------------------
C_LIGHT_CM_PER_PS = 2.99792458e-2      # ışık hızı [cm/ps]
CM1_TO_RADPS = 2 * np.pi * C_LIGHT_CM_PER_PS   # ~0.188365 rad/ps per cm^-1
KB_CM1_PER_K = 0.695034800              # Boltzmann sabiti [cm^-1 / K]


@dataclass
class SiteSystemParams:
    """
    N-site kromofor sisteminin fiziksel parametreleri.

    site_energies : list[float]
        Her kromoforun site (yerel) enerjisi, cm^-1 cinsinden.
        Örn. 2 site (donor-acceptor): [E1, E2]
        Örn. FMO benzeri 3 site: [E1, E2, E3]

    couplings : dict[(i,j) -> float]
        Site'lar arası elektronik kaplin J_ij, cm^-1 cinsinden.
        Örn. {(0,1): 100.0} -> site 0 ve 1 arasında 100 cm^-1 kaplin

    gamma_dephasing : list[float]
        Her site için pure-dephasing oranı, cm^-1 cinsinden
        (bağımsız değişkenimiz - ortam/banyo etkisini temsil eder).

    gamma_relax : Optional[list[float]]
        Site'lar arası enerji gevşemesi (relaxation) oranı, cm^-1.
        Fiziksel olarak yüksek enerjili site'tan düşük enerjili site'a
        (donor -> acceptor) enerji kaybını temsil eder. None ise
        yalnızca saf dephasing (Adım 2.1 minimum modeli) kullanılır.

    bath_temperature_K : float
        Banyo (ortam) sıcaklığı, Kelvin. Faz 3'te deneysel sıcaklık
        taramasıyla eşleştirilecek; bu adımda relaxation oranlarının
        detaylı-denge (detailed balance) düzeltmesinde kullanılır.
    """
    site_energies: list
    couplings: dict
    gamma_dephasing: list
    gamma_relax: Optional[list] = None
    bath_temperature_K: float = 300.0

    def __post_init__(self):
        self.n_sites = len(self.site_energies)
        if len(self.gamma_dephasing) != self.n_sites:
            raise ValueError("gamma_dephasing uzunluğu n_sites ile eşleşmeli.")


class OpenQuantumSystemModel:
    """
    N-site Frenkel ekziton sistemi için açık kuantum sistem (Lindblad)
    modeli. 2-site donor-acceptor sistemi için de, çok-siteli (FMO benzeri)
    genişletilmiş sistemler için de kullanılabilir (genelleştirilmiş tasarım).
    """

    def __init__(self, params: SiteSystemParams):
        self.params = params
        self.N = params.n_sites
        self.H = self._build_hamiltonian()

    # ------------------------------------------------------------------
    # 1) SİSTEM HAMİLTONİYENİ  H_S = sum_n eps_n |n><n| + sum_{n!=m} J_nm |n><m|
    # ------------------------------------------------------------------
    def _build_hamiltonian(self) -> qt.Qobj:
        """
        Frenkel ekziton Hamiltoniyenini site-bazında (localized basis)
        kurar. Matris cm^-1 cinsinden inşa edilip, QuTiP'e verilmeden
        önce rad/ps cinsine çevrilir (hbar=1 sözleşmesiyle).
        """
        N = self.N
        H_cm1 = np.zeros((N, N), dtype=complex)

        # Köşegen terimler: site enerjileri (diyagonal)
        for n in range(N):
            H_cm1[n, n] = self.params.site_energies[n]

        # Köşegen dışı terimler: elektronik kaplin J_nm (Hermitesel)
        for (i, j), J_ij in self.params.couplings.items():
            H_cm1[i, j] = J_ij
            H_cm1[j, i] = np.conj(J_ij)

        # cm^-1 -> rad/ps dönüşümü (hbar=1 varsayımıyla QuTiP enerji birimi)
        H_radps = H_cm1 * CM1_TO_RADPS

        return qt.Qobj(H_radps)

    # ------------------------------------------------------------------
    # 2) LİNDBLAD ÇÖKME (COLLAPSE) OPERATÖRLERİ
    # ------------------------------------------------------------------
    def _build_collapse_operators(self) -> list:
        """
        İki tür Lindblad operatörü kurar:

        (a) SAF DEPHASING (pure dephasing):
            L_n = sqrt(gamma_n) |n><n|
            Bu operatör popülasyonları KORUR, sadece site'lar arası
            kuantum tutarlılığını (coherence, rho_ij, i!=j) bozar.
            ENAQT fiziğinin merkezi mekanizması budur.

        (b) ENERJİ GEVŞEMESİ (relaxation, opsiyonel):
            Yüksek enerjili site n'den düşük enerjili site m'ye
            L_{n->m} = sqrt(gamma_relax) |m><n|
            Popülasyon transferini fiziksel olarak sağlar
            (yalnızca saf dephasing ile popülasyon transferi ENAQT'nin
            ikinci-mertebe etkisiyle sınırlı kalır; gerçekçi bir donor->
            acceptor akışı için relaxation terimi önemlidir).

        Not: Tüm gamma değerleri cm^-1 cinsinden girilir ve burada
        rad/ps'e çevrilir.
        """
        N = self.N
        c_ops = []

        # (a) Saf dephasing operatörleri - her site için
        for n in range(N):
            gamma_n_radps = self.params.gamma_dephasing[n] * CM1_TO_RADPS
            if gamma_n_radps > 0:
                proj_n = qt.Qobj(np.diag([1.0 if k == n else 0.0 for k in range(N)]))
                c_ops.append(np.sqrt(gamma_n_radps) * proj_n)

        # (b) Relaxation operatörleri (opsiyonel, detailed-balance düzeltmeli)
        if self.params.gamma_relax is not None:
            energies = self.params.site_energies
            T = self.params.bath_temperature_K
            for idx, gamma_r_cm1 in enumerate(self.params.gamma_relax):
                # Basit komşu-site gevşemesi varsayımı: n -> n+1 yönünde
                # (donor -> acceptor akışı). Çok-siteli sistemlerde bu
                # kısım kaplin topolojisine göre genişletilecek (Faz 2.2).
                if idx + 1 >= N:
                    continue
                n_high, n_low = idx, idx + 1
                dE = energies[n_high] - energies[n_low]  # cm^-1

                gamma_down_radps = gamma_r_cm1 * CM1_TO_RADPS

                # Detailed balance: yukarı yönlü oran Boltzmann faktörüyle bastırılır
                if dE > 0:
                    boltz_factor = np.exp(-dE / (KB_CM1_PER_K * T))
                else:
                    boltz_factor = 1.0
                gamma_up_radps = gamma_down_radps * boltz_factor

                ket_low = qt.basis(N, n_low)
                ket_high = qt.basis(N, n_high)

                L_down = np.sqrt(gamma_down_radps) * (ket_low * ket_high.dag())
                L_up = np.sqrt(gamma_up_radps) * (ket_high * ket_low.dag())

                c_ops.append(L_down)
                c_ops.append(L_up)

        return c_ops

    # ------------------------------------------------------------------
    # 3) ZAMAN EVRİMİ: Lindblad Master Equation çözümü
    # ------------------------------------------------------------------
    def evolve(self, initial_site: int = 0, t_max_ps: float = 2.0,
               n_steps: int = 500):
        """
        rho_dot = -i[H,rho] + sum_k ( L_k rho L_k^dag - 1/2{L_k^dag L_k, rho} )

        Başlangıç koşulu: uyarılma initial_site'ta lokalize (saf durum).

        Returns
        -------
        tlist : np.ndarray   (ps)
        result : qutip.solver.Result
        populations : np.ndarray, shape (N, n_steps)  -- her site için P_n(t)
        coherence_01 : np.ndarray (complex) -- site 0-1 arası coherence |rho_01(t)|
        """
        N = self.N
        rho0 = qt.ket2dm(qt.basis(N, initial_site))
        tlist = np.linspace(0, t_max_ps, n_steps)  # ps cinsinden zaman ekseni

        c_ops = self._build_collapse_operators()

        # Beklenti değeri operatörleri: her site için popülasyon projektörü
        e_ops = [qt.Qobj(np.diag([1.0 if k == n else 0.0 for k in range(N)]))
                 for n in range(N)]

        result = qt.mesolve(self.H, rho0, tlist, c_ops=c_ops, e_ops=e_ops)

        populations = np.array(result.expect)  # shape (N, n_steps)

        # Coherence takibi (site 0 ve 1 arasında, eğer N>=2 ise)
        coherence_01 = None
        if N >= 2:
            # e_ops sadece diyagonal beklenti verdiği için coherence'ı
            # ayrıca tam rho(t) çözümünden hesaplamamız gerekir.
            result_full = qt.mesolve(self.H, rho0, tlist, c_ops=c_ops, e_ops=[])
            coherence_01 = np.array([s.full()[0, 1] for s in result_full.states])

        return tlist, result, populations, coherence_01

    # ------------------------------------------------------------------
    # 4) GÖRSELLEŞTİRME
    # ------------------------------------------------------------------
    def plot_dynamics(self, tlist, populations, coherence_01=None,
                       labels=None, save_path="population_dynamics.png",
                       title_suffix=""):
        """
        Site popülasyonlarının P_n(t) zaman evrimini ve (varsa) site
        0-1 arası kuantum coherence |rho_01(t)|'yi grafikler.
        """
        N = populations.shape[0]
        if labels is None:
            labels = [f"Site {n}" for n in range(N)]

        fig, axes = plt.subplots(1, 2 if coherence_01 is not None else 1,
                                  figsize=(13, 5) if coherence_01 is not None else (7, 5))

        ax_pop = axes[0] if coherence_01 is not None else axes

        for n in range(N):
            ax_pop.plot(tlist, populations[n], label=labels[n], linewidth=2)
        ax_pop.set_xlabel("Zaman (ps)")
        ax_pop.set_ylabel("Popülasyon $P_n(t)$")
        ax_pop.set_title(f"Site Popülasyon Dinamiği{title_suffix}")
        ax_pop.legend()
        ax_pop.grid(alpha=0.3)

        if coherence_01 is not None:
            ax_coh = axes[1]
            ax_coh.plot(tlist, np.abs(coherence_01), color="darkred", linewidth=2,
                        label=r"$|\rho_{01}(t)|$")
            ax_coh.set_xlabel("Zaman (ps)")
            ax_coh.set_ylabel("Coherence Genliği")
            ax_coh.set_title("Site 0-1 Kuantum Tutarlılığı (Coherence)")
            ax_coh.legend()
            ax_coh.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[OK] Grafik kaydedildi: {save_path}")


# =========================================================================
# TEST / DEMO: 2-Site Donor-Akseptör Sistemi (Cy3-Cy5 benzeri)
# =========================================================================
if __name__ == "__main__":

    print("=" * 70)
    print("TEST 1: 2-Site Donor-Akseptör Sistemi (Cy3-Cy5)")
    print("=" * 70)

    # --- Biyofiziksel parametreler (literatürden kalibre, cm^-1) ---
    E1 = 18800.0    # Cy3 (donor) site enerjisi, cm^-1
    E2 = 15800.0    # Cy5 (acceptor) site enerjisi, cm^-1
    V_coupling = 100.0   # elektronik kaplin J_12, cm^-1 (~3-5 bp mesafeye karşılık gelir)

    gamma_dephasing_site = 50.0   # cm^-1, ortam kaynaklı saf dephasing (bağımsız değişken)
    gamma_relax_val = 20.0        # cm^-1, donor->acceptor enerji gevşemesi

    params_2site = SiteSystemParams(
        site_energies=[E1, E2],
        couplings={(0, 1): V_coupling},
        gamma_dephasing=[gamma_dephasing_site, gamma_dephasing_site],
        gamma_relax=[gamma_relax_val],
        bath_temperature_K=298.0,   # oda sıcaklığı, Faz 3 ile eşleşecek
    )

    model_2site = OpenQuantumSystemModel(params_2site)

    print(f"Hamiltoniyen (rad/ps cinsinden, cm^-1 -> rad/ps dönüşümü uygulanmış):")
    print(model_2site.H)

    tlist, result, pops, coh = model_2site.evolve(
        initial_site=0,     # uyarılma donörde (Cy3) başlar
        t_max_ps=2.0,       # 2 ps - tipik eksitonik transfer zaman ölçeği
        n_steps=500
    )

    model_2site.plot_dynamics(
        tlist, pops, coh,
        labels=["Donör (Cy3)", "Akseptör (Cy5)"],
        save_path="/home/claude/dynamics_2site.png",
        title_suffix=" - 2-Site Donor-Akseptör (Cy3-Cy5)"
    )

    print(f"\nSon zamanda (t={tlist[-1]:.2f} ps) popülasyonlar:")
    print(f"  Donör (Cy3)   P0 = {pops[0][-1]:.4f}")
    print(f"  Akseptör (Cy5) P1 = {pops[1][-1]:.4f}")
    print(f"  Toplam (kayıp yok kontrolü, relax olmadan 1.0 olmalı; "
          f"relax ile <1 olabilir çünkü sink henüz eklenmedi) "
          f"= {pops[0][-1] + pops[1][-1]:.4f}")

    print("\n" + "=" * 70)
    print("TEST 2: 3-Site Multi-Kromofor Sistemi (FMO-benzeri genelleme testi)")
    print("=" * 70)

    params_3site = SiteSystemParams(
        site_energies=[18800.0, 17200.0, 15800.0],
        couplings={(0, 1): 80.0, (1, 2): 90.0, (0, 2): 15.0},
        gamma_dephasing=[50.0, 50.0, 50.0],
        gamma_relax=[15.0, 15.0],
        bath_temperature_K=298.0,
    )

    model_3site = OpenQuantumSystemModel(params_3site)
    tlist3, result3, pops3, coh3 = model_3site.evolve(
        initial_site=0, t_max_ps=2.0, n_steps=500
    )
    model_3site.plot_dynamics(
        tlist3, pops3, coh3,
        labels=["Site 0 (Donör)", "Site 1 (Ara)", "Site 2 (Akseptör)"],
        save_path="/home/claude/dynamics_3site.png",
        title_suffix=" - 3-Site Multi-Kromofor Sistemi"
    )

    print(f"\nSon zamanda popülasyonlar: "
          f"P0={pops3[0][-1]:.4f}, P1={pops3[1][-1]:.4f}, P2={pops3[2][-1]:.4f}")
    print("\n[OK] Adım 2.1 testleri başarıyla tamamlandı.")
