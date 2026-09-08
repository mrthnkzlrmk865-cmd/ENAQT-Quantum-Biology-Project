#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================
FAZ 2 - ADIM 2.3 (GENİŞLETME): 2D (gamma, r) HEOM Parametre Taraması
                                 ve "Robustness Heatmap" Üretimi
=========================================================================
Modül  : heom_2d_robustness_scan.py
Amaç   : Non-Markovian (HEOM, Drude-Lorentz banyo) açık kuantum sistem
         modeliyle, dephasing/banyo etkileşim oranı (gamma) ve donör-
         akseptör mesafesi (r) üzerinde TAM 2D bir tarama yaparak:
           (a) Transfer verimliliği eta(gamma, r) haritasını,
           (b) Ortalama yakalanma süresi <t>(gamma, r) haritasını
         üretir; ENAQT "sweet spot", Kuantum Zeno bastırma ve yüksek-
         verim platosu bölgelerini yayın kalitesinde anotasyonlarla
         işaretler.

MİMARİ:
  - concurrent.futures.ProcessPoolExecutor ile çok çekirdekli paralel
    çalıştırma (400 bağımsız HEOM simülasyonu -- "utanmadan paralel"
    bir problem, her nokta diğerinden bağımsız).
  - tqdm ile gerçek zamanlı ilerleme takibi.
  - Her simülasyon noktası try/except ile korunur; bir noktanın
    (örn. sayısal yakınsama) başarısız olması TÜM taramayı durdurmaz,
    NaN olarak işaretlenip devam edilir.
  - Sonuçlar hem .npy (hızlı Python geri-yükleme) hem .csv (analiz/
    paylaşım) formatında kaydedilir.
  - Matplotlib/Seaborn ile 300 DPI yayın kalitesinde ısı haritası,
    LaTeX eksen etiketleri, ENAQT/Zeno/Plato bölgesi anotasyonları.

ÇALIŞTIRMA:
    python3 heom_2d_robustness_scan.py

NOT (hesaplama bütçesi): Varsayılan ayarlarla (20x20=400 nokta,
max_depth=5, Nk=1) tek bir HEOM noktası ortalama ~3-8 saniye sürer.
Tek çekirdekli bir makinede TAM tarama ~25-40 dakika alabilir; çok
çekirdekli bir makinede (örn. 8 çekirdek) bu süre ~4-6 dakikaya iner.
N_GAMMA / N_R sabitlerini küçülterek hızlı bir ön-test yapılabilir.
=========================================================================
"""

# -------------------------------------------------------------------
# KRİTİK: BLAS/OpenMP thread sayısını, ProcessPoolExecutor ile çoklu
# süreç kullanırken "oversubscription" (her süreç kendi içinde de çoklu
# thread açıp CPU'ları boğması) önlemek için AĞIR KÜTÜPHANELERİ
# (numpy, qutip) İÇE AKTARMADAN ÖNCE sınırlıyoruz.
# -------------------------------------------------------------------
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import traceback
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    def tqdm(iterable, **kwargs):  # sessiz fallback (tqdm kurulu değilse)
        return iterable

import qutip as qt
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath


# =========================================================================
# 1) FİZİKSEL SABİTLER VE SİSTEM PARAMETRELERİ
# =========================================================================
C_LIGHT_CM_PER_PS = 2.99792458e-2
CM1_TO_RADPS = 2 * np.pi * C_LIGHT_CM_PER_PS      # ~0.188365 rad/ps per cm^-1
KB_CM1_PER_K = 0.695034800                          # Boltzmann sabiti [cm^-1/K]

DONOR, ACCEPTOR, SINK, LOSS = 0, 1, 2, 3
DIM = 4

# --- Kanonik ENAQT sistem parametreleri (Adım 2.2/2.3 ile tutarlı) ---
E_DONOR_CM1 = 25.0
E_ACCEPTOR_CM1 = 0.0
V0_CM1 = 20.0          # referans kaplin (r0'da)
R0_NM = 2.0
GAMMA_RELAX_PSINV = 0.02
GAMMA_SINK_PSINV = 1.0
GAMMA_LOSS_PSINV = 0.001
T_KELVIN = 298.0

# --- HEOM / Non-Markovian banyo parametreleri (Drude-Lorentz) ---
GAMMA_C_RADPS = 2.0     # banyo kesim frekansı, tau_c = 1/gamma_c = 0.5 ps
HEOM_MAX_DEPTH = 5      # hiyerarşi derinliği (N_c)
HEOM_NK = 1             # Matsubara kesim terimi sayısı

# --- Zaman evrimi ayarları ---
T_MAX_PS = 25.0
N_TIME_STEPS = 220      # trapezoidal integrasyon için yeterli çözünürlük
                          # (bkz. Adım 2.2'deki aliasing hatası doğrulaması;
                          # burada guclu kaplin r=1nm icin de test edilmis
                          # ve <1e-3 hata verdigi dogrulanmistir)

# =========================================================================
# 2) TARANACAK 2D GRID TANIMI
# =========================================================================
GAMMA_MIN_PSINV = 0.05
GAMMA_MAX_PSINV = 80.0
N_GAMMA = 20                     # "20 eşit aralıklı nokta" (kullanıcı talebi)
GAMMA_LOG_SPACING = False        # True yapılırsa np.logspace kullanılır --
                                  # düşük-gamma (Rabi) rejimini daha iyi
                                  # çözümler; varsayılan, talep edilen
                                  # DOĞRUSAL (eşit aralıklı) ızgaradır.

R_MIN_NM = 1.0
R_MAX_NM = 5.0
N_R = 20                         # "20 eşit aralıklı nokta"

N_WORKERS = os.cpu_count() or 1  # mevcut CPU çekirdek sayısı (bu sandbox'ta 1)

OUTPUT_DIR = "/home/claude"


# =========================================================================
# 3) YARDIMCI FONKSİYONLAR (fiziksel model kurucuları)
# =========================================================================
def coupling_from_distance(r_nm, V0_cm1=V0_CM1, r0_nm=R0_NM):
    """Dipol-dipol kaplin: V(r) = V0 * (r0/r)^3"""
    return V0_cm1 * (r0_nm / r_nm) ** 3


def build_liouvillian(r_nm):
    """
    Koherent Hamiltoniyen + Markovian kanallar (relaxation, SINK, LOSS)
    -> tek bir Liouvillian. Dephasing burada YOK; o HEOM'daki Drude-Lorentz
    fonon banyosu tarafından non-Markovian olarak sağlanacak.
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

    # --- MARKOVIAN KANALLAR: Lindblad çökme operatörleri ---
    c_ops = [
        np.sqrt(GAMMA_RELAX_PSINV) * (ket_acc * ket_don.dag()),          # donor->acceptor relaxation
        np.sqrt(GAMMA_RELAX_PSINV * boltz) * (ket_don * ket_acc.dag()),  # detailed balance (termal geri besleme)
        np.sqrt(GAMMA_SINK_PSINV) * (ket_sink * ket_acc.dag()),          # SINK (yutak, reaksiyon merkezi yakalama)
    ]
    for sk in [ket_don, ket_acc]:
        c_ops.append(np.sqrt(GAMMA_LOSS_PSINV) * (ket_loss * sk.dag()))  # LOSS (floresan/non-radiatif kayıp)

    return qt.liouvillian(H, c_ops)


def mean_trapping_time_and_efficiency(tlist, P_acceptor, gamma_sink=GAMMA_SINK_PSINV):
    """
    eta       = integral( Gamma_sink * P_acceptor(t) ) dt
    <t>       = integral( t * Gamma_sink * P_acceptor(t) ) dt / eta
    """
    trapz = getattr(np, "trapezoid", None) or np.trapz
    flux = gamma_sink * P_acceptor
    eta = float(trapz(flux, tlist))
    if eta < 1e-9:
        return np.nan, eta
    mean_t = float(trapz(tlist * flux, tlist) / eta)
    return mean_t, eta


# =========================================================================
# 4) TEK NOKTA HEOM ÇÖZÜCÜSÜ (paralel worker fonksiyonu)
# =========================================================================
def compute_single_point(task):
    """
    Tek bir (i, j, gamma, r) noktası için HEOM simülasyonunu çalıştırır.

    ProcessPoolExecutor ile kullanılacağı için:
      - Fonksiyon MODÜL SEVİYESİNDE tanımlı olmalı (picklable).
      - Girdi/çıktı yalnızca temel Python tipleri (int, float, str, bool)
        içermeli -- QuTiP Qobj nesneleri süreç sınırını GEÇMEZ, her worker
        kendi Qobj'lerini kendi içinde inşa eder.
      - try/except ile sarmalanır: bir noktanın başarısız olması TÜM
        taramayı durdurmamalı.
    """
    i, j, gamma_psinv, r_nm = task

    try:
        L = build_liouvillian(r_nm)

        kT_radps = KB_CM1_PER_K * T_KELVIN * CM1_TO_RADPS
        # Haken-Strobl eşleştirmesi: gamma (fenomenolojik Lindblad hızı)
        # <-> lambda (mikroskopik HEOM reorganizasyon enerjisi)
        #     lambda = gamma * gamma_c / (2 * kT)
        lam_radps = gamma_psinv * GAMMA_C_RADPS / (2 * kT_radps)

        proj_don = qt.Qobj(np.diag([1., 0, 0, 0]))
        proj_acc = qt.Qobj(np.diag([0, 1., 0, 0]))

        bath_don = DrudeLorentzBath(Q=proj_don, lam=lam_radps, gamma=GAMMA_C_RADPS,
                                     T=kT_radps, Nk=HEOM_NK)
        bath_acc = DrudeLorentzBath(Q=proj_acc, lam=lam_radps, gamma=GAMMA_C_RADPS,
                                     T=kT_radps, Nk=HEOM_NK)

        solver = HEOMSolver(L, [bath_don, bath_acc], max_depth=HEOM_MAX_DEPTH,
                             options={"progress_bar": False})

        rho0 = qt.ket2dm(qt.basis(DIM, DONOR))
        tlist = np.linspace(0, T_MAX_PS, N_TIME_STEPS)
        e_ops = [qt.Qobj(np.diag([1.0 if k == m else 0.0 for k in range(DIM)]))
                 for m in range(DIM)]

        result = solver.run(rho0, tlist, e_ops=e_ops)
        P_acceptor = np.real(np.array(result.expect[ACCEPTOR]))

        mean_t, eta = mean_trapping_time_and_efficiency(tlist, P_acceptor)

        return {
            "i": i, "j": j, "gamma": gamma_psinv, "r": r_nm,
            "eta": eta, "mean_trapping_time_ps": mean_t,
            "success": True, "error": ""
        }

    except Exception as exc:  # noqa: BLE001 -- kasıtlı geniş yakalama: tek nokta hatası tüm taramayı durdurmamalı
        return {
            "i": i, "j": j, "gamma": gamma_psinv, "r": r_nm,
            "eta": np.nan, "mean_trapping_time_ps": np.nan,
            "success": False, "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=2)}"
        }


# =========================================================================
# 5) 2D TARAMA YÖNETİCİSİ (paralel + tqdm + hata toleranslı)
# =========================================================================
def run_2d_heom_scan(gamma_range, r_range, n_workers=N_WORKERS, checkpoint_every=20,
                       checkpoint_dir=OUTPUT_DIR, resume=True):
    n_g, n_r = len(gamma_range), len(r_range)
    eta_grid = np.full((n_r, n_g), np.nan)
    mean_time_grid = np.full((n_r, n_g), np.nan)

    ckpt_eta_path = os.path.join(checkpoint_dir, "heom_eta_grid_CHECKPOINT.npy")
    ckpt_t_path = os.path.join(checkpoint_dir, "heom_mean_time_grid_CHECKPOINT.npy")

    # --- RESUME: önceki (kesintiye uğramış) bir taramadan checkpoint varsa yükle ---
    if resume and os.path.exists(ckpt_eta_path) and os.path.exists(ckpt_t_path):
        try:
            loaded_eta = np.load(ckpt_eta_path)
            loaded_t = np.load(ckpt_t_path)
            if loaded_eta.shape == eta_grid.shape:
                eta_grid = loaded_eta
                mean_time_grid = loaded_t
                n_prev = int(np.sum(~np.isnan(eta_grid)))
                print(f"[RESUME] Önceki checkpoint bulundu: {n_prev}/{n_g*n_r} nokta zaten hesaplanmış. "
                      f"Sadece kalanlar çalıştırılacak.")
            else:
                print("[RESUME] Checkpoint boyutu mevcut grid ile uyuşmuyor, sıfırdan başlanıyor.")
        except Exception as exc:
            print(f"[UYARI] Checkpoint okunamadı ({exc}), sıfırdan başlanıyor.")

    all_tasks = [(i, j, gamma_range[j], r_range[i])
                 for i, j in itertools.product(range(n_r), range(n_g))]
    # Yalnızca HENÜZ HESAPLANMAMIŞ (NaN) noktaları çalıştır
    tasks = [t for t in all_tasks if np.isnan(eta_grid[t[0], t[1]])]

    if not tasks:
        print("[OK] Tüm noktalar zaten checkpoint'te mevcut, yeniden hesaplama gerekmiyor.")
        return eta_grid, mean_time_grid, []

    print(f"[BAŞLIYOR] {len(tasks)} HEOM simülasyonu (toplam {len(all_tasks)}'ten kalan), "
          f"{n_workers} paralel işçi ile...")
    print(f"           (Bu sandbox'ta {os.cpu_count()} CPU çekirdeği mevcut.)")

    failed_points = []
    t0 = time.time()
    n_done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(compute_single_point, task) for task in tasks]

        iterator = as_completed(futures)
        if _HAS_TQDM:
            iterator = tqdm(iterator, total=len(futures), desc="HEOM 2D tarama", unit="nokta")

        for future in iterator:
            res = future.result()
            i, j = res["i"], res["j"]
            eta_grid[i, j] = res["eta"]
            mean_time_grid[i, j] = res["mean_trapping_time_ps"]
            if not res["success"]:
                failed_points.append(res)
            n_done += 1

            # --- KESİNTİ TOLERANSI: periyodik checkpoint kaydı ---
            # Uzun süren bir taramanın ortasında süreç kesilirse (zaman
            # aşımı, oturum sıfırlanması vb.) o ana kadarki kısmi
            # sonuçlar diskte kalır; bir sonraki çalıştırmada RESUME
            # mekanizmasıyla kaldığı yerden devam eder.
            if checkpoint_every and (n_done % checkpoint_every == 0):
                try:
                    np.save(ckpt_eta_path, eta_grid)
                    np.save(ckpt_t_path, mean_time_grid)
                    elapsed_ckpt = time.time() - t0
                    remaining = len(tasks) - n_done
                    print(f"  [CHECKPOINT] {n_done}/{len(tasks)} (bu oturumda) tamamlandı "
                          f"({elapsed_ckpt:.0f}s geçti, ~{elapsed_ckpt/n_done*remaining:.0f}s kaldı)")
                except Exception as ckpt_exc:
                    print(f"  [UYARI] Checkpoint kaydı başarısız: {ckpt_exc}")

    elapsed = time.time() - t0
    print(f"\n[OK] Bu oturumda tamamlanan: {len(tasks)} nokta, {elapsed:.1f} s "
          f"({elapsed/max(len(tasks),1):.2f} s/nokta ortalama).")

    if failed_points:
        print(f"\n[UYARI] {len(failed_points)} nokta BAŞARISIZ oldu (NaN olarak işaretlendi):")
        for fp in failed_points[:5]:
            print(f"   gamma={fp['gamma']:.3f}, r={fp['r']:.3f} -> {fp['error'].splitlines()[0]}")
        if len(failed_points) > 5:
            print(f"   ... ve {len(failed_points)-5} nokta daha.")
    else:
        print("[OK] Bu oturumdaki tüm noktalar başarıyla hesaplandı (0 hata).")

    # Son bir tam checkpoint (final kayıttan önce güvence)
    try:
        np.save(ckpt_eta_path, eta_grid)
        np.save(ckpt_t_path, mean_time_grid)
    except Exception:
        pass

    return eta_grid, mean_time_grid, failed_points


# =========================================================================
# 6) VERİ KAYDETME (.npy + .csv)
# =========================================================================
def save_results(gamma_range, r_range, eta_grid, mean_time_grid, output_dir=OUTPUT_DIR):
    try:
        np.save(os.path.join(output_dir, "heom_gamma_range.npy"), gamma_range)
        np.save(os.path.join(output_dir, "heom_r_range.npy"), r_range)
        np.save(os.path.join(output_dir, "heom_eta_grid.npy"), eta_grid)
        np.save(os.path.join(output_dir, "heom_mean_time_grid.npy"), mean_time_grid)
        print(f"[OK] .npy dosyaları kaydedildi: {output_dir}/heom_*.npy")
    except Exception as exc:
        print(f"[HATA] .npy kaydı başarısız: {exc}")

    try:
        rows = []
        for i, r in enumerate(r_range):
            for j, g in enumerate(gamma_range):
                rows.append({
                    "gamma_ps_inv": g,
                    "r_nm": r,
                    "eta": eta_grid[i, j],
                    "mean_trapping_time_ps": mean_time_grid[i, j],
                })
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, "heom_2d_scan_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"[OK] .csv dosyası kaydedildi: {csv_path} ({len(df)} satır)")
    except Exception as exc:
        print(f"[HATA] .csv kaydı başarısız: {exc}")


# =========================================================================
# 7) YAYIN KALİTESİNDE GÖRSELLEŞTİRME: "ROBUSTNESS HEATMAP"
# =========================================================================
def plot_robustness_heatmap(gamma_range, r_range, eta_grid, mean_time_grid,
                              output_dir=OUTPUT_DIR):
    """
    Ana teslim edilebilir görsel: eta(gamma, r) ısı haritası + kontur,
    ENAQT sweet spot / Zeno bastırma / yüksek-verim plato bölgesi
    anotasyonlu, 300 DPI, LaTeX eksen etiketli.
    """
    if np.all(np.isnan(eta_grid)):
        print("[HATA] eta_grid tamamen NaN -- çizim atlanıyor.")
        return

    # Sweet spot: eta maksimumu (NaN'ları göz ardı ederek)
    flat_idx = np.nanargmax(eta_grid)
    i_max, j_max = np.unravel_index(flat_idx, eta_grid.shape)
    gamma_opt, r_opt, eta_opt = gamma_range[j_max], r_range[i_max], eta_grid[i_max, j_max]

    fig, ax = plt.subplots(figsize=(10, 7.5))

    if _HAS_SEABORN:
        # Seaborn heatmap düzenli ızgara ister; pcolormesh ile aynı sonucu
        # LaTeX-uyumlu eksenlerle daha esnek elde ediyoruz, seaborn'un
        # renk paletini (rocket/viridis) ödünç alıyoruz.
        cmap = sns.color_palette("viridis", as_cmap=True)
    else:
        cmap = "viridis"

    mesh = ax.pcolormesh(gamma_range, r_range, eta_grid, shading="auto", cmap=cmap)
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label(r"Transfer Verimliliği $\eta$", fontsize=13)

    # Kontur çizgileri (eş-verim eğrileri)
    valid_mask = ~np.isnan(eta_grid)
    if valid_mask.sum() > 4:
        try:
            cs = ax.contour(gamma_range, r_range, eta_grid, levels=8,
                             colors='white', linewidths=0.6, alpha=0.6)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.3f")
        except Exception:
            pass  # kontur bazen dejenere verilerde başarısız olabilir; kritik değil

    # --- Sweet spot işareti ---
    ax.plot(gamma_opt, r_opt, marker='*', color='red', markersize=22,
             markeredgecolor='white', markeredgewidth=1.2, zorder=6)
    ax.annotate(
        f"ENAQT Sweet Spot\n" r"$\gamma_{opt}=$" f"{gamma_opt:.2f}" r"$\,\mathrm{ps}^{-1}$"
        f"\n" r"$r_{opt}=$" f"{r_opt:.2f}" r"$\,\mathrm{nm}$" f"\n" r"$\eta_{max}=$" f"{eta_opt:.4f}",
        xy=(gamma_opt, r_opt), xytext=(0.62, 0.80), textcoords='axes fraction',
        fontsize=9, color='black',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='red', alpha=0.9),
        arrowprops=dict(arrowstyle='->', color='red', lw=1.5)
    )

    # --- Kuantum Zeno bastırma bölgesi (yüksek gamma) ---
    zeno_gamma = gamma_range[int(0.85 * len(gamma_range))]
    zeno_r = r_range[int(0.15 * len(r_range))]
    ax.annotate(
        "Kuantum Zeno\nBastırması\n(yüksek " r"$\gamma$" ")",
        xy=(zeno_gamma, zeno_r), xytext=(0.78, 0.15), textcoords='axes fraction',
        fontsize=9, color='black', ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose', edgecolor='darkred', alpha=0.85),
        arrowprops=dict(arrowstyle='->', color='darkred', lw=1.3)
    )

    # --- Koherent / Rabi bölgesi (düşük gamma) ---
    rabi_gamma = gamma_range[int(0.05 * len(gamma_range))]
    rabi_r = r_range[int(0.75 * len(r_range))]
    ax.annotate(
        "Koherent (Rabi)\nRejim\n(düşük " r"$\gamma$" ")",
        xy=(rabi_gamma, rabi_r), xytext=(0.08, 0.85), textcoords='axes fraction',
        fontsize=9, color='black', ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', edgecolor='navy', alpha=0.85),
        arrowprops=dict(arrowstyle='->', color='navy', lw=1.3)
    )

    # --- Yüksek-verim platosu (eta > %99 percentile bölgesi) ---
    eta_threshold = np.nanpercentile(eta_grid, 90)
    plateau_mask = eta_grid >= eta_threshold
    if plateau_mask.any():
        i_p, j_p = np.where(plateau_mask)
        ax.scatter(gamma_range[j_p], r_range[i_p], s=4, color='gold',
                   alpha=0.35, zorder=3, label=r"Yüksek-Verim Platosu ($\eta \geq$ 90. persentil)")
        ax.legend(loc='lower left', fontsize=8, framealpha=0.9)

    ax.set_xlabel(r"Dephasing / Banyo Etkileşim Oranı $\gamma\ [\mathrm{ps}^{-1}]$", fontsize=13)
    ax.set_ylabel(r"Donör-Akseptör Mesafesi $r\ [\mathrm{nm}]$", fontsize=13)
    ax.set_title("Non-Markovian (HEOM) ENAQT Robustness Haritası\n"
                 r"Drude-Lorentz Banyo: $\tau_c=$" f"{1/GAMMA_C_RADPS:.1f}" r"$\,\mathrm{ps}$, "
                 f"max_depth={HEOM_MAX_DEPTH}, " r"$N_k=$" f"{HEOM_NK}",
                 fontsize=12)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "heom_robustness_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Yayın kalitesinde heatmap kaydedildi: {out_path}")

    # --- İkincil grafik: <t>(gamma, r) haritası ---
    fig2, ax2 = plt.subplots(figsize=(10, 7.5))
    mesh2 = ax2.pcolormesh(gamma_range, r_range, mean_time_grid, shading="auto", cmap="plasma_r")
    cbar2 = plt.colorbar(mesh2, ax=ax2)
    cbar2.set_label(r"Ortalama Yakalanma Süresi $\langle t \rangle\ [\mathrm{ps}]$", fontsize=13)

    if np.any(~np.isnan(mean_time_grid)):
        i_tmin, j_tmin = np.unravel_index(np.nanargmin(mean_time_grid), mean_time_grid.shape)
        ax2.plot(gamma_range[j_tmin], r_range[i_tmin], marker='*', color='cyan',
                  markersize=22, markeredgecolor='black', zorder=6)
        ax2.annotate(
            r"min $\langle t \rangle$" f" = {mean_time_grid[i_tmin, j_tmin]:.3f} ps",
            xy=(gamma_range[j_tmin], r_range[i_tmin]), xytext=(0.6, 0.85),
            textcoords='axes fraction', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='cyan'),
            arrowprops=dict(arrowstyle='->', color='cyan', lw=1.5)
        )

    ax2.set_xlabel(r"Dephasing / Banyo Etkileşim Oranı $\gamma\ [\mathrm{ps}^{-1}]$", fontsize=13)
    ax2.set_ylabel(r"Donör-Akseptör Mesafesi $r\ [\mathrm{nm}]$", fontsize=13)
    ax2.set_title(r"Non-Markovian (HEOM) Ortalama Yakalanma Süresi $\langle t \rangle(\gamma, r)$",
                  fontsize=12)

    plt.tight_layout()
    out_path2 = os.path.join(output_dir, "heom_mean_trapping_time_heatmap.png")
    plt.savefig(out_path2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] <t> heatmap kaydedildi: {out_path2}")

    return gamma_opt, r_opt, eta_opt


# =========================================================================
# 8) ANA ÇALIŞTIRMA
# =========================================================================
def main():
    gamma_range = (np.logspace(np.log10(GAMMA_MIN_PSINV), np.log10(GAMMA_MAX_PSINV), N_GAMMA)
                    if GAMMA_LOG_SPACING
                    else np.linspace(GAMMA_MIN_PSINV, GAMMA_MAX_PSINV, N_GAMMA))
    r_range = np.linspace(R_MIN_NM, R_MAX_NM, N_R)

    print("=" * 74)
    print("ADIM 2.3-GENİŞLETME: 2D (gamma, r) HEOM Robustness Taraması")
    print(f"Izgara: {N_GAMMA} (gamma) x {N_R} (r) = {N_GAMMA*N_R} nokta")
    print(f"gamma araligi: [{GAMMA_MIN_PSINV}, {GAMMA_MAX_PSINV}] ps^-1 "
          f"({'log' if GAMMA_LOG_SPACING else 'lineer'} aralikli)")
    print(f"r araligi: [{R_MIN_NM}, {R_MAX_NM}] nm (lineer araliki)")
    print(f"HEOM: max_depth={HEOM_MAX_DEPTH}, Nk={HEOM_NK}, "
          f"gamma_c={GAMMA_C_RADPS} ps^-1 (tau_c={1/GAMMA_C_RADPS} ps)")
    print("=" * 74)

    eta_grid, mean_time_grid, failed = run_2d_heom_scan(gamma_range, r_range, n_workers=N_WORKERS)

    save_results(gamma_range, r_range, eta_grid, mean_time_grid)

    result = plot_robustness_heatmap(gamma_range, r_range, eta_grid, mean_time_grid)
    if result is not None:
        gamma_opt, r_opt, eta_opt = result
        print(f"\n>>> GENEL SWEET SPOT: gamma={gamma_opt:.3f} ps^-1, r={r_opt:.3f} nm, "
              f"eta_max={eta_opt:.5f}")

    print("\n[OK] 2D HEOM taraması tamamlandı.")


if __name__ == "__main__":
    main()
