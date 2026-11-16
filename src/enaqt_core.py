#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enaqt_core.py — Revizyon (v2) için ortak, bağımlılığı az çekirdek modül
=======================================================================

Bu modül yalnızca numpy + scipy kullanır (QuTiP gerekmez). Orijinal
scriptlerle (src/heom_2d_robustness_scan.py, QuTiP 5.3) AYNI fiziksel
modeli kurar:

  * 2 site (donör D, akseptör A), H = [[E_D, V(r)], [V(r), E_A]] (cm^-1)
  * V(r) = V0 (r0/r)^3
  * Markovian kanallar: D<->A gevşeme (detailed balance), SINK (A'dan
    Gamma_sink ile yakalama), LOSS (her iki siteden Gamma_loss ile kayıp)
  * Dephasing: (a) Lindblad: sqrt(gamma) * |n><n| her sitede, veya
               (b) HEOM: her sitede bağımsız Drude–Lorentz banyosu,
                   lambda = gamma * gamma_c / (2 kT) (orijinal eşleştirme)

Neden 2 boyut? Orijinal kod 4 boyutlu uzay (D, A, Sink, Loss) kullanır.
Sink ve Loss "emici" durumlardır; oradan D/A'ya geri akış yoktur. Bu
yüzden P_D(t), P_A(t) ve D–A koheransı, sink/loss kanallarını yalnızca
"sönüm" (iz-azaltan, -1/2 {c^dag c, rho}) olarak yazan 2 boyutlu modelde
BİREBİR aynıdır. Avantajı: hareket denklemi matrisi M tersinir olur ve
verim ile ortalama yakalanma süresi SONSUZ ZAMANA kadar analitik olarak
hesaplanabilir:

    int_0^inf x(t) dt   = -M^{-1} x0
    int_0^inf t x(t) dt =  M^{-2} x0

    eta  = Gamma_sink * int P_A dt
    <t>  = Gamma_sink * int t P_A dt / eta

Bu, orijinal kodda T_MAX = 25 ps'de kesilen trapez integralinin büyük r'de
yarattığı yapay düşük-eta / tavanlı-<t> sorununu ortadan kaldırır.

Doğrulama: src/validate_against_legacy_csv.py bu modülün zaman-alanı
çözümünü (T_MAX=25 ps, 220 nokta, trapez) orijinal QuTiP CSV'sinin
400 noktasıyla karşılaştırır.
"""

from dataclasses import dataclass, replace
from itertools import product
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---------------------------------------------------------------------
# Birimler (orijinal kodla aynı)
# ---------------------------------------------------------------------
C_LIGHT_CM_PER_PS = 2.99792458e-2
CM1_TO_RADPS = 2 * np.pi * C_LIGHT_CM_PER_PS   # ~0.188365 rad/ps per cm^-1
KB_CM1_PER_K = 0.695034800

D, A = 0, 1          # site indeksleri
DIM = 2


@dataclass(frozen=True)
class ModelParams:
    """Orijinal 'kanonik' ENAQT parametreleri (src/heom_2d_robustness_scan.py)."""
    E_donor_cm1: float = 25.0
    E_acceptor_cm1: float = 0.0
    V0_cm1: float = 20.0
    r0_nm: float = 2.0
    gamma_relax_psinv: float = 0.02
    gamma_sink_psinv: float = 1.0
    gamma_loss_psinv: float = 0.001
    T_kelvin: float = 298.0

    def with_(self, **kw):
        return replace(self, **kw)


CANONICAL = ModelParams()


def coupling_cm1(r_nm, p: ModelParams):
    return p.V0_cm1 * (p.r0_nm / r_nm) ** 3


def kT_radps(p: ModelParams):
    return KB_CM1_PER_K * p.T_kelvin * CM1_TO_RADPS


# ---------------------------------------------------------------------
# Süperoperatör yardımcıları (sütun-yığma vec: vec(A rho B) = (B^T ⊗ A) vec rho)
# ---------------------------------------------------------------------
_I = np.eye(DIM)


def _spre(X):
    return np.kron(_I, X)


def _spost(X):
    return np.kron(X.T, _I)


def _dissipator(c):
    cdc = c.conj().T @ c
    return np.kron(c.conj(), c) - 0.5 * _spre(cdc) - 0.5 * _spost(cdc)


def _proj(n):
    P = np.zeros((DIM, DIM))
    P[n, n] = 1.0
    return P


def vec_index(i, j):
    """rho[i, j] elemanının vec(rho) içindeki indeksi (sütun-yığma)."""
    return i + DIM * j


IDX_DD = vec_index(D, D)
IDX_AA = vec_index(A, A)


def system_liouvillian(r_nm, p: ModelParams = CANONICAL, gamma_deph_psinv=0.0):
    """
    2x2 iz-azaltan Liouvillian (4x4 matris, rad/ps):
      koherent H + gevşeme (D->A, A->D detailed balance)
      + SINK sönümü (A) + LOSS sönümü (D, A)
      + isteğe bağlı Lindblad saf dephasing sqrt(gamma)|n><n| (n = D, A).
    HEOM için gamma_deph_psinv = 0 bırakılır (dephasing banyodan gelir).
    """
    V = coupling_cm1(r_nm, p)
    H = np.array([[p.E_donor_cm1, V], [V, p.E_acceptor_cm1]], dtype=complex) * CM1_TO_RADPS
    L = -1j * (_spre(H) - _spost(H))

    dE = p.E_donor_cm1 - p.E_acceptor_cm1
    boltz = np.exp(-dE / (KB_CM1_PER_K * p.T_kelvin))
    sig_AD = np.zeros((DIM, DIM)); sig_AD[A, D] = 1.0      # |A><D|
    sig_DA = np.zeros((DIM, DIM)); sig_DA[D, A] = 1.0      # |D><A|
    L = L + _dissipator(np.sqrt(p.gamma_relax_psinv) * sig_AD)
    L = L + _dissipator(np.sqrt(p.gamma_relax_psinv * boltz) * sig_DA)

    # SINK ve LOSS: yalnızca sönüm kısmı (atlama terimi emici duruma gider)
    PA = _proj(A)
    L = L - 0.5 * p.gamma_sink_psinv * (_spre(PA) + _spost(PA))
    L = L - p.gamma_loss_psinv * np.eye(DIM * DIM)

    if gamma_deph_psinv > 0:
        for n in (D, A):
            L = L + _dissipator(np.sqrt(gamma_deph_psinv) * _proj(n))
    return L


def initial_state_vec():
    x0 = np.zeros(DIM * DIM, dtype=complex)
    x0[IDX_DD] = 1.0
    return x0


# ---------------------------------------------------------------------
# HEOM (Drude–Lorentz, Matsubara açılımı) — QuTiP DrudeLorentzBath ile
# aynı üstel katsayılar:
#   c_0 = lam*gc*(cot(gc/(2T)) - i),  nu_0 = gc
#   c_k = 4*lam*gc*T*nu_k/(nu_k^2 - gc^2),  nu_k = 2*pi*k*T   (k=1..Nk)
# (T burada enerji birimindedir: kT, rad/ps)
# ---------------------------------------------------------------------
def drude_lorentz_exponents(lam, gc, T, Nk):
    ck = [lam * gc * (1.0 / np.tan(gc / (2.0 * T)) - 1j)]
    vk = [gc]
    for k in range(1, Nk + 1):
        nu = 2.0 * np.pi * k * T
        ck.append(4.0 * lam * gc * T * nu / (nu ** 2 - gc ** 2))
        vk.append(nu)
    return np.array(ck, dtype=complex), np.array(vk, dtype=float)


def _ado_labels(K, depth):
    labels = [n for n in product(range(depth + 1), repeat=K) if sum(n) <= depth]
    labels.sort(key=lambda n: (sum(n), n))
    return labels


def heom_generator(r_nm, gamma_psinv, p: ModelParams = CANONICAL,
                   gamma_c_radps=2.0, depth=5, Nk=1, lam_override=None):
    """
    Tam HEOM hareket denklemi matrisi M (seyrek, CSC). Sistem Liouvillian'ı
    (H + Markovian kanallar) TÜM yardımcı yoğunluk operatörlerine (ADO)
    uygulanır — QuTiP HEOMSolver ile aynı.

    lam = gamma * gamma_c / (2 kT) (orijinal 'Haken–Strobl eşleştirmesi').
    Döndürür: (M, n_ados)
    """
    T = kT_radps(p)
    lam = gamma_psinv * gamma_c_radps / (2.0 * T) if lam_override is None else lam_override
    ck, vk = drude_lorentz_exponents(lam, gamma_c_radps, T, Nk)

    Ls = system_liouvillian(r_nm, p, gamma_deph_psinv=0.0)

    # Her site için bağımsız banyo -> toplam K = 2 * (Nk + 1) üstel terim
    exps = []
    for n in (D, A):
        Q = _proj(n)
        comm = _spre(Q) - _spost(Q)
        acomm = _spre(Q) + _spost(Q)
        for c, nu in zip(ck, vk):
            exps.append((comm, acomm, c.real, c.imag, nu))
    K = len(exps)

    labels = _ado_labels(K, depth)
    index = {lab: i for i, lab in enumerate(labels)}
    nA = len(labels)
    b = DIM * DIM

    rows, cols, vals = [], [], []

    def add_block(i, j, blk):
        blk = sp.coo_matrix(blk)
        rows.append(blk.row + i * b)
        cols.append(blk.col + j * b)
        vals.append(blk.data)

    for i, n in enumerate(labels):
        diag = Ls - sum(n[k] * exps[k][4] for k in range(K)) * np.eye(b)
        add_block(i, i, diag)
        for k in range(K):
            comm, acomm, cr, ci, nu = exps[k]
            # yukarı bağlantı: -i [Q, rho_{n+e_k}]
            up = list(n); up[k] += 1; up = tuple(up)
            if up in index:
                add_block(i, index[up], -1j * comm)
            # aşağı bağlantı: -i n_k (c_R [Q, .] + i c_I {Q, .}) rho_{n-e_k}
            if n[k] > 0:
                dn = list(n); dn[k] -= 1; dn = tuple(dn)
                add_block(i, index[dn], -1j * n[k] * (cr * comm + 1j * ci * acomm))

    M = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nA * b, nA * b))
    return M, nA


def heom_initial_vec(n_ados):
    x0 = np.zeros(n_ados * DIM * DIM, dtype=complex)
    x0[IDX_DD] = 1.0          # yalnızca fiziksel rho (ADO 0) dolu
    return x0


# ---------------------------------------------------------------------
# Metrikler
# ---------------------------------------------------------------------
def infinite_time_metrics(M, x0, p: ModelParams = CANONICAL):
    """
    Sonsuz zamana kadar tam eta ve <t> (zaman kesmesi yok).
    M: (seyrek veya yoğun) hareket denklemi matrisi; ilk 4 bileşen fiziksel rho.
    """
    Ms = sp.csc_matrix(M)
    lu = spla.splu(Ms)
    x_int = lu.solve(-x0)            # int x dt
    x_tint = lu.solve(-x_int)        # int t x dt
    eta = p.gamma_sink_psinv * x_int[IDX_AA].real
    mean_t = p.gamma_sink_psinv * x_tint[IDX_AA].real / eta
    return float(eta), float(mean_t)


def time_domain_PA(M, x0, tlist):
    """P_A(t) zaman serisi (expm_multiply, eşit aralıklı tlist)."""
    from scipy.linalg import expm
    Md = M.toarray() if sp.issparse(M) else np.asarray(M)
    dt = tlist[1] - tlist[0]
    U = expm(Md * dt)                 # tek adım propagatörü (eşit aralık)
    X = np.empty((len(tlist), len(x0)), dtype=complex)
    x = x0.copy()
    for k in range(len(tlist)):
        X[k] = x
        x = U @ x
    return X[:, IDX_AA].real, X[:, IDX_DD].real


def truncated_trapz_metrics(tlist, PA, p: ModelParams = CANONICAL):
    """Orijinal kodun tanımı (T_MAX'ta kesilen trapez) — yalnızca karşılaştırma için."""
    flux = p.gamma_sink_psinv * PA
    eta = np.trapezoid(flux, tlist)
    return float(eta), float(np.trapezoid(tlist * flux, tlist) / eta)


# Kısa yollar ----------------------------------------------------------
def lindblad_metrics(r_nm, gamma_psinv, p: ModelParams = CANONICAL):
    M = system_liouvillian(r_nm, p, gamma_deph_psinv=gamma_psinv)
    return infinite_time_metrics(M, initial_state_vec(), p)


def heom_metrics(r_nm, gamma_psinv, p: ModelParams = CANONICAL, **kw):
    M, nA = heom_generator(r_nm, gamma_psinv, p, **kw)
    return infinite_time_metrics(M, heom_initial_vec(nA), p)


def n_ados(K, depth):
    return math.comb(K + depth, depth)
