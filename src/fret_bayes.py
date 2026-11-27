#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fret_bayes.py — Laboratuvar yolu (B): Bayesçi Förster ileri modeli
==================================================================
Kararlı-hal (ansambl) FRET verimi E için fizik-temelli, belirsizliği nicelenmiş
bir model. Simülasyondan alınan şey sayısal R0 DEĞİL, fonksiyonel formdur:
v2'de (src/forster_limit.py) modelin FRET-benzeri rejimde
E(r) = 1 / (1 + (r/R0)^6) formuna indiği gösterildi.

Model (her yapı i, tekrar j):
    E_obs,ij = clip(<E>_i + δ, 0, 1) + ε_ij,     ε_ij ~ N(0, σ²)
    (ortalama verim fiziksel olarak [0, 1]'dedir; tekil ölçüm gürültü nedeniyle biraz dışına çıkabilir)
    <E>_i    = ∫ p(r | yapı i, θ) · 1/(1 + (r/R0_eff)^6) dr

Mesafe dağılımı p(r): iki ucun göreli konum vektörü = μ·û + Gauss(0, s²·I₃)
→ "3B Rice" (merkez dışı chi, 3 serbestlik derecesi) dağılımı:
    p(r) = r / (μ s √(2π)) · [exp(-(r-μ)²/2s²) - exp(-(r+μ)²/2s²)]
Aileler:
  * "poliprolin": μ = d0 + RISE·n  (PPII sarmal; RISE = 0.31 nm/kalıntı),
                  s = S_DYE (boya bağlayıcılarının konum yayılımı, sabit)
  * "fp_baglayici": esnek bağlayıcı (N kalıntı) solucan-benzeri zincir (WLC):
                  L = 0.38·N nm,  <R²> = 2 lp L [1 - (lp/L)(1 - e^{-L/lp})]
                  s² = <R²>/3 + S_FP²,  μ = a_eff (kromofor–bağlanma noktası
                  ofsetlerinin etkin büyüklüğü)

κ²: varsayılan dinamik ortalama (κ² = 2/3 → R0_eff = R0). "static" seçeneği
izotropik ama donmuş yönelimler için ⟨E⟩'yi κ² dağılımı üzerinden ortalar
(yalnızca duyarlılık analizi için).

Örnekleyici: Goodman & Weare (2010) afin-değişmez "stretch move" topluluk
örnekleyicisi (emcee'nin varsayılan algoritmasıyla aynı). Bu ortamda emcee/PyMC
kurulamadığı için kısa bir numpy uygulaması yazıldı; `run_ensemble` arayüzü,
emcee.EnsembleSampler ile kolayca değiştirilebilir. Tanılar: bütünleşik
özilinti süresi τ (Sokal/emcee yöntemi), ESS, bölünmüş R̂.

ÖNEMLİ VARSAYIMLAR (deneyden önce ODTÜ ekibiyle güncellenmeli):
  RISE = 0.31 nm, S_DYE = 0.6 nm, kalıntı kontur uzunluğu 0.38 nm, S_FP = 0.5 nm
  ve öncellerin merkez/genişlikleri. Bunlar tipik literatür büyüklükleridir;
  belirli bir boya/protein çifti için ölçülmüş değerler DEĞİLDİR.
"""
from dataclasses import dataclass, field
import numpy as np

# ----------------------------------------------------------------------
# Geometrik sabitler (varsayım; bkz. modül belgesi)
# ----------------------------------------------------------------------
RISE_PPII_NM = 0.31
S_DYE_NM = 0.6
RES_CONTOUR_NM = 0.38
S_FP_NM = 0.5

R_GRID = np.linspace(1e-3, 25.0, 2500)          # nm
_DR = R_GRID[1] - R_GRID[0]

KAPPA2_DYNAMIC = 2.0 / 3.0


def kappa2_static_samples(n=4000, seed=0):
    """İzotropik, donmuş yönelimler için κ² örnekleri (rastgele birim vektörler)."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(3, n, 3))
    v /= np.linalg.norm(v, axis=2, keepdims=True)
    d, a, rhat = v
    k = np.einsum("ij,ij->i", d, a) - 3 * np.einsum("ij,ij->i", d, rhat) * np.einsum("ij,ij->i", a, rhat)
    return k ** 2


# ----------------------------------------------------------------------
# Mesafe dağılımı ve ortalama verim
# ----------------------------------------------------------------------
def rice3d_pdf(r, mu, s):
    """3B Rice yoğunluğu; mu, s yayınlanabilir (…, 1) biçimlerde olabilir."""
    mu = np.maximum(mu, 1e-6)
    z = 1.0 / (mu * s * np.sqrt(2 * np.pi))
    return z * r * (np.exp(-0.5 * ((r - mu) / s) ** 2) - np.exp(-0.5 * ((r + mu) / s) ** 2))


def wlc_mean_sq(N_res, lp_nm):
    L = RES_CONTOUR_NM * np.asarray(N_res, dtype=float)
    lp = np.asarray(lp_nm, dtype=float)
    return 2 * lp * L * (1 - (lp / L) * (1 - np.exp(-L / lp)))


def distance_params(family, n, theta):
    """(mu, s) — theta: dict, değerleri (W,) dizileri (W = yürüyücü sayısı)."""
    if family == "poliprolin":
        mu = theta["d0"] + RISE_PPII_NM * n
        s = np.full_like(mu, S_DYE_NM)
    elif family == "fp_baglayici":
        s = np.sqrt(wlc_mean_sq(n, theta["lp"]) / 3.0 + S_FP_NM ** 2)
        mu = theta["a_eff"] * np.ones_like(s)
    else:
        raise ValueError(f"bilinmeyen aile: {family}")
    return mu, s


def mean_efficiency(mu, s, R0, kappa2="dynamic", k2_samples=None):
    """<E> = ∫ p(r) E(r) dr; mu, s, R0: (W,) diziler. Dönüş: (W,)."""
    mu, s, R0 = (np.atleast_1d(x)[:, None] for x in (mu, s, R0))
    p = rice3d_pdf(R_GRID[None, :], mu, s)
    norm = p.sum(axis=1) * _DR
    if kappa2 == "dynamic":
        E = 1.0 / (1.0 + (R_GRID[None, :] / R0) ** 6)
        return (p * E).sum(axis=1) * _DR / norm
    # statik izotropik: R0_eff^6 = R0^6 · κ²/(2/3); κ² üzerinden ortalama
    k2 = kappa2_static_samples() if k2_samples is None else k2_samples
    q = np.quantile(k2, np.linspace(0.0025, 0.9975, 200))      # 200 kuantil düğümü
    out = np.zeros(mu.shape[0])
    for kk in q:
        R0e6 = R0 ** 6 * (kk / KAPPA2_DYNAMIC)
        E = R0e6 / (R0e6 + R_GRID[None, :] ** 6)
        out += (p * E).sum(axis=1) * _DR
    return out / len(q) / norm


# ----------------------------------------------------------------------
# Öncel (prior) tanımları
# ----------------------------------------------------------------------
@dataclass
class Prior:
    kind: str           # "lognormal" | "normal" | "halfnormal" | "truncnormal"
    a: float
    b: float = 0.0
    lo: float = -np.inf
    hi: float = np.inf

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        if self.kind == "lognormal":            # a = medyan, b = log-ölçek sd
            ok = x > 0
            lx = np.log(np.where(ok, x, 1.0))
            return np.where(ok, -0.5 * ((lx - np.log(self.a)) / self.b) ** 2 - lx, -np.inf)
        if self.kind == "normal":
            return -0.5 * ((x - self.a) / self.b) ** 2
        if self.kind == "halfnormal":           # a = ölçek
            return np.where(x > 0, -0.5 * (x / self.a) ** 2, -np.inf)
        if self.kind == "truncnormal":
            ok = (x > self.lo) & (x < self.hi)
            return np.where(ok, -0.5 * ((x - self.a) / self.b) ** 2, -np.inf)
        raise ValueError(self.kind)

    def sample(self, rng, n):
        if self.kind == "lognormal":
            return self.a * np.exp(self.b * rng.normal(size=n))
        if self.kind == "normal":
            return self.a + self.b * rng.normal(size=n)
        if self.kind == "halfnormal":
            return np.abs(self.a * rng.normal(size=n))
        if self.kind == "truncnormal":
            out = np.empty(0)
            while out.size < n:
                x = self.a + self.b * rng.normal(size=4 * n)
                out = np.concatenate([out, x[(x > self.lo) & (x < self.hi)]])
            return out[:n]
        raise ValueError(self.kind)

    def describe(self):
        if self.kind == "lognormal":
            return f"log-normal(medyan {self.a:g}, log-sd {self.b:g})"
        if self.kind == "normal":
            return f"normal({self.a:g}, {self.b:g})"
        if self.kind == "halfnormal":
            return f"yarı-normal(ölçek {self.a:g})"
        return f"kesik normal({self.a:g}, {self.b:g}; [{self.lo:g}, {self.hi:g}])"


def default_priors(R0_lit_nm, R0_rel_sd=0.07, families=("poliprolin",)):
    """
    Varsayılan öncel seti. R0: literatür değeri ± %7 (Hellenkamp ve ark. 2018,
    çoğunlukla κ² belirsizliği). Diğerleri geniş, fiziksel olarak makul aralıklar.
    """
    pri = {
        "R0": Prior("lognormal", R0_lit_nm, R0_rel_sd),
        "delta": Prior("normal", 0.0, 0.05),
        "sigma": Prior("halfnormal", 0.05),
    }
    if "poliprolin" in families:
        pri["d0"] = Prior("truncnormal", 0.6, 0.5, lo=-1.0, hi=3.0)
    if "fp_baglayici" in families:
        pri["lp"] = Prior("lognormal", 0.45, 0.5)          # Evers ve ark. 2006: 4.5 Å
        pri["a_eff"] = Prior("truncnormal", 3.0, 1.5, lo=0.0, hi=10.0)
    return pri


# ----------------------------------------------------------------------
# Model nesnesi
# ----------------------------------------------------------------------
@dataclass
class FretModel:
    constructs: list                     # [{"name","family","n"}]
    priors: dict                         # isim -> Prior
    fixed: dict = field(default_factory=dict)
    kappa2: str = "dynamic"

    @property
    def free(self):
        return [k for k in self.priors if k not in self.fixed]

    def unpack(self, X):
        th = {k: X[:, i] for i, k in enumerate(self.free)}
        W = X.shape[0]
        for k, v in self.fixed.items():
            th[k] = np.full(W, float(v))
        return th

    def construct_means(self, X):
        """(W, n_yapı) — δ HARİÇ yapı başına <E>."""
        th = self.unpack(X)
        cols = []
        for c in self.constructs:
            mu, s = distance_params(c["family"], c["n"], th)
            cols.append(mean_efficiency(mu, s, th["R0"], kappa2=self.kappa2))
        return np.stack(cols, axis=1)

    def log_prior(self, X):
        th = self.unpack(X)
        lp = np.zeros(X.shape[0])
        for k in self.free:
            lp = lp + self.priors[k].logpdf(th[k])
        return lp

    def log_prob(self, X, data_idx, data_E):
        """data_idx: her ölçümün yapı indeksi; data_E: ölçülen E değerleri."""
        lp = self.log_prior(X)
        ok = np.isfinite(lp)
        out = np.full(X.shape[0], -np.inf)
        if not ok.any():
            return out
        Xo = X[ok]
        th = self.unpack(Xo)
        m = np.clip(self.construct_means(Xo) + th["delta"][:, None], 0.0, 1.0)   # ortalama verim [0, 1]
        sig = th["sigma"][:, None]
        resid = (data_E[None, :] - m[:, data_idx]) / sig
        ll = -0.5 * (resid ** 2).sum(axis=1) - len(data_E) * np.log(th["sigma"])
        out[ok] = lp[ok] + ll
        return out

    def sample_prior(self, rng, n):
        return np.column_stack([self.priors[k].sample(rng, n) for k in self.free])


# ----------------------------------------------------------------------
# Goodman–Weare topluluk örnekleyicisi (emcee'nin "stretch move"u)
# ----------------------------------------------------------------------
def run_ensemble(log_prob, X0, n_steps, a=2.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    W, D = X0.shape
    assert W % 2 == 0 and W >= 2 * D
    X = X0.copy()
    lp = log_prob(X)
    chain = np.empty((n_steps, W, D))
    lps = np.empty((n_steps, W))
    acc = 0
    half = W // 2
    for t in range(n_steps):
        for s0, s1 in ((slice(0, half), slice(half, W)), (slice(half, W), slice(0, half))):
            S, C = X[s0], X[s1]
            z = ((a - 1.0) * rng.random(half) + 1.0) ** 2 / a
            partners = C[rng.integers(0, half, size=half)]
            Y = partners + z[:, None] * (S - partners)
            lpY = log_prob(Y)
            logr = (D - 1) * np.log(z) + lpY - lp[s0]
            accept = np.log(rng.random(half)) < logr
            S[accept] = Y[accept]
            lp[s0] = np.where(accept, lpY, lp[s0])
            X[s0] = S
            acc += accept.sum()
        chain[t], lps[t] = X, lp
    return chain, lps, acc / (n_steps * W)


def autocorr_time(chain, c=5.0):
    """emcee yöntemiyle (Goodman & Weare 2010; Sokal) parametre başına τ."""
    n, W, D = chain.shape
    taus = []
    for d in range(D):
        x = chain[:, :, d] - chain[:, :, d].mean(axis=0)
        f = np.fft.rfft(x, n=2 * n, axis=0)
        acf = np.fft.irfft(f * np.conj(f), axis=0)[:n].mean(axis=1)
        acf /= acf[0] if acf[0] > 0 else 1.0
        taus_w = 2.0 * np.cumsum(acf) - 1.0
        m = np.arange(len(taus_w)) < c * taus_w
        win = np.argmin(m) if not m.all() else len(taus_w) - 1
        taus.append(taus_w[win])
    return np.array(taus)


def split_rhat(chain):
    """Yürüyücüleri zincir sayarak bölünmüş R̂ (Gelman ve ark.). (n, W, D) → (D,)."""
    n = chain.shape[0] // 2
    c = np.concatenate([chain[:n], chain[n:2 * n]], axis=1)        # (n, 2W, D)
    m = c.shape[1]
    means = c.mean(axis=0)
    B = n * means.var(axis=0, ddof=1)
    Wv = c.var(axis=0, ddof=1).mean(axis=0)
    var = (n - 1) / n * Wv + B / n
    return np.sqrt(var / Wv)


@dataclass
class FitResult:
    model: FretModel
    samples: np.ndarray          # (N, D) yanma sonrası, inceltilmiş değil
    acc: float
    tau: np.ndarray
    rhat: np.ndarray
    n_steps: int
    burn: int

    @property
    def ess(self):
        n_post = self.n_steps - self.burn
        W = self.samples.shape[0] // n_post
        return n_post * W / self.tau

    def converged(self):
        """emcee önerisi: zincir uzunluğu > 50 τ. Yürüyücüler bağımsız zincir
        olmadığından R̂ yalnızca ikincil ölçüttür (eşik 1.05)."""
        return bool((self.rhat < 1.05).all() and ((self.n_steps - self.burn) > 50 * self.tau).all())

    def summary(self, q=(0.05, 0.5, 0.95)):
        rows = {}
        for i, k in enumerate(self.model.free):
            v = self.samples[:, i]
            rows[k] = dict(zip(["q05", "medyan", "q95"], np.quantile(v, q)), ort=v.mean(), sd=v.std(),
                           rhat=self.rhat[i], ess=self.ess[i], tau=self.tau[i])
        return rows


def fit(model: FretModel, data_idx, data_E, n_walkers=32, n_steps=3000, burn=1000, seed=0):
    rng = np.random.default_rng(seed)
    D = len(model.free)
    n_walkers = max(n_walkers, 2 * D + (2 * D) % 2)
    # başlangıç: öncelden çekilip olabilirliği sonlu olan noktalar
    X0 = model.sample_prior(rng, 20 * n_walkers)
    lp0 = model.log_prob(X0, data_idx, data_E)
    order = np.argsort(-lp0)
    X0 = X0[order[:n_walkers]]
    chain, lps, acc = run_ensemble(lambda X: model.log_prob(X, data_idx, data_E), X0, n_steps, rng=rng)
    post = chain[burn:]
    tau = autocorr_time(post)
    rh = split_rhat(post)
    return FitResult(model, post.reshape(-1, D), acc, tau, rh, n_steps, burn)


def predict(model: FretModel, X, constructs, rng=None, with_noise=False):
    """Verilen parametre örnekleri X için yeni yapıların E tahmini (δ dahil)."""
    rng = np.random.default_rng(1) if rng is None else rng
    sub = FretModel(constructs, model.priors, model.fixed, model.kappa2)
    th = sub.unpack(X)
    m = np.clip(sub.construct_means(X) + th["delta"][:, None], 0.0, 1.0)
    if with_noise:
        m = m + th["sigma"][:, None] * rng.normal(size=m.shape)
    return m
