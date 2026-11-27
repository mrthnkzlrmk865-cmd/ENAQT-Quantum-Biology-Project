#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fret_lab_io.py — Laboratuvar dosyalarını okuma/yazma yardımcıları
=================================================================
Dosya biçimleri (şablonlar: data/lab/*_SABLON.*):

1) Yapılar (constructs) CSV — ölçümden ÖNCE doldurulur:
     name,family,n,n_rep,notes
     Pro6,poliprolin,6,3,"Cys–Pro6–Cys, Alexa 488 / Alexa 594"
   family: "poliprolin" (n = prolin sayısı) veya "fp_baglayici" (n = bağlayıcı kalıntı sayısı)
   n_rep : planlanan bağımsız tekrar sayısı (tahmin aralığı buna göre hesaplanır)

2) Model ayarı (config) JSON:
     {
       "pair": "Alexa 488 / Alexa 594",
       "R0_lit_nm": 5.4,  "R0_rel_sd": 0.07,  "R0_source": "kaynak + koşullar",
       "fixed": {},                       # ör. {"delta": 0.0}
       "prior_overrides": {"d0": ["truncnormal", 0.6, 0.5, -1.0, 3.0]},
       "success_rule": {"min_fraction_in_90": 0.67, "max_mae": 0.10}
     }

3) Ölçümler CSV — ölçümden SONRA:
     construct,replicate,E            (doğrudan verim)   veya
     construct,replicate,F_DA,F_D     (E = 1 − F_DA/F_D hesaplanır)
   F_DA ve F_D, aynı koşullarda, arka plan/sızma düzeltmesi yapılmış donör
   floresans şiddetleridir (bkz. README "Deney planı").
"""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import fret_bayes as fb

FAMILIES = {"poliprolin", "fp_baglayici"}


def load_constructs(path):
    df = pd.read_csv(path)
    need = {"name", "family", "n"}
    if not need <= set(df.columns):
        raise ValueError(f"yapı dosyasında eksik sütun: {need - set(df.columns)}")
    bad = set(df.family) - FAMILIES
    if bad:
        raise ValueError(f"bilinmeyen aile(ler): {bad}; izin verilen: {FAMILIES}")
    if "n_rep" not in df:
        df["n_rep"] = 3
    if df.name.duplicated().any():
        raise ValueError("yapı adları benzersiz olmalı")
    return df


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in ("pair", "R0_lit_nm"):
        if k not in cfg:
            raise ValueError(f"config içinde '{k}' eksik")
    cfg.setdefault("R0_rel_sd", 0.07)
    cfg.setdefault("fixed", {})
    cfg.setdefault("prior_overrides", {})
    cfg.setdefault("success_rule", {"min_fraction_in_90": 0.67, "max_mae": 0.10})
    return cfg


def build_model(cons_df, cfg, kappa2="dynamic"):
    fams = tuple(sorted(set(cons_df.family)))
    pri = fb.default_priors(cfg["R0_lit_nm"], cfg["R0_rel_sd"], families=fams)
    for name, spec in cfg["prior_overrides"].items():
        kind, *args = spec
        if kind == "truncnormal":
            pri[name] = fb.Prior(kind, args[0], args[1], lo=args[2], hi=args[3])
        else:
            pri[name] = fb.Prior(kind, *args)
    constructs = [dict(name=r.name, family=r.family, n=float(r.n)) for r in cons_df.itertuples()]
    return fb.FretModel(constructs, pri, fixed=dict(cfg["fixed"]), kappa2=kappa2)


def load_measurements(path):
    df = pd.read_csv(path)
    if "E" not in df:
        if {"F_DA", "F_D"} <= set(df.columns):
            df["E"] = 1.0 - df.F_DA / df.F_D
        else:
            raise ValueError("ölçüm dosyasında 'E' ya da 'F_DA' + 'F_D' sütunları olmalı")
    if not {"construct", "replicate"} <= set(df.columns):
        raise ValueError("ölçüm dosyasında 'construct' ve 'replicate' sütunları olmalı")
    return df


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit(root):
    try:
        h = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return h + (" (kaydedilmemiş değişiklikler VAR)" if dirty else "")
    except Exception:
        return "bilinmiyor (git yok)"


def replicate_mean_predictive(model, X, constructs, n_rep, rng):
    """Planlanan n_rep tekrarın ORTALAMASI için öngörü örnekleri: clip(<E> + δ) + σ/√n · ε."""
    m = fb.predict(model, X, constructs)
    th = model.unpack(X)
    return m + (th["sigma"][:, None] / np.sqrt(np.asarray(n_rep, float))[None, :]) * rng.normal(size=m.shape)
