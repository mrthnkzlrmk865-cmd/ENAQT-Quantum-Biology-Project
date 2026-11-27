#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fret_compare.py — Faz 4: ölçümleri ön-kayıtlı tahminlerle karşılaştırma
========================================================================
Üç adım, bu SIRAYLA:
  1. DONMUŞ KARŞILAŞTIRMA (güncelleme yok): ölçülen tekrar ortalamaları,
     ön-kayıtta yazılmış %90 aralıklarla ve önceden belirlenen başarı
     kuralıyla karşılaştırılır. Asıl test budur.
  2. POSTERIOR GÜNCELLEME: tüm ölçümlerle MCMC; R0, yapı parametresi, δ, σ.
     δ'nın büyüklüğü bilimsel bulgudur: δ ≈ 0 → Förster + basit geometri
     yeterli; |δ| belirgin → modelde eksik fizik (κ², cis-prolin, bağlayıcı
     etkileşimleri, kalibrasyon sistematiği…).
  3. BİR-YAPIYI-DIŞARIDA-BIRAK (LOCO): her yapı, onu görmeden uydurulan
     modelle tahmin edilir. Modelin yeni yapılara genellemesini ölçer.

Kullanım:
  python src/fret_compare.py --prereg results/fret/prereg/<etiket> \
                             --measurements data/lab/olcumler.csv \
                             --out results/fret/compare/<etiket>

Çıktılar: REPORT.md, frozen_comparison.csv, posterior_summary.csv, loco.csv,
          compare_predictions.png, posterior_delta.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fret_bayes as fb
import fret_lab_io as io

ROOT = Path(__file__).resolve().parents[1]
FIT_KW = dict(n_walkers=32, n_steps=5000, burn=1500)


def fit_subset(cons, cfg, meas, keep_names, seed):
    sub = cons[cons.name.isin(keep_names)].reset_index(drop=True)
    model = io.build_model(sub, cfg)
    name_to_i = {n: i for i, n in enumerate(sub.name)}
    mm = meas[meas.construct.isin(keep_names)]
    idx = mm.construct.map(name_to_i).values
    return model, fb.fit(model, idx, mm.E.values.astype(float), seed=seed, **FIT_KW)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--measurements", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--banner", default="")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pre_dir = Path(a.prereg)

    meta = json.loads((pre_dir / "prereg_meta.json").read_text(encoding="utf-8"))
    pred = pd.read_csv(pre_dir / "prereg_predictions.csv")
    cons = io.load_constructs(meta["constructs_path"])
    cfg = meta["config"]
    changed = [k for k, p in (("yapılar", meta["constructs_path"]), ("ayar", meta["config_path"]))
               if io.sha256(p) != meta["sha256"][k]]
    meas = io.load_measurements(a.measurements)
    unknown = set(meas.construct) - set(cons.name)
    if unknown:
        raise ValueError(f"ön-kayıtta olmayan yapı(lar): {unknown}")

    # ---------------- 1) donmuş karşılaştırma ----------------
    g = meas.groupby("construct").E
    obs = pd.DataFrame({"E_obs_mean": g.mean(), "E_obs_sd": g.std(ddof=1), "n_obs": g.size()})
    obs["E_obs_se"] = obs.E_obs_sd / np.sqrt(obs.n_obs)
    fz = pred.set_index("name").join(obs, how="inner")
    fz["in_90"] = (fz.E_obs_mean >= fz.E_repmean_q05) & (fz.E_obs_mean <= fz.E_repmean_q95)
    fz["abs_err"] = (fz.E_obs_mean - fz.E_med).abs()
    fz["n_rep_mismatch"] = fz.n_obs != fz.n_rep
    fz.to_csv(out / "frozen_comparison.csv")
    rule = cfg["success_rule"]
    frac, mae = fz.in_90.mean(), fz.abs_err.mean()
    success = (frac >= rule["min_fraction_in_90"]) and (mae < rule["max_mae"])

    # ---------------- 2) posterior ----------------
    model, res = fit_subset(cons, cfg, meas, list(cons.name), seed=11)
    summ = pd.DataFrame(res.summary()).T
    summ.to_csv(out / "posterior_summary.csv")
    d_i = model.free.index("delta") if "delta" in model.free else None
    rng = np.random.default_rng(5)
    S = res.samples[rng.choice(len(res.samples), 6000, replace=False)]
    post_mean = fb.predict(model, S, model.constructs)
    prior_X = model.sample_prior(np.random.default_rng(0), 20000)

    # ---------------- 3) LOCO ----------------
    loco = []
    if len(cons) >= 3:
        for k, name in enumerate(cons.name):
            keep = [n for n in cons.name if n != name]
            mk, rk = fit_subset(cons, cfg, meas, keep, seed=100 + k)
            Sk = rk.samples[rng.choice(len(rk.samples), 4000, replace=False)]
            target = [c for c in io.build_model(cons, cfg).constructs if c["name"] == name]
            n_obs = int(obs.loc[name, "n_obs"]) if name in obs.index else 3
            pk = io.replicate_mean_predictive(mk, Sk, target, [n_obs], rng)[:, 0]
            lo, med, hi = np.quantile(pk, [0.05, 0.5, 0.95])
            e = obs.loc[name, "E_obs_mean"] if name in obs.index else np.nan
            loco.append(dict(construct=name, pred_med=med, pred_q05=lo, pred_q95=hi, E_obs_mean=e,
                             in_90=lo <= e <= hi, abs_err=abs(e - med), converged=rk.converged()))
    loco = pd.DataFrame(loco)
    loco.to_csv(out / "loco.csv", index=False)

    # ---------------- rapor ----------------
    def r3(x): return f"{x:.3f}"
    lines = []
    if a.banner:
        lines += [f"> **{a.banner}**", ""]
    lines += ["# FRET karşılaştırma raporu", "",
              f"Ön-kayıt: `{pre_dir}` (oluşturulma {meta['created']}, commit `{meta['git_commit']}`)",
              f"Ölçümler: `{a.measurements}` ({len(meas)} ölçüm, {meas.construct.nunique()} yapı)", ""]
    if changed:
        lines += [f"**UYARI:** ön-kayıttan sonra değişmiş girdi dosyaları: {', '.join(changed)}. "
                  "Donmuş karşılaştırma ön-kayıttaki tahminleri kullanır, ancak bu durum raporlanmalıdır.", ""]
    if fz.n_rep_mismatch.any():
        lines += ["**Not:** bazı yapılarda ölçülen tekrar sayısı planlanandan farklı: "
                  + ", ".join(f"{n} ({int(r.n_obs)}/{int(r.n_rep)})" for n, r in fz[fz.n_rep_mismatch].iterrows()), ""]
    lines += ["## 1. Donmuş karşılaştırma (ön-kayıtlı tahminler, güncelleme yok)", "",
              "| Yapı | Tahmin ⟨E⟩ medyan | %90 aralık (tekrar ort.) | Ölçülen ort. ± SE | Aralıkta mı? | Mutlak hata |",
              "|---|---|---|---|---|---|"]
    for n, r in fz.iterrows():
        lines.append(f"| {n} | {r3(r.E_med)} | [{r3(r.E_repmean_q05)}, {r3(r.E_repmean_q95)}] | "
                     f"{r3(r.E_obs_mean)} ± {r3(r.E_obs_se)} | {'evet' if r.in_90 else '**hayır**'} | {r3(r.abs_err)} |")
    lines += ["", f"Aralık içinde: **{int(fz.in_90.sum())}/{len(fz)}** (kural ≥ %{100*rule['min_fraction_in_90']:.0f}); "
              f"ortalama mutlak hata **{mae:.3f}** (kural < {rule['max_mae']}).",
              f"**Önceden belirlenen başarı kuralı: {'SAĞLANDI' if success else 'SAĞLANMADI'}.**", "",
              "## 2. Posterior (tüm ölçümlerle güncelleme)", "",
              f"Yakınsama: {'evet' if res.converged() else '**HAYIR — sonuçlar güvenilmez, zinciri uzatın**'} "
              f"(maks R̂ {res.rhat.max():.3f}; min adım/τ {((res.n_steps - res.burn) / res.tau).min():.0f}; kabul oranı {res.acc:.2f})", "",
              "| Parametre | Öncel | Posterior medyan | %90 aralık | Daralma (post sd / öncel sd) | Öncelden kayma (öncel sd) |",
              "|---|---|---|---|---|---|"]
    shifted = []
    for i, k in enumerate(model.free):
        s = res.summary()[k]
        shift = (s["medyan"] - np.median(prior_X[:, i])) / prior_X[:, i].std()
        if abs(shift) > 1.0 and k not in ("sigma",):
            shifted.append((k, shift))
        lines.append(f"| {k} | {model.priors[k].describe()} | {s['medyan']:.3f} | [{s['q05']:.3f}, {s['q95']:.3f}] | "
                     f"{s['sd'] / prior_X[:, i].std():.2f} | {shift:+.1f} |")
    if shifted:
        lines += ["", "**UYARI — öncelden belirgin kayma:** " + ", ".join(f"{k} ({v:+.1f} sd)" for k, v in shifted) +
                  ". Özellikle R0'ın literatür değerinden uzaklaşması, modelin başka bir eksikliği (ör. mesafe "
                  "dağılımı, κ², cis-prolin) R0'ı değiştirerek telafi ettiğine işaret edebilir. Bu durumda δ ≈ 0 "
                  "çıkması 'model doğru' anlamına GELMEZ; LOCO sonuçlarına bakın."]
    if d_i is not None:
        dv = res.samples[:, d_i]
        lines += ["", f"**δ (model uyuşmazlığı):** medyan {np.median(dv):+.3f}; P(|δ| > 0.05) = {np.mean(np.abs(dv) > 0.05):.2f}. "
                  + (("Küresel δ belirgin bir sapma göstermiyor"
                      + ("; ANCAK yukarıdaki öncel-kayması uyarısı nedeniyle bu, modelin doğru olduğu anlamına gelmez."
                         if shifted else "; basit Förster + geometri modeli bu veriyle tutarlı."))
                     if np.mean(np.abs(dv) > 0.05) < 0.5 else
                     "Veri, basit modelden belirgin bir sapma gösteriyor; eksik fizik adayları: κ², cis-prolin kıvrımları, "
                     "bağlayıcı–boya etkileşimleri, kalibrasyon sistematiği.")]
    if len(loco):
        lines += ["", "## 3. Bir-yapıyı-dışarıda-bırak (LOCO)", "",
                  "| Dışarıda bırakılan | Tahmin medyan | %90 aralık | Ölçülen ort. | Aralıkta mı? | Mutlak hata |", "|---|---|---|---|---|---|"]
        for r in loco.itertuples():
            lines.append(f"| {r.construct} | {r3(r.pred_med)} | [{r3(r.pred_q05)}, {r3(r.pred_q95)}] | {r3(r.E_obs_mean)} | "
                         f"{'evet' if r.in_90 else '**hayır**'} | {r3(r.abs_err)} |")
        lines += ["", f"LOCO kapsama {int(loco.in_90.sum())}/{len(loco)}, ortalama mutlak hata {loco.abs_err.mean():.3f}."]
    lines += ["", "## Raporlama dili", "",
              "- Bu karşılaştırma yalnızca modelin **inkoherent (Förster) sınırını** test eder; ENAQT, koherans veya ps dinamiği ölçülmemiştir.",
              "- R0 literatürden alınmıştır; simülasyonun oyuncak R0'ı (4.11 nm) kullanılmamıştır.",
              f"- {len(cons)} yapı = {len(cons)} bağımsız mesafe koşulu; tekrarlar yalnızca teknik gürültüyü ölçer."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---------------- şekiller ----------------
    x = np.arange(len(pred)); names = list(pred.name)
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=200)
    ax.fill_between(x, pred.E_repmean_q05, pred.E_repmean_q95, color="#2a78d6", alpha=0.22, lw=0, label="ön-kayıt %90 (tekrar ort.)")
    ax.plot(x, pred.E_med, "o-", color="#2a78d6", lw=2, ms=4, label="ön-kayıt medyan")
    pm = {c["name"]: post_mean[:, i] for i, c in enumerate(model.constructs)}
    q = np.array([np.quantile(pm[n], [0.05, 0.5, 0.95]) for n in names])
    ax.errorbar(x + 0.12, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]], fmt="s", color="#1baf7a", ms=4, capsize=3,
                label="posterior ⟨E⟩ + δ (%90)")
    for i, n in enumerate(names):
        e = meas[meas.construct == n].E.values
        ax.plot(np.full(len(e), i - 0.12), e, "x", color="0.55", ms=5, label="tekil ölçüm" if i == 0 else None)
        if n in obs.index:
            ax.errorbar(i - 0.12, obs.loc[n, "E_obs_mean"], yerr=obs.loc[n, "E_obs_se"], fmt="D", color="black", ms=5,
                        capsize=3, label="ölçüm ort. ± SE" if i == 0 else None)
    ax.set_xticks(x, names); ax.set_ylim(0, 1); ax.set_ylabel("FRET verimi E"); ax.grid(alpha=0.25)
    ax.set_title(("SENTETİK DEMO — " if a.banner else "") + "Ön-kayıt vs ölçüm vs posterior", fontsize=10)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout(); fig.savefig(out / "compare_predictions.png"); plt.close(fig)

    if d_i is not None:
        fig, ax = plt.subplots(figsize=(5, 3.2), dpi=200)
        ax.hist(prior_X[:, d_i], bins=60, density=True, color="0.75", label="öncel")
        ax.hist(res.samples[:, d_i], bins=60, density=True, color="#2a78d6", alpha=0.8, label="posterior")
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("δ (model uyuşmazlığı, E birimi)"); ax.set_ylabel("yoğunluk"); ax.legend(fontsize=8)
        ax.set_title(("SENTETİK DEMO — " if a.banner else "") + "δ: Förster modeli yeterli mi?", fontsize=10)
        fig.tight_layout(); fig.savefig(out / "posterior_delta.png"); plt.close(fig)


if __name__ == "__main__":
    main()
