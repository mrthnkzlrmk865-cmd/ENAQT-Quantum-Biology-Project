#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fret_prereg.py — Faz 3: ölçümden ÖNCE kaydedilen FRET tahminleri
=================================================================
Her yapı için, HİÇBİR ölçüm görülmeden (yalnızca öncellerle) E tahmini ve
%90 tahmin aralığı üretir; başarı kuralını ve tüm girdilerin parmak izlerini
(SHA-256, git commit) bir ön-kayıt belgesine yazar.

Kullanım:
  python src/fret_prereg.py --constructs data/lab/yapilar.csv \
                            --config data/lab/model_ayari.json \
                            --out results/fret/prereg/<etiket>

ÖN-KAYIT ADIMLARI (ölçümden önce, sırayla):
  1. Yapı ve ayar dosyalarını ODTÜ ekibiyle kesinleştir.
  2. Bu scripti çalıştır.
  3. Çıktı klasörünü commit'le ve bir etiket (tag) at:
        git add results/fret/prereg/<etiket> data/lab && git commit -m "ön-kayıt" && git tag prereg-<etiket>
        git push && git push --tags
     (İsteğe bağlı: aynı PDF/MD'yi OSF'ye zaman damgalı yükle.)
  4. Ancak bundan SONRA ölçüm yap. Ön-kayıt dosyaları bir daha değiştirilmez.

Çıktılar (<out>/):
  prereg_predictions.csv   yapı başına: ⟨E⟩ medyan, %90 aralık; n_rep ortalaması ve tek ölçüm için aralıklar;
                           statik-κ² duyarlılık medyanı
  PREREG.md                ön-kayıt belgesi (tarih, parmak izleri, öncel tablosu, başarı kuralı)
  prereg_predictions.png
  prereg_meta.json         karşılaştırma scriptinin okuduğu makine-okunur kopya
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fret_bayes as fb
import fret_lab_io as io

ROOT = Path(__file__).resolve().parents[1]
N_PRIOR = 20000
SEED = 2026


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--constructs", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--banner", default="", help="belgenin başına eklenecek uyarı satırı (ör. ÖRNEK)")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    cons = io.load_constructs(a.constructs)
    cfg = io.load_config(a.config)
    model = io.build_model(cons, cfg)
    model_static = io.build_model(cons, cfg, kappa2="static")
    rng = np.random.default_rng(SEED)
    X = model.sample_prior(rng, N_PRIOR)
    clist = model.constructs

    mean_E = fb.predict(model, X, clist)                                       # ⟨E⟩ + δ
    rep_E = io.replicate_mean_predictive(model, X, clist, cons.n_rep.values, rng)
    one_E = fb.predict(model, X, clist, rng=rng, with_noise=True)
    stat_E = fb.predict(model_static, X[:4000], clist)

    rows = []
    for i, c in enumerate(cons.itertuples()):
        q = lambda v: np.quantile(v, [0.05, 0.5, 0.95])
        m5, m50, m95 = q(mean_E[:, i]); r5, _, r95 = q(rep_E[:, i]); o5, _, o95 = q(one_E[:, i])
        rows.append(dict(name=c.name, family=c.family, n=c.n, n_rep=c.n_rep,
                         E_med=m50, E_mean_q05=m5, E_mean_q95=m95,
                         E_repmean_q05=r5, E_repmean_q95=r95,
                         E_single_q05=o5, E_single_q95=o95,
                         E_med_static_kappa2=np.median(stat_E[:, i])))
    pred = pd.DataFrame(rows)
    pred.to_csv(out / "prereg_predictions.csv", index=False)

    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    files = {"yapılar": a.constructs, "ayar": a.config,
             "model kodu": str(ROOT / "src" / "fret_bayes.py"), "bu script": __file__}
    meta = dict(created=now, git_commit=io.git_commit(ROOT),
                sha256={k: io.sha256(v) for k, v in files.items()},
                constructs_path=str(a.constructs), config_path=str(a.config),
                config=cfg, n_prior_samples=N_PRIOR, seed=SEED,
                free_params=model.free, fixed=model.fixed,
                priors={k: model.priors[k].describe() for k in model.priors})
    (out / "prereg_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    rule = cfg["success_rule"]
    tbl = "\n".join(
        f"| {r.name} | {r.family} | {r.n:g} | {r.n_rep} | {r.E_med:.3f} | [{r.E_mean_q05:.3f}, {r.E_mean_q95:.3f}] | "
        f"[{r.E_repmean_q05:.3f}, {r.E_repmean_q95:.3f}] | [{r.E_single_q05:.3f}, {r.E_single_q95:.3f}] | {r.E_med_static_kappa2:.3f} |"
        for r in pred.itertuples())
    pri = "\n".join(f"| {k} | {'SABİT = ' + str(model.fixed[k]) if k in model.fixed else model.priors[k].describe()} |"
                    for k in model.priors)
    md = f"""{('> **' + a.banner + '**' + chr(10) + chr(10)) if a.banner else ''}# FRET ön-kayıt belgesi

**Oluşturulma:** {now}  ·  **git commit:** `{meta['git_commit']}`
**FRET çifti:** {cfg['pair']}  ·  **R0 (literatür):** {cfg['R0_lit_nm']} nm ± %{100*cfg['R0_rel_sd']:.0f}  ·  kaynak: {cfg.get('R0_source', 'belirtilmedi')}

Bu tahminler **hiçbir ölçüm görülmeden**, yalnızca aşağıdaki öncellerden üretilmiştir
(öncel öngörü dağılımı, {N_PRIOR} örnek, tohum {SEED}). Bu belge ölçümden önce commit'lenip
etiketlenmelidir ve sonradan değiştirilmez.

## Tahminler

| Yapı | Aile | n | Planlanan tekrar | ⟨E⟩ medyan | ⟨E⟩ %90 aralık | {'n_rep'} tekrar ortalaması %90 | Tek ölçüm %90 | Statik κ² ile medyan (duyarlılık) |
|---|---|---|---|---|---|---|---|---|
{tbl}

Başarı kuralında kullanılan aralık: **n_rep tekrar ortalaması %90** sütunu.

Not: Aralıkların genişliği büyük ölçüde geometri öncellerinden (bağlanma ofseti, bağlayıcı esnekliği)
ve R0 belirsizliğinden gelir. Geniş aralıkta "içinde kalmak" kolaydır; bu yüzden ortalama mutlak hata
kuralı daha sıkı testtir. Aralıkları daraltmanın dürüst yolu, ölçümden ÖNCE daha iyi geometrik bilgi
(yapı modeli, erişilebilir hacim hesabı) eklemektir — ölçümden sonra değil.

## Model ve öncel tablosu

Model: E = clip(∫ p(r) / (1 + (r/R0)⁶) dr + δ, 0, 1) + gürültü; p(r) için bkz. `src/fret_bayes.py`. κ² = 2/3 (dinamik ortalama).

| Parametre | Öncel |
|---|---|
{pri}

Geometrik varsayımlar: PPII yükselmesi {fb.RISE_PPII_NM} nm/kalıntı, boya konum yayılımı {fb.S_DYE_NM} nm,
kalıntı kontur uzunluğu {fb.RES_CONTOUR_NM} nm, FP ofset yayılımı {fb.S_FP_NM} nm.

## Önceden belirlenen başarı kuralı

Model **başarılı** sayılır, ancak ve ancak:
1. Yapıların en az **%{100*rule['min_fraction_in_90']:.0f}**'inde ölçülen tekrar ortalaması, yukarıdaki %90 aralığın içindedir; **ve**
2. Yapılar üzerinden ortalama mutlak hata |E_ölçülen − ⟨E⟩ medyan| < **{rule['max_mae']}**.

Kural tutmazsa bu da aynı ayrıntıyla raporlanacaktır.

## Parmak izleri (SHA-256)

""" + "\n".join(f"- {k}: `{v}`" for k, v in meta["sha256"].items()) + """

## Bu tahminlerin İDDİA ETMEDİĞİ şeyler

- Kararlı-hal FRET; ENAQT, kuantum koherans veya ps dinamiğini ölçmez. Bu karşılaştırma yalnızca
  modelin inkoherent (Förster) sınırını test eder.
- R0 literatürden alınmıştır; simülasyondaki oyuncak R0 (4.11 nm) burada kullanılmaz.
"""
    (out / "PREREG.md").write_text(md, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6.4, 4), dpi=200)
    x = np.arange(len(pred))
    ax.fill_between(x, pred.E_single_q05, pred.E_single_q95, color="#2a78d6", alpha=0.12, lw=0, label="tek ölçüm %90")
    ax.fill_between(x, pred.E_repmean_q05, pred.E_repmean_q95, color="#2a78d6", alpha=0.28, lw=0, label="tekrar ortalaması %90")
    ax.plot(x, pred.E_med, "o-", color="#2a78d6", lw=2, ms=5, label="⟨E⟩ medyan (κ² = 2/3)")
    ax.plot(x, pred.E_med_static_kappa2, "s--", color="#eb6834", lw=1.5, ms=4, label="statik κ² (duyarlılık)")
    ax.set_xticks(x, pred.name); ax.set_ylim(0, 1)
    ax.set_ylabel("FRET verimi E"); ax.grid(alpha=0.25)
    ax.set_title(("ÖRNEK — " if a.banner else "") + f"Ön-kayıtlı tahminler: {cfg['pair']}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "prereg_predictions.png"); plt.close(fig)
    print(md)


if __name__ == "__main__":
    main()
