> **SENTETİK DEMO — ölçümler yapay olarak üretildi (src/fret_make_synthetic_demo.py); gerçek veri değildir**

# FRET karşılaştırma raporu

Ön-kayıt: `results/fret/prereg/ornek` (oluşturulma 2026-11-27T20:55:06+03:00, commit `4501986d65e7404fda4555d460e840319d443940 (kaydedilmemiş değişiklikler VAR)`)
Ölçümler: `data/lab/ornek/SENTETIK_olcumler.csv` (12 ölçüm, 4 yapı)

## 1. Donmuş karşılaştırma (ön-kayıtlı tahminler, güncelleme yok)

| Yapı | Tahmin ⟨E⟩ medyan | %90 aralık (tekrar ort.) | Ölçülen ort. ± SE | Aralıkta mı? | Mutlak hata |
|---|---|---|---|---|---|
| Pro6 | 0.971 | [0.861, 1.028] | 0.974 ± 0.011 | evet | 0.003 |
| Pro10 | 0.864 | [0.650, 0.998] | 0.869 ± 0.018 | evet | 0.005 |
| Pro15 | 0.529 | [0.269, 0.786] | 0.604 ± 0.003 | evet | 0.075 |
| Pro20 | 0.214 | [0.056, 0.434] | 0.330 ± 0.011 | evet | 0.116 |

Aralık içinde: **4/4** (kural ≥ %67); ortalama mutlak hata **0.050** (kural < 0.1).
**Önceden belirlenen başarı kuralı: SAĞLANDI.**

## 2. Posterior (tüm ölçümlerle güncelleme)

Yakınsama: evet (maks R̂ 1.014; min adım/τ 60; kabul oranı 0.58)

| Parametre | Öncel | Posterior medyan | %90 aralık | Daralma (post sd / öncel sd) | Öncelden kayma (öncel sd) |
|---|---|---|---|---|---|
| R0 | log-normal(medyan 5.4, log-sd 0.07) | 6.138 | [5.571, 6.688] | 0.90 | +2.0 |
| delta | normal(0, 0.05) | -0.010 | [-0.045, 0.041] | 0.52 | -0.2 |
| sigma | yarı-normal(ölçek 0.05) | 0.034 | [0.021, 0.055] | 0.35 | +0.0 |
| d0 | kesik normal(0.6, 0.5; [-1, 3]) | 0.883 | [0.277, 1.475] | 0.73 | +0.6 |

**UYARI — öncelden belirgin kayma:** R0 (+2.0 sd). Özellikle R0'ın literatür değerinden uzaklaşması, modelin başka bir eksikliği (ör. mesafe dağılımı, κ², cis-prolin) R0'ı değiştirerek telafi ettiğine işaret edebilir. Bu durumda δ ≈ 0 çıkması 'model doğru' anlamına GELMEZ; LOCO sonuçlarına bakın.

**δ (model uyuşmazlığı):** medyan -0.010; P(|δ| > 0.05) = 0.06. Küresel δ belirgin bir sapma göstermiyor; ANCAK yukarıdaki öncel-kayması uyarısı nedeniyle bu, modelin doğru olduğu anlamına gelmez.

## 3. Bir-yapıyı-dışarıda-bırak (LOCO)

| Dışarıda bırakılan | Tahmin medyan | %90 aralık | Ölçülen ort. | Aralıkta mı? | Mutlak hata |
|---|---|---|---|---|---|
| Pro6 | 0.965 | [0.900, 1.025] | 0.974 | evet | 0.009 |
| Pro10 | 0.936 | [0.877, 0.987] | 0.869 | **hayır** | 0.067 |
| Pro15 | 0.656 | [0.584, 0.712] | 0.604 | evet | 0.052 |
| Pro20 | 0.257 | [0.216, 0.297] | 0.330 | **hayır** | 0.073 |

LOCO kapsama 2/4, ortalama mutlak hata 0.050.

## Raporlama dili

- Bu karşılaştırma yalnızca modelin **inkoherent (Förster) sınırını** test eder; ENAQT, koherans veya ps dinamiği ölçülmemiştir.
- R0 literatürden alınmıştır; simülasyonun oyuncak R0'ı (4.11 nm) kullanılmamıştır.
- 4 yapı = 4 bağımsız mesafe koşulu; tekrarlar yalnızca teknik gürültüyü ölçer.
