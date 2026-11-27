> **ÖRNEK — varsayımsal yapılarla araç gösterimi; gerçek bir ön-kayıt değildir**

# FRET ön-kayıt belgesi

**Oluşturulma:** 2026-11-27T20:55:06+03:00  ·  **git commit:** `4501986d65e7404fda4555d460e840319d443940 (kaydedilmemiş değişiklikler VAR)`
**FRET çifti:** Alexa 488 / Alexa 594 (ÖRNEK)  ·  **R0 (literatür):** 5.4 nm ± %7  ·  kaynak: ÖRNEK: Schuler ve ark. 2005 (PNAS 102:2754) çalışmasında bu çift için kullanılan değer; gerçek çiftiniz ve koşullarınız için güncelleyin

Bu tahminler **hiçbir ölçüm görülmeden**, yalnızca aşağıdaki öncellerden üretilmiştir
(öncel öngörü dağılımı, 20000 örnek, tohum 2026). Bu belge ölçümden önce commit'lenip
etiketlenmelidir ve sonradan değiştirilmez.

## Tahminler

| Yapı | Aile | n | Planlanan tekrar | ⟨E⟩ medyan | ⟨E⟩ %90 aralık | n_rep tekrar ortalaması %90 | Tek ölçüm %90 | Statik κ² ile medyan (duyarlılık) |
|---|---|---|---|---|---|---|---|---|
| Pro6 | poliprolin | 6 | 3 | 0.971 | [0.871, 1.000] | [0.861, 1.028] | [0.842, 1.056] | 0.876 |
| Pro10 | poliprolin | 10 | 3 | 0.864 | [0.655, 0.995] | [0.650, 0.998] | [0.640, 1.006] | 0.695 |
| Pro15 | poliprolin | 15 | 3 | 0.529 | [0.273, 0.783] | [0.269, 0.786] | [0.261, 0.794] | 0.401 |
| Pro20 | poliprolin | 20 | 3 | 0.214 | [0.066, 0.430] | [0.056, 0.434] | [0.047, 0.444] | 0.177 |

Başarı kuralında kullanılan aralık: **n_rep tekrar ortalaması %90** sütunu.

Not: Aralıkların genişliği büyük ölçüde geometri öncellerinden (bağlanma ofseti, bağlayıcı esnekliği)
ve R0 belirsizliğinden gelir. Geniş aralıkta "içinde kalmak" kolaydır; bu yüzden ortalama mutlak hata
kuralı daha sıkı testtir. Aralıkları daraltmanın dürüst yolu, ölçümden ÖNCE daha iyi geometrik bilgi
(yapı modeli, erişilebilir hacim hesabı) eklemektir — ölçümden sonra değil.

## Model ve öncel tablosu

Model: E = clip(∫ p(r) / (1 + (r/R0)⁶) dr + δ, 0, 1) + gürültü; p(r) için bkz. `src/fret_bayes.py`. κ² = 2/3 (dinamik ortalama).

| Parametre | Öncel |
|---|---|
| R0 | log-normal(medyan 5.4, log-sd 0.07) |
| delta | normal(0, 0.05) |
| sigma | yarı-normal(ölçek 0.05) |
| d0 | kesik normal(0.6, 0.5; [-1, 3]) |

Geometrik varsayımlar: PPII yükselmesi 0.31 nm/kalıntı, boya konum yayılımı 0.6 nm,
kalıntı kontur uzunluğu 0.38 nm, FP ofset yayılımı 0.5 nm.

## Önceden belirlenen başarı kuralı

Model **başarılı** sayılır, ancak ve ancak:
1. Yapıların en az **%67**'inde ölçülen tekrar ortalaması, yukarıdaki %90 aralığın içindedir; **ve**
2. Yapılar üzerinden ortalama mutlak hata |E_ölçülen − ⟨E⟩ medyan| < **0.1**.

Kural tutmazsa bu da aynı ayrıntıyla raporlanacaktır.

## Parmak izleri (SHA-256)

- yapılar: `fbb0622b8ba9abc5481be05c64d8b88e569e8e50871bf898054a097ad868d799`
- ayar: `b9f32204dc5378ea4f1d44b1850d6396d7962b2e3c9555a3ec69d0b981e0ca77`
- model kodu: `967f8aa03af80250e74676b6fd33f31e004079f61ae348db2d441789f24a3760`
- bu script: `855256f06138f79a8c253df924cf5a326f6eb30e1750504e334dbe152803560d`

## Bu tahminlerin İDDİA ETMEDİĞİ şeyler

- Kararlı-hal FRET; ENAQT, kuantum koherans veya ps dinamiğini ölçmez. Bu karşılaştırma yalnızca
  modelin inkoherent (Förster) sınırını test eder.
- R0 literatürden alınmıştır; simülasyondaki oyuncak R0 (4.11 nm) burada kullanılmaz.
