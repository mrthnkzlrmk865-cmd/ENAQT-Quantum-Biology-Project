# Simülasyondan Laboratuvara: Yapay Zekâ / Makine Öğrenmesi Yol Haritası (v3)

Bu belge, projenin v3 aşamasının planını, **şu anki durumunu** ve ulaşılan sonuçları özetler.
Her sayının kaynağı parantez içinde verilmiştir (`results/` altındaki dosyalar).
Proje herhangi bir yarışma takviminden bağımsız yürütülmektedir.

---

## 1. Temel karar: iki ayrı yol, iki ayrı soru

| | (A) In silico yol | (B) Laboratuvar yolu |
|---|---|---|
| Soru | HEOM/Lindblad sonuçlarını hızlı ve belirsizliğiyle tahmin etmek | Belirli bir yapı için kararlı-hal FRET verimini (E) **ölçümden önce** tahmin etmek |
| Veri | Bol, ucuz (kendi simülasyonumuz) | Çok az (~3–6 yapı × 3 tekrar) |
| Yöntem | GP / MLP emülatörü, Δ-öğrenme, seyrek regresyon | Fizik-temelli Bayesçi Förster modeli + ön-kayıt + dışarıda-bırakma doğrulaması |
| Geçerlilik iddiası | "Emülatör, test kümesinde HEOM'u şu hatayla taklit ediyor" | "Önceden kaydedilmiş tahmin aralıkları ölçümlerin şu kadarını kapsadı" |
| İddia ETMEDİĞİ | Gerçek biyomoleküllerde ENAQT | Koherans, ENAQT, ps dinamiği |

İki yol arasındaki tek köprü fonksiyonel formdur. v2'de model, FRET-benzeri rejimde E(r) = 1/(1+(r/R0)⁶) formuna iniyordu (`results/tables/forster_limit_summary.txt`). (B)'deki model bu formu kullanır. Ancak R0'ı simülasyondan değil, **literatürden/ölçümden** alır.

**PINN neden ana model değil?** Denklemlerimiz biliniyor ve kesin çözülebiliyor: η ve ⟨t⟩, hareket denklemi matrisinin tersiyle (∫x dt = −M⁻¹x₀, ∫t x dt = M⁻²x₀) milisaniyede hesaplanıyor. Sistem katı (stiff): zaman ölçekleri 0,001–200 ps⁻¹ arasında. Bu, PINN'lerin bilinen başarısızlık rejimidir (Krishnapriyan ve ark. 2021). Ayrıca laboratuvar verisi zaman serisi değil, tek bir sayıdır (E). PINN yalnızca opsiyonel bir eğitsel ek olarak düşünülebilir (§7).

---

## 2. Durum tablosu

| Faz | İçerik | Durum | Çıktı |
|---|---|---|---|
| 0 | Kodun dondurulması, kapsam | ⏳ Etiket elle atılacak: GitHub → Releases → "Draft a new release" → tag `v2.0`, hedef commit `4501986` (v2 birleştirmesi) | git tag `v2.0` |
| 0 | ODTÜ ile yapıların ve FRET çiftinin kesinleştirilmesi | ⏳ **Bekliyor (laboratuvar)** | `data/lab/yapilar.csv`, `model_ayari.json` |
| 1A | In silico sentetik veri seti | ✅ | `results/ml/data/` + `DATA_CARD.md` |
| 1B | FRET ileri simülatörü (mesafe dağılımı, R0/κ², gürültü) | ✅ | `src/fret_bayes.py` |
| 2A | Emülatör, rejim analizi, Δ-öğrenme, seyrek regresyon | ✅ | `results/ml/emulator/`, `delta/`, `sparse/` |
| 2B | Bayesçi FRET modeli + sahte deneylerle kalibrasyon | ✅ | `results/fret/calibration/` |
| 3 | Ön-kayıt aracı | ✅ araç hazır, ÖRNEK ile denendi · ⏳ gerçek ön-kayıt yapılar kesinleşince | `src/fret_prereg.py`, `results/fret/prereg/ornek/` |
| 4 | Ölçüm karşılaştırma + posterior + LOCO | ✅ araç hazır, SENTETİK veriyle denendi · ⏳ gerçek ölçüm | `src/fret_compare.py`, `results/fret/compare/ornek_sentetik/` |

---

## 3. (A) In silico sonuçlar

### 3.1 Veri seti (`results/ml/data/DATA_CARD.md`)
- 20 000 Lindblad + 1 000 HEOM (derinlik 11, γ ≤ 40 ps⁻¹) noktası. Latin hiperküp örnekleme: r = 0,8–8 nm, ΔE = 0–1000 cm⁻¹, γ = 0,1–200 ps⁻¹.
- Rastgele 40 HEOM noktasında derinlik 11 ile 13 arasındaki fark en fazla 1,8×10⁻⁶. Ancak v2'deki en kötü durum (ΔE = 25, γ = 40, r = 3,3 nm: %0,87) hâlâ geçerli.
- **Model özelliği:** D→A gevşeme kanalı (0,02 ps⁻¹) r'den bağımsızdır. Büyük r'de ⟨t⟩ ≈ 51 ps'de doyar, η ≈ 0,95'in altına inmez. Bu fiziksel bir mesafe yasası değil, model varsayımıdır.

### 3.2 Emülatör (`results/ml/emulator/emulator_summary.txt`)

| Model | ⟨t⟩ medyan hata | ⟨t⟩ %95'lik hata | %90 aralık kapsaması | Tahmin süresi |
|---|---|---|---|---|
| Lindblad-GP (2 000 nokta) | %0,07 | %1,1 | %98 | ~130 µs |
| Lindblad-MLP (16 000 nokta) | %0,50 | %2,6 | — | 1 µs |
| HEOM-GP (800 nokta) | %0,14 | %3,7 | %96 | ~40 µs |
| HEOM-MLP (800 nokta) | %1,2 | %7,7 | — | 2 µs |

- Doğrudan HEOM ~585 ms/nokta. HEOM-GP bu hesabı **~15 000 kat** hızlandırır. Lindblad zaten <1 ms olduğu için orada hız kazancı anlamsızdır.
- **Uygulama:** HEOM-GP ile kanonik modelde (ΔE = 25) S(r) eğrisi yeniden üretildi. v2 doğrudan taramasıyla fark medyan %2,2, en fazla %7,0; S_max 3,23 (emülatör) / 3,18 (doğrudan), ikisi de r = 3,30 nm'de.
- Az veriyle (800 nokta) GP, MLP'den belirgin biçimde iyidir ve belirsizlik verir. GP'nin %90 aralıkları genelde biraz fazla geniştir (%96–98 kapsama).

### 3.3 Emülatörün zayıf noktası: koherent köşe (`emulator_regimes_summary.txt`)

En büyük hatalar rezonansa yakın, zayıf dephasing'li köşede toplanır (ΔE < 50 cm⁻¹, γ < 1 ps⁻¹; girdi uzayının ~%1'i). Burası ENAQT açısından en ilginç bölgedir.

| Eğitim (bütçe 2 000) | Köşe testinde medyan hata | %95'lik | %90 kapsama | Genel testte medyan |
|---|---|---|---|---|
| Rastgele 2 000 | %3,7 | %41 | **%32** (aşırı özgüvenli) | %0,04 |
| Hedefli 1 500 + 500 köşe | %0,02 | %1,0 | %94 | %0,74 |

**Ders:** Genel ortalama hata, ilginç rejimdeki hatayı gizleyebilir. Emülatör güvenilirliği rejim bazında raporlanmalıdır. Simülasyon bütçesi de zor bölgelere yönlendirilmelidir.

### 3.4 Lindblad → HEOM Δ-öğrenme: negatif sonuç (`results/ml/delta/delta_summary.txt`)

| HEOM eğitim noktası | Doğrudan GP(HEOM) | Lindblad(γ) + GP(Δ) | Lindblad(2γ) + GP(Δ) |
|---|---|---|---|
| 25 | %5,9 | %8,1 | %11,4 |
| 50 | %2,7 | %4,5 | %5,3 |
| 100 | %1,3 | %1,9 | %2,6 |
| 800 | %0,12 | %0,08 | %0,12 |

- Ucuz modeli temel alıp üzerine düzeltme öğrenmek, bu problemde **yardımcı olmadı**. HEOM'u doğrudan öğrenmek az veriyle daha iyi.
- v2'deki "~2 çarpanı" (koherans sönümü için doğru) transfer süresi için sabit bir düzeltme değildir. Kanonik modelde t_HEOM/t_Lindblad(2γ) oranı 0,28 ile 1,51 arasında değişir (`heom_lindblad_ratio_map.png`).
- Bu, raporlanması gereken dürüst bir sonuçtur: "fizik ön bilgisi" her zaman kazandırmaz; doğru ön bilgi olması gerekir.

### 3.5 Veriden mesafe yasası (`results/ml/sparse/sparse_regression_summary.txt`)

| Rejim | Bulunan en seyrek model | Maks hata |
|---|---|---|
| ΔE = 1000, γ = 2 (zayıf kaplin) | k ∝ r⁻⁶ (tek terim) | %0,10 |
| FRET-benzeri ΔE = 1000, γ = 200 | r⁻⁶ + küçük r⁻⁸ düzeltmesi (tek terim: %1,3) | %0,53 |
| Kanonik ΔE = 25, r = 3–8 nm | 3 terim (tek terim r⁻⁶: %4,2) | %0,47 |
| Kanonik ΔE = 25, r = 1–8 nm | **basit kuvvet yasası yok** (3 terimle bile %29) | — |

- Yöntem: SINDy fikriyle en seyrek modeli seçen alt-küme araması. SINDy'nin standart eşikli algoritması (STLSQ) r⁻ᵖ terimleri neredeyse doğrusal bağımlı olduğu için anlamsız 9 terimli modeller üretti; bu da raporlandı.
- **Dürüstlük notu:** V ∝ r⁻³ modele girdi olarak verildi. r⁻⁶'nın bulunması yeni fizik değil, bir **doğrulama** testidir. İlginç olan, yöntemin tek terimli yasanın bozulduğu rejimi (kısa mesafe, rezonansa yakın) açıkça göstermesidir.

---

## 4. (B) Laboratuvar yolu: Bayesçi FRET modeli

### 4.1 Model (`src/fret_bayes.py`)

E_ölçülen = clip(∫ p(r) · 1/(1+(r/R0)⁶) dr + δ, 0, 1) + gürültü

- **p(r), mesafe dağılımı:** 3B Rice dağılımı.
  - Poliprolin: μ = d0 + 0,31·n nm, boya konum yayılımı 0,6 nm.
  - Floresan protein + esnek bağlayıcı: solucan-benzeri zincir (kalıcılık uzunluğu lp, öncel 0,45 nm; Evers ve ark. 2006).
- **Serbest parametreler (≤ 4–5):**
  - R0: literatür ± %7 (Hellenkamp ve ark. 2018)
  - Yapı parametresi: d0 veya lp, a_eff
  - δ: küresel model uyuşmazlığı
  - σ: ölçüm gürültüsü
- **κ²:** varsayılan dinamik ortalama (2/3); statik izotropik yönelim duyarlılık analizi olarak hesaplanır. Etkisi büyüktür: r = 4 nm, R0 = 5,4 nm için E = 0,82 (dinamik) ve 0,64 (statik).
- **Örnekleyici:** Goodman–Weare topluluk örnekleyicisi (emcee'nin algoritması). Bu ortamda emcee/PyMC kurulamadığı için kısa bir numpy uygulaması yazıldı. Bilinen bir 3B Gauss dağılımında ortalama ve kovaryansı doğru buldu. Tanılar: özilinti süresi τ, ESS, bölünmüş R̂.

### 4.2 Sahte deneylerle kalibrasyon (`results/fret/calibration/calibration_summary.txt`)

K = 120 sahte deney; örnek yapılar Pro6, Pro12, Pro20 × 3 tekrar; R0 = 5,4 nm (örnek). Gerçek parametreler öncelden çekildi, model bunları bilmeden uyduruldu. 120 uydurmanın 111'i τ ölçütüyle yakınsadı.

| Parametre | %90 aralık kapsaması (beklenen %90 ± 5,4) | Daralma (post sd / öncel sd) | SBC düzgünlük testi p |
|---|---|---|---|
| R0 | %87,5 ✅ | 0,73 | 0,65 |
| δ | %92,5 ✅ | 0,58 | 0,77 |
| d0 | %89,2 ✅ | 0,59 | 0,07 |
| σ | **%80,0** ⚠️ | 0,40 | 0,13 |

- **Dışarıda bırakılan yapı (Pro15):** Ortalama E için %90 aralık kapsaması %95, tek yeni ölçüm için %84. Aralık genişliği medyanı 0,080, mutlak hata medyanı 0,011.
- **Daralma:** R0'ın daralması 0,73. Yani 3 yapılık veri R0'ı literatür öncelinin ötesinde pek az öğretiyor. R0'ın kendisini ölçmek bu deneyin amacı olamaz.
- **σ zayıf noktası:** Gerçek σ < 0,018 iken kapsama %63'e düşüyor; posterior σ'yı öncelin ölçeğine (0,05) doğru fazla tahmin ediyor. 9 ölçümle σ zayıf belirlenir. **Öneri:** σ'yı tekrar ölçümlerinin saçılmasından (ve donör-yalnız tekrarlardan) ayrıca tahmin edip config'te öncel olarak verin. Örnek: `"prior_overrides": {"sigma": ["halfnormal", 0.02]}`.
- Bu test yalnızca model varsayımları **doğruyken** dürüstlüğü gösterir. Model hatasını §4.3'teki demo ve gerçek deneydeki LOCO gösterir.

### 4.3 Uçtan uca demo: ÖRNEK ön-kayıt + SENTETİK ölçüm

> Bu bölümdeki yapılar varsayımsal, "ölçümler" yapaydır (`src/fret_make_synthetic_demo.py`). Amaç araçları ve modelin zayıf noktalarını göstermektir.

Gizli "gerçekliğe" modelde **olmayan** bir etki bilerek kondu: uzun poliprolinlerde moleküllerin %25'inde cis-kıvrımı. Bu, Best ve ark. 2007'deki etkinin niteliksel bir taklididir.

| Kontrol | Sonuç | Yorum |
|---|---|---|
| Ön-kayıtlı başarı kuralı (≥ %67 aralıkta, MAE < 0,1) | **Sağlandı** (4/4, MAE 0,050) | Aralıklar geniş, bu yüzden kural kolay geçildi |
| Küresel δ | medyan −0,010, P(\|δ\| > 0,05) = 0,06 | δ sorunu **yakalamadı** |
| R0 posterior | 6,14 nm (gizli gerçek 5,56; öncelden +2,0 sd kayma) | Model, eksik fiziği R0'ı şişirerek telafi etti → rapor otomatik **uyarı** verir |
| Bir-yapıyı-dışarıda-bırak | **2/4** aralıkta | Sorunu **yakalayan** kontrol bu |

**Dersler (gerçek deney için):**
1. Yalnızca "aralıkta mı?" kuralına güvenmeyin. Ön-kayıt aralıkları geometri öncelleri nedeniyle geniştir.
2. δ ≈ 0, "model doğru" demek değildir. R0'ın literatürden kayması ve LOCO sonucu birlikte okunmalıdır.
3. Aralıkları daraltmanın dürüst yolu, ölçümden **önce** daha iyi geometri bilgisi eklemektir (yapı modeli + erişilebilir hacim).

---

## 5. Gerçek deney için iş akışı

1. **Yapıları ve çifti kesinleştir** (ODTÜ ile): `data/lab/yapilar_SABLON.csv` ve `model_ayari_SABLON.json` dosyalarını kopyalayıp doldur. R0 için kaynak, kuantum verimi, kırılma indisi ve κ² varsayımı yazılmalı.
2. **Etki büyüklüğünü kontrol et:** Yapılar arası beklenen E farkı, ölçüm hatasından (σ_E ≈ 0,02–0,05 varsayımı) belirgin biçimde büyük olmalı. E'nin 0,3–0,7 aralığında kaldığı mesafeler en bilgilendiricidir.
3. **Başarı kuralını gözden geçir:** Config'deki `success_rule` (varsayılan: ≥ %67 aralıkta ve MAE < 0,10). Daha sıkı bir kural (ör. MAE < 0,05) düşünülebilir. Ölçümden sonra değiştirilmez.
4. **Ön-kayıt:**
   ```bash
   python src/fret_prereg.py --constructs data/lab/yapilar.csv --config data/lab/model_ayari.json --out results/fret/prereg/<etiket>
   git add data/lab results/fret/prereg/<etiket> && git commit -m "ön-kayıt <etiket>" && git tag prereg-<etiket> && git push --tags
   ```
5. **Ölç:** donör-yalnız ve akseptör-yalnız kontroller; sızma ve doğrudan uyarılma düzeltmeleri; her yapı için en az 3 bağımsız tekrar. `olcumler_SABLON.csv` biçiminde kaydet (E veya F_DA, F_D).
6. **Karşılaştır:**
   ```bash
   python src/fret_compare.py --prereg results/fret/prereg/<etiket> --measurements data/lab/olcumler.csv --out results/fret/compare/<etiket>
   ```
   Rapor, sırasıyla şunları verir: donmuş karşılaştırma (asıl test), posterior (δ ve öncel-kayması uyarıları) ve LOCO.

---

## 6. Ortam notu ve araç değişimleri

Bu aşamanın hesaplandığı ortamda PyPI erişimi yoktu. Bu yüzden emcee/PyMC, PySR, PyTorch, sbi kurulamadı. Kullanılanlar: numpy, scipy, pandas, matplotlib, scikit-learn.

| Plandaki araç | Kullanılan | Colab'da değiştirme |
|---|---|---|
| emcee / PyMC | `fret_bayes.run_ensemble` (aynı algoritma) | `emcee.EnsembleSampler(nwalkers, ndim, log_prob)`; `FretModel.log_prob` doğrudan kullanılabilir |
| ArviZ tanıları | `autocorr_time`, `split_rhat` | `arviz.summary` |
| PySR | alt-küme araması (`ml_sparse_regression.py`) | `pysr.PySRRegressor` |
| PyTorch MLP | `sklearn.neural_network.MLPRegressor` | — (bu ölçekte gerek yok) |
| sbi (NPE) | yapılmadı | opsiyonel çapraz kontrol (§7) |

---

## 7. Yapılmayanlar ve sonraki adımlar (öncelik sırasıyla)

1. **Gerçek ön-kayıt:** ODTÜ yapıları kesinleşince (§5).
2. **Geometri önceli:** Yapı modeli (AlphaFold/PDB) ve erişilebilir hacim (FPS/LabelLib) ile p(r) öncelini fiziksel olarak kurmak. Ön-kayıt aralıklarını dürüstçe daraltmanın ana yolu budur.
3. **Hedefli (aktif) örnekleme:** HEOM emülatörü için koherent köşeye daha fazla HEOM noktası (§3.3'teki dersin HEOM'a uygulanması).
4. **Yapı başına rastgele etki:** Global δ yerine (veya δ'ya ek olarak) yapıya özgü uyuşmazlık. Ancak yalnızca yapı sayısı ≥ 5 ise anlamlıdır.
5. **sbi / NPE çapraz kontrolü:** p(r) örnekleme ile üretildiğinde.
6. **PINN eğitsel eki:** Lindblad'dan seyrek gürültülü popülasyonlarla γ geri çıkarımı ve klasik en küçük kareler uydurmasıyla dürüst karşılaştırma.
7. **NARGP** (doğrusal olmayan çok-doğruluklu GP): Δ-öğrenmenin başarısız olduğu bu problemde daha esnek bir alternatif olarak denenebilir.

---

## 8. Raporlama dili

| Söylemeyin | Bunun yerine |
|---|---|
| "Yapay zekâ ENAQT'yi laboratuvarda kanıtladı." | "Deney yalnızca inkoherent Förster rejimini test etti; ENAQT bir model sonucudur." |
| "PINN ile fiziği öğrendik." | "Denklemleri bilinen bir sistemde ML'yi hızlandırıcı (emülatör) ve belirsizlik nicelendirici olarak kullandık." |
| "Emülatör %99,9 doğru." | "HEOM-GP test kümesinde medyan %0,14 hata verdi; ancak koherent köşede rastgele örneklemeyle %41'e varan hata ve aşırı özgüven gözlendi." |
| "Model deneyi doğruladı." | "Ön-kayıtlı %90 aralıklar X/Y yapıda ölçümü kapsadı; LOCO kapsaması Z/Y; R0 posterior literatürden şu kadar kaydı." |
| "Simülasyon R0 = 4,11 nm'yi öngördü." | "4,11 nm oyuncak parametrelerin sonucudur; deneyde kullanılan R0 literatürdendir." |
| "Fizik bilgisi eklemek modeli iyileştirdi." | Yalnızca gösterilmişse. Δ-öğrenmede iyileştirmedi (§3.4). |

---

## 9. Sınırlamalar

- (A) yolunun bütün sonuçları **oyuncak model** içindir (kanonik dimer, seçilmiş sink/kayıp/gevşeme hızları).
- Emülatörler yalnızca eğitim aralığında geçerlidir (r 0,8–8 nm, ΔE 0–1000 cm⁻¹, γ 0,1–200 / HEOM için 0,1–40 ps⁻¹).
- Bayesçi modelin geometrik sabitleri (PPII yükselmesi 0,31 nm, boya yayılımı 0,6 nm, kontur 0,38 nm, FP ofset yayılımı 0,5 nm) tipik literatür büyüklükleridir; belirli bir yapı için ölçülmemiştir.
- Kalibrasyon testi modelin **kendi varsayımları doğruyken** dürüst olduğunu gösterir; gerçek moleküllerin bu varsayımlara uyduğunu göstermez.
- 3–4 yapı = 3–4 bağımsız mesafe koşulu. Tekrarlar yalnızca teknik gürültüyü ölçer.
