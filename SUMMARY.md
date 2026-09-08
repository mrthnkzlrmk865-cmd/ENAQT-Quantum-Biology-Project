# Faz 2 (In Silico) Bilimsel Özet

## Hipotez

DNA-iskeleli donör-akseptör kromofor çiftlerinde, orta düzeyde ortam
dephasing'i (γ), tamamen koherent (γ→0) ve tamamen dekohere (γ→∞) rejimlerden
daha hızlı eksiton transferi sağlar — Çevre-Yardımlı Kuantum Taşınımı (ENAQT).
Bu, mesafe (r) ve dephasing hızının (γ) birlikte ayarlanabilir olduğu sentetik
bir sistemde test edilmiştir.

---

## Adım 2.1 — Temel Model

N-site Frenkel ekziton Hamiltoniyeni (`H_S = Σε_n|n⟩⟨n| + ΣJ_nm|n⟩⟨m|`) ve
Lindblad açık kuantum sistem çerçevesi kuruldu: saf dephasing (coherence'ı
bozar, popülasyonu korur) ve detailed-balance'lı relaxation (donor→acceptor
enerji akışı, Boltzmann faktörüyle termodinamik tutarlı). 2-site (Cy3-Cy5) ve
3-site (FMO-benzeri) sistemlerde doğrulandı; toplam popülasyonun korunduğu
teyit edildi.

**Sonuç:** Donörden akseptöre net, fiziksel olarak tutarlı popülasyon akışı;
coherence zamanla dekohere oluyor — beklenen davranış.

---

## Adım 2.2 — Sink/Loss Dinamiği ve İlk Parametre Taraması

Sistem, Sink (RC, "kullanılan" enerji) ve Loss (floresan/non-radiatif kayıp,
Γ_loss≈0.001 ps⁻¹) durumlarıyla genişletildi; V(r)=V₀(r₀/r)³ dipol-dipol
kaplin yasası eklendi. Verim η ve ortalama yakalanma süresi ⟨t⟩ iki bağımsız
yöntemle (doğrudan Sink popülasyonu vs. akı integrali) çapraz doğrulandı.

**Kritik bulgu #1 — η doygunluğu:** Gerçekçi Cy3/Cy5 parametreleriyle,
Γ_sink (1 ps⁻¹) >> Γ_loss (0.001 ps⁻¹) olduğundan η her yerde ~0.997-0.998'e
doygunlaşıyor; ENAQT'nin asıl imzası η'da görünmüyor. **Çözüm:** ⟨t⟩
(ortalama yakalanma süresi, Rebentrost ve ark. 2009 standardı) metriği
eklendi — bu, verim doygunlaşsa bile transfer HIZINDAKİ ENAQT tepe/çukur
noktasını netçe ortaya çıkarıyor.

**Kritik bulgu #2 — enerji skalası sorunu:** Kubo/motional-narrowing
formülüne göre (k~2V²γ/(γ²+ΔE²)) optimal γ, ΔE'ye eşittir. Gerçek Cy3/Cy5
çifti için ΔE~3000 cm⁻¹~565 ps⁻¹ — deneysel olarak anlamsız bir γ aralığı.
**Çözüm:** near-resonant "kanonik ENAQT" toy dimer (ΔE=25 cm⁻¹, V₀=20 cm⁻¹)
kullanıldı; bu konfigürasyonda üç rejim (Rabi salınımı → optimal ENAQT →
Kuantum Zeno) net biçimde ortaya çıktı: **r=1.96nm kesitinde γ_opt≈3.81
ps⁻¹'de ⟨t⟩_min=2.29 ps.**

**Metodolojik not:** n_steps=300'de kısa mesafede (güçlü kaplin, hızlı
salınım) trapezoidal integrasyon %11 hataya (aliasing) yol açtı; n_steps=800'e
çıkarılarak doğrulama hatası 4.6×10⁻⁴'e indirildi.

---

## Adım 2.3 — Non-Markovian (HEOM) Doğrulama

Lindblad'ın "banyo hafızasız" (τ_c→0) varsayımı, Drude-Lorentz spektral
yoğunluklu (J(ω)=2λγ_c ω/(ω²+γ_c²)) HEOM simülasyonuyla gevşetildi
(max_depth=5, Nk=1; yakınsama max_depth=6,Nk=2'ye göre <%0.2 fark ile
doğrulandı). Haken-Strobl eşleştirmesi (λ(γ)=γγ_c/2kT) ile Lindblad ve HEOM
aynı γ ekseninde adil biçimde karşılaştırıldı (τ_c=0.5 ps).

**Bulgu — banyo hafızası hızlandırmıyor, SAĞLAMLAŞTIRIYOR:**

| Metrik | Lindblad | HEOM (τ_c=0.5 ps) |
|---|---|---|
| Sweet spot γ_opt | 3.29 ps⁻¹ | 22.33 ps⁻¹ (+578%) |
| Minimum ⟨t⟩ | 2.33 ps | 2.44 ps (+4.7%) |
| Yüksek-γ davranışı | Keskin Zeno çöküşü (⟨t⟩=4.65ps @ γ=80) | Yumuşak (⟨t⟩=2.82ps @ γ=80) |

Banyo hafızası ham transfer hızını artırmıyor (minimum ⟨t⟩ aslında hafifçe
daha yüksek), ama sweet spot'u genişleterek ve Zeno-tipi çöküşü bastırarak
sistemi **dephasing gücüne karşı çok daha dayanıklı (robust)** hale
getiriyor — yapılandırılmış/renkli gürültü, beyaz gürültünün aksine, ani
projeksiyon-benzeri lokalizasyona yol açmıyor.

---

## Adım 2.3-Genişletme — Tam 2D HEOM Robustness Haritası

20×20=400 nokta (γ∈[0.05,80] ps⁻¹, r∈[1,5] nm, doğrusal aralıklı), tam
Liouvillian+HEOM çerçevesiyle (Sink/Loss/Relaxation Markovian kanal olarak
entegre) tarandı. `ProcessPoolExecutor` + checkpoint/resume mimarisiyle,
tek-çekirdekli hesaplama ortamında dahi 9 ayrı oturumda kesintisiz
tamamlandı (0 hata).

**Sonuç:** Genel sweet spot γ_opt=16.88 ps⁻¹, r_opt=1.42 nm, η_max=0.9977;
minimum ⟨t⟩=2.05 ps. Haritada **r'nin etkisi γ'dan baskın** (V(r)~1/r³
nedeniyle), ama kontur eğrilerinin γ ekseninde hafif kavisli olması, r'nin
gölgesinde ikinci-mertebe bir ENAQT imzasının hâlâ var olduğunu gösteriyor.
⟨t⟩ haritası bunu çok daha net ortaya koyuyor: γ∈[10,20] ps⁻¹ bandında geniş
bir "hızlı transfer platosu" — Adım 2.3'teki tek-kesit bulgusuyla tutarlı.

---

## Genel Değerlendirme ve Faz 3 Bağlantısı

1. **r=1.4 nm, γ=10-20 ps⁻¹** bölgesi, hem tek-kesit hem tam 2D HEOM
   taramasında tutarlı biçimde "hızlı transfer" rejimi olarak işaretlendi —
   Faz 3'teki in vitro hedef parametreler doğrudan bu bulgudan türetildi.
2. Gerçek Cy3/Cy5 sisteminin (büyük ΔE) neden yalnızca ENAQT eğrisinin
   yükselen (fonon-yardımlı) kolunda yaşadığı ve bu yüzden gerçek FRET'in
   neden geleneksel olarak inkoherent (Förster) kabul edildiği açıklandı —
   bu, projenin temel bilim katkısına doğrudan katkı sağlıyor.
3. Banyo hafızasının sistemi Zeno-tipi bozulmaya karşı koruyucu etkisi,
   biyolojik sistemlerin neden geniş bir parametre aralığında güvenilir
   çalışabildiğine dair bir tasarım ilkesi öneriyor (mühendislik çıkarımı).

## Açık Sorular (Faz 4 için)

- 2D haritada r ve γ etkilerini ayrıştırmak için ek analiz (kısmi
  korelasyon veya sabit-r kesitlerinin sistematik karşılaştırması)
- Gerçek deneysel veriyle (Faz 3) model karşılaştırması: hangi model
  (Förster/Redfield/Lindblad/HEOM) gerçek sistemi en iyi açıklıyor? (AIC/BIC)
- Homodimer (Cy3-Cy3) kontrolünün eksitonik kaplin varlığını doğrulayıp
  doğrulamadığı
