# DNA-İskeleli Kromofor Dizilimlerinde ENAQT — In Silico Modül Seti

**Proje:** Titreşimsel Rezonans Mühendisliği ile Çevre-Yardımlı Kuantum Taşınımının (ENAQT)
Deneysel Doğrulanması — MIT Maker Portfolio / TÜBİTAK 2204-A / Regeneron ISEF

Bu depo, projenin **Faz 2 (In Silico)** bacağında üretilen tüm Python modüllerini,
ürettikleri veri/grafik çıktılarını ve bunların nasıl çalıştırılacağını belgeler.
Faz 1 (literatür/matematiksel model) ve Faz 3 (in vitro laboratuvar protokolü)
ayrı belgelerde ele alınmıştır; bilimsel bulguların anlatısı için `SUMMARY.md`
dosyasına bakınız.

---

## 1) Gereksinimler

```bash
pip install qutip numpy matplotlib pandas seaborn tqdm --break-system-packages
```

Test edilen sürümler: `qutip==5.3.0`, Python 3.12. HEOM modülleri (`qutip.solver.heom`)
QuTiP ≥5.0 gerektirir.

---

## 2) Dosya Yapısı ve Çalıştırma Sırası

```
ENAQT_Quantum_Biology_Project/
├── README.md
├── SUMMARY.md
├── requirements.txt
├── src/                              <- tüm Python modülleri burada
│   ├── quantum_biology_module.py
│   ├── enaqt_parameter_scan.py
│   ├── heom_nonmarkovian_validation.py
│   └── heom_2d_robustness_scan.py
└── results/
    ├── figures/                      <- tüm PNG çıktıları
    └── data/                         <- tüm npy/npz/csv çıktıları
```

Modüller **birbirinin üzerine inşa edilecek şekilde** (Adım 2.1 → 2.2 → 2.3)
tasarlanmıştır; her biri bağımsız olarak da çalıştırılabilir (`if __name__ ==
"__main__":` bloğu kendi test/demo senaryosunu içerir).

**Not:** Modüller varsayılan olarak çıktılarını `/home/claude/` gibi mutlak
yollara kaydedecek şekilde yazılmıştı (orijinal geliştirme ortamı). Kendi
bilgisayarınızda çalıştırırken dosya sonundaki `save_path` / `OUTPUT_DIR`
değişkenlerini kendi `results/figures/` ve `results/data/` yollarınıza
güncelleyin (her dosyada bu değişkenler dosyanın başında/ilgili fonksiyon
imzasında kolayca bulunur).

| # | Dosya | Adım | Ne Yapar | Çalıştırma |
|---|---|---|---|---|
| 1 | `src/quantum_biology_module.py` | 2.1 | N-site Frenkel ekziton Hamiltoniyeni + Lindblad dephasing/relaxation; 2-site ve 3-site test senaryoları | `python3 src/quantum_biology_module.py` |
| 2 | `src/enaqt_parameter_scan.py` | 2.2 | Sink+Loss dinamiği, V(r)=V₀(r₀/r)³ kaplin yasası, γ×r 2D Lindblad taraması, η ve ⟨t⟩ hesaplama | `python3 src/enaqt_parameter_scan.py` |
| 3 | `src/heom_nonmarkovian_validation.py` | 2.3 | Sabit r=2nm'de Lindblad vs HEOM (Drude-Lorentz banyo) karşılaştırması, Haken-Strobl eşleştirmesi | `python3 src/heom_nonmarkovian_validation.py` |
| 4 | `src/heom_2d_robustness_scan.py` | 2.3-genişletme | Tam 2D (γ,r) HEOM taraması (20×20=400 nokta), paralelleştirilmiş, checkpoint/resume destekli, yayın-kalitesi heatmap | `python3 src/heom_2d_robustness_scan.py` |

### Çıktı Dosyaları (`results/` altında, önceden üretilmiş halleriyle depoda mevcut)

| Dosya (`results/figures/` veya `results/data/`) | Üreten Modül | İçerik |
|---|---|---|
| `figures/dynamics_2site.png`, `figures/dynamics_3site.png` | (1) | Site popülasyon dinamiği + coherence |
| `figures/single_point_sink_loss_dynamics.png` | (2) | Sink/Loss dahil tam sistem dinamiği (tek nokta) |
| `figures/enaqt_dual_heatmap.png` | (2) | η(γ,r) ve ⟨t⟩(γ,r) — kanonik (near-resonant) Lindblad modeli |
| `figures/enaqt_1d_regime_slice.png` | (2) | r=1.96nm kesitinde 3 rejim (Rabi/ENAQT/Zeno) etiketli |
| `data/eta_scan_data.npz` | (2) | Ham 2D Lindblad tarama verisi (`gamma_range`, `r_range`, `eta_grid`, `mean_time_grid`) |
| `figures/lindblad_vs_heom_comparison.png` | (3) | Sabit r'de Lindblad/HEOM overlay (sweet spot kayması) |
| `data/lindblad_vs_heom_data.npz` | (3) | Karşılaştırma ham verisi |
| `figures/heom_robustness_heatmap.png` | (4) | **Ana teslim edilebilir görsel** — 20×20 HEOM η(γ,r) haritası, anotasyonlu |
| `figures/heom_mean_trapping_time_heatmap.png` | (4) | 20×20 HEOM ⟨t⟩(γ,r) haritası |
| `data/heom_2d_scan_results.csv` | (4) | Tidy-format tam tarama verisi (400 satır) |
| `data/heom_*_grid.npy`, `data/heom_*_range.npy` | (4) | Hızlı Python geri-yükleme için ham grid'ler |

---

## 3) Birim Sistemi (KRİTİK)

Tüm modüllerde tutarlı bir birim sözleşmesi kullanılır:

- **Enerjiler / kaplinler**: girdi olarak cm⁻¹ (spektroskopik standart), dahili
  olarak `CM1_TO_RADPS = 2π × 2.99792458×10⁻² ≈ 0.188365` sabitiyle rad/ps'ye
  çevrilir (ħ=1 sözleşmesi, QuTiP'in beklediği format).
- **Zaman**: ps (picosaniye).
- **Dephasing/relaxation/sink/loss hızları**: doğrudan ps⁻¹ (rad/ps ile aynı
  boyutta, ek dönüşüm gerekmez).
- **Sıcaklık**: Kelvin girilir, `KB_CM1_PER_K = 0.695034800` ile enerji
  birimine (cm⁻¹) çevrilip ardından rad/ps'ye taşınır.

---

## 4) Model Mimarisi Özeti

```
Faz 2.1: H_S (N-site) + Lindblad(dephasing, relaxation+detailed balance)
              │
              ▼
Faz 2.2: + Sink operatörü (akseptör→RC) + Loss operatörü (her site→floresan kaybı)
         + V(r) = V₀(r₀/r)³ mesafe-kaplin yasası
         + η ve ⟨t⟩ = ∫t·Γ_sink·P_acc(t)dt / ∫Γ_sink·P_acc(t)dt tanımları
              │
              ▼
Faz 2.3: Dephasing kanalı Lindblad'dan HEOM'a taşınır (Drude-Lorentz banyo,
         J(ω)=2λγ_c ω/(ω²+γ_c²)); Sink/Loss/Relaxation Liouvillian üzerinden
         HEOM'a Markovian kanal olarak enjekte edilir
         (qutip.liouvillian(H, c_ops) → HEOMSolver(L, baths, max_depth))
```

---

## 5) Bilinen Sınırlamalar / Dikkat Edilmesi Gerekenler

1. **İki farklı parametre seti kullanılır** (bilinçli tasarım kararı):
   - *Gerçekçi Cy3/Cy5* (E₁=18800, E₂=15800 cm⁻¹): deneysel bağlantı için, ama
     büyük enerji farkı nedeniyle ENAQT'nin Zeno kolu γ∈[0.01,100] ps⁻¹
     aralığında görünmez (bkz. `enaqt_parameter_scan.py` içindeki tasarım notu).
   - *Kanonik near-resonant toy dimer* (E₁=25, E₂=0 cm⁻¹): ENAQT fiziğini
     (Rabi→optimum→Zeno) net izole etmek için, literatürdeki standart
     yaklaşımı (Rebentrost ve ark. 2009) izler.
2. **Trapezoidal integrasyon çözünürlüğü**: kısa mesafede (güçlü kaplin,
   hızlı salınım) düşük `n_steps` aliasing hatasına yol açar — doğrulama
   için `eta_direct` vs `eta_integral` farkı her taramada raporlanır
   (hedef: <1e-3).
3. **Haken-Strobl eşleştirmesi** (`λ(γ)=γ·γ_c/(2kT)`) yalnızca γ_c→∞
   (hızlı banyo) limitinde Lindblad ile HEOM'u tam örtüştürür; sonlu γ_c'de
   aradaki fark banyo hafızasının fiziksel imzasıdır (bug değil, ölçüm hedefi).
4. **HEOM hesaplama maliyeti**: `max_depth=5, Nk=1` hız/doğruluk dengesi
   için seçilmiştir (yakınsama testi: `max_depth=6,Nk=2`'ye göre <%0.2 fark).
   Tam 400 nokta tarama tek çekirdekte ~25-30 dk sürer; `heom_2d_robustness_scan.py`
   checkpoint/resume mekanizmasıyla kesintilere karşı korumalıdır.

---

## 6) Hızlı Doğrulama

Herhangi bir modülün doğru kurulduğunu test etmek için:

```bash
python3 src/quantum_biology_module.py   # ~5 saniyede tamamlanmalı, 2 PNG üretmeli
```

Çıktıda `[OK] Adım 2.1 testleri başarıyla tamamlandı.` görülmeli ve toplam
popülasyonun 1.0'da korunduğu doğrulanmalıdır (sink olmadığı için).

**Not:** `quantum_biology_module.py` dosyasındaki `save_path` argümanlarını
(`/home/claude/dynamics_2site.png` gibi) kendi `results/figures/` yolunuza
güncellemeniz gerekebilir (bkz. Bölüm 2'deki not).
