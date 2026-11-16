# CHANGELOG

## v2 — Dürüstlük ve sayısal doğruluk revizyonu

### Geri çekilen veya düzeltilen iddialar

| v1 iddiası | Durum | v2'deki doğru ifade | Kaynak |
|---|---|---|---|
| "Pratik hedef pencere: r ≈ 1.4 nm, γ ∈ [10, 20] ps⁻¹" | **Geri çekildi** | Global min ⟨t⟩ r = 1.0 nm'de (v1 CSV: 2.0527 ps, γ = 12.67; v2: 2.020 ps, γ ≈ 0). r = 1.4 nm'de ENAQT kazancı yalnızca S = 1.022. En büyük kazanç S_max = 3.18 @ r = 3.30 nm. | `results/tables/legacy_csv_global_min.txt`, `scan_v2_headline.txt` |
| "Global minimum ⟨t⟩ ≈ 2.05 ps" (r = 1.4 nm ile birlikte sunulmuştu) | **Düzeltildi** | 2.020 ps @ r = 1.00 nm (ızgara kenarı), sonsuz zaman; v1'deki 2.053 ps kısmen zaman-adımı hatasıydı | `scan_v2_headline.txt`, `truncation_error_by_r.csv` |
| "Deneyde transfer süresinin ~2 ps'ye yaklaşıp yaklaşmadığı test edilecek" | **Geri çekildi** | En kısa ⟨t⟩ ≈ 2/Γ_sink (Γ_sink = 0.5/1/2 → 3.99/2.02/1.01 ps); yani ~2 ps, Γ_sink = 1 ps⁻¹ seçiminin sonucudur. Kararlı-hal FRET zaten ps ölçeğini ölçmez | `sink_sensitivity.csv` |
| "Fast window boyunca η ≳ 0.99" / büyük r'de düşük η haritası | **Düzeltildi** | v1'de r ≳ 2 nm'deki düşük η, T_MAX = 25 ps kesme artefaktıydı (r = 5 nm: 0.41–0.59 → gerçek 0.96–0.97) | `validation_summary.txt` |
| "HEOM derinliği artırmak sonucu <%0.2 değiştirir" | **Geri çekildi** | Yalnızca kısa r'de doğru; derinlik 5'in hatası %6.4 (γ ≤ 40) – %36 (γ = 100). Destekleyen script yoktu. | `heom_convergence_summary.txt` |
| "220 zaman adımı <1e-3 hata verir" (kod yorumu) | **Geri çekildi** | r = 1 nm'de η hatası 0.0092 | `truncation_error_by_r.csv` |
| "Yüksek γ'da Zeno yükselişi banyo hafızasıyla belirgin biçimde yumuşar" | **Desteklenmiyor** | γ > 40 ps⁻¹'de HEOM bu kurulumla yakınsamıyor/kararsız; yalnızca γ ≤ 40'ta HEOM'un daha az yavaşladığı söylenebilir | `heom_convergence_summary.txt`, `lindblad_vs_heom_v2.txt` |
| Lindblad ve HEOM aynı γ ekseninde doğrudan karşılaştırılabilir | **Düzeltildi** | λ = γγ_c/2k_BT eşleştirmesi HEOM'da ~2γ koherans sönümü verir (oran 1.97–1.98) | `mapping_check.txt` |
| "DNA yapılarında Cy3–Cy5 ~1.4 nm (4 bp) + TCSPC/streak kamera; viskoziteyle dephasing ayarı" | **Değiştirildi** | Peptit/protein kararlı-hal FRET; yalnızca Förster sınırı test edilir; viskozite ↔ γ iddiası yok | README §6–7 |
| "Tüm kod ve şekiller tek başına, bağımsız bir lise öğrencisi tarafından üretildi" | **Kaldırıldı** | Yapay zekâ asistanı kullanımı beyan edildi | README §10 |
| Proje adı "DNA-iskeleli … ENAQT'nin Deneysel Doğrulanması" | **Değiştirildi** | "Biyomoleküler iskeleli kromofor çiftlerinde çevre-yardımlı enerji transferi: simülasyon ve FRET ile sınır-rejim kontrolü" | README, `src/quantum_biology_module.py` |

### Korunan (v2'de yeniden doğrulanan) bulgular

- Kanonik modelde ENAQT imzası vardır: orta γ, ⟨t⟩'yi kısaltır (S > 1).
- Mesafe r en güçlü kontrol parametresidir.
- HEOM optimumu, Lindblad'a göre daha yüksek γ'dadır. Bu, 2 çarpanı düzeltildikten sonra da geçerlidir.
- Gerçek FRET çiftleri (büyük ΔE) inkoherent/Förster rejimindedir. Artık `forster_limit.py` ile sayısal olarak gösterilmiştir.

### Yeni dosyalar

- `src/enaqt_core.py`: numpy/scipy Lindblad + HEOM; sonsuz-zaman η ve ⟨t⟩.
- `src/analyze_legacy_csv.py`, `src/validate_against_legacy_csv.py`, `src/validate_lindblad_v1.py`
- `src/heom_convergence.py`, `src/lindblad_heom_mapping_check.py`
- `src/heom_2d_scan_v2.py`, `src/lindblad_vs_heom_v2.py`, `src/sink_sensitivity.py`, `src/forster_limit.py`
- `run_all_v2.sh`
- `results/tables/*` (her README sayısının kaynağı), `results/data/heom_2d_scan_v2.csv`
- `results/figures/enaqt_gain_S_vs_r.png`, `heom_v2_mean_time_heatmap.png`, `lindblad_vs_heom_v2.png`, `forster_limit_check.png`

### Değişen dosyalar

- Klasör yapısı README ile hizalandı: scriptler `src/`, şekiller `results/figures/`, veriler `results/data/`.
- v1 scriptleri: mutlak `/home/claude/...` yolları → repo köküne göre göreli `results/...` yolları; geliştirme ortamına özgü yorumlar ("sandbox", "kullanıcı talebi") kaldırıldı; yanlış çıkan yakınsama/hata iddialarına v2 notları eklendi. **Fizik ve hesap değiştirilmedi.**
- `requirements.txt`: v2 için gereken minimum paketler ayrıldı.
- `README.md`, `SUMMARY.md`: yeniden yazıldı (Türkçe).
- v1 veri dosyaları (`results/data/*.csv|npy|npz`) **değiştirilmeden** korundu.
