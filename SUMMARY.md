# Bilimsel Özet — In Silico Aşama (revizyon v2)

**TÜBİTAK 2204-A lise araştırma projesi**
Biyomoleküler iskeleli kromofor çiftlerinde çevre-yardımlı enerji transferi (ENAQT)

Tüm sayılar `results/tables/` altındaki dosyalardan alınmıştır; her birinin kaynağı parantez içinde verilmiştir. Neyin değiştiği için `CHANGELOG.md` dosyasına bakın.

---

## 1. Soru

İki seviyeli bir donör–akseptör modelinde, orta düzeyde dephasing (γ) ortalama yakalanma süresini ⟨t⟩ kısaltıyor mu (ENAQT)? Bu etki hangi mesafede en büyük? Banyo hafızası (HEOM) bunu nasıl değiştiriyor? Model, kararlı-hal FRET deneyinin ölçeceği inkoherent sınıra doğru biçimde iniyor mu?

## 2. Yöntem (özet)

- Frenkel ekziton dimeri, V(r) = V₀(r₀/r)³; Markov sink/loss/gevşeme kanalları.
- Dephasing: Lindblad (√γ|n⟩⟨n|) veya HEOM (her sitede Drude–Lorentz, γ_c = 2 rad/ps, λ = γγ_c/2k_BT).
- v2: η ve ⟨t⟩ sonsuz zamana kadar analitik (∫x dt = −M⁻¹x₀); log-γ ızgarası; HEOM derinlik 11.
- v2 HEOM kodu (numpy/scipy), v1 QuTiP sonuçlarını 400 noktada |Δη| ≤ 7×10⁻⁵ ile yeniden üretir (`validation_summary.txt`).
- Yeni metrik: **ENAQT kazancı S(r) = ⟨t⟩(γ=0) / min_γ⟨t⟩**.

## 3. Bulgular (kanonik oyuncak model içinde)

1. **En hızlı nokta ≠ en çok yardım eden gürültü.** Global min ⟨t⟩ = 2.020 ps, r = 1.00 nm'de ve γ ≈ 0'da (ızgara kenarı). Orada S = 1.000, yani gürültü yardım etmiyor. En büyük kazanç S_max = 3.18, r = 3.30 nm ve γ* ≈ 7.9 ps⁻¹'de. Kazancın yarı-yükseklik aralığı r = 2.5–4.4 nm (`scan_v2_headline.txt`). v1'deki "r ≈ 1.4 nm hedef pencere" iddiası veriyle uyuşmuyordu (v1 CSV'nin kendi minimumu da r = 1.0 nm'deydi) ve geri çekildi.
2. **~2 ps, Γ_sink'e bağlıdır.** Γ_sink = 0.5 / 1 / 2 ps⁻¹ için en kısa ⟨t⟩ = 3.99 / 2.02 / 1.01 ps, yani ≈ 2/Γ_sink. S_max'ın konumu (r ≈ 3.2–3.4 nm) değişmez, büyüklüğü 2.3–3.6 arasında değişir (`sink_sensitivity.csv`). Bu nedenle ⟨t⟩'nin mutlak değeri deneysel bir tahmin olarak sunulmaz.
3. **Lindblad ↔ HEOM.** Kullanılan λ–γ eşleştirmesi HEOM'da ~2γ koherans sönümü verir (ölçülen oran 1.97–1.98; `mapping_check.txt`). Bu ölçek farkı düzeltildikten sonra da r = 2 nm'de HEOM optimumu (γ* = 20 ps⁻¹) Lindblad'a göre ~10 kat yüksek γ'dadır ve ⟨t⟩_min biraz daha uzundur (2.465 vs 2.324 ps). γ = 39.8 ps⁻¹'de HEOM 2.52 ps, Markov-eşdeğer Lindblad 4.66 ps verir (`lindblad_vs_heom_v2.txt`). Bu sonuç, *güvenilir aralıkta* yavaş banyonun güçlü gürültüye karşı sistemi daha az yavaşlattığını gösterir.
4. **Sayısal sınırlar.** v1'in T_MAX = 25 ps kesmesi r ≳ 2 nm'de η ve ⟨t⟩'yi ciddi biçimde bozuyordu (r = 5 nm'de η 0.41–0.59 yerine 0.96–0.97; `validation_summary.txt`). HEOM derinlik 5, büyük r/γ'da %36'ya kadar hatalıdır. γ > 40 ps⁻¹'de kesilmiş HEOM güvenilmezdir (`heom_convergence_summary.txt`). Bu nedenle v1'deki "Zeno rejimi banyo hafızasıyla yumuşar" iddiası bu kurulumla desteklenmez.
5. **Förster köprüsü.** FRET-benzeri açıklayıcı parametrelerde (ΔE = 1000 cm⁻¹, γ = 200 ps⁻¹) aynı model k ∝ r^−5.996 verir ve E(r), 1/(1+(r/R₀)⁶) eğrisine 0.0005 içinde oturur (R₀ = 4.11 nm; `forster_limit_summary.txt`). Kanonik ENAQT rejimi ise kısa mesafede bu yasadan belirgin biçimde sapar (etkin üs 2 nm'de 3.76).

## 4. Laboratuvar aşamasının rolü

Planlanan peptit/protein kararlı-hal FRET deneyi, modelin **inkoherent (Förster) sınırını** test eder: ölçülen E değerleri, ölçümden önce yazılmış E(r) tahminleriyle karşılaştırılacaktır. Deney ENAQT'yi, koheransı veya ps dinamiğini ölçmez; ENAQT bulguları bir **model tahmini** olarak raporlanır.

## 5. Açık noktalar

- Kanonik parametreler gerçek bir boya çiftine ait değildir. Gerçek çift için ENAQT kazancının ne olduğu ayrı bir sorudur (büyük ΔE'de kazanç beklenmez).
- Nokta-dipol yaklaşımı < 2 nm'de geçersizdir; κ² modelde yoktur.
- Yüksek γ'da güvenilir HEOM sonuçları için farklı bir yöntem (ör. terminatörlü HEOM ya da daha kararlı bir banyo açılımı) gerekir.
