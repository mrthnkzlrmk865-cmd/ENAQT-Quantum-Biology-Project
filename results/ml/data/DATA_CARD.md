# Veri kartı — in silico ML veri seti (Faz 1A)

Üreten: `src/ml_dataset.py` · tohum `20261127` · tam sürüm

| Dosya | Satır | İçerik |
|---|---|---|
| `lindblad_lhs.csv` | 20000 | Lindblad η, ⟨t⟩ ve aynı noktada γ = 0 referansı |
| `heom_lhs.csv` | 1000 | HEOM (derinlik 11, Nk = 1) + aynı noktada Lindblad(γ) ve Lindblad(2γ) |
| `heom_depth_check.csv` | 40 | Rastgele alt kümede derinlik 9 / 11 / 13 karşılaştırması |

**Girdi aralıkları:** r ∈ (0.8, 8.0) nm (log-düzgün), ΔE ∈ (0.0, 1000.0) cm⁻¹ (düzgün),
γ ∈ (0.1, 200.0) ps⁻¹ Lindblad / (0.1, 40.0) ps⁻¹ HEOM (log-düzgün). Latin hiperküp.

**Sabit parametreler:** V0 = 20 cm⁻¹ @ r0 = 2 nm (V ∝ r⁻³), Γ_sink = 1 ps⁻¹, Γ_loss = 0.001 ps⁻¹,
gevşeme 0.02 ps⁻¹ (detailed balance), T = 298 K; HEOM: γ_c = 2 rad/ps, λ = γγ_c/(2k_BT).

**Yakınsama kontrolü (alt küme):** derinlik 11 vs 13, ⟨t⟩ göreli farkı medyan
1.05e-15, maksimum 1.84e-06.
Fiziksel olmayan (η ∉ [0, 1]) HEOM noktası: 0.

**Uyarılar:**
- Bu bir **oyuncak model** veri setidir; gerçek bir kromofor çiftinin parametreleri değildir.
- HEOM yakınsaması v2'de yalnızca ΔE = 25 cm⁻¹ için test edilmişti; burada geniş ΔE aralığı için
  yalnızca yukarıdaki alt küme kontrolü vardır.
- V ∝ r⁻³ nokta-dipol yaklaşımı r < ~2 nm'de fiziksel olarak güvenilmezdir (sayısal olarak sorun yok).
- ⟨t⟩'nin mutlak değeri Γ_sink seçimine bağlıdır (v2 bulgusu: min ⟨t⟩ ≈ 2/Γ_sink).
- Modeldeki D→A gevşeme kanalı (0.02 ps⁻¹) **r'den bağımsızdır**. Büyük r'de kaplin çok zayıfladığında
  transferi bu kanal belirler: ⟨t⟩ yaklaşık 1/0.02 + 1/Γ_sink ≈ 51 ps'de doyar, η ise ≈ 0.02/(0.02+0.001) ≈ 0.95'in
  altına inmez. Bu, fiziksel bir mesafe yasası değil, modelin bir varsayımıdır.

Süre: Lindblad 7 s, HEOM 305 s, derinlik kontrolü 56 s (2 çekirdek).
