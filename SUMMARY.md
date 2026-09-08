# Scientific Summary — In-Silico Phase

**TÜBİTAK 2204-A high-school research project**  
Environment-Assisted Quantum Transport (ENAQT) in DNA-scaffolded chromophore systems

This document summarises the computational results. All figures referred to below are located in `results/figures/`. The corresponding numerical data are in `results/data/`.

---

## 1. Research question

In a DNA-scaffolded donor–acceptor chromophore pair, does an intermediate amount of environmental dephasing accelerate excitonic energy transfer compared with the fully coherent (γ → 0) and fully incoherent (γ → ∞) limits?  
If so, what is the practical window of chromophore distance *r* and dephasing rate *γ* in which this Environment-Assisted Quantum Transport (ENAQT) occurs, and how does bath memory (non-Markovian effects) change the robustness of that window?

---

## 2. Computational approach (step by step)

### Step 2.1 – Basic open quantum system
- N-site Frenkel exciton Hamiltonian.
- Lindblad master equation with pure dephasing and detailed-balance relaxation.
- Verified on a two-site (Cy3–Cy5-like) and a three-site (FMO-inspired) system.

**Figures:** `dynamics_2site.png`, `dynamics_3site.png`  
Population flows from donor to acceptor while coherences decay, as expected. Total population is conserved.

### Step 2.2 – Sink, Loss and first parameter scan
- Added a Sink (reaction-centre) channel that irreversibly traps excitation at the acceptor and a weak Loss channel (fluorescence / non-radiative decay).
- Introduced distance-dependent dipole–dipole coupling \( V(r) = V_0 (r_0/r)^3 \).
- Defined two complementary figures of merit:
  - Transfer efficiency η
  - Mean trapping time ⟨t⟩

**Key observation:** With realistic Cy3/Cy5 parameters the Sink is much faster than Loss, so η saturates near 1 everywhere. The ENAQT signature therefore appears clearly only in the **speed** metric ⟨t⟩.

A near-resonant “canonical” dimer (small energy gap) was also studied so that the full ENAQT curve — coherent Rabi oscillations → optimal noise-assisted transfer → Quantum Zeno suppression — becomes visible.

**Figures:**  
- `single_point_sink_loss_dynamics.png` – full dynamics at one parameter point  
- `enaqt_dual_heatmap.png` – Lindblad maps of η(γ,r) and ⟨t⟩(γ,r)  
- `enaqt_1d_regime_slice.png` – 1-D cut showing the three regimes

On the near-resonant model, a clear minimum of ⟨t⟩ appears at intermediate dephasing.

### Step 2.3 – Non-Markovian validation (HEOM)
The Markovian (memory-less) assumption of the Lindblad equation was relaxed by replacing the pure-dephasing channel with a Hierarchical Equations of Motion (HEOM) treatment using a Drude–Lorentz spectral density. Sink, Loss and relaxation were kept as Markovian channels injected into the HEOM Liouvillian.

**Figure:** `lindblad_vs_heom_comparison.png`

| Quantity              | Lindblad          | HEOM (τ_c ≈ 0.5 ps)      |
|-----------------------|-------------------|--------------------------|
| Optimal γ             | lower             | shifted to higher values |
| Minimum ⟨t⟩           | slightly lower    | slightly higher          |
| High-γ behaviour      | sharp Quantum-Zeno rise | much softer         |

**Interpretation:** Bath memory does **not** make the transfer faster at the optimum; it makes the system **more robust**. The beneficial window widens and the destructive Quantum-Zeno regime is strongly suppressed.

### Step 2.3-extension – Full 2-D HEOM robustness map
A complete 20 × 20 grid (γ ∈ [0.05, 80] ps⁻¹, r ∈ [1, 5] nm) was computed with the full HEOM model (400 independent simulations, parallelised and checkpointed).

**Figures (main results):**  
- `heom_robustness_heatmap.png` – efficiency map η(γ,r)  
- `heom_mean_trapping_time_heatmap.png` – mean trapping-time map ⟨t⟩(γ,r)

**Quantitative findings from the 2-D HEOM scan:**
- Global minimum of mean trapping time: ⟨t⟩_min ≈ 2.05 ps  
- Corresponding location: r ≈ 1.4 nm, γ in the approximate range 10–20 ps⁻¹  
- Efficiency remains high (η ≳ 0.99) throughout this window  
- Distance *r* has the strongest effect (because V ∝ 1/r³), yet a clear secondary dependence on γ is visible — the ENAQT signature survives in the full non-Markovian treatment.

---

## 3. Main conclusions

1. **Target experimental window**  
   Chromophore separation r ≈ 1.4 nm (roughly 4 DNA base pairs) together with environmental dephasing in the range γ ∈ [10, 20] ps⁻¹ yields the fastest transfer under the HEOM model. This window is the primary prediction to be tested in the laboratory.

2. **Role of bath memory**  
   Non-Markovian effects do not dramatically accelerate the optimal transfer rate; they protect the system against excessive noise and widen the useful parameter region. This offers a design principle: structured noise can be less harmful than white noise.

3. **Why ordinary FRET looks incoherent**  
   Realistic Cy3/Cy5 energy gaps place the system on the rising (phonon-assisted) flank of the ENAQT curve. Consequently the Quantum-Zeno branch is experimentally inaccessible for that pair, which explains why conventional FRET is usually described as an incoherent Förster process.

---

## 4. What the laboratory phase should test

- Synthesise short DNA constructs that fix a Cy3–Cy5 (or equivalent) pair at approximately 1.4 nm.  
- Measure time-resolved fluorescence (TCSPC / streak camera) under controlled solvent viscosity or other dephasing-tuning conditions.  
- Check whether the observed transfer time approaches the ~2 ps scale predicted by the simulations and whether moderate environmental noise improves rather than degrades performance.

---

## 5. Limitations and open points

- Hierarchy depth and bath parameters in HEOM are a practical compromise; deeper hierarchies change the numbers only at the sub-percent level.  
- The realistic Cy3/Cy5 parameter set never shows a pronounced Quantum-Zeno regime inside experimentally relevant dephasing rates.  
- Direct experimental comparison (which theoretical model — Förster, Redfield, Lindblad or HEOM — best describes the real DNA construct) remains future work.  
- A homodimer (Cy3–Cy3) control would help confirm the presence of coherent excitonic coupling.

---

## 6. Reproducibility

All Python source files, numerical grids and figures are included in this repository. A high-school student can re-run the entire pipeline with the commands listed in `README.md` (QuTiP 5.3, ordinary laptop hardware). The most expensive step (full 20×20 HEOM map) takes a few tens of minutes on a single core and is fully checkpointed.
