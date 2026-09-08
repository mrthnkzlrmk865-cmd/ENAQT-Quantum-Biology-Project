# Environment-Assisted Quantum Transport (ENAQT) in DNA-Scaffolded Chromophore Systems

**In-silico module set for a TÜBİTAK 2204-A high-school research project**

This repository contains the complete computational (in-silico) phase of a high-school research project on environment-assisted quantum transport (ENAQT) in DNA-scaffolded donor–acceptor chromophore systems. The work was carried out independently by a high-school student using open-source Python tools (primarily QuTiP).

The goal of the computational phase is to identify a practical parameter window (chromophore distance *r* and environmental dephasing rate *γ*) in which excitonic energy transfer is accelerated by intermediate noise — the classic ENAQT effect — and to quantify how bath memory (non-Markovian effects) changes the robustness of that window. These results are intended to guide a subsequent laboratory validation experiment (time-resolved fluorescence / TCSPC on Cy3–Cy5 labelled DNA constructs).

---

## Project structure

```
ENAQT_Quantum_Biology_Project/
├── README.md                 ← this file (how to run the code)
├── SUMMARY.md                ← scientific narrative and key findings
├── requirements.txt
├── src/
│   ├── quantum_biology_module.py      # Step 2.1 – Hamiltonian + basic Lindblad dynamics
│   ├── enaqt_parameter_scan.py        # Step 2.2 – Sink/Loss model + 2-D Lindblad scan
│   ├── heom_nonmarkovian_validation.py# Step 2.3 – Lindblad vs HEOM comparison
│   └── heom_2d_robustness_scan.py     # Step 2.3-ext – full 20×20 HEOM map
└── results/
    ├── figures/              # all generated plots (PNG)
    └── data/                 # numerical grids and CSV
```

All modules can be run independently. They were written to build on one another (2.1 → 2.2 → 2.3).

---

## Requirements

```bash
pip install qutip==5.3.0 numpy matplotlib pandas seaborn tqdm
```

Python ≥ 3.10 and QuTiP ≥ 5.0 are required (HEOM solver lives in `qutip.solver.heom`).

**Note on paths:** Some scripts originally wrote to absolute paths used during development. Before re-running, set the output directories inside each script to your local `results/figures/` and `results/data/` folders.

---

## How to run (recommended order)

| Step | Script | What it does | Approx. runtime |
|------|--------|--------------|-----------------|
| 2.1  | `python src/quantum_biology_module.py` | Builds N-site Frenkel exciton Hamiltonian + Lindblad dephasing/relaxation; produces 2-site and 3-site population & coherence dynamics | ~5–10 s |
| 2.2  | `python src/enaqt_parameter_scan.py` | Adds Sink (reaction-centre) and Loss channels, distance-dependent coupling \(V(r)\propto 1/r^3\), computes transfer efficiency \(\eta\) and mean trapping time \(\langle t\rangle\) on a 2-D grid | a few minutes |
| 2.3  | `python src/heom_nonmarkovian_validation.py` | Fixed-distance comparison of Markovian Lindblad vs non-Markovian HEOM (Drude–Lorentz bath) | ~1–2 min |
| 2.3-ext | `python src/heom_2d_robustness_scan.py` | Full 20×20 HEOM scan over \(\gamma\) and \(r\) (400 independent simulations, parallelised, checkpointed) | ~25–40 min (single core) / much faster with multiple cores |

---

## Generated figures (all included in `results/figures/`)

| File | Description |
|------|-------------|
| `dynamics_2site.png` | Population and coherence dynamics of a two-site (Cy3–Cy5-like) system |
| `dynamics_3site.png` | Three-site (FMO-inspired) population dynamics |
| `single_point_sink_loss_dynamics.png` | Full dynamics including Sink and Loss at a single parameter point |
| `enaqt_dual_heatmap.png` | Lindblad 2-D maps of \(\eta(\gamma,r)\) and \(\langle t\rangle(\gamma,r)\) (near-resonant model) |
| `enaqt_1d_regime_slice.png` | 1-D cut at fixed *r* showing the three regimes (Rabi → ENAQT → Quantum Zeno) |
| `lindblad_vs_heom_comparison.png` | Direct overlay of Lindblad and HEOM mean trapping times at fixed distance |
| `heom_robustness_heatmap.png` | **Main result** – 20×20 HEOM efficiency map \(\eta(\gamma,r)\) with annotations |
| `heom_mean_trapping_time_heatmap.png` | 20×20 HEOM mean trapping-time map \(\langle t\rangle(\gamma,r)\) |

Numerical data corresponding to the heatmaps are stored in `results/data/` (`.npy`, `.npz`, `.csv`).

---

## Unit conventions (important)

- Energies and couplings are entered in **cm⁻¹** (spectroscopic convention) and converted internally to rad/ps via  
  `CM1_TO_RADPS = 2π × 2.99792458×10⁻² ≈ 0.188365`.
- Time is in **picoseconds (ps)**.
- Rates (dephasing, relaxation, sink, loss) are in **ps⁻¹**.
- Temperature is given in kelvin and converted with \(k_B = 0.695\) cm⁻¹ K⁻¹.

This is the standard convention used in photosynthetic open-quantum-system literature (e.g. Ishizaki–Fleming).

---

## Model hierarchy (short)

1. **System Hamiltonian** – N-site Frenkel exciton model.  
2. **Markovian open system** – Lindblad dephasing + detailed-balance relaxation + Sink + Loss.  
3. **Distance dependence** – dipole–dipole coupling \(V(r) = V_0 (r_0/r)^3\).  
4. **Non-Markovian bath** – Hierarchical Equations of Motion (HEOM) with Drude–Lorentz spectral density; Sink/Loss/relaxation kept as Markovian channels injected into the HEOM Liouvillian.

Two complementary parameter sets are used deliberately:
- Realistic Cy3/Cy5 energies (large \(\Delta E\)) – for experimental relevance.
- Near-resonant “canonical” dimer (small \(\Delta E\)) – to make the full ENAQT curve (including the Quantum-Zeno branch) clearly visible.

---

## Limitations (honest)

- The realistic Cy3/Cy5 energy gap places the system mainly on the phonon-assisted rising flank of the ENAQT curve; the Zeno regime is not experimentally accessible for that pair.
- Trapezoidal integration of fast oscillations at short distances requires a sufficiently large number of time steps (validated by cross-checking two independent definitions of \(\eta\)).
- HEOM hierarchy depth (`max_depth=5`, `Nk=1`) is a speed/accuracy compromise; deeper hierarchies change results by <0.2 %.
- Absolute output paths in the original scripts must be adjusted for local re-runs.

---

## Quick smoke test

```bash
python src/quantum_biology_module.py
```

You should see a success message and two PNG files; total population must remain conserved at 1.0 (no Sink in this basic test).

---

## Contact / context

This computational work forms the in-silico phase of a TÜBİTAK 2204-A high-school research project. The next planned step is laboratory validation of the identified parameter window (\(r \approx 1.4\) nm, \(\gamma \in [10,20]\) ps⁻¹) using dye-labelled DNA constructs and time-resolved fluorescence spectroscopy.
