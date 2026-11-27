#!/usr/bin/env bash
# v3 — yapay zekâ / makine öğrenmesi ve laboratuvar araçları zincirini yeniden üretir.
# Gerekli: numpy, scipy, pandas, matplotlib, scikit-learn (QuTiP, PyTorch, emcee GEREKMEZ).
# Süre (2 çekirdek): ~45–60 dk; en uzun adımlar ml_dataset, ml_emulator ve fret_calibration_test.
set -euo pipefail
cd "$(dirname "$0")"
# (A) In silico yol
python src/ml_dataset.py                 # Faz 1A: Lindblad 20 000 + HEOM 1 000 nokta (LHS)
python src/ml_emulator.py                # Faz 2A: GP / MLP emülatörleri + S(r) uygulaması
python src/ml_emulator_regimes.py        # emülatör hatası nerede? hedefli örnekleme
python src/ml_delta_learning.py          # Lindblad → HEOM Δ-öğrenme + oran haritası
python src/ml_sparse_regression.py       # veriden mesafe yasası (r^-6) — doğrulama testi
# (B) Laboratuvar yolu
python src/fret_calibration_test.py      # Faz 2B: Bayesçi FRET modeli, sahte deneylerle kalibrasyon
python src/fret_make_synthetic_demo.py   # SENTETİK demo ölçümleri (gerçek veri değil)
python src/fret_prereg.py --constructs data/lab/ornek/ORNEK_yapilar.csv \
    --config data/lab/ornek/ORNEK_model_ayari.json --out results/fret/prereg/ornek \
    --banner "ÖRNEK — varsayımsal yapılarla araç gösterimi; gerçek bir ön-kayıt değildir"
python src/fret_compare.py --prereg results/fret/prereg/ornek \
    --measurements data/lab/ornek/SENTETIK_olcumler.csv --out results/fret/compare/ornek_sentetik \
    --banner "SENTETİK DEMO — ölçümler yapay olarak üretildi (src/fret_make_synthetic_demo.py); gerçek veri değildir"
