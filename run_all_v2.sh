#!/usr/bin/env bash
# v2 analiz zincirini baştan sona yeniden üretir (QuTiP gerekmez).
# Toplam süre sıradan bir dizüstü bilgisayarda yaklaşık 20–40 dakikadır;
# en uzun adımlar heom_2d_scan_v2.py ve sink_sensitivity.py'dir.
set -euo pipefail
cd "$(dirname "$0")"
python src/analyze_legacy_csv.py            # eski CSV'den doğrudan sayılar
python src/validate_against_legacy_csv.py   # numpy HEOM == QuTiP? + kesme hatası
python src/validate_lindblad_v1.py          # v1 Lindblad haritası da kesmeden etkileniyor mu?
python src/heom_convergence.py              # hiyerarşi derinliği / Matsubara testi
python src/lindblad_heom_mapping_check.py   # γ ↔ λ eşleştirmesi (~2 çarpanı)
python src/heom_2d_scan_v2.py               # sonsuz-zaman, log-γ, derinlik 11 tarama + S(r)
python src/lindblad_vs_heom_v2.py           # r = 2 nm Lindblad–HEOM karşılaştırmasının tekrarı
python src/sink_sensitivity.py              # Γ_sink = 0.5 / 1 / 2 ps^-1
python src/forster_limit.py                 # FRET-benzeri rejimde Förster limiti
