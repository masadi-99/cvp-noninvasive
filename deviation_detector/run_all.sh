#!/bin/bash
# Reproduce everything, in order. ~4 h wall on 20 cores. Then scripts/run_gated.sh and scripts/run_notime.sh for the two variants.
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
$PY src/s00_scale_ref.py                                   # check the frozen amplitude reference
$PY src/s01_prepare_vitaldb.py                             # data/
$PY src/s02_tracker_oof.py                                 # results/vitaldb_tracker_oof.npz
for cfg in "10 15" "8 12" "10 20" "5 15" "15 30"; do set -- $cfg
  WLIST=$1 SDUR=$2 $PY src/s03_nested_cv.py; done          # results/nested_*.json (+ vitaldb_oof.npz)
CTRL=clock $PY src/s03_nested_cv.py; CTRL=perm $PY src/s03_nested_cv.py
for s in 7 99 1234; do SEED=$s $PY src/s03_nested_cv.py; done
$PY src/s04_vitaldb_examples.py                            # figures/vitaldb_*.png
bash scripts/run_mimic.sh                                  # steps 5-6, sharded: results/mimic_feat/
$PY src/s07_transfer.py                                    # results/transfer.json
$PY src/s08_mimic_analysis.py                              # results/mimic_analysis.json, mimic_scores.npz
$PY src/s09_mimic_examples.py                              # figures/mimic_*.png
$PY src/s10_perwindow_scores.py                            # results/perwindow_*.npz
$PY src/s11_investigation.py                               # results/investigation.json (~90 min)
$PY src/s13_mimic_extra.py                                 # results/mimic_extra.json
$PY src/s14_label_quality.py                               # results/label_quality.json
$PY src/s15_causal_offset.py                               # results/causal_offset.json
$PY src/s16_feature_reduction.py                           # results/feature_reduction.json (~40 min)
for S in 15 20 30 45 60; do SDUR=$S $PY src/s17_longer_stretches.py; done   # results/longer_S*.json (~25 min each)
$PY src/s18_one_feature_horizons.py                        # results/one_feature_horizons.json
$PY src/s19_mimic_cvp_gate.py measure && $PY src/s19_mimic_cvp_gate.py score   # results/mimic_gated.json
$PY src/s20_ablate_spread.py                               # results/ablate_spread.json
$PY src/s12_figures.py all                                 # figures/f*.png
bash scripts/run_followup.sh                               # steps 21-32 on the gated variant (after scripts/run_gated.sh)
