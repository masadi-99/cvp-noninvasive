#!/bin/bash
# The 'gated' variant: every MIMIC number with the CVP-waveform gate. VitalDB artifacts are linked from the base tree.
cd "$(dirname "$0")/.."; PY=${PY:-python}; mkdir -p logs; $PY src/s19_mimic_cvp_gate.py measure all > logs/s19_measure_all.log 2>&1; echo "exit $?" >> logs/s19_measure_all.log
export VARIANT=gated; PY=${PY:-python}; mkdir -p logs; J=logs/gated; mkdir -p $J results/gated figures/gated; for f in results/nested_*.json results/nested_*_scores.npz results/vitaldb_oof.npz results/causal_offset.json results/causal_offset_*_scores.npz results/vitaldb_patient_table.json; do ln -sfn ../$(basename $f) results/gated/$(basename $f); done
( export OMP_NUM_THREADS=4; $PY src/s07_transfer.py > $J/s07.log 2>&1; $PY src/s08_mimic_analysis.py > $J/s08.log 2>&1; $PY src/s09_mimic_examples.py > $J/s09.log 2>&1; $PY src/s10_perwindow_scores.py > $J/s10.log 2>&1
  while ! grep -q "^exit" logs/s19_measure_all.log; do sleep 30; done; $PY src/s13_mimic_extra.py > $J/s13.log 2>&1; echo done > $J/chainA.flag ) &
( export OMP_NUM_THREADS=6; $PY src/s11_investigation.py > $J/s11.log 2>&1; echo done > $J/chainB.flag ) &
( export OMP_NUM_THREADS=6; $PY src/s16_feature_reduction.py > $J/s16.log 2>&1; $PY src/s18_one_feature_horizons.py > $J/s18.log 2>&1; echo done > $J/chainC.flag ) &
( export OMP_NUM_THREADS=2; for S in 15 20 30 45 60; do SDUR=$S $PY src/s17_longer_stretches.py > $J/s17_S$S.log 2>&1 & done; wait; echo done > $J/chainD.flag ) &
wait; $PY src/s04_vitaldb_examples.py > $J/s04.log 2>&1; $PY src/s12_figures.py all > $J/s12.log 2>&1; echo "all gated chains done $(date)" > $J/all.flag
