#!/bin/bash
# The 'gated_notime' variant: the MIMIC CVP-waveform gate everywhere and no elapsed-time product columns (119 columns).
cd "$(dirname "$0")/.."; export VARIANT=gated_notime; PY=${PY:-python}; mkdir -p logs; J=logs/gated_notime; mkdir -p $J results/gated_notime figures/gated_notime
export OMP_NUM_THREADS=6
for cfg in "10 15" "8 12" "10 20" "5 15" "15 30"; do set -- $cfg; WLIST=$1 SDUR=$2 $PY src/s03_nested_cv.py > $J/s03_W$1_S$2.log 2>&1; done
CTRL=clock $PY src/s03_nested_cv.py > $J/s03_clock.log 2>&1; CTRL=perm $PY src/s03_nested_cv.py > $J/s03_perm.log 2>&1
for s in 7 99 1234; do SEED=$s $PY src/s03_nested_cv.py > $J/s03_seed$s.log 2>&1; done
$PY src/s04_vitaldb_examples.py > $J/s04.log 2>&1; $PY src/s15_causal_offset.py > $J/s15.log 2>&1; echo done > $J/vitaldb.flag
while ! grep -q "^exit" logs/s19_measure_all.log; do sleep 30; done
( export OMP_NUM_THREADS=4; $PY src/s07_transfer.py > $J/s07.log 2>&1; $PY src/s08_mimic_analysis.py > $J/s08.log 2>&1; $PY src/s09_mimic_examples.py > $J/s09.log 2>&1; $PY src/s10_perwindow_scores.py > $J/s10.log 2>&1; $PY src/s13_mimic_extra.py > $J/s13.log 2>&1; echo done > $J/chainA.flag ) &
( export OMP_NUM_THREADS=6; $PY src/s11_investigation.py > $J/s11.log 2>&1; echo done > $J/chainB.flag ) &
( export OMP_NUM_THREADS=6; $PY src/s16_feature_reduction.py > $J/s16.log 2>&1; $PY src/s18_one_feature_horizons.py > $J/s18.log 2>&1; echo done > $J/chainC.flag ) &
( export OMP_NUM_THREADS=2; for S in 15 20 30 45 60; do SDUR=$S $PY src/s17_longer_stretches.py > $J/s17_S$S.log 2>&1 & done; wait; echo done > $J/chainD.flag ) &
wait; $PY src/s12_figures.py all > $J/s12.log 2>&1; echo "all gated_notime chains done $(date)" > $J/all.flag
