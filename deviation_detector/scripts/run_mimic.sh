#!/bin/bash
# Steps 5-6 over both MIMIC archives, sharded. Logs to logs/.
cd "$(dirname "$0")/.."
PY=${PY:-python}; mkdir -p logs
export OMP_NUM_THREADS=1
for s in 0 1 2 3 4 5 6 7; do SHARD=$s NSHARD=8 $PY src/s05_mimic_survey.py mimic3wdb > logs/s05_m3_$s.log 2>&1 & done
$PY src/s05_mimic_survey.py mimic4wdb > logs/s05_m4.log 2>&1 &
wait
echo "survey done $(date)"
for s in $(seq 0 15); do SHARD=$s NSHARD=16 $PY src/s06_mimic_extract.py mimic3wdb > logs/s06_m3_$s.log 2>&1 & done
$PY src/s06_mimic_extract.py mimic4wdb > logs/s06_m4.log 2>&1 &
wait
echo "extraction done $(date)"
ls results/mimic_feat | wc -l
