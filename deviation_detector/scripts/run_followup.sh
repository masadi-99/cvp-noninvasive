#!/bin/bash
# Steps 21-32 (the Q&A round and the follow-up round), all on the gated variant. Needs the base tree and
# scripts/run_gated.sh to have run. Step 24 downloads the public MIMIC-III numerics from PhysioNet.
set -e; cd "$(dirname "$0")/.."; PY=${PY:-python}; export VARIANT=gated; mkdir -p logs/gated
$PY src/s21_extra_inputs.py                                 # demographics / arterial pressure as inputs; the read reduced to its median
$PY src/s23_cardiac_output.py                               # EV1000 cardiac output / index / stroke volume as the label; CVP on the same patients
$PY src/s24_mimic_numerics.py headers && $PY src/s24_mimic_numerics.py fetch   # MIMIC-III records with a PA-catheter cardiac output
$PY src/s25_cardiac_output_mimic.py                         # cardiac output on those records, zero-shot and MIMIC-only
$PY src/s26_co_head_start.py                                # stretches already outside the band at the end of the read, removed from every arm
$PY src/s22_qa_figures.py                                   # figures for steps 21-26
$PY src/s27_waveform_only.py                                # ECG/PPG features alone; every arm by minutes since the read
$PY src/s28_mimic_arms.py                                   # MIMIC-III zero-shot arms, pooled and per record
for S in 15 20 30 45 60; do SDUR=$S $PY src/s29_horizon_arms.py; done   # every arm per tracking horizon, filter over the stretch or the read
$PY src/s30_absolute_cvp.py                                 # absolute CVP from the raw features, no calibration
$PY src/s32_absolute_cvp_mimic.py                           # the same on MIMIC-III, applied without retraining and trained within
$PY src/s31_review_figures.py                               # figures for steps 27-32
