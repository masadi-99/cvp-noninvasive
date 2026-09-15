# Deviation detector: has the CVP moved more than n mmHg since the last read?

A CVP line is read for 10 minutes in every 25. Between reads, a gradient-boosting regressor scores
each 30-s window from 91 finger-PPG / surface-ECG features (differenced against the read), the minutes
since the read, and the read's own CVP shape; the score is the causal running maximum of the predicted
max |CVP - baseline|. Trained on VitalDB (surgical), tested zero-shot on MIMIC-III/IV ICU records.

## Results (AUC at |drift| > 2 / 3 / 4 mmHg)

| | VitalDB, 5-fold by patient | MIMIC-III zero-shot | MIMIC-trained ceiling | MIMIC per-record mean |
|---|---|---|---|---|
| 123 columns, MIMIC label without a CVP-waveform gate | 0.781 / 0.780 / 0.791 | 0.834 / 0.796 / 0.757 | 0.862 / 0.819 / 0.776 | 0.801 / 0.775 / 0.742 |
| 123 columns, MIMIC CVP-waveform gate (`VARIANT=gated`) | 0.781 / 0.780 / 0.791 | 0.814 / 0.793 / 0.769 | 0.855 / 0.839 / 0.823 | 0.804 / 0.785 / 0.759 |
| 119 columns, no elapsed-time product columns (`VARIANT=gated_notime`) | 0.786 / 0.790 / 0.800 | 0.811 / 0.790 / 0.766 | 0.856 / 0.840 / 0.825 | 0.801 / 0.779 / 0.753 |

Controls on VitalDB: clock-only 0.650 / 0.653 / 0.674, permuted targets 0.545 / 0.509 / 0.490.
What carries it: the calibration read's own CVP (spread, last-minus-median, slope) gives most of the
AUC (a learned model on the non-waveform columns alone: 0.745 / 0.756 / 0.768); the ECG/PPG columns add
+0.035 / +0.024 / +0.022 on VitalDB and +0.014 / +0.017 / +0.015 on MIMIC. Sixteen columns match the
full model; eight keep 0.763 / 0.829 (VitalDB / MIMIC at n>2).

## Layout

```
src/config.py            paths, schedule (10-min read, 15-min tracking), model, variant switches
src/lib/features.py      the 91 window features (verbatim VitalDB extractors)
src/lib/build.py         schedule, artefact filter, baseline, 123-column vector, model, scoring
src/s00..s20_*.py        the ordered steps (data -> tracker -> nested CV -> examples -> MIMIC survey /
                         extraction / transfer / analysis -> investigation -> extra records ->
                         label quality -> causal alignment -> feature reduction -> horizons -> gate -> ablations)
run_all.sh               everything in order; scripts/run_gated.sh and scripts/run_notime.sh for the two variants
```

Inputs are not in the repo: the VitalDB gated window table and 30-s tile table, raw VitalDB cases,
VitalDB `clinical_data.csv`, and the MIMIC-III/IV waveform records as `.npz` (see `src/config.py`).
Python 3.12; `pip install -r requirements.txt`.
