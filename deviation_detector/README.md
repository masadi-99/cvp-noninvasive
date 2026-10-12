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
src/s21..s32_*.py        the follow-up steps (extra inputs; cardiac output on VitalDB and MIMIC-III with the
                         head-start control; ECG/PPG alone with time bins; MIMIC-III arms; horizons with the
                         filter span; absolute CVP without calibration on VitalDB and MIMIC-III; figures)
run_all.sh               everything in order; scripts/run_gated.sh and scripts/run_notime.sh for the two variants,
                         scripts/run_followup.sh for steps 21-32
```

Inputs are not in the repo: the VitalDB gated window table and 30-s tile table, raw VitalDB cases,
VitalDB `clinical_data.csv`, and the MIMIC-III/IV waveform records as `.npz` (see `src/config.py`).
Python 3.12; `pip install -r requirements.txt`.

## Follow-up steps 21-32 (gated variant; `scripts/run_followup.sh`)

Questions raised after the presentation, each answered on the same folds and model:

- **Extra inputs (s21).** Demographics appended to the 123 columns change the AUC by -0.006 / +0.004 / +0.009; the arterial-line MAP by +0.006 / +0.005 / +0.004. The read reduced to its median + clock scores 0.642 / 0.660 / 0.666, the clock alone 0.650 / 0.653 / 0.674: the read's value is its variability and trend, not its level.
- **Cardiac output as the label (s23-s26).** EV1000 cardiac output on VitalDB (146 patients, |drift| > 0.5 / 0.75 / 1 L/min): full model 0.830 / 0.840 / 0.838; CVP on 140 of the same patients 0.737 / 0.759 / 0.734. PA-catheter cardiac output on 31 MIMIC-III records (public mimic3wdb numerics): zero-shot 0.803 / 0.808 / 0.829 on all stretches, but 0.658 / 0.669 / 0.710 once the stretches already outside the band at the end of the read are removed, against a clock at 0.690 / 0.668 / 0.659. The same control leaves VitalDB cardiac output at 0.798 / 0.829 / 0.832 and the main CVP system at 0.767 / 0.776 / 0.792 (VitalDB) / 0.795 / 0.784 / 0.762 (MIMIC-III).
- **ECG/PPG features alone (s27).** No clock, no read columns: 0.693 / 0.696 / 0.673 (clock alone 0.650 / 0.653 / 0.674, both 0.709 / 0.721 / 0.713). Inside 0-5 / 5-10 / 10-15 min since the read, single-window prediction at n>2: ECG/PPG alone 0.648 / 0.647 / 0.635, clock alone 0.594 / 0.527 / 0.514.
- **MIMIC-III arms (s28).** Zero-shot, pooled: clock 0.652 / 0.666 / 0.673, ECG/PPG alone 0.713 / 0.728 / 0.744, both 0.715 / 0.732 / 0.745, read columns + clock 0.791 / 0.766 / 0.737, full 0.814 / 0.793 / 0.769; per record the full model averages 0.804 / 0.785 / 0.759.
- **Longer tracking (s29).** At 15 / 20 / 30 / 45 / 60 min after the read, n>2: full model 0.777 / 0.765 / 0.749 / 0.753 / 0.722 with the artefact filter over the whole stretch, 0.788 / 0.780 / 0.770 / 0.784 / 0.769 with the filter over the read only; ECG/PPG alone 0.693 / 0.699 / 0.670 / 0.686 / 0.675; clock 0.650 / 0.669 / 0.664 / 0.686 / 0.700. The whole-stretch filter keeps 1,027 / 1,013 / 884 / 688 / 559 of 1,145 / 1,173 / 1,082 / 906 / 773 stretches.
- **Absolute CVP without calibration (s30, s32).** Raw features against CVP > 8 / 10 / 12 mmHg on VitalDB: per window 0.803 / 0.766 / 0.744, per case 0.848 / 0.814 / 0.783, within a patient 0.548 / 0.556 / 0.573. The signal is the raw PPG DC level alone (without it: 0.554 / 0.538 / 0.532 per window). Applied to MIMIC-III without retraining: 0.582 / 0.597 / 0.582 per record; trained within MIMIC-III 0.667 / 0.653 / 0.618.
