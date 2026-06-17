# Non-invasive prediction of elevated central venous pressure

Predict **elevated central venous pressure (CVP > 12 mmHg)** from a finger pleth (PPG) waveform and
two pre-operative chart values — no central line. Built on the open [VitalDB](https://vitaldb.net)
surgical dataset.

A **six-feature** model reaches **AUC ≈ 0.79** (repeated nested grouped-by-patient cross-validation).

## The six features

| feature | source | what it measures |
|---|---|---|
| `ppg_alternans` | PPG | beat-to-beat pulse-amplitude alternation (mechanical pulsus alternans) |
| `ppg_upstroke` | PPG | systolic upstroke time (foot→peak) — slow upstroke ↔ high CVP |
| `ppg_ac_amp` | PPG | pulse amplitude |
| `ppg_pvi` | PPG | pleth variability index (respiratory amplitude variation) |
| `height` | chart | patient height |
| `asa` | chart | ASA physical-status score |

Each PPG value is the **median over a patient's clean 30-s windows**. See [cvp/extract.py](cvp/extract.py).

## Result — parsimony frontier

Adding one feature at a time (nested grouped CV, 40 repeats; 304 patients, 40 elevated):

| # features | + feature | AUC |
|---|---|---|
| 1 | ppg_alternans | 0.66 |
| 2 | + asa | 0.74 |
| 3 | + ppg_ac_amp | 0.78 |
| 4 | + ppg_upstroke | 0.79 |
| 5 | + height | 0.79 |
| 6 | + ppg_pvi | **0.79** |

**Full 6-feature model: AUC 0.794 [90% CI 0.765–0.815], sensitivity 0.77 / specificity 0.69.**

The single most useful feature is `ppg_alternans`; `asa` (one chart value) is the largest single jump.

## Cohort

| stage | cases | |
|---|---|---|
| full VitalDB | 6,388 | the open surgical dataset |
| downloaded | 1,256 | cases with PPG + ECG + an invasive CVP line |
| quality-passed | 333 | ≥1 clean 30-s window (good PPG/ECG **and** a calibrated CVP for the label) |
| **analysis cohort** | **304** | with a contemporaneous monitor CVP reading |

**304 patients — 40 elevated (CVP > 12) / 264 normal (13.2% prevalence).**

## Reproduce

```bash
pip install -r requirements.txt
python evaluate.py            # uses data/features.csv — no raw signals needed
```

`data/features.csv` is the per-patient feature table (six features + CVP label) derived from VitalDB.
To re-extract it from the raw waveforms, download VitalDB and run `python -m cvp.build` (see the paths
in [cvp/build.py](cvp/build.py)).

## Honest caveats

- **Small positive count.** With only 40 elevated patients, the cross-validated AUC is *in-cohort
  optimistic*. Treat ~0.79 as an upper estimate; a conservative, more generalizable figure is ~0.74–0.76.
- **This is a screening signal, not a CVP measurement.** AUC is threshold-free; the operating point
  (sens/spec) is chosen on an inner CV of the training data only.
- **Population.** VitalDB surgical patients; elevated-CVP cases are enriched for fluid-overload /
  transplant physiology. External validation on a separate population is needed before any clinical use.

## Layout

```
cvp/extract.py   four PPG features from one 30-s window
cvp/model.py     Ridge + HistGradientBoosting ensemble, repeated nested grouped CV
cvp/build.py     raw VitalDB cases -> data/features.csv
evaluate.py      features.csv -> AUC + parsimony frontier
data/features.csv  per-patient features + CVP label (304 rows)
```
