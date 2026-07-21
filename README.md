# Non-invasive prediction of elevated central venous pressure

Predict **elevated central venous pressure (CVP > 12 mmHg)** from a finger pleth (PPG) waveform and
one pre-operative chart value (height) — no central line. Built on the open [VitalDB](https://vitaldb.net)
surgical dataset.

A **five-feature** model reaches **AUC ≈ 0.75** (repeated nested grouped-by-patient cross-validation).

## The five features

| feature | source | what it measures |
|---|---|---|
| `ppg_pvi` | PPG | pleth variability index (respiratory amplitude variation) |
| `ppg_alternans` | PPG | beat-to-beat pulse-amplitude alternation (mechanical pulsus alternans) |
| `ppg_ac_amp` | PPG | pulse amplitude |
| `ppg_upstroke` | PPG | systolic upstroke time (foot→peak) — slow upstroke ↔ high CVP |
| `height` | chart | patient height (weak alone; helps only in combination) |

Each PPG value is the **median over a patient's clean 30-s windows**. See [cvp/extract.py](cvp/extract.py).

## Result — parsimony frontier

Adding one feature at a time (nested grouped CV, 40 repeats; 304 patients, 40 elevated):

| # features | + feature | AUC |
|---|---|---|
| 1 | ppg_pvi | 0.68 |
| 2 | + ppg_alternans | 0.71 |
| 3 | + ppg_ac_amp | 0.74 |
| 4 | + ppg_upstroke | 0.75 |
| 5 | + height | **0.75** |

**Full 5-feature model: AUC 0.754 [90% CI 0.720–0.778], sensitivity 0.66 / specificity 0.67.**

The signal is carried by the PPG waveform features (`ppg_pvi`, `ppg_alternans`, `ppg_ac_amp`); the
single most useful feature is `ppg_pvi`, and `height` adds little on its own.

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

`data/features.csv` is the per-patient feature table (five features + CVP label) derived from VitalDB.
To re-extract it from the raw waveforms, download VitalDB and run `python -m cvp.build` (see the paths
in [cvp/build.py](cvp/build.py)).

## Honest caveats

- **Small positive count.** With only 40 elevated patients, the cross-validated AUC is *in-cohort
  optimistic*. Treat ~0.75 as an upper estimate; a conservative, more generalizable figure is ~0.70–0.73.
- **This is a screening signal, not a CVP measurement.** AUC is threshold-free; the operating point
  (sens/spec) is chosen on an inner CV of the training data only.
- **Population.** VitalDB surgical patients; elevated-CVP cases are enriched for fluid-overload /
  transplant physiology. External validation on a separate population is needed before any clinical use.

## Enhancement studies

Four proposed additions were tested under the same CV — see [enhancements/](enhancements/). Three do
not help (lower CVP threshold → ~0.66; windows-as-samples → ~0.53; invasive arterial BP → 0.745), and
one does: a **categorical PPG-morphology feature (Dawber 4-class)** gives **0.754 → 0.780 (+0.026)**,
Sensitivity 0.68 → 0.75. All are reproducible from the repo:

```bash
python enhancements/reproduce.py            # threshold, arterial BP, morphology
python enhancements/reproduce_windowing.py  # windows-as-samples (a few minutes)
```

## Follow-up studies

A later round probed the model's design and scope — see [followup_studies/](followup_studies/)
(provenance only; scripts ran against the full private pipeline). In brief: **30 s is the optimal
analysis window**; the **ECG quality gate cannot be dropped** (it is a load-bearing artifact filter —
dropping it grows the cohort 304 → 357 but collapses AUC to chance); and **CVP is the only
hemodynamic target the PPG waveform genuinely predicts** — apparent Cardiac-Output / Stroke-Volume
prediction was a body-size/sex artifact. A confounder audit found CVP has real clinical correlates but
the PPG signal is independent of them (no sex/size confound).

## Layout

```
cvp/extract.py     four PPG features from one 30-s window
cvp/model.py       Ridge + HistGradientBoosting ensemble, repeated nested grouped CV
cvp/build.py       raw VitalDB cases -> data/features.csv
evaluate.py        features.csv -> AUC + parsimony frontier
data/features.csv  per-patient features + CVP label (304 rows)
enhancements/      four enhancement studies (arterial BP, windowing, morphology, threshold),
                   self-contained data + runnable reproductions + research scripts
followup_studies/  window length, PPG-only gating, other outcomes, confounder audit
                   (provenance only — not runnable from this repo)
```

```
cvp/extract.py     four PPG features from one 30-s window
cvp/model.py       Ridge + HistGradientBoosting ensemble, repeated nested grouped CV
cvp/build.py       raw VitalDB cases -> data/features.csv
evaluate.py        features.csv -> AUC + parsimony frontier
data/features.csv  per-patient features + CVP label (304 rows)
enhancements/      four enhancement studies (arterial BP, windowing, morphology, threshold),
                   self-contained data + runnable reproductions + research scripts
```
