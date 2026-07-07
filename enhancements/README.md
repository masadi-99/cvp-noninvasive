# Enhancement studies

Four proposed additions to the non-invasive model were implemented and evaluated under the **same**
honest nested grouped-by-patient cross-validation as the headline model. **Baseline** throughout is
the 5-feature model (`ppg_alternans, ppg_ac_amp, ppg_upstroke, ppg_pvi, height`), **AUC 0.754** at
CVP > 12 (304 patients / 40 elevated).

**Result: three of the four do not help; one does.**

| # | proposal | result | verdict |
|---|---|---|---|
| A | Lower the CVP threshold (9/8/7 mmHg) for a balanced cohort | AUC falls to **~0.66** | ✗ worse |
| B | Treat each 30-s window as its own training sample | collapses to **~0.53** (random-window CV leaks to 0.90) | ✗ worse |
| C | Add invasive arterial-line BP — sys/dia/mean **and** 15 rich waveform features | 0.754 → **0.745** (numerics); **~0.68** (rich waveform, hurts) | ✗ no help |
| D | Add a categorical PPG-morphology feature (Dawber 4-class) | 0.754 → **0.780** (+0.026, t = 13.2, perm p < 0.001) | ✔ **small gain** |

## Reproduce

Everything here runs from two self-contained data files (no raw signals needed):

```bash
pip install -r ../requirements.txt
python enhancements/reproduce.py                    # studies A (threshold), C (arterial BP), D (morphology)
python enhancements/reproduce_windowing.py          # study B (windowing) — takes a few minutes
python enhancements/reproduce_arterial_waveform.py  # study C rich waveform-morphology extension
```

- `data/features_ext.csv` — per-patient: the 5 model features + arterial numerics + Dawber class
  fractions + CVP (304 rows).
- `data/windows.csv` — per-window: the 5 features + arterial numerics + per-window CVP (8,909 rows),
  used by the windowing study.
- `data/artwave_features.csv` — per-patient: the 15 rich arterial-waveform features + CVP (304 rows).

The first two are derived once from the full VitalDB pipeline (generator:
`research/gen_repo_data.py`, provenance-only); the scripts above reproduce every headline
number from them using the repo's own Ridge + HistGradientBoosting ensemble (`cvp.model`).

## What each study found

**A — CVP threshold.** Sweeping the "elevated" cut from 12 down to 7 makes the cohort more balanced
(13% → 51% positive) but AUC falls to ~0.66: the borderline CVP 8–12 patients are not separable from
normals on the discriminative features, so balancing the labels does not create signal.

**B — windows as samples.** The per-patient **median** (the deployed design, 0.754) beats treating
each window as an independent training row (~0.53). This is **not** a label bug (per-window CVP
aggregates exactly to the patient label) and it is **not** that more windows hurt — aggregating *more*
windows by median monotonically *improves* AUC (0.62 at one window → 0.756 at all). The drop comes
from feeding raw windows to the model and decomposes into two ordinary, fixable artifacts:
(1) **window-count weighting** — long-recording patients contribute more rows and dominate the fit
(a control with zero feature noise still drops to ~0.65; weighting patients equally recovers ~0.72);
(2) **single-window feature noise** — e.g. pulsus alternans is only ~0.35-reliable in one 30-s window.
Grouped-by-patient CV is essential: random-window CV leaks to 0.90.

**C — invasive arterial BP.** Arterial systolic/diastolic/mean pressures (`art_*_n`, available in 100%
of the cohort) are uncorrelated with CVP (|r| ≤ 0.09) and each is ≈ chance alone (AUC 0.42–0.58).
Adding them mildly *hurts* (0.754 → 0.745). Arterial pressure is systemic afterload, not venous
preload.

*Waveform-morphology extension.* Because the simple numerics could be "too simplistic," we also
extracted **15 rich arterial-pressure WAVEFORM features analogous to the PPG set** — upstroke time,
systolic/notch fractions, augmentation index, dicrotic-notch height, systolic/diastolic area ratio,
Windkessel decay τ, pulse width, max dP/dt (up/down), reflected-wave transit, arterial pulsus
alternans, amplitude CV, PPV, SPV (`research/artwave.py`). Every one is near-chance (single-feature
AUC 0.44–0.64, correlations |r| ≤ 0.11), and adding them **significantly hurts** the model
(0.754 → ~0.68; paired t = −17.9, 0% of splits improve). So the null is not an artifact of simplistic
features — arterial waveform *shape* carries no CVP signal either. (`reproduce_arterial_waveform.py`,
`data/artwave_features.csv`.)

**D — categorical PPG morphology (Dawber 4-class).** Graded by dicrotic-notch development
(class 1 = notch present → class 4 = fully damped). The per-patient **class fractions** help: the
class-4 (fully-damped) fraction is ~2.6× higher in elevated patients (median 0.037 vs 0.014),
correlates +0.12 with CVP, and is orthogonal to the existing upstroke feature. Adding the class-1 and
class-4 fractions gives **0.754 → 0.780 (+0.026)**, Sensitivity 0.68 → 0.75; paired 95% of splits
improve, t = 13.2, permutation-null p < 0.001, stable across seeds. This is in-cohort like the
baseline — the *increment* is leakage-free and validated, but the absolute 0.78 still needs external
validation. Reference: Dawber's classification, Fig. 2 of Sensors 2021;21(13):4315.

## `research/`

The original research scripts (`enh.py`, `dawber.py`, `study_*.py`) are kept in `research/` for
provenance. They were run against the **full private VitalDB pipeline** (the per-window feature
matrix, arterial tracks, and raw beats) and reference local paths, so they are **not** runnable from
this repo as-is. The self-contained, runnable reproductions are `reproduce.py` /
`reproduce_windowing.py` above.
