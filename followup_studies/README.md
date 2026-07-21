# Follow-up studies

A second round of investigation into the non-invasive CVP model (the 5-feature finger-PPG model,
**AUC 0.756** at CVP > 12, 304 patients / 40 elevated), prompted by collaborator feedback. Four
questions:

1. **Analysis-window length** — is the 30-s window optimal? (`task1_*`)
2. **PPG-only gating** — can the ECG quality gate be dropped to recover more patients? (`sweep2.py`, `task2_*`)
3. **Other outcomes** — can the same PPG predict CO / SV / CI / SVRI / SVV / MAP? (`task3_*`)
4. **Confounders** — is the CVP signal confounded by demographics / clinical variables? (`task_confounders*`)

See **[REPORT.md](REPORT.md)** for the full step-by-step findings. Headlines:

- **30 s is the genuine optimum** — the peak of a fixed-cohort sweep (W=30 reproduces the deployed
  0.756 exactly); shorter windows are beat-starved (noisier amplitude features), longer windows are
  rarely clean and never win on like-for-like patients.
- **Dropping the ECG gate expands the cohort 304 → 357 but collapses AUC to chance (~0.53)** — the ECG
  gate is a load-bearing artifact filter even though the model is PPG-only (ECG-failed windows carry no
  CVP signal and outnumber the good ones ~2:1).
- **CVP is the only hemodynamic target the PPG waveform genuinely predicts.** The apparent Cardiac
  Output / Stroke Volume predictability was a **sex / body-size artifact** (height ≈ a sex proxy;
  low-CO prevalence 46 % female vs 7 % male). Cardiac Index, MAP/hypotension and fluid-responsiveness
  (SVV) are not predictable.
- **CVP has real clinical correlates** (renal function, ventilation, a transplant-heavy case-mix) but
  the PPG signal is **independent** of them — it adds over every confounder, survives within
  homogeneous subgroups, and predicts the confounder-residualised CVP. There is **no** body-size/sex
  confound. External validation on a different case-mix is the key next step.

All four conclusions were independently re-derived by four adversarial code audits; the evaluation
harness was confirmed leakage-free and one bug was found and fixed (a feature-column mix-up that had
overstated the Part-2 dilution: 0.556 → 0.608; conclusion unchanged).

## ⚠️ Provenance only — not runnable from this repo

These scripts were run against the **full private VitalDB pipeline**: the `cvpkit` package, the
signal-quality gate in the `cvp-ecg-ppg` package, the raw per-case waveforms, and the per-window
matrix `medium.npz`. They import that pipeline and reference local paths, so **they do not run from
this repo as-is** — they are included for transparency and provenance. `results/` holds the numeric
JSON outputs each script produced (the audited/corrected values).

## Scripts

| file | study |
|---|---|
| `common.py` | shared lean PPG feature extractor (validated **bit-exact** to the deployed features, r = 1.000) + eval helpers |
| `validate_baseline.py` | control — the extractor reproduces the 0.756 baseline before any change is trusted |
| **1 · Window length** | |
| `task1_sweep_v3.py` | **definitive** fixed-cohort window-length sweep (cohort + label fixed; W=30 anchored to 0.756) |
| `task1_fair.py` | same-cohort 30 s vs longer-W comparison (like-for-like, no imputation) |
| `task1_sweep_v2.py` | window sweep with recomputed labels (medium-oracle windows) |
| `task1_window.py` | first attempt — **superseded** (extended curated windows with un-gated tails; unfair) |
| `task1_diagnose.py`, `task1_diagnose2.py` | diagnostics for the initial (flawed) 90-s result |
| **2 · PPG-only gating (drop ECG)** | |
| `sweep2.py` | configurable re-tiler — any window length, `trimodal` or `ppgcvp` gate — reusing the upstream gate + labels |
| `task2_clean.py` | **headline** anchored dilution test (deployed vs +ECG-failed windows, deployed cohort) |
| `task2_diag.py` | expansion-cohort splits and window-provenance breakdown |
| `task2_ppg_quality.py` | neurokit PPG-quality (>0.7) filter + full-cohort expansion |
| `task2b_disentangle.py` | disentangles feature-window vs label effects |
| `diag_sweep.py` | diagnoses the config-drift between the current gate and the one that built `medium.npz` |
| **3 · Other outcomes** | |
| `task3c_outcomes.py` | CO / SV / CI / SVRI / SVV / MAP prediction + body-size (height/weight/BSA/sex) decomposition |
| `task3_co_ci.py` | Cardiac Output / Cardiac Index detail |
| `task3b_co_decompose.py` | CO decomposition (PPG-only vs height-only vs full) |
| `task3_verify.py` | the sex-vs-height verification (rules out a model artifact behind height-only) |
| **4 · Confounders** | |
| `task_confounders.py` | univariate confounder scan + adjustment + elevated-vs-normal demographics table |
| `task_confounders2.py` | PPG-carrier correlations, within-subgroup survival, confounder-residualised-CVP test |
