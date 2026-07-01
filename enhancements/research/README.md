# research/ — original analysis scripts (provenance only)

These are the scripts as they were run during the study, against the **full private VitalDB
pipeline** (the `cvpkit` package, the per-window feature matrix, arterial/vent tracks, and raw
beats). They import that pipeline and reference local paths, so **they do not run from this repo
as-is** — they are included for transparency and provenance.

For runnable, self-contained reproductions of every result, use `../reproduce.py` and
`../reproduce_windowing.py`, which work from `../data/*.csv` and the repo's own `cvp.model`.

| file | study |
|---|---|
| `enh.py` | shared harness (merged per-window matrix + threshold-parameterized nested grouped CV) |
| `study_a_threshold.py` | A — CVP-threshold sweep |
| `study_b_perwindow.py`, `study_b_audit.py`, `study_b_clean.py`, `study_b_fix.py`, `study_b_winpred.py`, `study_b_diag.py` | B — windows-as-samples + full audit/decomposition |
| `study_c_arterial.py` | C — invasive arterial BP (systolic/diastolic/mean numerics) |
| `artwave.py`, `study_e_build.py`, `study_e_eval.py`, `study_e_artwave.py` | C — rich arterial-pressure WAVEFORM morphology (15 features analogous to the PPG set) |
| `dawber.py`, `study_d_build.py`, `study_d_eval.py`, `study_d_robust.py`, `study_d_f4.py` | D — Dawber morphology classifier + evaluation |
