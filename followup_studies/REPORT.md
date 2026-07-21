# Collaborator follow-up — CORRECTED report

> Supersedes the earlier version. Two methodological errors in the first pass were found and fixed:
> (1) the window-length test extended curated 30-s windows out to 90 s, appending **un-gated** signal
> (an unfair test); (2) Task 2 dropped the **CVP** gate, not just the ECG gate. Both are redone
> against the deployed pipeline (`cvp/sweep.py` gate + labels), with the deployed **AUC 0.756 / 304
> cases anchored and reproduced exactly** as the sanity check. Code: `run/analyses/collab_followups/`.

The lean PPG extractor reproduces the deployed features **exactly** (per-feature r = 1.000) and the
full pipeline reproduces **AUC 0.756 at W=30** — every result below is anchored to that.

### Independent verification (4 adversarial code audits, each re-deriving numbers from scratch)
- **Eval harness — SOUND.** No train/test leakage (imputer/scaler in-fold, Youden op-point on inner
  CV, AUC strictly out-of-fold); independent re-derivation of 0.756 (0.74–0.77 across four CV variants),
  label-permuted null = 0.49.
- **Task 1 — CONFIRMED.** Every number reproduced to ±0.001; extractor bit-exact; no off-by-one / label
  drift. Nuance: on *like-for-like* cases 60 s ≈ 30 s (0.707 vs 0.711, a tie); 30 s's win over 60 s is
  driven by availability + the clear degradation of *shorter* windows, not a large per-case edge over 60 s.
- **Task 2 — CONFIRMED, one bug fixed.** An audit found a feature-column misalignment in the dilution
  test (window start-time was injected as the alternans feature, PVI dropped); **corrected the middle
  row 0.556 → 0.608** (table above already updated). Direction/conclusion unchanged (A=0.756, C=0.461).
  Minor: the PPG gate omits one MAD-spike check, so "ECG-only" attribution is ~90% precise.
- **Task 3 — CONFIRMED.** CO/SV=sex artifact reproduced exactly (46% vs 7% prevalence); CVP genuine &
  unconfounded survives independent code. **Caveat (applies to all Task-3 numbers):** they use
  rep-averaged-OOF, ~+0.02 more optimistic than the headline per-rep protocol — so e.g. "CVP PPG-only
  0.767" is ~0.72–0.75 under stricter CV. Relative comparisons are unaffected (same protocol throughout).

---

## Task 1 — Optimal analysis-window length → **30 s is the genuine optimum**

Method (fixed to isolate window length): the **deployed 304 cohort and CVP label are held fixed**;
only the feature-window length W changes. A W-window is built only from **gate-clean** signal (every
30-s tile it spans must have passed the deployed gate — no un-gated tails). W=30 = the deployed
pipeline exactly.

| W (s) | 8 | 10 | 15 | 20 | 25 | **30** | 40 | 45 | 60 | 90 | 120 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (fixed 304 cohort) | .670 | .649 | .688 | .662 | .723 | **.756** | .568 | .591 | .602 | .520 | .471 |
| cases with a clean W-window | 304 | 304 | 304 | 304 | 304 | 304 | 205 | 205 | 205 | 155 | 129 |

**30 s is the peak.** Two forces bracket it:
- **Shorter → worse.** Fewer beats make the amplitude-based features (pulsus alternans needs ≥15
  beats; PVI; ac-amp) noisier. AUC falls to ~0.65–0.69 by 8–20 s.
- **Longer → worse.** Fully-clean long stretches are rare, so the usable cohort shrinks fast
  (304 → 129 by 120 s). And on a **like-for-like cohort** (only the cases that *have* a clean
  W-window, so no imputation), 30 s still wins or ties every longer window:

  | cohort | 30 s | longer W | Δ(W−30) |
  |---|---|---|---|
  | cases-with-45s-window (N=205) | **0.711** | 45 s: 0.674 | −0.037 |
  | cases-with-60s-window (N=205) | **0.711** | 60 s: 0.707 | −0.003 |
  | cases-with-90s-window (N=155) | **0.603** | 90 s: 0.556 | −0.047 |

  (This 205-case subset scores 0.711 at 30 s, not 0.756, because long-recording cases are a slightly
  harder subset — which is why the raw "avail-only" numbers were never a like-for-like comparison.)

*(This corrects the earlier report's "90 s hurts, 0.72→0.59" — that number came from appending
un-gated signal to a gated 30-s window. Done properly, 30 s is simply optimal.)*
Files: `task1_sweep_v3.py` (fixed-cohort sweep), `task1_fair.py` (same-cohort), `task1_sweep_v2.py`.

---

## Task 2 — Drop **only** the ECG quality gate (keep PPG + CVP gates)

Corrected per your note: keep the PPG channel gate **and** the CVP-morphology gate; drop only the ECG
SQI/burst gate. (The CVP gate still uses ECG R-peaks for beat timing — the a/c/x/v/y fit is
R-peak-referenced and can't be done without them.)

**Result: it expands the cohort but destroys performance.**
- ppgcvp cohort = **357 cases** (+119 beyond the deployed 304), 13 074 windows (~37/case vs ~27
  deployed) — dropping the ECG gate lets ~2× more windows through.
- AUC = **0.525 — chance.**

**Clean, drift-free proof** (add exactly the ECG-gate-failed windows to the deployed 0.756 pipeline,
same 304 cohort + label):

| window pool | AUC |
|---|---|
| deployed windows only | **0.756** |
| deployed + ECG-gate-failed windows (46 % of pool) | **0.608** |
| ECG-gate-failed windows only | **0.461** (anti-signal) |

**Why:** an ECG-failed window is a *motion/artifact* flag. Those windows still pass the PPG-template
and CVP-morphology checks, but the artifact corrupts the PPG features and the R-peak-timed CVP label,
so they carry **no** CVP signal (0.46–0.49) — and they *outnumber* the good windows (46–69 % of the
pool), dominating the per-case median. **The ECG gate is load-bearing as a general artifact filter,
even though the model itself is PPG-only.** So the ECG gate can't simply be dropped to gain cases; a
replacement PPG/CVP-based artifact gate would be needed first.
Files: `sweep2.py` (configurable re-tiler), `task2_diag.py`, `task2_clean.py`.

---

## Task 3 — What else can the PPG predict? → **only CVP; CO/SV are a SEX artifact**

Same 5-feature model, every hemodynamic target. Verified three ways (raw Spearman, univariate raw-AUC
with NO model, and model-based) to rule out leaks; decomposed against height, weight, BSA **and sex**.

| target (clinical cut) | N | full AUC | PPG-only | height-only | sex-only | verdict |
|---|---|---|---|---|---|---|
| **CVP** (>12) | 304 | **0.773** | **0.767** | 0.385 | 0.449 | **genuine PPG waveform** ✔ |
| Cardiac Output (<4) | 131 | 0.802 | 0.661 | 0.765 | **0.756** | **sex artifact** ✗ |
| Stroke Volume (<60) | 131 | 0.737 | 0.444 | 0.768 | 0.736 | **sex artifact** ✗ |
| Cardiac Index (<2.5) | 131 | 0.626 | — | 0.651 | 0.697 | weak / sex |
| SVRI (<1970) | 65 | 0.644 | — | — | — | weak (small N) |
| SVV (>13, fluid-resp) | 131 | 0.474 | — | — | — | not predictable (4 pos) |
| MAP / hypotension (<65) | 304 | 0.527 | — | — | — | not predictable |

**The CO/SV "predictions" are a sex confound, not cardiac physiology.** The absolute cut-offs
(CO<4, SV<60) essentially select small/female patients: low-CO prevalence is **46% in women vs 7% in
men**; low-SV **64% vs 19%**. `sex`-only AUC (0.76) ≈ `height`-only AUC (0.77) — height is just a
proxy for sex. (No model artifact: raw-height univariate AUC 0.765 == model height-only 0.756.) The
body-size-**indexed** target, Cardiac Index, is near-chance (0.63) precisely because indexing removes
the confound. **So finger PPG does not genuinely predict cardiac output or stroke volume.**

**CVP is the one real result.** It has **zero** body-size/sex confound (Spearman(height,CVP) = −0.02,
sex-only AUC 0.45 = chance), so its **PPG-only AUC 0.767** is true venous-congestion waveform signal —
the only hemodynamic target the finger pulse genuinely encodes. MAP/hypotension, fluid-responsiveness
(SVV), and Cardiac Index are not captured.
Files: `task3_co_ci.py`, `task3b_co_decompose.py`, `task3c_outcomes.py`, `task3_verify.py`.

---

## Confounder audit + cohort demographics (CVP model)

Prompted by the CO/SV sex-artifact, I checked whether CVP prediction is similarly confounded.
Files: `task_confounders.py`, `task_confounders2.py`.

**Cohort demographics — elevated (n=40) vs normal (n=264):**
- **No body-size/sex confound** (the CO/SV killer is absent): sex 42% vs 53% male (p=0.23),
  height p=0.82, BMI p=0.37, MAP/HR/SpO2 n.s.
- Elevated patients are a **sicker, distinct case-mix**: ASA 3 vs 2 (p<0.001), creatinine 1.2 vs 0.8
  (p<0.001), BUN↑, Hb 11.2 vs 12.8 (p=0.002); 38% emergency vs 12% (p<0.001); **35% transplant vs 9%**
  (p<0.001); tidal volume 403 vs 340 mL (p<0.001); MAC 0.8 vs 0.0; more open surgery. All are real
  physiology of fluid overload / renal dysfunction.

**Several clinical variables predict CVP as well as the PPG does** (univariate AUC for CVP>12):
tidal volume 0.73, MAC 0.71, creatinine 0.71, remifentanil 0.71, ASA 0.69, BUN 0.69 — vs PPG upstroke
0.65, alternans 0.66. So there ARE strong non-PPG predictors.

**But the PPG signal is genuinely independent — NOT a proxy (unlike CO/SV):**
- PPG carriers barely correlate with any confounder (|Spearman| ≤ 0.13 with TV, MAC, Cr, ASA, BUN, Hb).
- Adjustment: PPG-only 0.767; all top-8 confounders-only 0.796; **PPG + all confounders 0.835** — PPG
  adds +0.04 over the *entire* confounder set. (Contrast CO/SV: PPG added nothing over sex.)
- Survives homogeneous subgroups: exclude transplant **0.774**, general-surgery only **0.752**,
  exclude emergency **0.764**. So not a surgery-type / sickness proxy.
- Predicts the confounder-**residualized** CVP (Spearman 0.25, vs 0.32 raw).

**Verdict:** CVP prediction is genuine venous-congestion waveform signal, independent of the measured
confounders — no body-size/sex artifact. **Caveat:** the elevated-CVP cohort is clinically
distinctive (transplant-heavy, sicker, higher-creatinine), so **external validation on a different
case-mix** is the key next step to confirm the signal generalizes beyond this specific population.

## One line each
1. **Window length:** 30 s is the genuine optimum (peak of a proper sweep; shorter = noisier
   features, longer = rare clean windows). Earlier "90 s" result was a windowing bug, now fixed.
2. **Drop ECG gate:** expands 304→357 cases but collapses to chance (0.53) — the ECG gate is a
   load-bearing artifact filter; ECG-failed windows are anti-signal and flood the median.
3. **Other outcomes:** CVP is the only genuine one (0.767 PPG-only, zero sex/size confound). The
   CO/SV "predictions" were a **sex artifact** (height ≈ sex proxy; low-CO 46% female vs 7% male);
   CI/MAP/SVV/fluid-responsiveness are not predictable. Finger PPG does not predict cardiac output.
