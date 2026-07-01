"""Study A — CVP threshold sweep.

The cohort is imbalanced at the canonical CVP>12 cut (40/304 = 13%). The proposal: lower the
threshold (9/8/7) for a more balanced cohort and see the effect on performance. We sweep
T in {7,8,9,10,11,12} on the 5-feature NON-INVASIVE model, reporting at each T:
  N positives, prevalence, AUC [90% CI], Sens, Spec (nested-Youden), permutation-null AUC.

Same model/CV as the headline. Honest read: more balance vs whether the signal survives.
"""
import json, numpy as np
import enh

THRS = [7, 8, 9, 10, 11, 12]
REPS = 40

X, y, cases = enh.per_case(enh.FEATS5)
print(f"cohort N={len(cases)}; per-case median CVP range [{y.min():.1f},{y.max():.1f}]\n", flush=True)

rows = []
for T in THRS:
    r = enh.evaluate(X, y, cases, threshold=float(T), reps=REPS)
    # permutation null at this threshold (shuffle continuous y), few reps
    null = []
    for s in range(20):
        rng = np.random.default_rng(9000 + s); yp = y.copy(); rng.shuffle(yp)
        null.append(enh.evaluate(X, yp, cases, threshold=float(T), reps=1)["auc"])
    nullm = float(np.mean(null))
    prev = 100 * r["npos"] / r["n"]
    rows.append(dict(thr=T, npos=r["npos"], n=r["n"], prev=prev, auc=r["auc"], ci=r["ci"],
                     sens=r["sens"], spec=r["spec"], nullauc=nullm))
    print(f"  CVP>{T:>2}: pos={r['npos']:>3}/{r['n']} ({prev:4.1f}%)  AUC={r['auc']:.3f} "
          f"[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]  Sens={r['sens']:.2f} Spec={r['spec']:.2f}  null={nullm:.3f}",
          flush=True)

json.dump(rows, open("results_a_threshold.json", "w"), indent=2)
print("\nsaved results_a_threshold.json", flush=True)
