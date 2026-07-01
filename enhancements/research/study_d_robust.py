"""Study D robustness — is the +morph(mean,f1,f4)=0.775 gain real, or winner's curse?

The eval showed base5=0.756, +morph_mean=0.753 (mean class is uninformative: corr w/ CVP +0.01),
but +morph(mean,f1,f4)=0.775. Before claiming a gain we: (1) decompose which class fraction drives
it, (2) PAIRED-test the full morph config vs base5 over identical splits, (3) permutation null,
(4) re-check at a second seed for stability.
"""
import json, numpy as np
import enh
from cvpkit import aggregate, config as C

THR = 12.0; REPS = 40
Xb, y, cases = enh.per_case(enh.FEATS5)
m = np.load("morph_features.npz"); mp = {int(c): i for i, c in enumerate(m["cid"])}
order = np.array([mp[int(c)] for c in cases])
f1 = m["morph_f1"][order].reshape(-1, 1); f4 = m["morph_f4"][order].reshape(-1, 1)
mn = m["morph_mean"][order].reshape(-1, 1)

cfgs = {
    "base5": Xb,
    "+f1 (class-1 frac)": np.column_stack([Xb, f1]),
    "+f4 (class-4 frac)": np.column_stack([Xb, f4]),
    "+f1+f4": np.column_stack([Xb, f1, f4]),
    "+mean+f1+f4": np.column_stack([Xb, mn, f1, f4]),
}
res = {}
for name, X in cfgs.items():
    for seed in (0, 1):
        r = enh.evaluate(X, y, cases, threshold=THR, reps=REPS, seed=seed)
        res.setdefault(name, {})[f"seed{seed}"] = round(r["auc"], 4)
    print(f"  {name:20} seed0={res[name]['seed0']:.3f}  seed1={res[name]['seed1']:.3f}", flush=True)

# paired test: +mean+f1+f4 vs base5 over identical splits (seed 0)
rb = enh.evaluate(Xb, y, cases, threshold=THR, reps=REPS, seed=0, return_reps=True)["reps_auc"]
rm = enh.evaluate(cfgs["+mean+f1+f4"], y, cases, threshold=THR, reps=REPS, seed=0, return_reps=True)["reps_auc"]
delta = rm - rb
paired = dict(mean_delta=float(delta.mean()), pct_splits_improve=float(np.mean(delta > 0)),
              t=float(delta.mean() / (delta.std(ddof=1) / np.sqrt(len(delta)) + 1e-12)), reps=len(delta))
print(f"\n  PAIRED +mean+f1+f4 vs base5: Δ={delta.mean():+.4f}  {100*paired['pct_splits_improve']:.0f}% improve  t={paired['t']:.2f}", flush=True)

# permutation null for the full morph config
nulls = []
for s in range(30):
    rng = np.random.default_rng(5000 + s); yp = y.copy(); rng.shuffle(yp)
    nulls.append(enh.evaluate(cfgs["+mean+f1+f4"], yp, cases, threshold=THR, reps=1)["auc"])
nulls = np.array(nulls)
perm = dict(null_mean=float(nulls.mean()), null_p95=float(np.percentile(nulls, 95)),
            real=res["+mean+f1+f4"]["seed0"], p_gt=float(np.mean(nulls >= res["+mean+f1+f4"]["seed0"])))
print(f"  permutation null: mean={perm['null_mean']:.3f} p95={perm['null_p95']:.3f}  real={perm['real']:.3f}  p={perm['p_gt']:.3f}", flush=True)

res["paired_full_vs_base"] = paired; res["perm_null"] = perm
json.dump(res, open("results_d_robust.json", "w"), indent=2)
print("\nsaved results_d_robust.json", flush=True)
