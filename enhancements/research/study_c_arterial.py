"""Study C — add INVASIVE arterial-line blood pressure to the feature set.

Proposal: add systolic / diastolic / mean BP from the arterial line. These are the Solar8000
monitor numerics art_sbp_n / art_dbp_n / art_mbp_n (available in 100% of this cohort's cases,
since every CVP-line case also has an A-line). This REQUIRES an arterial catheter, so it is an
INVASIVE branch — reported separately from the non-invasive headline.

Configs (per-case median aggregation, same Ridge+HGB ensemble, CVP>12, 40 reps):
  base5            5 non-invasive features                              (reference)
  +art3            base5 + art_sbp_n, art_dbp_n, art_mbp_n              (the faithful request)
  +art5            base5 + the 3 numerics + art_pp + art_ppv            (richer invasive upper bound:
                                                                         pulse pressure & PP-variation,
                                                                         a fluid-responsiveness index)
  art3 alone       just the 3 arterial numerics
Plus: single-feature CV AUC of each arterial feature, and a PAIRED test (+art3 vs base5 over
identical CV splits).
"""
import json, numpy as np
import enh

THR = 12.0; REPS = 40

# extend the merged matrix with art_pp + art_ppv (waveform-derived) for the richer set
d = enh.load_merged()
z = np.load("/home/masadi/run/data/medium_eng.npz", allow_pickle=True)
Me, ce = z["data"], [str(c) for c in z["cols"]]
cide = Me[:, ce.index("cid")].astype(int); starte = Me[:, ce.index("start")].astype(int)
idx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(cide, starte))}
order = np.array([idx[k] for k in d["key"]])
for f in ["art_pp", "art_ppv"]:
    d["F"][f] = Me[order, ce.index(f)].astype(float)

ART3 = ["art_sbp_n", "art_dbp_n", "art_mbp_n"]
ART5 = ART3 + ["art_pp", "art_ppv"]

configs = {
    "base5": enh.FEATS5,
    "+art3 (sys/dia/mean)": enh.FEATS5 + ART3,
    "+art5 (+PP,+PPV)": enh.FEATS5 + ART5,
    "art3 alone": ART3,
}
res = {}
for name, feats in configs.items():
    X, y, cases = enh.per_case(feats)
    r = enh.evaluate(X, y, cases, threshold=THR, reps=REPS)
    res[name] = dict(auc=r["auc"], ci=r["ci"], sens=r["sens"], spec=r["spec"], nfeat=r["nfeat"])
    print(f"  {name:22} ({r['nfeat']}f): AUC={r['auc']:.3f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}] "
          f"Sens={r['sens']:.2f} Spec={r['spec']:.2f}", flush=True)

# single-feature AUCs of the arterial features
print("\n  single-feature CV AUC (arterial):", flush=True)
sf = {}
for f in ART5:
    X, y, cases = enh.per_case([f])
    a = enh.evaluate(X, y, cases, threshold=THR, reps=REPS)["auc"]
    sf[f] = a; print(f"    {f:12} {a:.3f}", flush=True)
res["single_feature"] = sf

# paired test: +art3 vs base5 over identical CV splits
Xb, yb_, cb = enh.per_case(enh.FEATS5)
Xc, yc_, cc = enh.per_case(enh.FEATS5 + ART3)
rb = enh.evaluate(Xb, yb_, cb, threshold=THR, reps=REPS, return_reps=True)["reps_auc"]
rc = enh.evaluate(Xc, yc_, cc, threshold=THR, reps=REPS, return_reps=True)["reps_auc"]
delta = rc - rb
win = float(np.mean(delta > 0)); tval = float(delta.mean() / (delta.std(ddof=1) / np.sqrt(len(delta)) + 1e-12))
res["paired_art3_vs_base"] = dict(mean_delta=float(delta.mean()), pct_splits_improve=win, t=tval, reps=len(delta))
print(f"\n  PAIRED +art3 vs base5: Δ={delta.mean():+.3f}  {100*win:.0f}% of splits improve  t={tval:.1f}", flush=True)

json.dump(res, open("results_c_arterial.json", "w"), indent=2)
print("\nsaved results_c_arterial.json", flush=True)
