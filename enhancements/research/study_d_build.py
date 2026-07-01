"""Study D (build) — extract per-case Dawber morphology features from the raw beats.

For every clean window in the matrix (cid,start), rebuild the cvpkit window context, classify each
clean beat's descending limb into a Dawber class (dawber.classify), and aggregate per case:
  morph_mean (1..4 ordinal index), morph_mode, morph_f1 (frac class-1), morph_f4 (frac class-4), n_beats.
Caches to morph_features.npz so evaluation can re-run without re-extracting.
Also stashes a few example beats per class (for the diagnostic figure).
"""
import sys, warnings, json
sys.path.insert(0, "/home/masadi/cvp-ecg-ppg"); sys.path.insert(0, "/home/masadi/run")
warnings.filterwarnings("ignore")
import numpy as np
from cvpkit import signals, config as C
import enh, dawber

FS, WIN = C.FS, C.WIN
d = enh.load_merged(); cid, start = d["cid"], d["start"]
Xc, yc, cases = enh.per_case(enh.FEATS5)          # the 304-case cohort
case_set = set(cases.tolist())

starts_by_case = {}
for c, s in zip(cid.tolist(), start.tolist()):
    if c in case_set:
        starts_by_case.setdefault(c, []).append(s)

rows = []; examples = {1: [], 2: [], 3: [], 4: []}; clscount = {1: 0, 2: 0, 3: 0, 4: 0, None: 0}
for n, c in enumerate(cases):
    loaded = signals.load_case(int(c))
    ecg, ppg, anum, ncols = loaded
    classes = []
    for s in sorted(starts_by_case[int(c)]):
        e, p = ecg[s:s + WIN], ppg[s:s + WIN]
        if len(e) < WIN or np.isnan(e).any() or np.isnan(p).any():
            continue
        try:
            ctx = signals.make_context(e, p, signals.nibp_at(anum, ncols, s), {})
        except Exception:
            continue
        for b in ctx.beats:
            seg = ctx.pc[b["foot"]:b["foot"] + b["dur"]]
            cl = dawber.classify(seg)
            classes.append(cl); clscount[cl] = clscount.get(cl, 0) + 1
            if cl in examples and len(examples[cl]) < 8 and len(seg) > 12:
                examples[cl].append(seg.astype(float))
    f = dawber.case_features(classes)
    f["cid"] = int(c)
    rows.append(f)
    if (n + 1) % 50 == 0:
        print(f"  {n+1}/{len(cases)} cases done", flush=True)

cids = np.array([r["cid"] for r in rows])
np.savez_compressed("morph_features.npz",
                    cid=cids,
                    morph_mean=np.array([r["morph_mean"] for r in rows]),
                    morph_mode=np.array([r["morph_mode"] for r in rows]),
                    morph_f1=np.array([r["morph_f1"] for r in rows]),
                    morph_f4=np.array([r["morph_f4"] for r in rows]),
                    nbeats=np.array([r["n"] for r in rows]))
# save a few example beats per class for the figure
np.savez_compressed("morph_examples.npz",
                    **{f"c{k}": np.array(v, dtype=object) for k, v in examples.items()})
tot = sum(clscount[k] for k in (1, 2, 3, 4))
dist = {f"class{k}": round(clscount[k] / tot, 3) for k in (1, 2, 3, 4)}
json.dump({"class_distribution": dist, "n_beats_total": tot, "unclassified": clscount[None]},
          open("results_d_distribution.json", "w"), indent=2)
print("class distribution (per beat):", dist, " unclassified:", clscount[None], flush=True)
print("saved morph_features.npz, morph_examples.npz", flush=True)
