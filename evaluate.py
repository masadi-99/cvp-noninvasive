"""Reproduce the result: load the per-case feature table and report the model AUC and the
parsimony frontier (AUC as features are added one at a time). Needs only data/features.csv.

    python evaluate.py
"""
import csv
import numpy as np
from cvp.model import evaluate, THRESHOLD

FEATURES = ["ppg_alternans", "ppg_ac_amp", "ppg_upstroke", "ppg_pvi", "height", "asa"]
# order in which features are added for the parsimony frontier (most informative first)
FRONTIER = ["ppg_alternans", "asa", "ppg_ac_amp", "ppg_upstroke", "height", "ppg_pvi"]


def load(path="data/features.csv"):
    rows = list(csv.DictReader(open(path)))
    cases = np.array([int(r["caseid"]) for r in rows])
    y = np.array([float(r["cvp_numeric"]) for r in rows])
    col = lambda f: np.array([float(r[f]) if r[f] != "" else np.nan for r in rows])
    return {f: col(f) for f in FEATURES}, y, cases


def main():
    X, y, cases = load()
    mat = lambda fs: np.column_stack([X[f] for f in fs])
    npos = int((y > THRESHOLD).sum())
    print(f"cohort: {len(cases)} cases — {npos} elevated (CVP>12) / {len(cases)-npos} normal "
          f"({100*npos/len(cases):.1f}% prevalence)\n")

    full = evaluate(mat(FEATURES), y, cases, reps=40)
    print(f"FULL 6-feature model:  AUC = {full['auc']:.3f}  [90% CI {full['ci90'][0]:.3f}-{full['ci90'][1]:.3f}]"
          f"   Sens {full['sens']:.2f} / Spec {full['spec']:.2f}\n")

    print("parsimony frontier (add one feature at a time):")
    print(f"  {'k':>2}  {'feature added':16}  AUC")
    chosen = []
    for k, f in enumerate(FRONTIER, 1):
        chosen.append(f)
        r = evaluate(mat(chosen), y, cases, reps=40)
        print(f"  {k:>2}  + {f:14}  {r['auc']:.3f}")
    print("\nNote: nested grouped-by-patient CV, 40 repeats. With only 40 positives this estimate is "
          "in-cohort optimistic; treat ~0.79 as an upper estimate (see README).")


if __name__ == "__main__":
    main()
