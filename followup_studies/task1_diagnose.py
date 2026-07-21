"""TASK 1 (diagnosis) — WHY does the 90-s window hurt, and can it be fixed?

Hypothesis: 3 of the 4 features (ac_amp, upstroke, pvi) are timescale-invariant medians, but
pulsus-alternans = power fraction in ONE FFT bin (S[-1]/sum) is defined for the ~30-s beat count;
over 90 s it changes meaning. Test: recompute the 90-s alternans as the MEDIAN of the three 30-s
sub-window alternans values (native timescale, aggregated) and see if AUC recovers.

Arms (identical shared valid-90 anchors, per-case median, thr 12):
  30s              — baseline
  90s-naive        — all 4 features over the full 90 s (the failing arm)
  90s-fixed        — same but alternans = median over 3x 30-s sub-windows
  90s-no-alternans — 3 features (drop alternans) to confirm it's the culprit
"""
import json, time
import numpy as np
import common as K
from cvpkit import config as C
from collections import defaultdict

t0 = time.time()
FS = K.FS


def alternans_from_amps(amps):
    """The pulsus-alternans formula on a pulse-amplitude array (verbatim)."""
    if len(amps) < 15 or np.median(amps) < 1e-9:
        return np.nan
    mm = np.array([np.median(amps[max(0, i - 2):i + 3]) for i in range(len(amps))])
    r = amps - mm
    if np.std(r) < 1e-9:
        return np.nan
    S = np.abs(np.fft.rfft(r - r.mean())) ** 2
    return float(S[-1] / (S.sum() + 1e-12))


def feats90(beats):
    """naive-90 features + a 'fixed' alternans (median over 30-s thirds)."""
    d = K.feats_from_beats(beats)                       # naive over all 90 s beats
    if beats:
        foot = np.array([b[0] for b in beats]); amps = np.array([b[1] for b in beats])
        sub = []
        for t in range(3):
            m = (foot >= t * 30 * FS) & (foot < (t + 1) * 30 * FS)
            if m.sum() >= 15:
                sub.append(alternans_from_amps(amps[m]))
        sub = [x for x in sub if np.isfinite(x)]
        d_fix_alt = float(np.median(sub)) if sub else np.nan
    else:
        d_fix_alt = np.nan
    return d, d_fix_alt


d = np.load(C.WINDOWS_NPZ, allow_pickle=True); M = d["data"]; cols = [str(c) for c in d["cols"]]
cid = M[:, cols.index("cid")].astype(int); start = M[:, cols.index("start")].astype(int)
anchors = defaultdict(list)
for c, s in zip(cid, start):
    anchors[int(c)].append(int(s))

W30, W90 = 30 * FS, 90 * FS
r30, r90n, r90f, r90na = [], [], [], []
for n, c in enumerate(sorted(anchors)):
    lc = K.load_ppg_cvp(c)
    if lc is None:
        continue
    ppg, cvpn = lc; L = len(ppg)
    for s in anchors[c]:
        if s + W90 > L or np.isnan(ppg[s:s + W90]).any():
            continue
        lab30 = K.cvp_label(cvpn, s // FS, 30); lab90 = K.cvp_label(cvpn, s // FS, 90)
        if not (np.isfinite(lab30) and np.isfinite(lab90)):
            continue
        f30 = K.ppg_features(ppg[s:s + W30])
        b90 = K.ppg_beats(ppg[s:s + W90]); fn, fixalt = feats90(b90)
        r30.append((c, f30, lab30))
        r90n.append((c, dict(fn), lab90))
        r90f.append((c, {**fn, "ppg_alternans": fixalt}, lab90))
        r90na.append((c, {k: fn[k] for k in ["ppg_ac_amp", "ppg_upstroke", "ppg_pvi"]}, lab90))
    if (n + 1) % 80 == 0:
        print(f"  [{n+1}/{len(anchors)}] {time.time()-t0:.0f}s", flush=True)

# per-feature 30s-vs-90s per-case correlation (why it breaks)
print(f"\nextraction done {time.time()-t0:.0f}s\n--- per-case feature agreement 30s vs 90s-naive ---")
Xa, _, ca, fna = K.per_case_matrix(r30); Xb, _, cb, _ = K.per_case_matrix(r90n)
casemap = {int(c): i for i, c in enumerate(cb)}
for j, f in enumerate(K.FEATS):
    a = Xa[:, j]; b = np.array([Xb[casemap[int(c)], j] if int(c) in casemap else np.nan for c in ca])
    m = np.isfinite(a) & np.isfinite(b); r = np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 5 else np.nan
    print(f"    {f:16s} r(30,90)={r:+.3f}")

out = {}
import common as _K
for tag, rows, feats in [("30s", r30, _K.FEATS), ("90s-naive", r90n, _K.FEATS),
                         ("90s-fixed", r90f, _K.FEATS),
                         ("90s-no-alternans", r90na, ["ppg_ac_amp", "ppg_upstroke", "ppg_pvi"])]:
    # per_case_matrix uses the global FEATS; temporarily override for the reduced arm
    _K.FEATS = feats
    X, y, cases, fnames = _K.per_case_matrix(rows)
    r = _K.evaluate(X, y, cases, threshold=12.0, reps=40)
    print(f"{tag:18s} N={r['n']} pos={r['npos']} AUC={r['auc']:.3f} CI[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]")
    out[tag] = {k: r[k] for k in ["auc", "ci", "n", "npos"]}
    _K.FEATS = ["ppg_alternans", "ppg_ac_amp", "ppg_upstroke", "ppg_pvi"]
json.dump(out, open("results_task1_diagnose.json", "w"), indent=2)
print("saved results_task1_diagnose.json")
