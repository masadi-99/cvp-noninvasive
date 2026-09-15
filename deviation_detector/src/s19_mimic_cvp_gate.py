"""Step 19. A CVP-waveform quality gate for MIMIC, and every MIMIC number with it.

VitalDB's label comes from an a-wave detector that only passes windows with a real, pulsatile CVP
waveform. The MIMIC extraction (step 6) only required a plausible median (-5..40 mmHg) and
>= 90 % finite samples, so clamped or flushed transducers (flat lines at 24-37 mmHg for hours,
single-window spikes) entered the label. This step measures, for every MIMIC window, the CVP
waveform's standard deviation, its longest run of identical samples, and its beat-to-beat
pulsatility, then rebuilds the stretches from the windows that pass and rescores: zero-shot full
model, the one-column baseline, the CVP-line-only model, the waveform-only model and the
MIMIC-trained ceiling. The 400-record subset of the reported numbers is used throughout.
    python src/s19_mimic_cvp_gate.py measure     # per-window quality -> results/mimic_cvp_quality.npz
    python src/s19_mimic_cvp_gate.py score       # gated stretches, all arms -> results/mimic_gated.json
"""
import os, sys, glob, json, time
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES, RES_BASE, MIMIC_RECORDS, THRS, W_CAL, S_TRACK
FSM, WINM = 125, 3750
GATE = dict(std_min=0.5, std_max=12.0, flat_run_max_s=2.0, puls_min=0.5)   # see the deck's gate slide


def longest_run(x):
    if len(x) < 2: return len(x)
    chg = np.flatnonzero(np.diff(x) != 0)
    if len(chg) == 0: return len(x)
    b = np.concatenate(([-1], chg, [len(x) - 1])); return int(np.diff(b).max())


def measure(subset="paper"):
    from lib.build import mimic_files
    out = {}; qf = f"{RES_BASE}/mimic_cvp_quality.npz"
    if os.path.exists(qf):
        Q0 = np.load(qf, allow_pickle=True)
        for k in {x.rsplit("__", 1)[0] for x in Q0.files}: out[k] = {q: Q0[f"{k}__{q}"] for q in ("t", "std", "flat_run_s", "puls")}
    t0 = time.time()
    todo = [p for p in mimic_files(subset) if os.path.basename(p).replace(".npz", "") not in out]
    print(f"{len(out)} records already measured; measuring {len(todo)}", flush=True)
    for i, p in enumerate(todo):
        key = os.path.basename(p).replace(".npz", ""); src, rid = key.split("_", 1)
        zf = np.load(p, allow_pickle=True); t = zf["t"]
        z = np.load(f"{MIMIC_RECORDS}/{src}/{rid}.npz", allow_pickle=True); cvp = z["sig_cvp"]
        sd, run, pul = [], [], []
        for tt in t:
            a = int(round(tt * 2)) * WINM; c = cvp[a:a + WINM].astype(float); f = np.isfinite(c)
            cc = c[f] if f.any() else np.array([np.nan])
            sd.append(float(np.nanstd(c))); run.append(longest_run(np.round(cc, 3)) / FSM)
            # pulsatility: median over 1-s segments of (95th - 5th percentile) within the segment
            segs = [c[j:j + FSM] for j in range(0, WINM, FSM)]
            pul.append(float(np.nanmedian([np.nanpercentile(s, 95) - np.nanpercentile(s, 5) for s in segs if np.isfinite(s).sum() > 60] or [np.nan])))
        out[key] = dict(t=t, std=np.array(sd, np.float32), flat_run_s=np.array(run, np.float32), puls=np.array(pul, np.float32))
        if i % 40 == 0: print(f"  {i+1} records [{time.time()-t0:.0f}s]", flush=True)
    np.savez_compressed(qf, **{f"{k}__{q}": v for k, d in out.items() for q, v in d.items()})
    allsd = np.concatenate([d["std"] for d in out.values()]); allrun = np.concatenate([d["flat_run_s"] for d in out.values()]); allp = np.concatenate([d["puls"] for d in out.values()])
    print(f"{len(allsd):,} windows: std percentiles 5/25/50/75/95 = {np.nanpercentile(allsd, [5,25,50,75,95]).round(2)};  "
          f"longest flat run (s) 50/75/90/95 = {np.nanpercentile(allrun, [50,75,90,95]).round(1)};  pulsatility 5/25/50/75/95 = {np.nanpercentile(allp, [5,25,50,75,95]).round(2)}")
    ok = gate_mask(allsd, allrun, allp); print(f"gate {GATE}: {100*ok.mean():.1f}% of windows pass")


def gate_mask(sd, run, pul):
    return (sd >= GATE["std_min"]) & (sd <= GATE["std_max"]) & (run <= GATE["flat_run_max_s"]) & (pul >= GATE["puls_min"])


def load_gated(src="mimic3wdb", W=W_CAL, S=S_TRACK):
    from lib.build import build_record, mimic_files
    Q = np.load(f"{RES_BASE}/mimic_cvp_quality.npz", allow_pickle=True); R = {}; kept = tot = 0
    for p in mimic_files("paper"):
        key = os.path.basename(p).replace(".npz", "")
        if not key.startswith(src): continue
        z = np.load(p, allow_pickle=True); ok = gate_mask(Q[f"{key}__std"], Q[f"{key}__flat_run_s"], Q[f"{key}__puls"])
        tot += len(ok); kept += int(ok.sum())
        ss = build_record(z["t"].astype(float)[ok], z["y"].astype(float)[ok], z["X"].astype(float)[ok], key, W, S)
        if ss: R[key] = ss
    print(f"gate keeps {kept:,} of {tot:,} windows; {len(R)} records with stretches", flush=True)
    return R


def score_all():
    from lib.build import VitalDB, load_vitaldb_generic, fit, score, target, boot_ci, feature_names, build_vitaldb
    from lib.features import PPG_FEATS, ECG_FEATS, NF_FEATS
    t0 = time.time(); V = VitalDB(tracker=False); NAMES = feature_names(V.names); IDX = {n: i for i, n in enumerate(NAMES)}
    VD = load_vitaldb_generic(V); MI = load_gated(); keys = sorted(MI); segs = [s for k in keys for s in MI[k]]
    T = np.concatenate([target(s) for s in segs]); G = np.concatenate([[s["key"]] * len(s["dev"]) for s in segs])
    Y = {t: (T > t).astype(int) for t in THRS}
    WAVE = [IDX[f"d_{n}"] for n in PPG_FEATS + ECG_FEATS + NF_FEATS]; TRK = [IDX[n] for n in NAMES if n.startswith("trk_")]
    NOWAVE = [i for i in range(len(NAMES)) if i not in WAVE and i not in TRK]; WAVET = WAVE + [IDX["since_last_cvp"]]
    out = dict(gate=GATE, records=len(keys), stretches=len(segs), windows=int(len(T)), prevalence={f"n>{t:g}": float(Y[t].mean()) for t in THRS}, arms={}); SC = {}

    def rec(nm, P, per=True):
        d = {f"n>{t:g}": dict(auc=float(roc_auc_score(Y[t], P)), ci=list(boot_ci(P, Y[t], G, 200))) for t in THRS}
        if per:
            aus = [roc_auc_score(Y[2.0][G == r], P[G == r]) for r in keys if Y[2.0][G == r].min() != Y[2.0][G == r].max()]
            d["per_record_n2"] = float(np.mean(aus)); d["evaluable"] = len(aus)
        out["arms"][nm] = d; SC[nm] = P
        print(f"  {nm:34s} " + "  ".join(f"{d[k]['auc']:.3f}" for k in d if k.startswith("n>")) + (f"   per-record n>2 {d['per_record_n2']:.3f} (n={d['evaluable']})" if per else "") + f"  [{time.time()-t0:.0f}s]", flush=True)
    MS = fit(VD, list(VD), drop_nan_rows=True); PW = score(MS, segs); rec("zero-shot full model", np.concatenate(PW))
    # example figures on the gated data, same layout as step 9
    from lib.plots import draw_records
    for s_, p_ in zip(segs, PW): s_["p"] = p_; s_["ev"] = target(s_) > 3.0
    REC = {k.replace("mimic3wdb_", ""): MI[k] for k in keys}
    ok = {}
    for k in keys:
        yy = np.concatenate([(target(s_) > 3.0).astype(int) for s_ in MI[k]]); pp = np.concatenate([s_["p"] for s_ in MI[k]])
        if yy.min() != yy.max(): ok[k.replace("mimic3wdb_", "")] = float(roc_auc_score(yy, pp))
    # show records with at least 8 stretches and 2 event stretches, so a 'best' record is not a two-stretch record with one event
    rich = [k for k in ok if len(REC[k]) >= 8 and sum(int(s_["ev"].any()) for s_ in REC[k]) >= 2]
    srt = sorted(rich, key=lambda k: -ok[k]); q80 = float(np.quantile(np.concatenate(PW), .80)); med = srt[len(srt) // 2 - 1: len(srt) // 2 + 1]
    draw_records(REC, ok, q80, srt[:2], "mimic_gated_best", "MIMIC ICU with the CVP-waveform gate: the two best records (n>3), among records with >= 8 stretches and >= 2 events", "Windows whose CVP waveform is flat, clamped or flushed are removed before the stretches are formed. Same layout as before.")
    draw_records(REC, ok, q80, med, "mimic_gated_typical", "MIMIC ICU with the CVP-waveform gate: two typical records", "Same layout.")
    draw_records(REC, ok, q80, srt[-2:], "mimic_gated_worst", "MIMIC ICU with the CVP-waveform gate: the two worst records", "Same layout.")
    out["per_record_n3"] = dict(mean=float(np.mean(list(ok.values()))), evaluable=len(ok))
    rec("one column: block spread x since (no model)", np.concatenate([np.maximum.accumulate(np.nan_to_num(s["X"][:, IDX["std_x_since"]])) for s in segs]), per=False)
    rec("clock: minutes since the read", np.concatenate([s["X"][:, IDX["since_last_cvp"]] for s in segs]), per=False)
    rec("CVP-line columns only (no ECG/PPG)", np.concatenate(score(fit(VD, list(VD), drop_nan_rows=True, cols=NOWAVE), segs, cols=NOWAVE)), per=False)
    rec("ECG/PPG + clock only", np.concatenate(score(fit(VD, list(VD), drop_nan_rows=True, cols=WAVET), segs, cols=WAVET)), per=False)
    rng = np.random.default_rng(11); ids = np.array(keys); rng.shuffle(ids); PF = {}
    for te in np.array_split(ids, 5):
        tr = [k for k in keys if k not in set(te.tolist())]; ms = fit(MI, tr, nseed=2, drop_nan_rows=True); sg = [s for k in te.tolist() for s in MI[k]]
        for s, p in zip(sg, score(ms, sg)): PF[id(s)] = p
    rec("MIMIC-trained ceiling (5-fold by record)", np.concatenate([PF[id(s)] for s in segs]))
    np.savez_compressed(f"{RES}/mimic_gated_scores.npz", T=T, G=G, **{f"arm{i}": SC[k] for i, k in enumerate(SC)}, arm_names=np.array(list(SC)))
    json.dump(out, open(f"{RES}/mimic_gated.json", "w"), indent=1); print(f"[{time.time()-t0:.0f}s] -> results/mimic_gated.json")


if __name__ == "__main__":
    if sys.argv[1] == "measure": measure(sys.argv[2] if len(sys.argv) > 2 else "paper")
    else: score_all()
