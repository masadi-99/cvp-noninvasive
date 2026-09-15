"""Step 12. Every deck figure that is not an example plot (steps 4 and 9 draw those).

Part A (data and method, needs only data/ and the raw archives):
  f01_schedule        the calibration / tracking cycle on one real VitalDB case
  f02_pipeline        the whole system on one slide
  f03_cohorts         VitalDB OR vs MIMIC ICU: how much data each patient gives
  f04_window          one 30-s window: PPG, ECG, and the hidden CVP; the same for a MIMIC window
  f05_ppg_features    the 39 PPG features on an annotated pulse and ensemble beat
  f06_ecg_features    the 20 ECG features on the ensemble beat
  f07_nf_features     the 32 nf features: SDPPG, pulse arrival, coupling, dynamics, two-Gaussian
  f08_differencing    why every feature is differenced against the calibration block
  f09_block_features  the 32 non-waveform columns, drawn on one calibration block
  f10_target          the target, the dense baseline, the artefact filter
  f11_protocol        nested CV, controls, clustered bootstrap
Part B (results, needs results/):
  f13_headline, f14_schedules, f16_parity, f17_transfer, f21_operating, f22_baselines,
  f23_ablations, f24_construction, f25_importance, f26_single, f27_confounders,
  f28_demographics, f29_mimic_strata, f30_correlates, f31_limits
    python src/s12_figures.py A|B|all
"""
import os, sys, json, csv, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (DATA, RES, RES_BASE, FIG, THRS, W_CAL, S_TRACK, VITALDB_CASES, MIMIC_RECORDS, SCALE_REF,
                    DESC_NAMES, HIST_NAMES, EXTRA_NAMES, HGB, MAX_STEP, DROP_COLS, VARIANT)
from lib.build import VitalDB, build_vitaldb, target, desc, worst_step, feature_names, ACTIVE as ACTIVE_COLS
from lib import features as F
from lib import style as C
PART = sys.argv[1] if len(sys.argv) > 1 else "all"
FS = 500


def box(ax, x, y, w, h, text, fc, ec=None, fs=9.5, bold=False, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", fc=fc, ec=ec or fc, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            color=tc or C.INK, wrap=True)


def arrow(ax, x0, y0, x1, y1, color=C.INK):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=1.4, color=color))


# ===================================================================================================
def part_a():
    V = VitalDB(); R = build_vitaldb(V)
    # ---- f01 schedule on one real case ---------------------------------------------------------
    best = max(R, key=lambda c: (len(R[c]), sum(int((target(s) > 3).any()) for s in R[c])))
    c = sorted(R, key=lambda c: -(len(R[c]) * (1 + sum(int((target(s) > 3).any()) for s in R[c]))))[0]
    ss = R[c]; o = V.ORDER[c]; t, y = V.T[o], V.Y[o]; u, yd = V.DEN[c]
    off = np.median([np.median(s["yb"]) - s["base"] for s in ss]) * 0
    fig = plt.figure(figsize=(17, 6.6)); gs = fig.add_gridspec(2, 1, height_ratios=[2.6, 1.1], hspace=.12)
    ax = fig.add_subplot(gs[0]); axb = fig.add_subplot(gs[1], sharex=ax)
    from matplotlib.patches import Patch
    for i, s in enumerate(ss):
        ax.axvspan(s["tb"][0], s["tb"][-1], color=C.BLUE, alpha=.14, lw=0)
        axb.axvspan(s["tb"][0], s["tb"][-1], color=C.BLUE, alpha=.14, lw=0)
        ax.plot(s["tb"], s["yb"], "s", color=C.CVPC, ms=4, mec="#fff", mew=.5)
        ax.plot([s["t"][0], s["t"][-1]], [s["base"]] * 2, "-", color=C.GREY, lw=2.4)
        ax.fill_between([s["t"][0], s["t"][-1]], s["base"] - 3, s["base"] + 3, color=C.GREY, alpha=.13, lw=0)
        ax.plot(s["t"], s["y"], "-o", color=C.CVPC, lw=1.3, ms=3.6, mew=0)
        bad = s["dev"] > 3
        if bad.any(): ax.plot(s["t"][bad], s["y"][bad], "o", color=C.RED, ms=7, mew=0, zorder=5)
        axb.step(s["t"], target(s), where="post", color=C.INK, lw=2)
        axb.fill_between(s["t"], 0, target(s), step="post", color=C.INK, alpha=.12, lw=0)
    ax.set_ylabel("CVP (mmHg)"); ax.set_ylim(np.nanmin(y) - 2, np.nanmax(y) + 3)
    ax.set_title(f"VitalDB case {c}: a 10-min calibration read, then 15 min of tracking on ECG/PPG only, repeating for the whole case", fontsize=12.5, color=C.INK, pad=26)
    from matplotlib.lines import Line2D
    ax.legend([Patch(color=C.BLUE, alpha=.14), Line2D([], [], color=C.CVPC, marker="s", lw=0, ms=5), Line2D([], [], color=C.CVPC, marker="o", lw=1.3, ms=4),
               Line2D([], [], color=C.GREY, lw=2.4), Line2D([], [], color=C.RED, marker="o", lw=0, ms=6)],
              ["10-min calibration read (CVP line in use)", "CVP readings inside the read", "CVP during tracking (hidden from the model, used to score)",
               "baseline = median of the read, +/-3 mmHg band", "windows truly beyond 3 mmHg"], loc="upper center", bbox_to_anchor=(.5, 1.09), ncol=5, fontsize=8.8, frameon=False)
    ax.tick_params(labelbottom=False)
    axb.set_ylabel("target: max |CVP - baseline|\nso far (mmHg)", fontsize=9.5); axb.set_xlabel("minutes into the case")
    axb.axhline(3, color=C.RED, lw=1.2, ls=":"); axb.text(axb.get_xlim()[1], 3.1, "n = 3 mmHg", color=C.RED, fontsize=9, ha="right")
    fig.text(.5, .005, "The bottom trace is what the model is asked to predict: it can only rise within a stretch and resets at every calibration read.", ha="center", va="bottom", fontsize=9.5, color=C.MUTE)
    C.save(fig, "f01_schedule")

    # ---- f02 pipeline ----------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(17, 5.4)); ax.set_xlim(0, 17); ax.set_ylim(0, 5.4); ax.axis("off")
    cols = [(0.3, "30-s window\nECG + PPG\n500 Hz", "#eef1f4"), (2.9, "91 waveform\nfeatures\n39 PPG / 20 ECG / 32 nf", "#e6f3ec"),
            (5.5, "minus the calibration\nblock's median\n(within-patient)", "#e3edf7"), (8.1, "+ 32 CVP-line columns\ntime since reading,\nblock shape, history", "#fdf1e3"),
            (10.7, "gradient boosting\nmagnitude regressor\n3 seeds, |error| loss", "#efe6f4"), (13.3, "causal running max\n= predicted drift\nso far (mmHg)", "#fbe9e7"),
            (15.2, "rank /\nalarm", "#fbe9e7")]
    for i, (x, tx, fc) in enumerate(cols):
        w = 2.3 if i < 6 else 1.5
        box(ax, x, 2.6, w, 1.7, tx, fc, fs=9.8, bold=(i in (1, 4)))
        if i < 6: arrow(ax, x + w + .02, 3.45, cols[i + 1][0] - .02, 3.45)
    box(ax, 0.3, 0.45, 7.6, 1.45, "GROUND TRUTH: the invasive CVP catheter, read for 10 min per 25-min cycle\n(the calibration), then hidden. Target = largest |CVP - baseline| seen so far\nin the stretch. Used to train, to calibrate and to score; never an input while tracking.",
        "#fbe9e7", fs=9.6, tc="#8c2d1e")
    box(ax, 8.4, 0.45, 8.3, 1.45, "TWO COHORTS: VitalDB (Seoul, anaesthetised surgery, 815 patients) to train,\nnested cross-validation by patient. MIMIC-III/IV (Boston, ICU, 417 records,\n423k windows): zero-shot external test, same code, same 91 features.", "#e3edf7", fs=9.6)
    ax.text(8.5, 4.95, "Everything at minute t uses only minutes <= t. The only thing the model ever learns about THIS patient's CVP is what the line said during calibration.",
            ha="center", fontsize=10.5, color=C.INK, fontweight="bold")
    C.save(fig, "f02_pipeline")

    # ---- f03 cohorts -------------------------------------------------------------------------------
    span = [np.ptp(V.T[V.ORDER[c]]) for c in V.ORDER]; nwin = [len(V.ORDER[c]) for c in V.ORDER]
    sv = []
    for f in glob.glob(f"{RES_BASE}/mimic/survey_mimic3wdb_*.json") + glob.glob(f"{RES_BASE}/mimic/survey_mimic4wdb_*.json"):
        sv += [r for r in json.load(open(f)) if "hours" in r]
    from lib.build import mimic_files
    used = {os.path.basename(p).split("_", 1)[1].replace(".npz", "") for p in mimic_files("paper")}
    mh = [r["hours"] for r in sv if r["rec"] in used]; mrun = [r["longest_run_min"] for r in sv if r["rec"] in used]
    fig, AX = plt.subplots(1, 3, figsize=(17, 4.6))
    ax = AX[0]; ax.hist(span, bins=np.arange(0, 400, 10), color=C.BLUE, edgecolor="white")
    ax.axvline(25, color=C.RED, lw=2, ls="--"); ax.text(27, ax.get_ylim()[1] * .9, "one full 25-min cycle", color=C.RED, fontsize=9.5)
    ax.set_xlabel("gated span per VitalDB patient (min)"); ax.set_ylabel("patients")
    ax.set_title(f"VitalDB: {len(span)} patients, median span {np.median(span):.0f} min", fontsize=11.5, color=C.INK)
    ax = AX[1]; ax.hist(nwin, bins=np.arange(0, 300, 8), color=C.BLUE, edgecolor="white")
    ax.set_xlabel("gated 30-s windows per VitalDB patient"); ax.set_ylabel("patients")
    ax.set_title(f"VitalDB: {sum(nwin):,} windows, median {np.median(nwin):.0f} per patient", fontsize=11.5, color=C.INK)
    ax = AX[2]; ax.hist(mh, bins=np.arange(0, 120, 4), color=C.ORANGE, edgecolor="white")
    ax.set_xlabel("record length (hours), MIMIC records used"); ax.set_ylabel("records")
    ax.set_title(f"MIMIC ICU: {len(mh)} records, median {np.median(mh):.0f} h, median usable run {np.median(mrun):.0f} min", fontsize=11.5, color=C.INK)
    fig.suptitle("Two cohorts: short gated OR cases to learn from, long ICU records to test on", fontsize=13, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .93]); C.save(fig, "f03_cohorts")

    # ---- f04 one window, both cohorts ------------------------------------------------------------
    # a VitalDB window: pick a gated window of the f01 case
    s0 = ss[0]; tw = float(s0["t"][3]); z = np.load(f"{VITALDB_CASES}/case_{c}.npz")
    a = int(round(tw * 60 * FS)); e = z["ecg"][a:a + 15000].astype(float); g = z["ppg"][a:a + 15000].astype(float); cv = z["cvp_waveform"][a:a + 15000].astype(float)
    mrec = sorted(glob.glob(f"{MIMIC_RECORDS}/mimic3wdb/*.npz"))[0]; zm = np.load(mrec, allow_pickle=True)
    zf = np.load(f"{RES_BASE}/mimic_feat/mimic3wdb_{os.path.basename(mrec)}", allow_pickle=True)
    i0 = int(zf["t"][10] * 2) * 3750
    from s06_mimic_extract import rescale
    emr, _ = rescale(zm["sig_ecg"].astype(np.float32), SCALE_REF["ecg_med"], SCALE_REF["ecg_iqr"]); gmr, _ = rescale(zm["sig_ppg"].astype(np.float32), SCALE_REF["ppg_med"], SCALE_REF["ppg_iqr"])
    em = emr[i0:i0 + 3750].astype(float); gm = gmr[i0:i0 + 3750].astype(float); cm = zm["sig_cvp"][i0:i0 + 3750].astype(float)
    gm5 = resample_poly(np.nan_to_num(gm, nan=np.nanmedian(gm)), 4, 1); em5 = resample_poly(np.nan_to_num(em, nan=np.nanmedian(em)), 4, 1)
    fig, AX = plt.subplots(3, 2, figsize=(17, 7.6), sharex="col")
    tt = np.arange(15000) / FS
    for j, (sig, lab, col, ttl) in enumerate([((g, gm5), "PPG (a.u.)", C.PPGC, "INPUT 1  finger photoplethysmogram"),
                                              ((e, em5), "ECG (mV)", C.ECGC, "INPUT 2  surface ECG"),
                                              ((cv, np.repeat(cm, 4)), "CVP (mmHg)", C.CVPC, "HIDDEN TRUTH  invasive CVP (never an input while tracking)")]):
        for k in range(2):
            ax = AX[j, k]; ax.plot(tt[:5000], sig[k][:5000], color=col, lw=1.0); ax.set_ylabel(lab, fontsize=9.5)
            ax.set_title(ttl + ("   VitalDB, 500 Hz native" if k == 0 else "   MIMIC-III, 125 Hz resampled x4 and amplitude-normalised"), fontsize=10.5, color=C.INK, loc="left")
    AX[2, 0].set_xlabel("seconds (first 10 of the 30-s window)"); AX[2, 1].set_xlabel("seconds")
    fig.suptitle("One window: two non-invasive inputs and a hidden truth, in both cohorts", fontsize=13, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .95]); C.save(fig, "f04_window")

    # ---- f05 PPG features ---------------------------------------------------------------------------
    pc = F.morph_ppg(g); feet = F.find_feet(pc); tmpl = F.window_template(pc, feet)
    beats = [(fi, fj, F.beat_fiducials(pc, fi, fj, template=tmpl)) for fi, fj in zip(feet[:-1], feet[1:])]
    beats = [b for b in beats if b[2] is not None and b[2]["notch"] is not None and b[2]["dia"] is not None]
    fi, fj, fid = beats[len(beats) // 2]; seg = pc[fi:fj]; ts = np.arange(len(seg)) / FS * 1000
    fig = plt.figure(figsize=(17, 8.2)); gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=.45, wspace=.28)
    ax = fig.add_subplot(gs[0, 0]); ax.plot(ts, seg, color=C.PPGC, lw=2.2)
    sp = fid["sys"] - fi; ni = fid["notch"] - fi; di = fid["dia"] - fi; a = seg[sp] - seg[0]
    ax.plot([0, ts[sp], ts[ni], ts[di]], [seg[0], seg[sp], seg[ni], seg[di]], "o", color=C.INK, ms=7, zorder=5)
    ax.annotate("foot", (0, seg[0]), xytext=(-4, 10), textcoords="offset points", fontsize=9, color=C.INK, fontweight="bold")
    ax.annotate("systolic peak", (ts[sp], seg[sp]), xytext=(8, 2), textcoords="offset points", fontsize=9, color=C.INK, fontweight="bold")
    ax.annotate("dicrotic notch (e-wave)", (ts[ni], seg[ni]), xytext=(8, 10), textcoords="offset points", fontsize=9, color=C.INK, fontweight="bold")
    ax.annotate("dicrotic peak", (ts[di], seg[di]), xytext=(8, -14), textcoords="offset points", fontsize=9, color=C.INK, fontweight="bold")
    ybot = seg[0] - .10 * a
    ax.annotate("", (ts[sp], ybot), (0, ybot), arrowprops=dict(arrowstyle="<->", color=C.ORANGE, lw=1.6))
    ax.text(ts[sp] / 2, ybot - .07 * a, "ppg_upstroke", ha="center", va="top", fontsize=9, color=C.ORANGE, fontweight="bold")
    ax.annotate("", (ts[-1], ybot), (ts[sp], ybot), arrowprops=dict(arrowstyle="<->", color=C.BLUE, lw=1.6))
    ax.text((ts[sp] + ts[-1]) / 2, ybot - .07 * a, "ppg_crest", ha="center", va="top", fontsize=9, color=C.BLUE, fontweight="bold")
    ax.annotate("", (ts[sp] - 60, seg[sp]), (ts[sp] - 60, seg[0]), arrowprops=dict(arrowstyle="<->", color=C.RED, lw=1.6))
    ax.text(ts[sp] - 66, seg[0] + a * .75, "ppg_ac_amp", fontsize=9, color=C.RED, fontweight="bold", ha="right")
    th = seg[0] + .5 * a; ab = np.where(seg >= th)[0]; ax.plot([ts[ab[0]], ts[ab[-1]]], [th, th], "-", color=C.ACC, lw=2.4)
    ax.text(ts[ab[-1]] + 8, th, "ppg_w50", fontsize=9, color=C.ACC, fontweight="bold", va="center")
    ax.fill_between(ts[ni:], seg[0], seg[ni:], color=C.GREY, alpha=.25); ax.text(ts[ni] + 110, seg[0] + .16 * a, "ppg_dia_frac\n(diastolic area share)", fontsize=8.5, color=C.MUTE)
    ax.set_ylim(ybot - .25 * a, seg[sp] + .15 * a)
    ax.set_xlabel("ms from the foot"); ax.set_ylabel("PPG, 0.5-10 Hz band"); ax.set_title("One accepted pulse: geometry, timing, notch", fontsize=10.5, color=C.INK)
    # ensemble + dct
    ax = fig.add_subplot(gs[0, 1]); norm = []
    for fi_, fj_, fd in beats:
        sg = pc[fi_:fj_]; aa = sg.max() - sg.min(); gg = np.interp(np.linspace(0, len(sg) - 1, 100), np.arange(len(sg)), sg); norm.append((gg - gg.min()) / aa)
        ax.plot(np.arange(100), norm[-1], color=C.PPGC, alpha=.18, lw=.9)
    Tm = np.median(np.asarray(norm), 0); ax.plot(np.arange(100), Tm, color=C.INK, lw=2.6, label="median beat (100 points)")
    k = np.arange(100); dct = [np.sum(Tm * np.cos(np.pi * q * (2 * k + 1) / 200.0)) * 2 / 100.0 for q in range(1, 13)]
    ax.set_xlabel("% of the beat"); ax.set_ylabel("amplitude-normalised"); ax.set_title("Median beat: ppg_dct1-5, ppg_tau, ppg_awave\n(and nf_dct6-12, see input 3)", fontsize=10.5, color=C.INK)
    ax.legend(fontsize=9); ax.text(52, .82, "dct1..5 = " + ", ".join(f"{v:+.2f}" for v in dct[:5]), fontsize=8.5, color=C.MUTE)
    # slow signal
    ax = fig.add_subplot(gs[0, 2]); slow = F.sosfiltfilt(F._SLOW, g - np.mean(g)); respb = F.sosfiltfilt(F._RESP, slow)
    ax.plot(tt, pc, color=C.PPGC, lw=.7, alpha=.7, label="pulse (0.5-10 Hz)"); ax.plot(tt, respb, color=C.RED, lw=2.2, label="respiratory baseline swing (0.1-0.5 Hz)")
    ax.set_xlabel("seconds"); ax.set_title("Slow modulation over the window: ppg_riiv, ppg_vci,\nppg_pvi, ppg_resp_rate, ppg_lf_frac, ppg_alternans, ...", fontsize=10.5, color=C.INK)
    ax.legend(fontsize=8.5, loc="upper right")
    # table of the 39
    ax = fig.add_subplot(gs[1, :]); ax.axis("off")
    rows = [("ppg_upstroke / ppg_crest / ppg_up_norm", "foot-to-peak time; peak-to-next-foot time; upstroke / inter-beat interval"),
            ("ppg_ac_amp / ppg_dc / ppg_area / ppg_amp_slope", "pulse height (median); mean level; area under the pulse; drift of pulse height over the window"),
            ("ppg_w50 / ppg_skew / ppg_kurt / ppg_rate / ppg_nbeat", "width at half height; skewness / kurtosis of the pulse band; beats per min; accepted beats"),
            ("ppg_notch_frac / ppg_notch_rel / ppg_dic_height / ppg_dia_frac", "fraction of beats with a notch; its position in the beat; dicrotic wave height; diastolic area share"),
            ("ppg_morph_f1 / ppg_morph_f4 / ppg_morph_mean", "Dawber classes: fraction with a true notch; fraction fully damped; mean class 1-4"),
            ("ppg_riiv / ppg_riiv_norm / ppg_baseline_amp / ppg_vci / ppg_resp_pow_frac", "respiratory swing of the slow baseline, raw and / pulse height; foot-level swing; venous index; resp. power share"),
            ("ppg_pvi / ppg_resp_rate / ppg_lf_frac", "pleth variability index (4-s max-min of pulse height); respiratory rate; 0.04-0.5 Hz power fraction"),
            ("ppg_alternans / ppg_alt_amp / ppg_amp_cv / ppg_ibi_cv / ppg_ibi_rmssd", "period-2 fraction and amplitude of the de-respired pulse-height series; CV of height; CV / RMSSD of intervals"),
            ("ppg_dct1..5 / ppg_tau / ppg_awave", "cosine-basis shape of the median beat; diastolic decay constant; pre-systolic hump above the late-diastolic line")]
    tb = ax.table(cellText=[[a_, b_] for a_, b_ in rows], colLabels=["features (39 in all)", "what is measured on the accepted beats"], loc="center", cellLoc="left", colWidths=[.42, .58])
    tb.auto_set_font_size(False); tb.set_fontsize(8.4); tb.scale(1, 1.32)
    for (r_, c_), cell in tb.get_celld().items():
        cell.set_edgecolor("#dfe3e8")
        if r_ == 0: cell.set_facecolor("#e3edf7"); cell.set_text_props(fontweight="bold", color=C.INK)
    fig.suptitle("Input 1: the 39 PPG features. Beats are accepted only if they correlate >= 0.85 with the window's median beat.", fontsize=13, fontweight="bold", color=C.INK)
    C.save(fig, "f05_ppg_features")

    # ---- f06 ECG features ---------------------------------------------------------------------------
    rp = F.detect_rpeaks(e); ec = F.pw_clean(e)
    idx = [k_ for k_ in rp if k_ - F.PRE >= 0 and k_ + F.POST < F.WIN]; B = np.array([ec[k_ - F.PRE:k_ + F.POST] for k_ in idx])
    med = np.median(B, 0); keep = np.array([np.corrcoef(b, med)[0, 1] for b in B]) > 0.90; B = B[keep]; avg = B.mean(0)
    rr = np.diff(rp) / FS * 1000; rmed = float(np.median(rr[(rr > 250) & (rr < 2500)])); p = F.find_p(avg, rmed)
    tms = (np.arange(len(avg)) - F.PRE) / FS * 1000
    fig = plt.figure(figsize=(17, 7.6)); gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=.42, wspace=.22)
    ax = fig.add_subplot(gs[0, 0])
    for b in B: ax.plot(tms, b, color=C.ECGC, alpha=.12, lw=.8)
    ax.plot(tms, avg, color=C.INK, lw=2.4, label=f"ensemble of {len(B)} beats, corr >= 0.90 with the median")
    ax.axvline(0, color=C.RED, lw=1, ls=":"); ax.set_xlabel("ms relative to the R peak"); ax.set_ylabel("ECG (band 0.5-40 Hz)")
    ax.set_title("Beats aligned on R: r_amp, r_amp_cv, qrs_width (15 % of peak |deflection|), t_amp", fontsize=10.5, color=C.INK); ax.legend(fontsize=9)
    ax = fig.add_subplot(gs[0, 1])
    if p is not None:
        i0, i1 = p["i0"], F.PRE + F.P_END; s_ = F.detrend_span(avg[i0:i1]); tx = (np.arange(i0, i1) - F.PRE) / FS * 1000
        ax.plot(tx, s_, color=C.INK, lw=2.4); ax.axvspan(tx[p["on"] - i0], tx[p["off"] - i0], color=C.ORANGE, alpha=.25)
        ax.plot(tx[p["k"] - i0], p["amp"], "o", color=C.RED, ms=8)
        ax.annotate(f"p_amp {1000*p['amp']:.0f} uV\np_dur {p['dur']:.0f} ms\np_pr {p['pr']:.0f} ms\np_sym {p['sym']:.2f}", (tx[p["k"] - i0], p["amp"]), xytext=(10, -10), textcoords="offset points", fontsize=9.5, color=C.INK, va="top")
        ax.set_title("The P wave in the RR-adaptive, detrended search span (-300 ms or -0.42 RR .. -80 ms)", fontsize=10.5, color=C.INK)
    ax.set_xlabel("ms relative to the R peak"); ax.set_ylabel("detrended ECG")
    ax = fig.add_subplot(gs[1, :]); ax.axis("off")
    rows = [("p_amp / p_absamp / p_dur / p_area / p_pr", "P amplitude (signed / absolute), duration at 30 % of peak, area, PR interval on the ensemble beat"),
            ("p_snr / p_present_frac / p_amp_sd / p_amp_cv", "P amplitude over beat-to-beat residual noise; fraction of beats with |P| > 2 x noise; spread of per-beat P"),
            ("p_rise / p_sym / p_notch / p_r_ratio", "onset-to-peak time; its share of the duration; bifid P; P over R amplitude"),
            ("rr_mean / rr_cv / rr_rmssd", "mean RR (ms), its coefficient of variation, RMSSD / mean (heart rate and its variability)"),
            ("af_score", "0-3: irregular RR + weak or absent P + high RR CV, a crude atrial-fibrillation flag"),
            ("qrs_width / r_amp / r_amp_cv", "QRS width at 15 % of peak |deflection|; R amplitude and its beat-to-beat CV")]
    tb = ax.table(cellText=[[a_, b_] for a_, b_ in rows], colLabels=["features (20 in all)", "what is measured"], loc="center", cellLoc="left", colWidths=[.34, .66])
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.4)
    for (r_, c_), cell in tb.get_celld().items():
        cell.set_edgecolor("#dfe3e8")
        if r_ == 0: cell.set_facecolor("#e3edf7"); cell.set_text_props(fontweight="bold", color=C.INK)
    fig.suptitle("Input 2: the 20 ECG features. The a-wave of the CVP is right-atrial contraction; the P wave is its electrical antecedent.", fontsize=13, fontweight="bold", color=C.INK)
    C.save(fig, "f06_ecg_features")

    # ---- f07 nf features ------------------------------------------------------------------------------
    d2 = np.gradient(np.gradient(Tm)); pk = int(np.argmax(Tm))
    fig = plt.figure(figsize=(17, 7.6)); gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=.45, wspace=.28)
    ax = fig.add_subplot(gs[0, 0]); ax.plot(np.arange(100), d2 / np.abs(d2).max(), color=C.ACC, lw=2.2); ax.plot(np.arange(100), Tm - .5, color=C.PPGC, lw=1.2, alpha=.7)
    ax.set_title("Median beat (green) and its 2nd derivative (purple):\nnf_sd_b..e, nf_curv, nf_asym, nf_area1-3", fontsize=10, color=C.INK); ax.set_xlabel("% of the beat")
    ax = fig.add_subplot(gs[0, 1]); feet_i = np.array([b[0] for b in beats], float)
    ax.plot(tt[:4000], (e[:4000] - e.mean()) / e.std() * .3 + 1.2, color=C.ECGC, lw=.9); ax.plot(tt[:4000], (pc[:4000]) / pc.std() * .3, color=C.PPGC, lw=.9)
    for r_ in rp[rp < 4000]:
        nxt = feet_i[feet_i > r_]
        if len(nxt) and (nxt[0] - r_) / FS < .6: ax.annotate("", (nxt[0] / FS, .0), (r_ / FS, 1.2), arrowprops=dict(arrowstyle="->", color=C.RED, lw=1.2))
    ax.set_title("R peak to PPG foot (red arrows):\nnf_pat, nf_pat_cv, nf_pat_rr, nf_slope_delay", fontsize=10, color=C.INK); ax.set_xlabel("seconds"); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 2]); amps = np.array([pc[b[0]:b[1]][b[2]["sys"] - b[0]] - pc[b[0]] for b in beats])
    ax.plot(amps[:-1], amps[1:], "o", color=C.PPGC, ms=4, alpha=.7); ax.set_xlabel("pulse height, beat i"); ax.set_ylabel("pulse height, beat i+1")
    ax.set_title("Beat-to-beat dynamics (Poincare): nf_sd1, nf_sd2, nf_sd12,\nnf_dfa, nf_ibi_sd1, nf_amp_ibi_xcorr", fontsize=10, color=C.INK)
    ax = fig.add_subplot(gs[1, :]); ax.axis("off")
    rows = [("nf_dct6..12", "higher cosine coefficients of the median beat (finer shape than ppg_dct1-5)"),
            ("nf_curv / nf_asym / nf_area1 / nf_area2 / nf_area3", "total curvature; systolic-peak position; area shares of the three thirds of the beat"),
            ("nf_sd_b / nf_sd_c / nf_sd_d / nf_sd_e", "acceleration-plethysmogram waves relative to the a-wave (stiffness / reflection indices)"),
            ("nf_pat / nf_pat_cv / nf_pat_rr / nf_slope_delay", "R-peak-to-foot pulse arrival time, its CV, its share of the RR; R to steepest PPG upslope"),
            ("nf_resp_phase / nf_resp_coh / nf_mod_ratio", "phase and coherence between pulse-height and baseline modulation at the respiratory peak; their amplitude ratio"),
            ("nf_sd1 / nf_sd2 / nf_sd12 / nf_ibi_sd1 / nf_amp_ibi_xcorr / nf_dfa", "Poincare short/long-term variability of pulse height, of intervals, their correlation; detrended fluctuation exponent"),
            ("nf_g_ratio / nf_g_sep / nf_g_width", "two-Gaussian decomposition of the median beat: reflected / forward height, separation, width ratio")]
    tb = ax.table(cellText=[[a_, b_] for a_, b_ in rows], colLabels=["features (32 in all)", "what is measured"], loc="center", cellLoc="left", colWidths=[.34, .66])
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.4)
    for (r_, c_), cell in tb.get_celld().items():
        cell.set_edgecolor("#dfe3e8")
        if r_ == 0: cell.set_facecolor("#e3edf7"); cell.set_text_props(fontweight="bold", color=C.INK)
    fig.suptitle("Input 3: the 32 'nf' features: finer pulse shape, pulse arrival vs the ECG, respiratory coupling, beat-to-beat dynamics", fontsize=13, fontweight="bold", color=C.INK)
    C.save(fig, "f07_nf_features")

    # ---- f08 differencing ---------------------------------------------------------------------------------
    NAMES = feature_names(V.names)
    fig, AX = plt.subplots(2, 3, figsize=(17, 6.4))
    sdemo = max(ss, key=lambda s: s["dev"].max())
    for j, nm in enumerate(["ppg_w50", "ppg_dct2", "rr_mean", "p_pr", "nf_sd_b", "ppg_riiv_norm"]):
        ax = AX.flat[j]; i = V.names.index(nm); o_ = V.ORDER[c]; tt_ = V.T[o_]; v = V.X[o_][:, i]
        blk = (tt_ >= sdemo["tb"][0]) & (tt_ <= sdemo["tb"][-1]); ref = np.nanmedian(v[blk])
        ax.axvspan(sdemo["tb"][0], sdemo["tb"][-1], color=C.BLUE, alpha=.14, lw=0)
        m_ = (tt_ >= sdemo["tb"][0] - 1) & (tt_ <= sdemo["t"][-1] + 1)
        ax.plot(tt_[m_], v[m_], "-o", color=C.PPGC if nm.startswith("ppg") or nm.startswith("nf") else C.ECGC, ms=3.5, lw=1.2)
        ax.axhline(ref, color=C.INK, lw=1.6, ls="--"); ax.set_title(f"{nm}  ->  d_{nm} = value - block median ({ref:.3g})", fontsize=10, color=C.INK)
        ax2 = ax.twinx(); ax2.plot(sdemo["t"], sdemo["y"], "-", color=C.CVPC, lw=1.0, alpha=.6); ax2.set_ylabel("CVP", color=C.CVPC, fontsize=8); ax2.tick_params(labelsize=7, colors=C.CVPC); ax2.grid(False)
        ax.set_xlabel("minutes")
    fig.suptitle(f"Every waveform feature enters as a CHANGE since this patient's own calibration block (case {c}, one cycle; red = CVP for orientation)", fontsize=12.5, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .94]); C.save(fig, "f08_differencing")

    # ---- f09 block features ---------------------------------------------------------------------------------
    sdemo = ss[min(2, len(ss) - 1)]; u_, yd_ = V.DEN[c]; offc = np.median(V.Y[V.ORDER[c]][np.isfinite(V.Y[V.ORDER[c]])]) - np.median(yd_)
    g2 = (u_ >= sdemo["tb"][0]) & (u_ <= sdemo["tb"][-1]); ub, yb_ = u_[g2], yd_[g2] + offc; dd = desc(ub, yb_)
    fig = plt.figure(figsize=(17, 6.2)); gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0]); ax.axvspan(ub[0], ub[-1], color=C.BLUE, alpha=.14, lw=0)
    ax.plot(ub, yb_, "o", color=C.CVPC, ms=4, label="30-s CVP tiles inside the read")
    t0_ = ub - ub[-1]; ax.plot(ub, np.polyval(np.polyfit(t0_, yb_, 1), t0_), "-", color=C.INK, lw=2.2, label=f"blk_slope {dd[0]:+.2f} mmHg/min")
    ax.plot(ub, np.polyval(np.polyfit(t0_, yb_, 2), t0_), "--", color=C.ACC, lw=1.8, label=f"blk_curv {dd[1]:+.3f}")
    h = len(yb_) // 2
    ax.plot(ub[:h], np.polyval(np.polyfit(t0_[:h], yb_[:h], 1), t0_[:h]), "-", color=C.ORANGE, lw=2.4, label=f"blk_slope_1st_half {dd[2]:+.2f}")
    ax.plot(ub[h:], np.polyval(np.polyfit(t0_[h:], yb_[h:], 1), t0_[h:]), "-", color=C.RED, lw=2.4, label=f"blk_slope_2nd_half {dd[3]:+.2f}  (change {dd[4]:+.2f})")
    ax.axhline(dd[10], color=C.GREY, lw=2.2, label=f"blk_median {dd[10]:.1f} = the baseline")
    ax.plot(ub[-1], dd[11], "s", color=C.INK, ms=9, label=f"blk_last {dd[11]:.1f}  (last - median {dd[5]:+.2f})")
    ax.plot(sdemo["t"], sdemo["y"], "-o", color=C.CVPC, lw=1.1, ms=3, alpha=.7, label="tracked stretch (not seen)")
    ax.set_xlabel("minutes"); ax.set_ylabel("CVP (mmHg)"); ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.set_title(f"The calibration read's own CVP: 13 shape descriptors (blk_std {dd[7]:.2f}, blk_ptp {dd[8]:.1f}, mean |step| {dd[9]:.2f}, n {int(dd[12])})", fontsize=12, color=C.INK)
    fig.suptitle("What the CVP line itself tells the model at calibration. All of it is known before tracking starts.", fontsize=12.5, fontweight="bold", color=C.INK)
    C.save(fig, "f09_block_plot")
    fig, ax = plt.subplots(figsize=(17, 6.4)); ax.axis("off")
    rows = [("since_last_cvp", "minutes since the last CVP reading of the block (the clock)"),
            ("blk_slope, blk_curv, blk_slope_1st/2nd_half, blk_slope_change", "trend, curvature and turning of the block's CVP"),
            ("blk_last_minus_median, blk_mean_minus_median", "where the block ended relative to its baseline"),
            ("blk_std, blk_ptp, blk_mean_abs_step, blk_median, blk_last, blk_n", "spread, level, endpoint, support of the block"),
            ("hist_delta_level, hist_rate, hist_gap_min, hist_n_blocks, hist_prev_slope", "this block vs the previous one: level change, rate, gap, count, previous slope"),
            ("gap_to_first_window, blk_n_dense, blk_n_gated, blk_std_gated", "gap to the first tracked window; dense tiles / gated windows / gated spread of the block"),
            *([("slope_x_since, abs_slope_x_since, std_x_since, rate_x_since", "the block trend / spread / history rate extrapolated over elapsed time")] if "std_x_since" not in DROP_COLS else []),
            ("trk_pred, trk_abs_pred, trk_cummean, trk_abs_cummean, trk_cummax_abs", "the auxiliary tracker's output and its causal summaries (VitalDB only; zero on MIMIC)")]
    rows = [(a_.replace(", ", ",  ").replace("hist_delta_level,  hist_rate,  hist_gap_min,  hist_n_blocks,  hist_prev_slope", "hist_delta_level,  hist_rate,  hist_gap_min,\nhist_n_blocks,  hist_prev_slope")
              .replace("trk_pred,  trk_abs_pred,  trk_cummean,  trk_abs_cummean,  trk_cummax_abs", "trk_pred,  trk_abs_pred,  trk_cummean,\ntrk_abs_cummean,  trk_cummax_abs")
              .replace("blk_std,  blk_ptp,  blk_mean_abs_step,  blk_median,  blk_last,  blk_n", "blk_std,  blk_ptp,  blk_mean_abs_step,\nblk_median,  blk_last,  blk_n"),
             b_.replace("dense tiles", "30-s CVP tiles").replace("gap to the first tracked window; 30-s CVP tiles / gated windows / gated spread of the block", "gap to the first tracked window; number of 30-s tiles,\nnumber of gated windows and gated spread of the read")) for a_, b_ in rows]
    tb = ax.table(cellText=[[a_, b_] for a_, b_ in rows], colLabels=[f"the {len(ACTIVE_COLS) - 91} non-waveform columns", "meaning"], loc="center", cellLoc="left", colWidths=[.46, .54])
    tb.auto_set_font_size(False); tb.set_fontsize(10.5); tb.scale(1, 2.9)
    for (r_, c_), cell in tb.get_celld().items():
        cell.set_edgecolor("#dfe3e8")
        if r_ == 0: cell.set_facecolor("#fdf1e3"); cell.set_text_props(fontweight="bold", color=C.INK)
    ax.set_title(f"The {len(ACTIVE_COLS) - 91} non-waveform columns", fontsize=13, fontweight="bold", color=C.INK, pad=8)
    C.save(fig, "f09_block_table")

    # ---- f10 target, dense baseline, artefact filter -----------------------------------------------------
    # find a patient with a sustained-step artefact that the filter drops
    ex = None
    for cc in sorted(V.ORDER):
        if cc not in V.DEN: continue
        uu, yy = V.DEN[cc]; oo = V.ORDER[cc]; tt_ = V.T[oo]
        cyc = W_CAL + S_TRACK; kk = np.floor((tt_ - tt_[0]) / cyc).astype(int); inb = ((tt_ - tt_[0]) - kk * cyc) < W_CAL
        for k in np.unique(kk):
            bm = (kk == k) & inb; sm = (kk == k) & ~inb & np.isfinite(V.Y[oo])
            if bm.sum() < 3 or sm.sum() < 8: continue
            ws = worst_step(uu, yy, float(tt_[bm][0]), float(tt_[sm][-1]))
            if 3.5 < ws < 12: ex = (cc, k, float(tt_[bm][0]), float(tt_[sm][-1]), ws); break
        if ex: break
    fig = plt.figure(figsize=(17, 5.8)); gs = fig.add_gridspec(1, 3, wspace=.25)
    ax = fig.add_subplot(gs[0])
    cand = [s for s in ss if (s["dev"] > 3).any() and 4 <= int(np.argmax(s["dev"] > 3)) <= len(s["dev"]) - 3]
    s = max(cand or ss, key=lambda s: s["dev"].max())
    ax.axvspan(s["tb"][0], s["tb"][-1], color=C.BLUE, alpha=.14, lw=0); ax.plot(s["tb"], s["yb"], "s", color=C.CVPC, ms=4, mec="#fff")
    g2 = (u >= s["tb"][0]) & (u <= s["tb"][-1]); offp = np.median(V.Y[o][np.isfinite(V.Y[o])]) - np.median(yd)
    ax.plot(u[g2], yd[g2] + offp, ".", color="#c9a5a0", ms=5, label=f"30-s tiles ({g2.sum()}) vs gated ({len(s['tb'])})")
    ax.axhline(s["base"], color=C.GREY, lw=2.4, label="baseline = median of the 30-s tiles"); ax.fill_between([s["t"][0], s["t"][-1]], s["base"] - 3, s["base"] + 3, color=C.GREY, alpha=.13)
    ax.plot(s["t"], s["y"], "-o", color=C.CVPC, ms=4, lw=1.3); bad = s["dev"] > 3; ax.plot(s["t"][bad], s["y"][bad], "o", color=C.RED, ms=7)
    ax.set_title("Baseline: all 30-s CVP tiles give ~3x the support\nof the gated windows", fontsize=10.5, color=C.INK); ax.legend(fontsize=8.5); ax.set_xlabel("minutes"); ax.set_ylabel("CVP (mmHg)")
    ax = fig.add_subplot(gs[1]); ax.plot(s["t"], s["dev"], "-o", color=C.MUTE, ms=3.5, lw=1, label="|CVP - baseline| at the window")
    ax.step(s["t"], target(s), where="post", color=C.INK, lw=2.4, label="TARGET: max so far (what is regressed)")
    ax.axhline(3, color=C.RED, ls=":", lw=1.4); ax.set_xlabel("minutes"); ax.set_ylabel("mmHg"); ax.legend(fontsize=8.5)
    ax.set_title("Target: cumulative, so the label is\n'has it deviated yet', not 'is it deviated now'", fontsize=10.5, color=C.INK)
    ax = fig.add_subplot(gs[2])
    if ex:
        cc, k, lo_, hi_, ws = ex; uu, yy = V.DEN[cc]; m_ = (uu >= lo_ - 2) & (uu <= hi_ + 2)
        ax.plot(uu[m_], yy[m_], ".", color=C.CVPC, ms=4); mm = (uu >= lo_) & (uu <= hi_)
        med = np.array([np.median(yy[(uu >= a_ - 2.5) & (uu <= a_ + 2.5)]) for a_ in uu[mm]]); ax.plot(uu[mm], med, "-", color=C.INK, lw=2.2, label="rolling 5-min median")
        ax.axvspan(lo_, lo_ + W_CAL, color=C.BLUE, alpha=.14, lw=0)
        ax.set_title(f"Artefact filter, case {cc}: worst 5-min median step\n{ws:.1f} mmHg > {MAX_STEP:g} -> stretch EXCLUDED", fontsize=10.5, color=C.INK); ax.legend(fontsize=8.5)
    ax.set_xlabel("minutes"); ax.set_ylabel("dense CVP (mmHg)")
    fig.suptitle("The label side: baseline from all 30-s CVP tiles, cumulative target, and a schedule-agnostic line-artefact filter", fontsize=12.5, fontweight="bold", color=C.INK)
    C.save(fig, "f10_target")

    # ---- f11 protocol -----------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(17, 5.6)); ax.set_xlim(0, 17); ax.set_ylim(0, 5.6); ax.axis("off")
    n = len(R); ax.text(8.5, 5.25, f"{n} VitalDB patients with at least one clean stretch, split into 5 outer folds BY PATIENT (seed 2024)", ha="center", fontsize=11.5, fontweight="bold", color=C.INK)
    for f_ in range(5):
        y0 = 4.3 - f_ * .62
        for j in range(5):
            ax.add_patch(Rectangle((1.2 + j * 1.55, y0), 1.45, .5, fc=C.ORANGE if j == f_ else "#dbe6f2", ec="white"))
        ax.text(0.9, y0 + .25, f"fold {f_+1}", ha="right", va="center", fontsize=10)
        # inner
        for j in range(3):
            ax.add_patch(Rectangle((9.4 + j * .95, y0 + .05), .9, .4, fc="#cfe3d5" if j != f_ % 3 else "#8cc9a3", ec="white"))
    ax.text(4.9, 1.05, "orange = held out and scored once; blue = training", ha="center", fontsize=9.5, color=C.MUTE)
    ax.text(10.8, 4.9, "3 inner folds choose W", ha="center", va="center", fontsize=9.5, color=C.INK)
    box(ax, 12.6, 3.3, 4.1, 1.5, "CONTROLS\nclock only: the same fit with time-since-calibration as the only input\npermutation: stretch targets shuffled among stretches\n4 outer-fold seeds", "#fdf1e3", fs=9)
    box(ax, 12.6, 1.55, 4.1, 1.5, "UNCERTAINTY\n600 patient-clustered bootstrap resamples of the pooled AUC;\npaired resamples for every A-vs-B difference", "#e3edf7", fs=9)
    box(ax, 12.6, 0.1, 4.1, 1.2, "EXTERNAL\nMIMIC: one fit on all of VitalDB, scored once on 417 records\nrecord-clustered bootstrap (400)", "#e6f3ec", fs=9)
    box(ax, 1.2, 0.1, 10.8, .8, "Scored at |drift| > 2, 3 and 4 mmHg. Every number in this deck is out-of-fold or out-of-cohort unless it says 'in-sample'.", "#fbe9e7", fs=10, tc="#8c2d1e")
    C.save(fig, "f11_protocol")

    # ---- f12 the two controls, on three real stretches ---------------------------------------------------
    from sklearn.ensemble import HistGradientBoostingRegressor
    z = np.load(f"{RES}/vitaldb_oof.npz", allow_pickle=True); OOF = {(int(a), int(k)): np.asarray(p, float) for a, k, p in zip(z["cid"], z["k"], z["p"])}
    A = np.vstack([s["X"][:, [91]] for cc in R for s in R[cc]]); tg = np.concatenate([target(s) for cc in R for s in R[cc]])
    clk = HistGradientBoostingRegressor(random_state=0, **HGB).fit(A, tg)
    allseg = [s for cc in sorted(R) for s in R[cc]]
    dev_c = [s for s in allseg if (target(s) > 3).any() and 4 <= int(np.argmax(target(s) > 3)) <= 9 and len(s["tb"]) >= 6 and OOF[(s["key"], s["k"])].max() > 2 and s["yb"].max() - s["yb"].min() < 4][:1]
    quiet = [s for s in allseg if not (target(s) > 1.5).any() and len(s["tb"]) >= 6 and OOF[(s["key"], s["k"])].max() < 1.2 and s["key"] != dev_c[0]["key"]]
    cand = dev_c + [quiet[0]] + [q for q in quiet[1:] if q["key"] != quiet[0]["key"]][:1]
    fig = plt.figure(figsize=(17, 6.4)); gs = fig.add_gridspec(2, 4, height_ratios=[1.6, 1], hspace=.35, wspace=.3)
    for j, s in enumerate(cand):
        ax = fig.add_subplot(gs[0, j]); axp = fig.add_subplot(gs[1, j], sharex=ax)
        ax.axvspan(s["tb"][0], s["tb"][-1], color=C.BLUE, alpha=.14, lw=0); ax.plot(s["tb"], s["yb"], "s", color=C.CVPC, ms=3.5, mec="#fff")
        ax.axhline(s["base"], color=C.GREY, lw=2); ax.fill_between([s["t"][0], s["t"][-1]], s["base"] - 3, s["base"] + 3, color=C.GREY, alpha=.13)
        ax.plot(s["t"], s["y"], "-o", color=C.CVPC, ms=3.5, lw=1.2); bad = s["dev"] > 3
        if bad.any(): ax.plot(s["t"][bad], s["y"][bad], "o", color=C.RED, ms=6)
        ax.set_title(f"case {s['key']}: {'deviates' if bad.any() else 'stays inside the band'}", fontsize=10.5, color=C.INK); ax.set_ylabel("CVP (mmHg)")
        axp.plot(s["t"], OOF[(s["key"], s["k"])], "-", color=C.GREEN, lw=2.2, label="model (out of fold)")
        axp.plot(s["t"], np.maximum.accumulate(clk.predict(s["X"][:, [91]])), "--", color=C.GREY, lw=2, label="clock only")
        axp.set_ylim(0, 3.4); axp.set_xlabel("minutes"); axp.set_ylabel("score (mmHg)")
        if j == 0: axp.legend(fontsize=8.5, loc="upper left")
    ax = fig.add_subplot(gs[:, 3]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    box(ax, 0.02, 0.55, 0.96, 0.42, "CLOCK CONTROL\nthe same regressor with one input:\nminutes since the read.\nIts score is the same curve on every\nstretch (dashed).\nAUC 0.650 / 0.653 / 0.674", "#eef1f4", fs=9.5)
    box(ax, 0.02, 0.03, 0.96, 0.46, "PERMUTATION CONTROL\nevery feature and window stays where\nit is; each stretch's target series is\nhanded to a different stretch before\nfitting. Same folds, same pipeline.\nAUC 0.545 / 0.509 / 0.490 = chance", "#fbe9e7", fs=9.5, tc="#8c2d1e")
    fig.suptitle("The two controls: what a clock-only score looks like, and what the permutation breaks", fontsize=12.5, fontweight="bold", color=C.INK)
    C.save(fig, "f12_controls")


if PART in ("A", "all"): part_a()


# ===================================================================================================
def _ci_bar(ax, x, v, ci, color, w=.26, label=None):
    ax.bar(x, v, w, color=color, label=label)
    ax.errorbar(x, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor=C.INK, elinewidth=1.3, capsize=3)


def part_b():
    J = json.load(open(f"{RES}/investigation.json")); TR = json.load(open(f"{RES}/transfer.json")); MA = json.load(open(f"{RES}/mimic_analysis.json"))
    K = [f"n>{t:g}" for t in THRS]
    # ---- f13 headline -----------------------------------------------------------------------------------
    full = json.load(open(f"{RES}/nested_full_W10_S15_seed2024.json")); clock = json.load(open(f"{RES}/nested_clock_W10_S15_seed2024.json"))
    perm = json.load(open(f"{RES}/nested_perm_W10_S15_seed2024.json"))
    seeds = {s: json.load(open(f"{RES}/nested_full_W10_S15_seed{s}.json")) for s in (2024, 7, 99, 1234)}
    fig, AX = plt.subplots(1, 2, figsize=(17, 5.6), width_ratios=[1.5, 1])
    ax = AX[0]
    for j, k in enumerate(K):
        for i, (nm, d, col) in enumerate((("full model", full, C.NAVY), ("clock only", clock, C.GREY), ("permuted targets", perm, "#d7dbe0"))):
            r = d["results"][k]; _ci_bar(ax, j + (i - 1) * .28, r["auc"], r["ci"], col, label=nm if j == 0 else None)
            ax.text(j + (i - 1) * .28, r["ci"][1] + .012, f"{r['auc']:.3f}", ha="center", fontsize=9.5, fontweight="bold" if i == 0 else "normal")
    ax.axhline(.75, color=C.RED, lw=1.4, ls=":"); ax.text(2.42, .755, "target 0.75", color=C.RED, fontsize=9, ha="right")
    ax.axhline(.5, color=C.INK, lw=1, ls="--"); ax.set_ylim(.4, .9); ax.set_xticks(range(3)); ax.set_xticklabels([f"|drift| {k[1:]} mmHg  (prev {100*full['results'][k]['prevalence']:.0f}%)" for k in K])
    ax.set_ylabel("nested-CV AUC (pooled, 600 patient-bootstrap CI)"); ax.legend(loc="upper left", fontsize=10, ncol=3)
    ax.set_title(f"VitalDB, {full['meta']['patients']} patients / {full['meta']['stretches']} stretches / {full['meta']['windows']:,} windows, W=10 / S=15 fixed", fontsize=11.5, color=C.INK)
    ax = AX[1]
    for j, k in enumerate(K):
        v = [seeds[s]["results"][k]["auc"] for s in seeds]
        ax.plot([j] * 4, v, "o", color=C.NAVY, ms=8, alpha=.7); ax.plot([j - .2, j + .2], [np.mean(v)] * 2, "-", color=C.RED, lw=2.5)
        ax.text(j + .25, np.mean(v), f"mean {np.mean(v):.3f}", va="center", fontsize=9.5, color=C.RED)
    ax.set_xticks(range(3)); ax.set_xticklabels(K); ax.set_ylim(.7, .85); ax.axhline(.75, color=C.RED, lw=1.4, ls=":")
    ax.set_title("Four outer-fold assignments (seeds 2024 / 7 / 99 / 1234)", fontsize=11.5, color=C.INK); ax.set_ylabel("nested-CV AUC")
    fig.suptitle("The VitalDB result: above 0.75 at every threshold, +0.13 over a clock, permutation at chance", fontsize=13.5, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .94]); C.save(fig, "f13_headline")

    # ---- f14 schedules --------------------------------------------------------------------------------------
    S = J["schedules"]; order = sorted(S, key=lambda k: (float(k.split("/S")[1]), float(k[1:].split("/")[0])))
    fig, AX = plt.subplots(1, 2, figsize=(17, 5.2), width_ratios=[1.6, 1])
    ax = AX[0]
    for j, k in enumerate(K):
        for i, sc in enumerate(order):
            r = S[sc][k]; x = i + (j - 1) * .27
            col = [C.GREEN, C.BLUE, C.ORANGE][j]
            _ci_bar(ax, x, r["auc"], r["ci"], col, label=k if i == 0 else None)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([f"{sc}\n{S[sc]['meta']['patients']} pts, {S[sc]['meta']['windows']:,} win" for sc in order], fontsize=9.5)
    ax.set_ylim(.6, .92); ax.axhline(.75, color=C.RED, lw=1.4, ls=":"); ax.legend(fontsize=10, ncol=3, loc="upper left"); ax.set_ylabel("nested-CV AUC")
    ax.set_title("Calibration W / tracking S (minutes): the chosen 10/15 is not the best cell at every threshold", fontsize=11, color=C.INK)
    ax = AX[1]; ax.axis("off")
    rows = [[sc, f"{S[sc]['meta']['patients']}", f"{S[sc]['meta']['stretches']}", f"{S[sc]['meta']['windows']:,}"] + [f"{100*S[sc][k]['prevalence']:.1f}%" for k in K] for sc in order]
    tb = ax.table(cellText=rows, colLabels=["W/S", "patients", "stretches", "windows", "prev >2", "prev >3", "prev >4"], loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1, 1.6)
    for (r_, c_), cell in tb.get_celld().items():
        cell.set_edgecolor("#dfe3e8")
        if r_ == 0: cell.set_facecolor("#e3edf7"); cell.set_text_props(fontweight="bold", color=C.INK)
    ax.set_title("Patients / stretches kept per cell (a stretch needs 8 gated windows)", fontsize=11, color=C.INK)
    fig.suptitle("Schedule sweep: every cell is a full five-fold cross-validation by patient; read for its direction, not its argmax", fontsize=13, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .93]); C.save(fig, "f14_schedules")

    # ---- f16 parity ------------------------------------------------------------------------------------------
    V = np.load(f"{DATA}/vitaldb_windows.npz", allow_pickle=True); names = [str(n) for n in V["names"]]; XV = V["X"].astype(float)
    from lib.build import mimic_files
    files = [p for p in mimic_files("paper") if "mimic3wdb" in p]; rng = np.random.default_rng(0)
    XM = np.vstack([np.load(f)["X"][rng.choice(len(np.load(f)["X"]), min(300, len(np.load(f)["X"])), replace=False)] for f in files]).astype(float)
    scl = np.array([np.load(f)["scale"] for f in files])
    shift = []
    for i, n in enumerate(names):
        a = XV[:, i][np.isfinite(XV[:, i])]; b = XM[:, i][np.isfinite(XM[:, i])]
        sig = (np.percentile(a, 75) - np.percentile(a, 25)) / 1.349 + 1e-9
        shift.append((np.median(b) - np.median(a)) / sig)
    shift = np.array(shift); o = np.argsort(-np.abs(shift))
    fig = plt.figure(figsize=(17, 7.2)); gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=.5, wspace=.3)
    ax = fig.add_subplot(gs[0, :]); ax.bar(range(91), shift[o], color=[C.RED if abs(v) > 2 else C.BLUE for v in shift[o]])
    ax.set_xticks(range(91)); ax.set_xticklabels([names[i] for i in o], rotation=90, fontsize=6.5); ax.axhline(0, color=C.INK, lw=1)
    ax.axhline(2, color=C.RED, lw=1, ls=":"); ax.axhline(-2, color=C.RED, lw=1, ls=":"); ax.set_ylabel("(MIMIC median - VitalDB median)\n/ VitalDB robust SD")
    ax.set_title(f"After per-record amplitude normalisation: {np.mean(np.abs(shift) < 2)*100:.0f}% of the 91 features sit within 2 robust SD; median |shift| {np.median(np.abs(shift)):.2f}", fontsize=11, color=C.INK)
    for j, n in enumerate(["ppg_dc", "ppg_w50", "rr_mean", "ppg_lf_frac"]):
        ax = fig.add_subplot(gs[1, j]); i = names.index(n)
        a = XV[:, i][np.isfinite(XV[:, i])]; b = XM[:, i][np.isfinite(XM[:, i])]
        lo, hi = np.percentile(np.concatenate([a, b]), [1, 99]); bins = np.linspace(lo, hi, 40)
        ax.hist(a, bins=bins, density=True, alpha=.55, color=C.BLUE, label="VitalDB"); ax.hist(b, bins=bins, density=True, alpha=.55, color=C.ORANGE, label="MIMIC-III")
        ax.set_title(f"{n}  (shift {shift[i]:+.1f} SD)", fontsize=10.5, color=C.INK); ax.set_yticks([])
        if j == 0: ax.legend(fontsize=9)
    fig.text(.5, .005, f"Raw PPG IQR across MIMIC records spans {np.nanpercentile(scl[:,1],5):.2f} to {np.nanpercentile(scl[:,1],95):.2f} a.u. (VitalDB {SCALE_REF['ppg_iqr']:.1f}): the pleth has no units, so each record's PPG and ECG are mapped affinely onto VitalDB's median / IQR before extraction. "
             "Remaining shifts (ppg_lf_frac, ppg_vci) are slow variability, which should differ between a ventilated OR patient and an awake ICU one.", ha="center", va="bottom", fontsize=9, color=C.MUTE, wrap=True)
    C.save(fig, "f16_parity")

    # ---- f17 transfer -----------------------------------------------------------------------------------------
    A = TR["arms"]; fig, ax = plt.subplots(figsize=(17, 5.8))
    arms = [("A zero-shot VitalDB->MIMIC-III", "A  zero-shot\nVitalDB -> MIMIC-III", C.NAVY), ("A zero-shot VitalDB->MIMIC-IV", "A  zero-shot\nVitalDB -> MIMIC-IV", "#4d6b8f"),
            ("B MIMIC-only ceiling (5-fold)", "B  MIMIC-only ceiling\n(5-fold by record)", C.ORANGE), ("C pooled VitalDB+MIMIC -> MIMIC", "C  pooled VitalDB\n+ half MIMIC -> other half", C.GREEN),
            ("  (MIMIC-half only, same split)", "   half MIMIC alone,\nsame split", "#7fc59b")]
    for j, k in enumerate(K):
        for i, (key, lab, col) in enumerate(arms):
            if key not in A: continue
            r = A[key][k]; x = j * 6 + i; _ci_bar(ax, x, r["auc"], r["ci"], col, w=.8, label=lab if j == 0 else None)
            ax.text(x, r["ci"][1] + .008, f"{r['auc']:.3f}", ha="center", fontsize=8.8, fontweight="bold" if i == 0 else "normal")
    ax.set_xticks([j * 6 + 2 for j in range(3)]); ax.set_xticklabels([f"|drift| {k[1:]} mmHg   (MIMIC prev {100*A['A zero-shot VitalDB->MIMIC-III'][k]['prevalence']:.1f}%)" for k in K], fontsize=10.5)
    ax.set_ylim(.55, .92); ax.axhline(.75, color=C.RED, lw=1.4, ls=":"); ax.legend(fontsize=9.5, ncol=5, loc="upper right"); ax.set_ylabel("AUC (record-clustered 95% CI)")
    c = TR["counts"]
    ax.set_title(f"MIMIC: {c['mimic3']} MIMIC-III + {c['mimic4']} MIMIC-IV records, {c['mimic_stretches']:,} stretches, {c['mimic_windows']:,} windows; VitalDB training set here: {c['vitaldb_patients']} patients, {c['vitaldb_windows']:,} windows",
                 fontsize=11, color=C.INK)
    fig.suptitle("External validation: the VitalDB model reaches 97-98% of what training on MIMIC itself achieves; pooling adds nothing", fontsize=13, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .94]); C.save(fig, "f17_transfer")

    # ---- f21 operating point -------------------------------------------------------------------------------------
    T = MA["thresholds"]; fig, AX = plt.subplots(1, 2, figsize=(17, 5.6), width_ratios=[1, 1.15])
    ax = AX[0]
    for nm, col, mk in (("global_p", C.GREY, "s"), ("own_past_p", C.NAVY, "o")):
        ks = [k for k in T if k.startswith(nm)]; ax.plot([1 - T[k]["spec"] for k in ks], [T[k]["sens"] for k in ks], "-" + mk, color=col, ms=9, lw=2,
                                                        label="one GLOBAL cut (percentile of all scores)" if nm == "global_p" else "PER-RECORD cut: percentile of the record's own past score (causal)")
        for k in ks: ax.annotate(k.split("_p")[-1] + "th", (1 - T[k]["spec"], T[k]["sens"]), xytext=(6, -10 if nm == "global_p" else 6), textcoords="offset points", fontsize=8.5, color=col)
    ax.plot(1 - T["own_youden"]["spec"], T["own_youden"]["sens"], "^", color=C.ORANGE, ms=11, label="Youden on the record's own past LABELS")
    for k in ("stable_mean_plus_1.0sd", "stable_mean_plus_1.5sd", "stable_mean_plus_2.0sd"):
        ax.plot(1 - T[k]["spec"], T[k]["sens"], "D", color=C.GREEN, ms=8, label="mean + k SD of own STABLE windows" if k.endswith("1.0sd") else None)
    ax.plot([0, 1], [0, 1], "--", color=C.INK, lw=1); ax.set_xlim(0, .5); ax.set_ylim(.2, .9); ax.set_xlabel("1 - specificity (per-record mean)"); ax.set_ylabel("sensitivity (per-record mean)")
    ax.legend(fontsize=8.8, loc="lower right"); ax.set_title("Operating points, n>3, averaged over MIMIC-III records", fontsize=11, color=C.INK)
    ax = AX[1]
    ks = [k for k in T if k.startswith("global_p")] + [k for k in T if k.startswith("own_past_p")]
    ax.bar(range(len(ks)), [100 * T[k]["records_spec_below_50"] for k in ks], color=[C.GREY] * 5 + [C.NAVY] * 5)
    ax.set_xticks(range(len(ks))); ax.set_xticklabels([("global " if k.startswith("global") else "own ") + k.split("_p")[-1] + "th" for k in ks], rotation=30, fontsize=9)
    ax.set_ylabel("% of records that alarm on >half of their STABLE windows"); ax.set_title("A per-record threshold removes the runaway records", fontsize=11, color=C.INK)
    for i, k in enumerate(ks): ax.text(i, 100 * T[k]["records_spec_below_50"] + .4, f"{100*T[k]['records_spec_below_50']:.1f}%", ha="center", fontsize=8.5)
    st = MA["structure"]["n>3"]
    fig.suptitle(f"From AUC to an alarm: the score needs a per-record threshold; the running max latches ({100*st['flagged_time_recovered']:.0f}% of flagged time at n>3 is after CVP has already recovered)", fontsize=12.5, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .93]); C.save(fig, "f21_operating")

    # ---- f22 baselines ------------------------------------------------------------------------------------------
    B = J["baselines"]; BM = J["mimic"]["baselines"]
    fig, AX = plt.subplots(1, 2, figsize=(17, 6.6), width_ratios=[1, 1])
    for ax, (D, fullr, ttl) in zip(AX, ((B, J["full"], f"VitalDB, out of fold ({J['vitaldb']['windows']:,} windows)"), (BM, J["mimic"]["full"], f"MIMIC-III, zero-shot ({J['mimic']['windows']:,} windows)"))):
        names_ = [f"FULL MODEL ({len(ACTIVE_COLS)} columns)"] + [n.replace("32 cols", f"{len(ACTIVE_COLS)-91} cols") for n in D]; vals = [fullr] + [D[n] for n in D]
        y = np.arange(len(names_))[::-1]
        for j, k in enumerate(K):
            ax.barh(y + (1 - j) * .27, [v[k]["auc"] for v in vals], .27, color=[C.GREEN, C.BLUE, C.ORANGE][j], label=k)
        ax.set_yticks(y); ax.set_yticklabels(names_, fontsize=9.5); ax.set_xlim(.45, .9); ax.axvline(.5, color=C.INK, lw=1, ls="--"); ax.axvline(.75, color=C.RED, lw=1.2, ls=":")
        ax.set_xlabel("AUC"); ax.set_title(ttl, fontsize=11.5, color=C.INK); ax.legend(fontsize=9, loc="lower right")
        for i, v in enumerate(vals): ax.text(v["n>2"]["auc"] + .004, y[i] + .27, f"{v['n>2']['auc']:.3f}", va="center", fontsize=8.5)
    fig.suptitle("What does a score with NO model get? The clock, the calibration block's own trend / spread, and the CVP level, each alone", fontsize=13, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .94]); C.save(fig, "f22_baselines")

    # ---- f23 ablations -----------------------------------------------------------------------------------------------
    AB = J["ablations"]; ABM = J["mimic"]["ablations"]
    fig, AX = plt.subplots(1, 2, figsize=(17, 6.8))
    for ax, (D, ttl) in zip(AX, ((AB, "VitalDB, out of fold, paired patient bootstrap"), (ABM, "MIMIC-III, zero-shot, paired record bootstrap"))):
        names_ = [n for n in D if not n.startswith("full")]; y = np.arange(len(names_))[::-1]
        for j, k in enumerate(K):
            d = [D[n]["vs_full"][k]["delta"] for n in names_]; lo = [D[n]["vs_full"][k]["ci"][0] for n in names_]; hi = [D[n]["vs_full"][k]["ci"][1] for n in names_]
            ax.errorbar(d, y + (1 - j) * .27, xerr=[np.array(d) - np.array(lo), np.array(hi) - np.array(d)], fmt="o", color=[C.GREEN, C.BLUE, C.ORANGE][j], ms=6, capsize=2.5, label=k)
        ax.axvline(0, color=C.INK, lw=1.4); ax.set_yticks(y); ax.set_yticklabels([f"{n}  [{D[n]['ncols']}]" for n in names_], fontsize=9.5); ax.set_xlabel("AUC change vs the full model (95% CI)")
        ax.set_title(ttl, fontsize=11.5, color=C.INK); ax.legend(fontsize=9, loc="lower left"); ax.set_xlim(-.2, .06)
        full_ = D[[n for n in D if n.startswith("full")][0]]
        ax.text(.055, y[0] + .6, "full: " + "  ".join(f"{k} {full_[k]['auc']:.3f}" for k in K), ha="right", fontsize=9, color=C.INK, fontweight="bold")
    fig.suptitle("Ablations: remove a family, refit, score on the same windows. [n] = columns left", fontsize=13, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .94]); C.save(fig, "f23_ablations")

    # ---- f24 construction ---------------------------------------------------------------------------------------------
    CO = J["construction"]; fig, ax = plt.subplots(figsize=(17, 5.2))
    names_ = [n for n in CO if "vs_full" in CO[n]]; y = np.arange(len(names_))[::-1]
    for j, k in enumerate(K):
        d = [CO[n]["vs_full"][k]["delta"] for n in names_]; lo = [CO[n]["vs_full"][k]["ci"][0] for n in names_]; hi = [CO[n]["vs_full"][k]["ci"][1] for n in names_]
        ax.errorbar(d, y + (1 - j) * .27, xerr=[np.array(d) - np.array(lo), np.array(hi) - np.array(d)], fmt="o", color=[C.GREEN, C.BLUE, C.ORANGE][j], ms=6, capsize=2.5, label=k)
    ax.axvline(0, color=C.INK, lw=1.4); ax.set_yticks(y); ax.set_yticklabels(names_, fontsize=10); ax.set_xlabel("AUC change vs the full construction (95% CI, paired)"); ax.legend(fontsize=9.5)
    g = CO["baseline from the gated block only (no dense grid)"]
    ax.set_title("Same features, different construction of target / score / loss / learner   |   gated-block baseline on its own stretch set: " +
                 "  ".join(f"{k} {g[k]['auc']:.3f}" for k in K) + f"  ({g['stretches']} stretches)", fontsize=10.5, color=C.INK)
    fig.suptitle("Construction choices: the cumulative target and the running max matter; regression vs classification and the loss barely do", fontsize=12.5, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .93]); C.save(fig, "f24_construction")

    # ---- f25 importance groups ------------------------------------------------------------------------------------------
    IG = J["importance_groups"]; IGM = J["mimic"]["importance_groups"]
    fig, AX = plt.subplots(1, 2, figsize=(17, 7.6))
    for ax, (D, ttl) in zip(AX, ((IG, "VitalDB, out of fold"), (IGM, "MIMIC-III, zero-shot"))):
        names_ = sorted(D, key=lambda n: -D[n]["n>2"]); y = np.arange(len(names_))[::-1]
        for j, k in enumerate(K):
            ax.barh(y + (1 - j) * .27, [D[n][k] for n in names_], .27, color=[C.GREEN, C.BLUE, C.ORANGE][j], label=k)
        ax.set_yticks(y); ax.set_yticklabels(names_, fontsize=8.6); ax.axvline(0, color=C.INK, lw=1); ax.set_xlabel("AUC drop when the family is permuted"); ax.set_title(ttl, fontsize=11.5, color=C.INK); ax.legend(fontsize=9)
    fig.suptitle("Permutation importance by feature family (families overlap: the first nine partition the columns, the rest split them finer)", fontsize=12.5, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .95]); C.save(fig, "f25_importance")

    # ---- f26 single features ---------------------------------------------------------------------------------------------
    IS = J["importance_single"]; ISM = J["mimic"]["importance_single"]; U = J["univariate"]
    fig, AX = plt.subplots(1, 3, figsize=(17, 6.8))
    for ax, (D, ttl) in zip(AX[:2], ((IS, "VitalDB, out of fold"), (ISM, "MIMIC-III, zero-shot"))):
        top = sorted(D, key=lambda n: -D[n]["n>2"])[:22]; y = np.arange(len(top))[::-1]
        ax.barh(y + .18, [D[n]["n>2"] for n in top], .36, color=C.GREEN, label="n>2"); ax.barh(y - .18, [D[n]["n>3"] for n in top], .36, color=C.BLUE, label="n>3")
        ax.set_yticks(y); ax.set_yticklabels(top, fontsize=9); ax.set_xlabel("AUC drop when permuted"); ax.set_title(f"top 22 single columns, {ttl}", fontsize=11, color=C.INK); ax.legend(fontsize=9)
    ax = AX[2]; top = sorted(U, key=lambda n: -U[n]["auc_n2"])[:22]; y = np.arange(len(top))[::-1]
    ax.barh(y, [U[n]["auc_n2"] - .5 for n in top], .6, color=C.ORANGE); ax.set_yticks(y); ax.set_yticklabels(top, fontsize=9); ax.set_xlabel("univariate AUC - 0.5 (n>2, running max of the column)")
    ax.set_title("Each column alone, VitalDB", fontsize=11, color=C.INK)
    fig.suptitle("Which columns carry the signal? Permutation importance of single columns, and each column's own discrimination", fontsize=12.5, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .94]); C.save(fig, "f26_single")

    # ---- f27 confounders ----------------------------------------------------------------------------------------------------
    CF = J["confounders"]; CFM = J["mimic"]["confounders"]
    fig, AX = plt.subplots(2, 4, figsize=(17, 8.8))
    panels = [(CF, "|dHR| (bpm, vs block median)", "VitalDB: |dHR| since calibration (bpm)"), (CF, "|dMAP| (mmHg, vs block median)", "VitalDB: |dMAP| since calibration (mmHg)"),
              (CF, "baseline CVP level (block median, mmHg)", "VitalDB: baseline CVP level (mmHg)"), (CF, "minutes since last CVP reading", "VitalDB: minutes since calibration"),
              (CFM, "|d rr_mean| (ms, vs block median)", "MIMIC: |dRR| since calibration (ms)"), (CFM, "|dMAP| (mmHg, vs first tracked window; art-line records)", "MIMIC: |dMAP| within the stretch (mmHg)"),
              (CFM, "baseline CVP level (block median, mmHg)", "MIMIC: baseline CVP level (mmHg)"), (CFM, "minutes since last CVP reading", "MIMIC: minutes since calibration")]
    for ax, (D, key, ttl) in zip(AX.flat, panels):
        d = D[key]; rows = d["model_within_strata"]; rc = d["clock_within_strata"]
        x = np.arange(len(rows)); lab = [r["stratum"].split(" ", 1)[1] for r in rows]
        for j, k in enumerate(K[:2]):
            ax.plot(x, [r[k] if r[k] is not None else np.nan for r in rows], "-o", color=[C.GREEN, C.BLUE][j], lw=2.2, ms=7, label=f"model {k}")
        ax.plot(x, [r["n>2"] if r["n>2"] is not None else np.nan for r in rc], "--s", color=C.GREY, lw=1.6, ms=5, label="clock n>2")
        ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8, rotation=20); ax.set_ylim(.45, .95); ax.axhline(.5, color=C.INK, lw=.8, ls=":")
        ax2 = ax.twinx(); ax2.bar(x, [r["n"] for r in rows], color="#e6e9ec", zorder=0, width=.6); ax2.set_yticks([]); ax2.grid(False); ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        cov = d.get("coverage", 1.0); asc = d["auc_as_score"]["n>2"] if d.get("auc_as_score") else np.nan
        ax.set_title(f"{ttl}\nalone {asc:.3f} | rho {d['spearman_with_score']:+.2f} | cov {100*cov:.0f}%", fontsize=9.3, color=C.INK)
        if ax is AX.flat[0]: ax.legend(fontsize=8, loc="lower left")
    AX[0, 0].set_ylabel("AUC inside the stratum"); AX[1, 0].set_ylabel("AUC inside the stratum")
    fig.suptitle("Confounders: does the model still rank deviations when heart rate, arterial pressure, CVP level or elapsed time are held fixed? (grey bars = windows)", fontsize=12.5, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .95]); C.save(fig, "f27_confounders")

    # ---- f28 / f29 forest plots ------------------------------------------------------------------------------------------------
    def forest(D, name, ttl, key_mean, key_n, ref):
        rows = []
        for grp, rs in D.items():
            rows.append((grp, None)); rows += [(r["stratum"], r) for r in rs]
        fig, ax = plt.subplots(figsize=(17, max(5.5, 0.26 * len(rows) + 1.6))); y = np.arange(len(rows))[::-1]
        for yy, (lab, r) in zip(y, rows):
            if r is None: ax.text(.30, yy, lab, fontsize=9.5, fontweight="bold", color=C.NAVY, va="center"); continue
            ax.plot([r["ci"][0], r["ci"][1]], [yy, yy], "-", color=C.INK, lw=1.4); ax.plot(r[key_mean], yy, "o", color=C.GREEN if r[key_mean] >= ref else C.RED, ms=6)
            ax.text(.31, yy, f"   {lab}", fontsize=8.6, va="center"); ax.text(1.02, yy, f"n={r[key_n]}   pooled {r['pooled']['n>2'] if r['pooled']['n>2'] is not None else float('nan'):.3f}   prev {100*r['prevalence_n2']:.0f}%", fontsize=8, va="center", ha="left", color=C.MUTE)
        ax.axvline(ref, color=C.GREY, lw=1.4, ls="--"); ax.set_xlim(.3, 1.25); ax.set_yticks([]); ax.set_ylim(-1, len(rows)); ax.set_xlabel("mean per-patient / per-record AUC at n>2 (bootstrap 95% CI); dashed = overall mean")
        ax.set_xticks([.3, .4, .5, .6, .7, .8, .9, 1.0])
        ax.set_title(ttl, fontsize=12, color=C.INK, fontweight="bold"); ax.grid(False)
        C.save(fig, name)
    def halves(D):
        ks = list(D); h = (len(ks) + 1) // 2
        return {k: D[k] for k in ks[:h]}, {k: D[k] for k in ks[h:]}
    ttl = f"VitalDB, out of fold: who is it good or bad for?  (overall per-patient mean {CF['per_patient_auc_n2']['mean']:.3f}, n={CF['per_patient_auc_n2']['n']})"
    forest(J["demographics_vitaldb"], "f28_demographics", ttl, "per_patient_mean", "patients", CF["per_patient_auc_n2"]["mean"])
    for tag, D in zip("ab", halves(J["demographics_vitaldb"])):
        forest(D, f"f28_demographics_{tag}", ttl + ("  [1/2]" if tag == "a" else "  [2/2]"), "per_patient_mean", "patients", CF["per_patient_auc_n2"]["mean"])
    prm = MA["per_record"]["n>2"]["mean"]
    ttl = f"MIMIC-III, zero-shot: record strata  (overall per-record mean {prm:.3f}, n={MA['per_record']['n>2']['evaluable']})"
    forest(J["mimic"]["strata"], "f29_mimic_strata", ttl, "per_record_mean", "records", prm)
    for tag, D in zip("ab", halves(J["mimic"]["strata"])):
        forest(D, f"f29_mimic_strata_{tag}", ttl + ("  [1/2]" if tag == "a" else "  [2/2]"), "per_record_mean", "records", prm)

    # ---- f30 correlates ----------------------------------------------------------------------------------------------------------
    fig, AX = plt.subplots(1, 2, figsize=(17, 4.8))
    for ax, (D, ttl) in zip(AX, ((J["per_patient_correlates"], "VitalDB: per-patient AUC (n>2) vs patient attributes"), (J["mimic"]["per_record_correlates"], "MIMIC-III: per-record AUC (n>2) vs record attributes"))):
        ks = sorted(D, key=lambda k: D[k]["spearman"]); v = [D[k]["spearman"] for k in ks]
        ax.barh(range(len(ks)), v, color=[C.RED if x < 0 else C.GREEN for x in v]); ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks, fontsize=9.5); ax.axvline(0, color=C.INK, lw=1)
        ax.set_xlabel("Spearman correlation with per-patient AUC"); ax.set_title(ttl, fontsize=11, color=C.INK); ax.set_xlim(-.5, .5)
    fig.suptitle("What predicts a bad patient? Rank correlations of the per-patient AUC with what we know about the patient and the record", fontsize=12.5, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .92]); C.save(fig, "f30_correlates")

    # ---- f31 limits ------------------------------------------------------------------------------------------------------------------
    cap = MA["cap"]; fig, AX = plt.subplots(1, 3, figsize=(17, 5.2))
    ax = AX[0]; q = [10, 25, 50, 75, 90, 95, 99]
    ax.plot(q, [cap["target_pct"][str(x)] for x in q], "-o", color=C.INK, lw=2.4, ms=7, label=f"training target (max {cap['target_max']:.1f})")
    ax.plot([50, 99], [cap["insample_pct"]["50"], cap["insample_pct"]["99"]], "s", color=C.RED, ms=10, label=f"in-sample prediction (max {cap['insample_max']:.1f})")
    ax.set_xlabel("percentile"); ax.set_ylabel("mmHg"); ax.legend(fontsize=9.5); ax.set_title("The regressor cannot reach its own large targets:\nleaf medians shrink the tail", fontsize=10.5, color=C.INK)
    ax = AX[1]; ls = cap["losses"]; names_ = list(ls)
    ax.bar(np.arange(3) - .2, [ls[n]["p99"] for n in names_], .4, color=C.GREY, label="99th pct of the MIMIC score"); ax.bar(np.arange(3) + .2, [ls[n]["slope"] * 10 for n in names_], .4, color=C.BLUE, label="slope vs truth x 10")
    ax2 = ax.twinx(); ax2.plot(np.arange(3), [ls[n]["auc2"] for n in names_], "o-", color=C.GREEN, ms=9, lw=2, label="AUC n>2"); ax2.set_ylim(.8, .86); ax2.set_ylabel("AUC n>2", color=C.GREEN); ax2.grid(False)
    ax.set_xticks(range(3)); ax.set_xticklabels(names_); ax.legend(fontsize=8.5, loc="upper left"); ax.set_title("Loss changes the spread, not the ordering", fontsize=10.5, color=C.INK)
    ax = AX[2]; st = MA["structure"]; dr = MA["direction"]
    xs = np.arange(3); ax.bar(xs - .3, [100 * st[k]["deviate_then_return"] for k in K], .28, color=C.ORANGE, label="events that return inside the band")
    ax.bar(xs, [100 * st[k]["flagged_time_recovered"] for k in K], .28, color=C.RED, label="flagged time already recovered")
    ax.bar(xs + .3, [100 * st[k]["both_directions"] for k in K], .28, color=C.BLUE, label="stretches with both up and down")
    ax.set_xticks(xs); ax.set_xticklabels(K); ax.set_ylabel("% (MIMIC-III stretches with an event)"); ax.legend(fontsize=8.5, loc="upper left"); ax.set_ylim(0, 100)
    ax.set_title(f"What a running max costs; direction at stretch level: AUC {dr['stretch_level_n3']['auc']:.2f}, accuracy {100*dr['stretch_level_n3']['acc']:.0f}% vs {100*dr['stretch_level_n3']['up_rate']:.0f}% rises", fontsize=10, color=C.INK)
    fig.suptitle("Limits: magnitude saturates near 2 mmHg, the score latches, and direction is only weakly recovered", fontsize=12.5, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .93]); C.save(fig, "f31_limits")

    # ---- f32 label quality (reported system only) ---------------------------------------------------------------------------------
    LQ = json.load(open(f"{RES}/label_quality.json"))["mimic_run>=5"] if os.path.exists(f"{RES}/label_quality.json") else None; arms = list(LQ["arms"]) if LQ else []
    fig, AX = plt.subplots(1, 3, figsize=(17, 5.6))
    for ax, k in (zip(AX, K) if LQ else []):
        y = np.arange(len(arms))[::-1]
        ax.barh(y + .2, [LQ["arms"][a][k]["pooled_all"] for a in arms], .38, color="#c9d2dc", label="all windows")
        ax.barh(y - .2, [LQ["arms"][a][k]["pooled_without_flat"] for a in arms], .38, color=C.NAVY, label=f"flat plateaus removed ({100*LQ['flat_fraction']:.0f}% of windows)")
        for i, a in enumerate(arms):
            ax.text(LQ["arms"][a][k]["pooled_all"] + .004, y[i] + .2, f"{LQ['arms'][a][k]['pooled_all']:.3f}", va="center", fontsize=8.5, color=C.MUTE)
            ax.text(LQ["arms"][a][k]["pooled_without_flat"] + .004, y[i] - .2, f"{LQ['arms'][a][k]['pooled_without_flat']:.3f}", va="center", fontsize=8.5, fontweight="bold")
        ax.set_yticks(y); ax.set_yticklabels(arms if ax is AX[0] else [""] * len(arms), fontsize=9.5); ax.set_xlim(.5, .9); ax.axvline(.75, color=C.RED, lw=1.2, ls=":")
        ax.set_xlabel("pooled AUC, MIMIC-III"); ax.set_title(f"|drift| {k[1:]} mmHg   (prev {100*LQ['arms'][arms[0]][k]['prevalence_all']:.1f}% -> {100*LQ['arms'][arms[0]][k]['prevalence_without_flat']:.1f}%)", fontsize=11, color=C.INK)
        if ax is AX[0]: ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(1.6, -0.12), ncol=2, frameon=False)
    if LQ:
        z = LQ["arms"]["zero-shot full model"]
        fig.suptitle(f"A quarter of MIMIC windows are a flat, clamped transducer. Removing them costs every arm; per-record zero-shot {z['n>2']['per_record_all']:.3f} -> {z['n>2']['per_record_without_flat']:.3f} at n>2",
                     fontsize=12.5, fontweight="bold", color=C.INK)
        fig.tight_layout(rect=[0, 0, 1, .93]); C.save(fig, "f32_label_quality")
    else: plt.close(fig)

    # ---- f33 the untouched extra MIMIC records ------------------------------------------------------------------------------------------
    EX = json.load(open(f"{RES}/mimic_extra.json"))
    fig, AX = plt.subplots(1, 2, figsize=(17, 4.8))
    for ax, (key, lab) in zip(AX, (("auc", "pooled AUC (record-clustered 95% CI)"), ("per_record_mean", "mean per-record AUC"))):
        for j, k in enumerate(K):
            for i, (sub, col) in enumerate((("paper", C.NAVY), ("extra", C.ORANGE))):
                r = EX[sub][k]; x = j + (i - .5) * .36
                if key == "auc": _ci_bar(ax, x, r["auc"], r["ci"], col, w=.34, label=f"{sub}: {EX[sub]['records']} records, {EX[sub]['windows']:,} windows" if j == 0 else None)
                else: ax.bar(x, r[key], .34, color=col, label=f"{sub}: {r['evaluable']} evaluable records" if j == 0 else None)
                ax.text(x, (r["ci"][1] if key == "auc" else r[key]) + .006, f"{r[key]:.3f}", ha="center", fontsize=9)
        ax.set_xticks(range(3)); ax.set_xticklabels(K); ax.set_ylim(.6, .9); ax.axhline(.75, color=C.RED, lw=1.2, ls=":"); ax.set_ylabel(lab); ax.legend(fontsize=9, loc="upper right")
    fig.suptitle("The other usable MIMIC-III records, never looked at until the end: the same model, the same numbers", fontsize=12.5, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .92]); C.save(fig, "f33_mimic_extra")


if PART in ("B", "all"): part_b()


# ===================================================================================================
def feature_tables():
    """Standalone, full-width feature tables (f05t / f06t / f07t) so every feature is legible on a slide."""
    from lib.features import PPG_FEATS, ECG_FEATS, NF_FEATS
    PPG = [("ppg_upstroke", "foot-to-systolic-peak time (s), median over accepted beats"), ("ppg_crest", "systolic-peak-to-next-foot time (s)"),
           ("ppg_up_norm", "upstroke time / mean inter-beat interval (heart-rate normalised)"), ("ppg_ac_amp", "pulse height, foot to peak (a.u.), median"),
           ("ppg_dc", "mean level of the raw PPG over the window"), ("ppg_area", "area under the pulse above its foot (a.u. x s)"),
           ("ppg_amp_slope", "linear drift of pulse height across the window"), ("ppg_w50", "pulse width at half height (s)"),
           ("ppg_skew / ppg_kurt", "skewness / kurtosis of the 0.5-10 Hz pulse band"), ("ppg_rate / ppg_nbeat", "accepted beats per minute / accepted beats in the window"),
           ("ppg_notch_frac", "fraction of accepted beats with a dicrotic notch (e-wave of the acceleration PPG)"), ("ppg_notch_rel", "notch position as a fraction of the beat"),
           ("ppg_dic_height", "dicrotic wave height / pulse height"), ("ppg_dia_frac", "area after the notch / area of the whole pulse"),
           ("ppg_morph_f1 / f4 / mean", "Dawber classes: fraction with a true notch / fraction fully damped / mean class 1-4"),
           ("ppg_riiv", "respiratory swing of the slow (0.04-10 Hz) baseline, 95th-5th percentile of its 0.1-0.5 Hz band"),
           ("ppg_riiv_norm", "ppg_riiv / pulse height"), ("ppg_baseline_amp", "95th-5th percentile of the slow baseline sampled at the beat feet"),
           ("ppg_vci", "ppg_baseline_amp / pulse height (venous index)"), ("ppg_resp_pow_frac", "0.1-0.6 Hz power fraction of the foot-level series"),
           ("ppg_pvi", "pleth variability index: median over 4-s windows of 100 (max-min)/(max+min) of pulse height"),
           ("ppg_resp_rate", "peak of the 0.1-0.5 Hz spectrum, in breaths per minute"), ("ppg_lf_frac", "0.04-0.5 Hz power / total power of the raw PPG"),
           ("ppg_alternans", "period-2 power fraction of the de-respired pulse-height series"), ("ppg_alt_amp", "period-2 amplitude / pulse height"),
           ("ppg_amp_cv", "coefficient of variation of pulse height"), ("ppg_ibi_cv / ppg_ibi_rmssd", "CV / RMSSD of the foot-to-foot intervals"),
           ("ppg_dct1 .. ppg_dct5", "cosine (DCT-II) coefficients 1-5 of the amplitude- and duration-normalised median beat"),
           ("ppg_tau", "exponential decay constant fitted to the last 45% of the median beat (diastolic run-off)"),
           ("ppg_awave", "largest positive residual above the linear late-diastolic trend in the last quarter of the beat")]
    ECG = [("p_amp / p_absamp", "P-wave amplitude on the ensemble beat, signed / absolute (mV)"), ("p_dur", "P duration at 30% of its peak (ms)"),
           ("p_area", "area of the P hump (mV x ms)"), ("p_pr", "P onset to R peak (ms)"), ("p_snr", "P amplitude / beat-to-beat residual SD in the P span"),
           ("p_amp_cv / p_amp_sd", "spread of the per-beat P amplitude at the ensemble's P position"), ("p_rise / p_sym", "P onset-to-peak time (ms) / its share of the duration"),
           ("p_notch", "1 if the P hump has more than one local maximum (bifid P)"), ("p_r_ratio", "|P| / |R| amplitude"),
           ("p_present_frac", "fraction of beats whose per-beat P exceeds 2 x the residual noise"), ("rr_mean", "mean RR interval (ms): heart rate"),
           ("rr_cv / rr_rmssd", "RR coefficient of variation / RMSSD divided by the mean"), ("af_score", "0-3: RMSSD > 0.15 + weak or absent P + RR CV > 0.12"),
           ("qrs_width", "width at 15% of the peak |deflection| around R (ms)"), ("r_amp / r_amp_cv", "R amplitude on the ensemble beat and its per-beat CV")]
    NF = [("nf_dct6 .. nf_dct12", "cosine coefficients 6-12 of the median beat"), ("nf_curv", "sum of squared second differences of the median beat"),
          ("nf_asym", "position of the systolic peak as a fraction of the beat"), ("nf_area1 / 2 / 3", "area share of the first / second / last third of the median beat"),
          ("nf_sd_b / c / d / e", "acceleration-plethysmogram b, c, d, e waves relative to the a wave"),
          ("nf_pat / nf_pat_cv / nf_pat_rr", "R-peak-to-PPG-foot pulse arrival time (s), its CV, its share of the RR"), ("nf_slope_delay", "R peak to the steepest PPG upslope (s)"),
          ("nf_resp_phase / nf_resp_coh", "phase and coherence between pulse-height and baseline modulation at the respiratory peak"),
          ("nf_mod_ratio", "baseline swing / pulse-height swing (95th-5th percentiles)"), ("nf_sd1 / nf_sd2 / nf_sd12", "Poincare short- / long-term variability of pulse height and their ratio"),
          ("nf_ibi_sd1", "Poincare short-term variability of the intervals"), ("nf_amp_ibi_xcorr", "correlation between pulse height and the preceding interval"),
          ("nf_dfa", "detrended-fluctuation exponent of the pulse-height series"), ("nf_g_ratio / nf_g_sep / nf_g_width", "two-Gaussian fit of the median beat: reflected / forward height, separation, width ratio")]
    for name, rows, ttl, n in (("f05t_ppg_table", PPG, "Input 1: the 39 PPG features, one line each", 39), ("f06t_ecg_table", ECG, "Input 2: the 20 ECG features, one line each", 20),
                               ("f07t_nf_table", NF, "Input 3: the 32 'nf' features, one line each", 32)):
        fig, ax = plt.subplots(figsize=(17, 0.36 * len(rows) + 1.2)); ax.axis("off")
        tb = ax.table(cellText=[[a, b] for a, b in rows], colLabels=["feature(s)", "what is measured"], loc="center", cellLoc="left", colWidths=[.27, .73])
        tb.auto_set_font_size(False); tb.set_fontsize(11); tb.scale(1, 1.55)
        for (r_, c_), cell in tb.get_celld().items():
            cell.set_edgecolor("#dfe3e8")
            if r_ == 0: cell.set_facecolor("#e3edf7"); cell.set_text_props(fontweight="bold", color=C.INK)
            elif r_ % 2 == 0: cell.set_facecolor("#f6f8fa")
        ax.set_title(ttl, fontsize=13, fontweight="bold", color=C.INK, pad=8)
        C.save(fig, name)


if PART in ("A", "all", "T"): feature_tables()


# ===================================================================================================
def feature_reduction_figs():
    """f34: AUC against the number of columns; f35: the minimal sets, listed."""
    FR = json.load(open(f"{RES}/feature_reduction.json")); K = [f"n>{t:g}" for t in THRS]
    st = FR["stages"]; ns = [d["n"] for d in st]
    fig, AX = plt.subplots(1, 2, figsize=(17, 5.6))
    for ax, (coh, ttl) in zip(AX, (("vitaldb", "VitalDB, out of fold (same five patient folds)"), ("mimic", "MIMIC-III, zero-shot"))):
        for j, k in enumerate(K):
            v = [d[coh][k]["auc"] for d in st]; lo = [d[coh][k]["ci"][0] for d in st]; hi = [d[coh][k]["ci"][1] for d in st]
            ax.fill_between(ns, lo, hi, color=[C.GREEN, C.BLUE, C.ORANGE][j], alpha=.12, lw=0)
            ax.plot(ns, v, "-o", color=[C.GREEN, C.BLUE, C.ORANGE][j], lw=2.2, ms=5, label=f"{k}  (118 cols: {v[0]:.3f})")
        for nm, d in FR["hand_picked"].items():
            for j, k in enumerate(K):
                ax.plot(d["n"], d[coh][k]["auc"], "*", color=[C.GREEN, C.BLUE, C.ORANGE][j], ms=14, mec=C.INK, mew=.8, zorder=5)
            ax.annotate(nm, (d["n"], d[coh]["n>2"]["auc"]), xytext=(8, 10), textcoords="offset points", fontsize=8.5, color=C.INK)
        ax.set_xscale("log"); ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 118]); ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 118])
        ax.axhline(.75, color=C.RED, lw=1.2, ls=":"); ax.set_xlabel("columns kept (recursive elimination by out-of-fold permutation importance)"); ax.set_ylabel("AUC")
        ax.set_title(ttl, fontsize=11.5, color=C.INK); ax.legend(fontsize=9, loc="lower right"); ax.set_ylim(.55, .9)
    fig.suptitle("How few columns are enough? Refit at every step, scored out of fold and zero-shot; stars = hand-picked sets", fontsize=13, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .93]); C.save(fig, "f34_feature_reduction")
    # f35: the sets at 16 / 8 / 4 / 2 / 1 and the hand-picked ones
    def wrap(cols, per=9):
        return "\n".join("  ".join(cols[i:i + per]) for i in range(0, len(cols), per))
    rows = []
    for n in (16, 12, 8, 6, 4, 3, 2, 1):
        d = [x for x in st if x["n"] == n]
        if d: d = d[0]; rows.append([f"RFE {n}", wrap(d["columns"]), " / ".join(f"{d['vitaldb'][k]['auc']:.3f}" for k in K), " / ".join(f"{d['mimic'][k]['auc']:.3f}" for k in K), f"{d['mimic_per_record_n2']:.3f}"])
    short = {"5 CVP-line columns": "5 line", "5 CVP-line + 5 waveform": "5 line + 5 wave", "5 CVP-line + 10 waveform": "5 line + 10 wave"}
    for nm, d in FR["hand_picked"].items():
        rows.append([short.get(nm, nm), wrap(d["columns"]), " / ".join(f"{d['vitaldb'][k]['auc']:.3f}" for k in K), " / ".join(f"{d['mimic'][k]['auc']:.3f}" for k in K), f"{d['mimic_per_record_n2']:.3f}"])
    full = st[0]; rows.insert(0, [f"all {full['n']}", "(every non-tracker column)", " / ".join(f"{full['vitaldb'][k]['auc']:.3f}" for k in K), " / ".join(f"{full['mimic'][k]['auc']:.3f}" for k in K), f"{full['mimic_per_record_n2']:.3f}"])
    fig, ax = plt.subplots(figsize=(17, 0.55 * len(rows) + 1.0)); ax.axis("off"); ax.set_position([0.01, 0.01, 0.98, 0.9])
    tb = ax.table(cellText=rows, colLabels=["set", "columns", "VitalDB OOF  n>2 / 3 / 4", "MIMIC 0-shot  n>2 / 3 / 4", "per-rec n>2"], loc="center", cellLoc="left", colWidths=[.10, .54, .14, .14, .08])
    tb.auto_set_font_size(False); tb.set_fontsize(8.6); tb.scale(1, 2.3)
    for (r_, c_), cell in tb.get_celld().items():
        cell.set_edgecolor("#dfe3e8")
        if r_ == 0: cell.set_facecolor("#e3edf7"); cell.set_text_props(fontweight="bold", color=C.INK)
        elif c_ == 1: cell.set_text_props(fontsize=7.6)
    ax.set_title("The reduced sets: which columns survive, and what they score", fontsize=13, fontweight="bold", color=C.INK, pad=8)
    C.save(fig, "f35_feature_sets")


if PART in ("B", "all", "R"): feature_reduction_figs()


# ===================================================================================================
def longer_figs():
    """f36: AUC against columns for each tracking length; f37: the eight-column sets per length."""
    L = {}
    for f in sorted(glob.glob(f"{RES}/longer_S*.json"), key=lambda p: float(p.split("_S")[1][:-5])):
        j = json.load(open(f)); L[j["meta"]["S"]] = j
    K = [f"n>{t:g}" for t in THRS]; cols = {15: C.NAVY, 20: C.BLUE, 30: C.GREEN, 45: C.ORANGE, 60: C.RED}
    fig, AX = plt.subplots(2, 2, figsize=(17, 9.4))
    for r, k in enumerate(("n>2", "n>3")):
        for c, (coh, ttl) in enumerate((("vitaldb", "VitalDB, out of fold"), ("mimic", "MIMIC-III, zero-shot"))):
            ax = AX[r, c]
            for S, j in L.items():
                ns = [d["n"] for d in j["stages"]]; v = [d[coh][k]["auc"] for d in j["stages"]]
                m = j["meta"]; lab = f"S={S:g} min  ({m['vitaldb_stretches'] if coh == 'vitaldb' else m['mimic_stretches']:,} stretches, prev {100*m[coh + '_prevalence'][k]:.0f}%)"
                ax.plot(ns, v, "-o", color=cols.get(S, C.GREY), lw=2.2, ms=5, label=lab)
            ax.set_xscale("log"); ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 118]); ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 118])
            ax.axhline(.75, color=C.RED, lw=1.2, ls=":"); ax.set_ylim(.6, .9); ax.set_ylabel(f"AUC, {k}"); ax.legend(fontsize=8.5, loc="lower right")
            ax.set_title(f"{ttl}, {k}", fontsize=11.5, color=C.INK)
            if r == 1: ax.set_xlabel("columns kept (recursive elimination)")
    fig.suptitle("Longer tracking after the 10-min read: the same elimination at S = 15, 20, 30, 45, 60 min (zero-shot model trained on all VitalDB rows here)", fontsize=13, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .95]); C.save(fig, "f36_longer_curves")
    rows = []; ONE = json.load(open(f"{RES}/one_feature_horizons.json"))
    for S, j in L.items():
        m = j["meta"]
        n0 = j["stages"][0]["n"]
        for n in (n0, 16, 8, 4, 1):
            d = [x for x in j["stages"] if x["n"] == n][0]
            rows.append([f"S={S:g}", f"{n}", ("(all shared columns)" if n == n0 else "\n".join("  ".join(d["columns"][i:i + 9]) for i in range(0, len(d["columns"]), 9))),
                         " / ".join(f"{d['vitaldb'][kk]['auc']:.3f}" for kk in K), " / ".join(f"{d['mimic'][kk]['auc']:.3f}" for kk in K), f"{d['mimic_per_record_n2']:.3f}"])
        o = ONE[f"S{S:g}"]
        rows.append([f"S={S:g}", "1, no model", "std_x_since used directly as the score (block CVP spread x minutes since the read)",
                     " / ".join(f"{o['vitaldb'][kk]['auc']:.3f}" for kk in K), " / ".join(f"{o['mimic'][kk]['auc']:.3f}" for kk in K), ""])
    fig, ax = plt.subplots(figsize=(17, 0.44 * len(rows) + 1.0)); ax.axis("off"); ax.set_position([0.01, 0.01, 0.98, 0.9])
    tb = ax.table(cellText=rows, colLabels=["stretch", "cols", "columns kept", "VitalDB OOF  n>2 / 3 / 4", "MIMIC 0-shot  n>2 / 3 / 4", "per-rec n>2"], loc="center", cellLoc="left", colWidths=[.07, .05, .52, .14, .14, .08])
    tb.auto_set_font_size(False); tb.set_fontsize(8.4); tb.scale(1, 1.9)
    for (r_, c_), cell in tb.get_celld().items():
        cell.set_edgecolor("#dfe3e8")
        if r_ == 0: cell.set_facecolor("#e3edf7"); cell.set_text_props(fontweight="bold", color=C.INK)
        elif c_ == 2: cell.set_text_props(fontsize=7.2)
        if r_ > 0 and rows[r_ - 1][2] == "(all shared columns)": cell.set_facecolor("#f6f8fa")
    ax.set_title("What survives at 16 / 8 / 4 / 1 columns for each tracking length, and the one-column no-model score", fontsize=13, fontweight="bold", color=C.INK, pad=8)
    C.save(fig, "f37_longer_sets")


if PART in ("B", "all", "L"): longer_figs()


# ===================================================================================================
def gate_fig():
    """f38: the CVP-waveform quality measures on MIMIC, three example windows, and every arm before / after the gate."""
    import glob as _g
    from s19_mimic_cvp_gate import GATE
    from lib.build import mimic_files
    Q = np.load(f"{RES_BASE}/mimic_cvp_quality.npz", allow_pickle=True); paper = {os.path.basename(p).replace(".npz", "") for p in mimic_files("paper")}
    keys = sorted({k.rsplit("__", 1)[0] for k in Q.files} & paper)
    sd = np.concatenate([Q[f"{k}__std"] for k in keys]); run = np.concatenate([Q[f"{k}__flat_run_s"] for k in keys]); pul = np.concatenate([Q[f"{k}__puls"] for k in keys])
    MG = json.load(open(f"{RES}/mimic_gated.json")) if os.path.exists(f"{RES}/mimic_gated.json") else None
    fig = plt.figure(figsize=(17, 6.6)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=.6, wspace=.28)
    for j, (v, lab, lo, hi, bins) in enumerate(((sd, "standard deviation of the 30-s CVP waveform (mmHg)", GATE["std_min"], GATE["std_max"], np.linspace(0, 14, 71)),
                                                (run, "longest run of identical samples (s)", None, GATE["flat_run_max_s"], np.linspace(0, 30, 61)),
                                                (pul, "pulsatility: median 1-s range (mmHg)", GATE["puls_min"], None, np.linspace(0, 20, 81)))):
        ax = fig.add_subplot(gs[0, j]); ax.hist(np.clip(v[np.isfinite(v)], bins[0], bins[-1]), bins=bins, color=C.BLUE, edgecolor="white")
        for x_ in (lo, hi):
            if x_ is not None: ax.axvline(x_, color=C.RED, lw=2, ls="--")
        ax.set_xlabel(lab); ax.set_yticks([]); ax.set_title(f"{100*np.mean(v < 0.05):.0f}% of windows at zero" if j != 1 else f"{100*np.mean(run >= 29.9):.0f}% of windows flat for all 30 s", fontsize=10.5, color=C.INK)
    # three example windows from the raw records: flat, flushed, pulsatile
    from config import MIMIC_RECORDS
    ex = []
    for k in keys:
        s_, r_, p_ = Q[f"{k}__std"], Q[f"{k}__flat_run_s"], Q[f"{k}__puls"]; t_ = Q[f"{k}__t"]
        for cond, nm in (((r_ >= 29.9), "flat: identical samples for 30 s"), ((s_ > 12) & (p_ > 0.5), "flush / step: std > 12 mmHg"), ((s_ > 1.5) & (s_ < 4) & (p_ > 3) & (r_ < 0.5), "pulsatile: passes the gate")):
            if any(e[0] == nm for e in ex): continue
            w = np.where(cond)[0]
            if len(w): ex.append((nm, k, float(t_[w[len(w) // 2]])))
        if len(ex) == 3: break
    for j, (nm, k, tt) in enumerate(ex):
        src, rid = k.split("_", 1); z = np.load(f"{MIMIC_RECORDS}/{src}/{rid}.npz", allow_pickle=True); a = int(round(tt * 2)) * 3750
        c = z["sig_cvp"][a:a + 3750].astype(float); ax = fig.add_subplot(gs[1, j]); ax.plot(np.arange(3750) / 125.0, c, color=C.CVPC, lw=.9)
        ax.set_title(f"{nm}   ({rid}, minute {tt:.0f})", fontsize=10.5, color=C.INK); ax.set_xlabel("seconds"); ax.set_ylabel("CVP (mmHg)")
    fig.suptitle("A CVP-waveform gate for MIMIC: what it measures and what it removes", fontsize=13, fontweight="bold", color=C.INK)
    C.save(fig, "f38_gate")
    if MG is None: return
    MA = json.load(open(f"{RES}/mimic_analysis.json")); TR = json.load(open(f"{RES}/transfer.json")); J = json.load(open(f"{RES}/investigation.json"))
    # before / after per arm
    fig, ax = plt.subplots(figsize=(17, 5.8)); K = [f"n>{t:g}" for t in THRS]
    before = {"zero-shot full model": {k: TR["arms"]["A zero-shot VitalDB->MIMIC-III"][k]["auc"] for k in K},
              "one column: block spread x since (no model)": {k: J["mimic"]["baselines"]["block spread: std x since"][k]["auc"] for k in K},
              "clock: minutes since the read": {k: J["mimic"]["baselines"]["clock: minutes since last CVP reading"][k]["auc"] for k in K},
              "CVP-line columns only (no ECG/PPG)": {k: J["mimic"]["ablations"]["- all ECG/PPG (91)"][k]["auc"] for k in K},
              "ECG/PPG + clock only": {k: J["mimic"]["ablations"]["ECG/PPG + time only (92)"][k]["auc"] for k in K},
              "MIMIC-trained ceiling (5-fold by record)": {k: TR["arms"]["B MIMIC-only ceiling (5-fold)"][k]["auc"] for k in K}}
    arms = list(MG["arms"]); x = np.arange(len(arms))
    for j, k in enumerate(K):
        ax.bar(x + (j - 1) * .27 - .07, [before[a][k] for a in arms], .12, color=[C.GREEN, C.BLUE, C.ORANGE][j], alpha=.35)
        ax.bar(x + (j - 1) * .27 + .07, [MG["arms"][a][k]["auc"] for a in arms], .12, color=[C.GREEN, C.BLUE, C.ORANGE][j], label=f"{k} with the gate" if True else None)
        for i, a in enumerate(arms): ax.text(x[i] + (j - 1) * .27 + .07, MG["arms"][a][k]["auc"] + .004, f"{MG['arms'][a][k]['auc']:.3f}", ha="center", fontsize=7.5, rotation=90, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels([a.replace(" (", "\n(") for a in arms], fontsize=9); ax.set_ylim(.5, .95); ax.axhline(.75, color=C.RED, lw=1.2, ls=":")
    ax.set_ylabel("pooled AUC, MIMIC-III"); ax.legend(fontsize=9, ncol=3, loc="upper left")
    ax.set_title(f"Every arm before (faint) and after the gate: {MG['windows']:,} windows in {MG['records']} records remain of 393,099 in 392; prevalence n>2 {100*MG['prevalence']['n>2']:.1f}%", fontsize=10.5, color=C.INK)
    fig.suptitle("Every MIMIC arm before and after the CVP-waveform gate", fontsize=13, fontweight="bold", color=C.INK)
    fig.tight_layout(rect=[0, 0, 1, .93]); C.save(fig, "f38b_gate_arms")


if PART in ("B", "all", "G"): gate_fig()
