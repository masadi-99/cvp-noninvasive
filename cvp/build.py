"""Build data/features.csv from raw VitalDB cases (transparency / re-extraction).

Requires the VitalDB cases downloaded locally and two inputs you provide:
  - a windows index: rows of (caseid, start_sample) for 30-s windows that passed quality control
    (clean PPG/ECG and a calibrated invasive CVP for the label);
  - the VitalDB clinical table (height) and the per-case CVP label.

Each case's feature value is the MEDIAN over its clean windows. Edit the paths/loaders below for
your layout, then:  python -m cvp.build
"""
import csv
import numpy as np
from .extract import extract_window, FS, WIN_SEC

# --- configure these for your local VitalDB layout -----------------------------
CASES_DIR = "/path/to/vitaldb/cases"     # case_<id>.npz with a 500 Hz 'ppg' array
WINDOWS = "/path/to/windows.csv"         # columns: caseid,start  (quality-passed 30-s windows)
CLINICAL = "/path/to/clinical.csv"       # columns: caseid,height,cvp_numeric
OUT = "data/features.csv"
FEATURES = ["ppg_alternans", "ppg_ac_amp", "ppg_upstroke", "ppg_pvi"]


def _load_ppg(caseid):
    return np.load(f"{CASES_DIR}/case_{caseid}.npz")["ppg"]


def main():
    windows = {}
    for r in csv.DictReader(open(WINDOWS)):
        windows.setdefault(int(r["caseid"]), []).append(int(r["start"]))
    clin = {int(r["caseid"]): r for r in csv.DictReader(open(CLINICAL))}

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["caseid"] + FEATURES + ["height", "cvp_numeric", "elevated"])
        for cid, starts in windows.items():
            ppg = _load_ppg(cid)
            per_win = [extract_window(ppg[s:s + FS * WIN_SEC]) for s in starts]
            med = {fe: np.nanmedian([w[fe] for w in per_win]) for fe in FEATURES}
            c = clin.get(cid, {})
            cvp = float(c.get("cvp_numeric", "nan"))
            w.writerow([cid] + [round(med[fe], 5) for fe in FEATURES] +
                       [c.get("height", ""), round(cvp, 2), int(cvp > 12)])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
