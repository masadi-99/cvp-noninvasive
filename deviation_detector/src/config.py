"""Every path and constant the pipeline uses, in one place.

Inputs that live OUTSIDE this repo (large archives, not copied):
  VITALDB_TABLE   the 39,466 gated VitalDB windows x 91 ECG/PPG features + a-wave CVP label
  VITALDB_TILES   every 30-s VitalDB tile with the dense CVP waveform summaries and monitor numerics
  VITALDB_CASES   raw 500 Hz VitalDB waveforms (only needed to recompute the amplitude reference)
  VITALDB_CLIN    VitalDB clinical_data.csv (demographics)
  MIMIC_RECORDS   MIMIC-III / MIMIC-IV waveform records as .npz (sig_cvp/sig_ecg/sig_ppg at 125 Hz)
Everything the scripts produce goes under data/, results/, figures/ inside the repo.
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# VARIANT selects a results / figures tree: "" is the reported system; "gated" applies the MIMIC
# CVP-waveform gate everywhere; "gated_notime" also removes the elapsed-time product columns.
VARIANT = os.environ.get("VARIANT", "")
DATA = f"{ROOT}/data"
RES_BASE, FIG_BASE = f"{ROOT}/results", f"{ROOT}/figures"
RES = f"{RES_BASE}/{VARIANT}" if VARIANT else RES_BASE
FIG = f"{FIG_BASE}/{VARIANT}" if VARIANT else FIG_BASE
for d in (DATA, RES, FIG, f"{RES_BASE}/mimic_feat"):
    os.makedirs(d, exist_ok=True)
# the MIMIC CVP-waveform gate (step 19): on for every variant except the reported one
MIMIC_GATE = os.environ.get("MIMIC_GATE", "1" if VARIANT else "0") == "1"
# columns removed from the feature set for this variant (names from feature_names())
DROP_COLS = [c for c in os.environ.get("DROP_COLS", "slope_x_since,abs_slope_x_since,std_x_since,rate_x_since" if VARIANT == "gated_notime" else "").split(",") if c]

VITALDB_TABLE = "/home/masadi/run/cvp_min/data.npz"
VITALDB_TILES = "/home/masadi/cvp-step-examples/features/tiles.npz"
VITALDB_CASES = "/home/masadi/cvp_data/cases"
VITALDB_CLIN = "/home/masadi/vitaldb_meta/clinical_data.csv"
MIMIC_RECORDS = "/home/masadi/cvp-track/data/records"        # DATA ONLY from that project

# The MIMIC-III subset used for every reported number: the first 400 usable records in sorted
# record-id order (an arbitrary, id-ordered subset chosen for compute, before any result was seen).
# Every other usable record (~800) is extracted too and kept as a further held-out set (step 13).
MIMIC3_PAPER_N = 400

# ---- the schedule -----------------------------------------------------------------------------
W_CAL = 10.0        # minutes of CVP calibration at the start of every cycle
S_TRACK = 15.0      # minutes of causal ECG/PPG-only tracking after it
MIN_N = 8           # a stretch needs at least this many scored windows
MAX_STEP = 3.0      # mmHg; a sustained step this large inside a stretch marks a line artefact
THRS = (2.0, 3.0, 4.0)

# ---- the model --------------------------------------------------------------------------------
HGB = dict(max_iter=500, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=2.0,
           loss="absolute_error", max_features=0.7)
NSEED = 3
FOLD_SEED = 2024    # outer-fold assignment for the VitalDB nested CV
BOOT_SEED = 7

# ---- feature-vector layout (123 columns) ------------------------------------------------------
DESC_NAMES = ["blk_slope", "blk_curv", "blk_slope_1st_half", "blk_slope_2nd_half", "blk_slope_change",
              "blk_last_minus_median", "blk_mean_minus_median", "blk_std", "blk_ptp",
              "blk_mean_abs_step", "blk_median", "blk_last", "blk_n"]
HIST_NAMES = ["hist_delta_level", "hist_rate", "hist_gap_min", "hist_n_blocks", "hist_prev_slope"]
EXTRA_NAMES = ["gap_to_first_window", "blk_n_dense", "blk_n_gated", "blk_std_gated",
               "slope_x_since", "abs_slope_x_since", "std_x_since", "rate_x_since",
               "trk_pred", "trk_abs_pred", "trk_cummean", "trk_abs_cummean", "trk_cummax_abs"]

# ---- MIMIC amplitude reference (VitalDB pooled median / IQR of the raw signals) ---------------
SCALE_REF = dict(ppg_med=35.97441101074219, ppg_iqr=10.664535522460938,
                 ecg_med=0.0205230712890625, ecg_iqr=0.09380865097045898)
