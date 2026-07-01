"""Dawber (1973) four-class PPG pulse morphology — the scheme in Fig. 2 of Sensors 2021,21(13):4315.

Classify a beat by the development of its dicrotic notch on the DESCENDING limb (systolic peak -> end):
  Class 1: a true dicrotic notch (incisura) — the downslope reverses (local min then a dicrotic peak).
  Class 2: NO notch, but the downslope becomes ~horizontal (a plateau / slope touches ~0).
  Class 3: NO notch & no plateau, but a well-defined change in the angle of descent (a prominent
           2nd-derivative 'shoulder'/inflection while the slope stays clearly negative).
  Class 4: no evidence of a notch — smooth monotone decay.
Ordinal 1->4 = increasing vascular stiffening / pulse damping. We work on the amplitude-normalised,
length-normalised (100-pt) descending limb so the thresholds are scale- and heart-rate-invariant.
"""
import numpy as np

# thresholds on the 100-pt, amplitude-normalised descending limb (calibrated in calibrate())
REGION = (6, 82)        # ignore the immediate post-peak samples and the very tail
RISE_MIN = 0.010        # Class-1: notch->dicrotic-peak rise, in fraction of systolic amplitude
FLAT_EPS = 0.0020       # Class-2: slope comes within this of 0 (per-sample, normalised units)
D2_MIN = 0.0010         # Class-3: prominence of the concavity 'shoulder'


def _descending(seg):
    """Amplitude-normalised, 100-pt descending limb (systolic peak -> end). None if unusable."""
    if len(seg) < 12:
        return None
    sp = int(np.argmax(seg)); A = seg[sp] - seg[0]
    if A <= 1e-9 or sp >= len(seg) - 6:
        return None
    desc = seg[sp:]
    sn = (desc - desc[0]) / A                       # 0 at the peak, negative going down
    g = np.linspace(0, len(sn) - 1, 100)
    return np.interp(g, np.arange(len(sn)), sn)


def classify(seg):
    """Return Dawber class (1-4) for one beat segment (foot..next foot), or None."""
    sn = _descending(seg)
    if sn is None:
        return None
    d1 = np.gradient(sn); d2 = np.gradient(d1)
    lo, hi = REGION
    r1, r2 = d1[lo:hi], d2[lo:hi]
    # Class 1: an upward zero-crossing of the slope (local min -> rise) with a real dicrotic peak
    up = np.where((r1[:-1] < 0) & (r1[1:] >= 0))[0]
    if len(up):
        i = up[0] + lo
        post = sn[i:hi]
        if post.size and (post.max() - sn[i]) > RISE_MIN:
            return 1
    # Class 2: slope becomes ~horizontal (touches near 0) but never reverses
    if r1.max() > -FLAT_EPS:
        return 2
    # Class 3: a prominent concavity 'shoulder' while still descending
    if r2.max() > D2_MIN:
        return 3
    # Class 4: smooth monotone decay
    return 4


def case_features(classes):
    """Per-case morphology features from the list of per-beat classes."""
    c = np.array([x for x in classes if x is not None], float)
    if len(c) < 5:
        return dict(morph_mean=np.nan, morph_mode=np.nan, morph_f1=np.nan, morph_f4=np.nan, n=len(c))
    fr = {k: float(np.mean(c == k)) for k in (1, 2, 3, 4)}
    mode = int(max(fr, key=fr.get))
    return dict(morph_mean=float(c.mean()), morph_mode=float(mode),
                morph_f1=fr[1], morph_f4=fr[4], n=int(len(c)))
