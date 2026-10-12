"""Step 24a. Which usable MIMIC-III records carry a cardiac-output NUMERIC, and fetch it.

The waveform archive on disk has no cardiac output. The public mimic3wdb numerics records (1 Hz
monitor trends: HR, ABP, CVP, ... and, on pulmonary-artery-catheter patients, a continuous
cardiac output 'CO') are fetched from PhysioNet for every usable record; for the records that
have 'CO', the numerics are downloaded and saved with their offset to the waveform record.
    python src/s24_mimic_numerics.py headers     # -> results/mimic_numerics/headers.json
    python src/s24_mimic_numerics.py fetch       # -> results/mimic_numerics/<rid>.npz
"""
import os, sys, csv, json, glob, subprocess, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RES_BASE, MIMIC_RECORDS
OUT = f"{RES_BASE}/mimic_numerics"; os.makedirs(OUT, exist_ok=True)
DIRS = {r["record_id"]: r["dir"] for r in csv.DictReader(open("/home/masadi/cvp-track/data/index/mimic3wdb_header_inventory.csv"))}


def parse_time(s):
    h, m, sec = s.split(":"); return int(h) * 3600 + int(m) * 60 + float(sec)


def headers():
    from s05_mimic_survey import usable
    ids = sorted(usable("mimic3wdb")); hd = f"{OUT}/hea"; os.makedirs(hd, exist_ok=True)
    urls = [f"https://physionet.org/files/mimic3wdb/1.0/{DIRS[r]}/{r}n.hea" for r in ids if not os.path.exists(f"{hd}/{r}n.hea")]
    open(f"{hd}/urls.txt", "w").write("\n".join(urls))
    if urls:
        subprocess.run(f"cd {hd} && cat urls.txt | xargs -P 8 -n 1 sh -c 'f=$(basename \"$0\"); curl -s -f -o \"$f\" \"$0\" || true'", shell=True)
    info = {}
    for r in ids:
        f = f"{hd}/{r}n.hea"
        if not os.path.exists(f): continue
        L = open(f).read().splitlines(); h = L[0].split(); names = [l.split()[-1] for l in L[1:] if l and not l.startswith("#") and len(l.split()) >= 9]
        wf = f"{MIMIC_RECORDS.replace('/records', '/raw')}/mimic3wdb/{DIRS[r]}/{r}.hea"; wh = open(wf).read().splitlines()[0].split() if os.path.exists(wf) else []
        info[r] = dict(fs=h[2], n=int(h[3]) if len(h) > 3 else None, start=h[-1] if len(h) > 4 else None, wave_start=wh[-1] if len(wh) > 4 else None, channels=names, has_co="CO" in names)
    json.dump(info, open(f"{OUT}/headers.json", "w"), indent=1)
    co = [r for r, d in info.items() if d["has_co"]]
    print(f"{len(info)} numerics headers of {len(ids)} usable records; {len(co)} carry a CO channel: {co}")


def fetch():
    import wfdb
    info = json.load(open(f"{OUT}/headers.json")); co = [r for r, d in info.items() if d["has_co"]]; t0 = time.time()
    for i, r in enumerate(co):
        o = f"{OUT}/{r}.npz"
        if os.path.exists(o): continue
        try:
            rec = wfdb.rdrecord(f"{r}n", pn_dir=f"mimic3wdb/1.0/{DIRS[r]}")
        except Exception as e:
            print(f"  {r}: FAILED {str(e)[:80]}", flush=True); continue
        sig = rec.p_signal; names = rec.sig_name; fs = float(rec.fs)
        keep = {n: sig[:, names.index(n)].astype(np.float32) for n in ("CO", "CVP", "HR", "ABPMean", "PAPMean") if n in names}
        offs = parse_time(info[r]["start"]) - parse_time(info[r]["wave_start"]) if info[r]["start"] and info[r]["wave_start"] else 0.0
        np.savez_compressed(o, fs=fs, offset_s=offs, **keep)
        print(f"  {i+1}/{len(co)} {r}: {sig.shape[0]:,} samples at {fs:g} Hz, CO finite {100*np.isfinite(keep['CO']).mean():.0f}%, offset {offs:+.1f} s  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    {"headers": headers, "fetch": fetch}[sys.argv[1]]()
