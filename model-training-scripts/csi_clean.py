#!/usr/bin/env python3
"""csi_clean.py - turn raw csi_logger.py sessions into windowed training data.

Input : data/<label>_<timestamp>.csv (+ matching .json from csi_logger.py)
Output: clean/<session_id>.npz   per-session windows
        clean/all_sessions.npz   everything concatenated, with session_id kept
        clean/report.json        per-phase counts, for sanity-checking the run
        clean/null_subcarrier_mask.npy   saved so future runs reuse the same mask

Needs: pip install numpy pandas scipy

Usage:
    python csi_clean.py --data-dir data --out-dir clean \
        --scale-shift 4 --empty-label empty \
        --window-sec 3 --overlap 0.5

Recommended for the 5-class motion model (see TRAINING_IMPROVEMENTS.md):
    python csi_clean.py --data-dir data --out-dir clean --empty-label empty \
        --window-sec 6 --overlap 0.75 --hampel-window 1 --min-fill 0.7

--min-fill: windows with fewer than this fraction of the expected records are
dropped rather than padded (padding repeats the last record and looks like
'no motion', which blurs approaching/leaving vs idle).

--scale-shift: the fixed-point exponent from AN14281's CSI format table for
the header/format you used in wlan-set-csi-param-header. Confirm this value
against the note before trusting absolute amplitude scale; it does not affect
per-record normalization (Phase 5), only cross-session absolute comparisons.

--empty-label: which --label value in your sessions represents an empty room.
Used once, on the longest such session, to find null/pilot subcarriers.

Firmware-compatible deployment mode (--window-mode records):
    python csi_clean.py --data-dir data --out-dir clean --window-mode records \
        --win-len 30 --hop 15 --val-frac 0.15 --test-frac 0.15 --seed 42

Reproduces csi_presence.c exactly: fixed radio filters (source MAC, channel,
payload size), no Hampel filter, a common TSF gap threshold, and windows of
--win-len real records emitted every --hop records after (re)filling, with no
padding. Sessions are assigned to train/val/test BEFORE the null-subcarrier
mask and gap threshold are fitted (training sessions only), so held-out
sessions never influence preprocessing. Writes clean/preprocess_config.json
and clean/split_manifest.json; reuse the latter with --split-manifest to keep
identical splits across runs. Then run csi_split.py --assignment.
"""
import argparse
import glob
import hashlib
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

HDR, TAIL = 48, 4
SIG = 0xABCD


# ---------- Phase 1: structural validation ----------------------------------

def validate_row(hexstr, csv_len):
    try:
        b = bytes.fromhex(hexstr)
    except ValueError:
        return None
    if len(b) < HDR + TAIL:
        return None
    if int.from_bytes(b[2:4], "little") != SIG:
        return None
    if int.from_bytes(b[0:2], "little") * 4 != len(b):
        return None
    data_bytes = (int.from_bytes(b[44:46], "little") - 1) * 4
    if HDR + data_bytes + TAIL != len(b):
        return None
    if data_bytes != csv_len:
        return None  # header's own data_bytes column disagrees with the blob
    return b, data_bytes


def phase1_validate(df):
    keep_idx, decoded = [], []
    for i, row in df.iterrows():
        r = validate_row(row["record_hex"], int(row["data_bytes"]))
        if r is not None:
            keep_idx.append(i)
            decoded.append(r)
    dropped = len(df) - len(keep_idx)
    out = df.loc[keep_idx].copy()
    out["_raw_bytes"] = [d[0] for d in decoded]
    out["_data_bytes"] = [d[1] for d in decoded]
    return out, dropped


# ---------- Phase 2: consistency filtering -----------------------------------

def phase2_filter(df, src_mac):
    n0 = len(df)
    if src_mac:
        df = df[df["src"].str.lower() == src_mac.lower()]
    if len(df) == 0:
        return df, n0
    mode_bytes = df["_data_bytes"].mode().iloc[0]
    mode_chan = df["channel"].mode().iloc[0]
    df = df[(df["_data_bytes"] == mode_bytes) & (df["channel"] == mode_chan)]
    return df, n0 - len(df)


# ---------- Phase 3: decode I/Q -> amplitude/phase ---------------------------

def decode_amp_phase(raw_bytes, data_bytes, scale_shift):
    payload = np.frombuffer(raw_bytes[HDR:HDR + data_bytes], dtype=np.int8).astype(np.float32)
    payload /= (2 ** scale_shift)
    I = payload[0::2]
    Q = payload[1::2]
    return np.sqrt(I ** 2 + Q ** 2), np.arctan2(Q, I)


def phase3_decode(df, scale_shift):
    amps, phases = [], []
    for b, db in zip(df["_raw_bytes"], df["_data_bytes"]):
        a, p = decode_amp_phase(b, db, scale_shift)
        amps.append(a)
        phases.append(p)
    n_sc = {len(a) for a in amps}
    if len(n_sc) != 1:
        raise ValueError(f"inconsistent subcarrier counts after Phase 2 filtering: {n_sc}")
    return np.stack(amps), np.stack(phases)


# ---------- Phase 4: null/pilot subcarrier mask ------------------------------

def build_null_mask(amp_matrix, var_pct=10, mean_pct=10):
    var = amp_matrix.var(axis=0)
    mean = amp_matrix.mean(axis=0)
    var_thresh = np.percentile(var, var_pct)
    mean_thresh = np.percentile(mean, mean_pct)
    return (var <= var_thresh) & (mean <= mean_thresh)


# ---------- Phase 5: per-record normalization --------------------------------

def normalize_rows(amp_matrix):
    denom = amp_matrix.mean(axis=1, keepdims=True) + 1e-9
    return amp_matrix / denom


# ---------- Phase 6: time alignment / stall detection ------------------------

def find_stalls(tsf, stall_factor=3.0):
    """Return boolean array, True where the record AFTER it follows a stall gap."""
    tsf = tsf.astype(np.float64)
    d = np.diff(tsf)
    d = np.where(d < 0, np.nan, d)  # guard against wrap/non-monotonic entries
    median = np.nanmedian(d)
    if not np.isfinite(median) or median <= 0:
        return np.zeros(len(tsf), dtype=bool)
    is_stall = np.zeros(len(tsf), dtype=bool)
    is_stall[1:] = np.where(np.isnan(d) | (d > stall_factor * median), True, False)
    return is_stall


# ---------- Phase 7: outlier suppression (Hampel, per subcarrier) -----------

def hampel_filter(matrix, window=7, n_sigmas=3.0):
    med = median_filter(matrix, size=(window, 1), mode="nearest")
    mad = median_filter(np.abs(matrix - med), size=(window, 1), mode="nearest")
    threshold = n_sigmas * 1.4826 * mad
    out = matrix.copy()
    outliers = np.abs(matrix - med) > threshold
    out[outliers] = med[outliers]
    return out, int(outliers.sum())


# ---------- Phase 8: windowing ------------------------------------------------

def make_windows(amp_matrix, tsf, is_stall, window_sec, overlap, target_len, min_fill=0.7):
    """Slice into time windows. Windows that contain a stall, or hold fewer than
    min_fill * target_len records, are dropped instead of being padded later -
    heavy padding repeats the last record and fakes a 'motionless' tail."""
    tsf_sec = (tsf.astype(np.float64) - tsf[0]) / 1e6  # assumes TSF in microseconds
    windows, starts = [], []
    t = 0.0
    step = window_sec * (1 - overlap)
    total = tsf_sec[-1] if len(tsf_sec) else 0.0
    min_records = max(1, int(np.ceil(min_fill * target_len)))
    n_underfilled = 0
    while t + window_sec <= total + 1e-9:
        idx = np.where((tsf_sec >= t) & (tsf_sec < t + window_sec))[0]
        if len(idx) > 0 and not is_stall[idx].any():
            if len(idx) >= min_records:
                windows.append(amp_matrix[idx])
                starts.append(t)
            else:
                n_underfilled += 1
        t += step
    return windows, starts, n_underfilled


def pad_or_trim(windows, target_len):
    out = []
    for w in windows:
        if len(w) >= target_len:
            out.append(w[:target_len])
        else:
            pad = np.repeat(w[-1:], target_len - len(w), axis=0)
            out.append(np.concatenate([w, pad], axis=0))
    return np.stack(out) if out else np.empty((0, target_len, windows[0].shape[1] if windows else 0))


# ---------- Firmware-compatible record-count mode (--window-mode records) ----
# Every function below mirrors csi_presence.c; change both together.

def record_fields(raw):
    """Header fields the firmware filters on (AN14281 offsets)."""
    return raw[26:32].hex(":"), raw[37]


def tsf_resets(tsf, stall_us):
    """True where csi_presence.c discards the partial window BEFORE adding the record:
    TSF went backwards or advanced by more than stall_us."""
    tsf = np.asarray(tsf, dtype=np.uint64)
    reset = np.zeros(len(tsf), dtype=bool)
    if len(tsf) > 1:
        prev, cur = tsf[:-1], tsf[1:]
        back = cur < prev
        delta = cur - np.where(back, cur, prev)  # avoid unsigned wrap; 'back' already resets
        reset[1:] = back | (delta > np.uint64(stall_us))
    return reset


def stream_window_starts(n, resets, win_len, hop):
    """Start indices of the windows the firmware classifies: the ring fills with
    win_len real records, then a result is produced every hop records."""
    starts, fill, since = [], 0, 0
    for i in range(n):
        if resets[i]:
            fill = since = 0
        if fill < win_len:
            fill += 1
        since += 1
        if fill == win_len and since >= hop:
            since = 0
            starts.append(i - win_len + 1)
    return starts


def decode_amp_f32(raw_list, data_bytes, iq_scale):
    """[n, raw_n_sc] float32 amplitudes, computed like the C code."""
    n_sc = data_bytes // 2
    out = np.empty((len(raw_list), n_sc), dtype=np.float32)
    for r, b in enumerate(raw_list):
        p = np.frombuffer(b[HDR:HDR + data_bytes], dtype=np.int8).astype(np.float32) * np.float32(iq_scale)
        out[r] = np.sqrt(p[0::2] * p[0::2] + p[1::2] * p[1::2])
    return out


def normalize_rows_c(amp):
    """row * (1 / (mean + 1e-9)) in float32, as csi_presence.c does."""
    inv = (np.float32(1.0) / (amp.mean(axis=1, dtype=np.float64).astype(np.float32) + np.float32(1e-9)))
    return (amp * inv[:, None]).astype(np.float32)


def file_sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load_raw_session(csv_path):
    """Phase 1 only; radio filters are applied later with dataset-wide values."""
    df = pd.read_csv(csv_path)
    n_raw = len(df)
    df, n_bad = phase1_validate(df)
    raws = list(df["_raw_bytes"])
    fields = [record_fields(b) for b in raws]
    return {"raw": raws, "row": df.index.to_numpy(dtype=np.int64),
            "data_bytes": np.array(df["_data_bytes"], dtype=int),
            "tsf": df["tsf"].to_numpy(dtype=np.uint64),
            "src": [f[0] for f in fields], "chan": np.array([f[1] for f in fields], dtype=int),
            "n_raw": n_raw, "n_bad": n_bad}


def apply_radio_filter(s, src_mac, channel, data_bytes):
    keep = np.ones(len(s["raw"]), dtype=bool)
    other_src = np.array([m != src_mac for m in s["src"]], dtype=bool) if s["raw"] else keep[:0]
    keep &= ~other_src
    other_chan = keep & (s["chan"] != channel)
    keep &= ~other_chan
    other_size = keep & (s["data_bytes"] != data_bytes)
    keep &= ~other_size
    return keep, {"other_src": int(other_src.sum()), "other_channel": int(other_chan.sum()),
                  "other_size": int(other_size.sum())}


def mode_of(values):
    vals, counts = np.unique(np.asarray(values), return_counts=True)
    return vals[np.argmax(counts)]


def run_records_mode(args):
    csv_paths = sorted(args.data_dir.glob("*.csv"))
    if not csv_paths:
        sys.exit(f"No CSV files found in {args.data_dir}")
    if args.hampel_window != 1:
        sys.exit("--window-mode records requires --hampel-window 1 (the firmware has no Hampel filter)")
    if not (1 <= args.hop <= args.win_len):
        sys.exit("--hop must be between 1 and --win-len")

    sessions, labels, hashes = {}, {}, {}
    for p in csv_paths:
        meta_path = p.with_suffix(".json")
        if not meta_path.exists():
            sys.exit(f"{p.name}: no matching .json (labels come from the metadata)")
        meta = json.loads(meta_path.read_text())
        if not meta.get("label"):
            sys.exit(f"{meta_path.name}: no 'label'")
        labels[p.stem] = meta["label"]
        hashes[p.name] = file_sha256(p)
        hashes[meta_path.name] = file_sha256(meta_path)
        sessions[p.stem] = load_raw_session(p)

    # ---- session assignment, decided before anything is fitted ------------
    if args.split_manifest:
        manifest = json.loads(pathlib.Path(args.split_manifest).read_text())
        assignment = manifest["session_assignment"]
        missing = sorted(set(sessions) - set(assignment))
        extra = sorted(set(assignment) - set(sessions))
        if missing or extra:
            sys.exit(f"--split-manifest does not match the data: missing={missing[:5]} extra={extra[:5]}")
        split_cfg = {k: manifest.get(k) for k in ("seed", "val_frac", "test_frac", "test_days")}
    else:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from csi_split import assign_sessions
        by_label = defaultdict(set)
        for sid in sorted(sessions):
            by_label[labels[sid]].add(sid)
        test_days = tuple(d for d in (args.test_days or "").split(",") if d)
        assignment = assign_sessions(by_label, args.val_frac, args.test_frac, args.seed, test_days)
        split_cfg = {"seed": args.seed, "val_frac": args.val_frac, "test_frac": args.test_frac,
                     "test_days": list(test_days)}
    train_ids = sorted(s for s in sessions if assignment[s] == "train")

    # ---- radio filters: explicit, or the dataset-wide mode (not label-dependent)
    all_src = [m for s in sessions.values() for m in s["src"]]
    all_chan = np.concatenate([s["chan"] for s in sessions.values()])
    all_db = np.concatenate([s["data_bytes"] for s in sessions.values()])
    src_mac = (args.src_mac or str(mode_of(all_src))).lower()
    channel = int(args.channel if args.channel is not None else mode_of(all_chan))
    data_bytes = int(args.data_bytes if args.data_bytes is not None else mode_of(all_db))
    raw_n_sc = data_bytes // 2
    iq_scale = 1.0 / (2 ** args.scale_shift)
    print(f"radio filter: src={src_mac} channel={channel} data_bytes={data_bytes} "
          f"({raw_n_sc} raw subcarriers)")

    filt = {}
    for sid, s in sessions.items():
        keep, counts = apply_radio_filter(s, src_mac, channel, data_bytes)
        raw = [b for b, k in zip(s["raw"], keep) if k]
        filt[sid] = {"amp": decode_amp_f32(raw, data_bytes, iq_scale) if raw else
                     np.empty((0, raw_n_sc), np.float32),
                     "tsf": s["tsf"][keep], "counts": counts}

    # ---- fitted on training sessions only -----------------------------------
    empty_train = [sid for sid in train_ids if labels[sid] == args.empty_label and len(filt[sid]["amp"])]
    if not empty_train:
        sys.exit(f"no training session labelled '{args.empty_label}' to fit the subcarrier mask")
    mask_sid = max(empty_train, key=lambda sid: (len(filt[sid]["amp"]), sid))
    mask = build_null_mask(filt[mask_sid]["amp"].astype(np.float32))
    kept_sc = np.where(~mask)[0]
    np.save(args.out_dir / "null_subcarrier_mask.npy", mask)

    deltas = []
    for sid in train_ids:
        d = np.diff(filt[sid]["tsf"].astype(np.int64))
        deltas.append(d[d > 0])
    deltas = np.concatenate(deltas) if deltas else np.array([], dtype=np.int64)
    if not len(deltas):
        sys.exit("no TSF intervals in the training sessions")
    median_gap_us = float(np.median(deltas))
    stall_us = int(args.stall_us) if args.stall_us else int(round(args.stall_factor * median_gap_us))
    print(f"mask from training session {mask_sid}: keeps {len(kept_sc)}/{raw_n_sc} subcarriers; "
          f"median TSF gap {median_gap_us:.0f} us -> stall threshold {stall_us} us")

    # ---- windows --------------------------------------------------------------
    report, all_w, all_y, all_s, all_start, all_split = [], [], [], [], [], []
    for sid in sorted(sessions):
        f, s = filt[sid], sessions[sid]
        amp = normalize_rows_c(f["amp"][:, kept_sc]) if len(f["amp"]) else \
            np.empty((0, len(kept_sc)), np.float32)
        resets = tsf_resets(f["tsf"], stall_us)
        starts = stream_window_starts(len(amp), resets, args.win_len, args.hop)
        w = np.stack([amp[i:i + args.win_len] for i in starts]) if starts else \
            np.empty((0, args.win_len, len(kept_sc)), np.float32)
        span = (float(f["tsf"][-1] - f["tsf"][0]) / 1e6) if len(f["tsf"]) > 1 else 0.0
        stats = {"session": sid, "label": labels[sid], "split": assignment[sid], "raw": s["n_raw"],
                 "bad": s["n_bad"], **f["counts"], "kept_records": int(len(amp)),
                 "gap_resets": int(resets.sum()),
                 "actual_rate_hz": round((len(amp) - 1) / span, 3) if span > 0 else None,
                 "n_windows": len(starts)}
        if not starts:
            stats["excluded"] = "no full window"
        report.append(stats)
        np.savez_compressed(args.out_dir / f"{sid}.npz", windows=w, labels=np.array([labels[sid]] * len(starts)),
                            session_id=np.array([sid] * len(starts)), window_start=np.array(starts, dtype=int))
        if starts:
            all_w.append(w)
            all_y += [labels[sid]] * len(starts)
            all_s += [sid] * len(starts)
            all_start += starts
            all_split += [assignment[sid]] * len(starts)

    np.savez_compressed(args.out_dir / "all_sessions.npz", windows=np.concatenate(all_w),
                        labels=np.array(all_y), session_id=np.array(all_s),
                        window_start=np.array(all_start, dtype=int), split=np.array(all_split))
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))

    classes = sorted(set(labels.values()))
    per_split = {sp: {c: 0 for c in classes} for sp in ("train", "val", "test")}
    for y, sp in zip(all_y, all_split):
        per_split[sp][y] += 1
    for sp, counts in per_split.items():
        missing = [c for c, n in counts.items() if n == 0]
        print(f"{sp:<5} windows per class: {counts}")
        if missing:
            print(f"WARNING: {sp} has no windows for {missing}", file=sys.stderr)

    (args.out_dir / "split_manifest.json").write_text(json.dumps(
        {**split_cfg, "session_assignment": assignment, "labels": labels}, indent=2))
    config = {
        "format": "presense-preprocess", "version": 1, "mode": "records",
        "win_len": args.win_len, "hop": args.hop, "n_sc": int(len(kept_sc)), "raw_n_sc": int(raw_n_sc),
        "kept_sc": kept_sc.tolist(), "mask_source_session": mask_sid,
        "mask_rule": "build_null_mask(var_pct=10, mean_pct=10) on longest training empty session",
        "src_mac": src_mac, "channel": channel, "data_bytes": data_bytes,
        "scale_shift": args.scale_shift, "iq_scale": iq_scale, "hampel": None,
        "normalization": "amp * 1/(mean(kept amp) + 1e-9), float32",
        "stall_us": stall_us, "stall_factor": args.stall_factor, "median_tsf_gap_us": median_gap_us,
        "measured_rate_hz": round(1e6 / median_gap_us, 3),
        "window_span_s": round(args.win_len * median_gap_us / 1e6, 3),
        "empty_label": args.empty_label, "classes": classes, "split": split_cfg,
        "n_windows": {sp: int(sum(c.values())) for sp, c in per_split.items()},
        "excluded_sessions": [r["session"] for r in report if r.get("excluded")],
        "dataset": {"dir": str(args.data_dir.resolve()), "files": hashes},
    }
    (args.out_dir / "preprocess_config.json").write_text(json.dumps(config, indent=2))
    print(f"\nDone. {len(all_y)} windows ({args.win_len} records, hop {args.hop}) from "
          f"{len(sessions)} sessions -> {args.out_dir}/ (preprocess_config.json, split_manifest.json)")


# ---------- Main per-session pipeline ----------------------------------------

def process_session(csv_path, meta, args, null_mask_holder, target_len):
    df = pd.read_csv(csv_path)
    n_raw = len(df)

    df, n_bad = phase1_validate(df)
    df, n_inconsistent = phase2_filter(df, args.src_mac)
    if len(df) == 0:
        return None, {"session": csv_path.stem, "raw": n_raw, "bad": n_bad,
                       "inconsistent": n_inconsistent, "kept": 0, "error": "no rows survived"}

    amp, phase = phase3_decode(df, args.scale_shift)
    tsf = df["tsf"].to_numpy()

    if null_mask_holder["mask"] is None and meta.get("label") == args.empty_label:
        null_mask_holder["mask"] = build_null_mask(amp)
        np.save(args.out_dir / "null_subcarrier_mask.npy", null_mask_holder["mask"])

    mask = null_mask_holder["mask"]
    if mask is not None and mask.shape[0] == amp.shape[1]:
        amp = amp[:, ~mask]

    amp = normalize_rows(amp)
    is_stall = find_stalls(tsf, args.stall_factor)
    amp, n_outliers = hampel_filter(amp, window=args.hampel_window, n_sigmas=args.hampel_sigmas)

    windows, starts, n_underfilled = make_windows(amp, tsf, is_stall, args.window_sec, args.overlap,
                                                  target_len, args.min_fill)
    windows_arr = pad_or_trim(windows, max(target_len, 1))

    labels = np.array([meta["label"]] * len(windows_arr))
    session_ids = np.array([csv_path.stem] * len(windows_arr))

    out_path = args.out_dir / f"{csv_path.stem}.npz"
    np.savez_compressed(out_path, windows=windows_arr, labels=labels, session_id=session_ids,
                        window_start_s=np.array(starts))

    stats = {
        "session": csv_path.stem, "label": meta["label"], "raw": n_raw, "bad": n_bad,
        "inconsistent": n_inconsistent, "kept_records": len(df), "outlier_points": n_outliers,
        "stall_count": int(is_stall.sum()), "underfilled_windows": n_underfilled,
        "n_windows": len(windows_arr),
        "n_subcarriers": amp.shape[1],
    }
    return out_path, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="clean")
    ap.add_argument("--scale-shift", type=int, default=0,
                    help="I/Q scale exponent, if your firmware/AN14281 read confirms one; "
                         "defaults to 0 (no scaling) since per-record normalization in Phase 5 "
                         "cancels a constant scale within a session anyway")
    ap.add_argument("--empty-label", default="empty",
                    help="--label value used for the empty-room null-subcarrier reference")
    ap.add_argument("--src-mac", default=None, help="keep only records from this AP MAC")
    ap.add_argument("--window-sec", type=float, default=3.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--rate-hz", type=float, default=10.0,
                    help="nominal records/sec used to fix every session's window length "
                         "(records per window = round(window_sec * rate_hz)), so all sessions "
                         "produce IDENTICAL window shapes and can be concatenated into "
                         "all_sessions.npz. Using each session's own measured rate (as earlier "
                         "versions of this script did) causes a shape mismatch across sessions "
                         "whose actual capture rate differed slightly. 10 matches the beacon-rate "
                         "capture used throughout this project; check your sessions' .json "
                         "'rate_hz' values and adjust if yours differ notably from 10.")
    ap.add_argument("--min-fill", type=float, default=0.7,
                    help="drop windows holding fewer than this fraction of the fixed "
                         "records/window (instead of padding them with repeated records)")
    ap.add_argument("--min-sessions", type=int, default=8,
                    help="warn when a label has fewer sessions than this; leave-one-session-out "
                         "evaluation and csi_split.py need several sessions per label")
    ap.add_argument("--stall-factor", type=float, default=3.0)
    ap.add_argument("--hampel-window", type=int, default=7)
    ap.add_argument("--hampel-sigmas", type=float, default=3.0)
    rec = ap.add_argument_group("firmware-compatible record mode (--window-mode records)")
    ap.add_argument("--window-mode", choices=["time", "records"], default="time",
                    help="time = original time windows (default); records = exactly what "
                         "csi_presence.c computes, for training a model you will deploy")
    rec.add_argument("--win-len", type=int, default=30, help="records per window (CSI_WIN_LEN)")
    rec.add_argument("--hop", type=int, default=15, help="records between classifications (CSI_HOP)")
    rec.add_argument("--channel", type=int, default=None,
                     help="keep only this header channel (default: dataset-wide mode)")
    rec.add_argument("--data-bytes", type=int, default=None,
                     help="keep only this CSI payload size (default: dataset-wide mode)")
    rec.add_argument("--stall-us", type=int, default=None,
                     help="TSF gap that restarts the window (default: --stall-factor x median "
                          "training-session TSF interval)")
    rec.add_argument("--val-frac", type=float, default=0.15)
    rec.add_argument("--test-frac", type=float, default=0.15)
    rec.add_argument("--seed", type=int, default=42)
    rec.add_argument("--test-days", default=None,
                     help="comma-separated YYYYMMDD; those sessions become the test set "
                          "(recording-day holdout)")
    rec.add_argument("--split-manifest", default=None,
                     help="reuse a split_manifest.json so runs share identical splits")
    args = ap.parse_args()

    args.data_dir = pathlib.Path(args.data_dir)
    args.out_dir = pathlib.Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.window_mode == "records":
        run_records_mode(args)
        return

    csv_paths = sorted(args.data_dir.glob("*.csv"))
    if not csv_paths:
        sys.exit(f"No CSV files found in {args.data_dir}")

    # process the empty-room reference session(s) first, so the null mask
    # exists before any other session tries to apply it
    def sort_key(p):
        meta_path = p.with_suffix(".json")
        try:
            meta = json.loads(meta_path.read_text())
        except FileNotFoundError:
            return (1, p.name)
        return (0 if meta.get("label") == args.empty_label else 1, -meta.get("records", 0))
    csv_paths.sort(key=sort_key)

    target_len = max(1, int(round(args.window_sec * args.rate_hz)))
    print(f"fixed window length: {target_len} records/window "
         f"(window_sec={args.window_sec} x rate_hz={args.rate_hz}) - applied to ALL sessions")

    null_mask_holder = {"mask": None}
    all_windows, all_labels, all_sessions = [], [], []
    report = []

    for csv_path in csv_paths:
        meta_path = csv_path.with_suffix(".json")
        if not meta_path.exists():
            print(f"skip {csv_path.name}: no matching .json", file=sys.stderr)
            continue
        meta = json.loads(meta_path.read_text())
        actual_rate = meta.get("rate_hz")
        if actual_rate and abs(actual_rate - args.rate_hz) / args.rate_hz > 0.25:
            print(f"  NOTE {csv_path.stem}: measured rate_hz={actual_rate} differs from "
                 f"--rate-hz={args.rate_hz} by >25% - its windows will be padded/trimmed more "
                 f"than usual to fit the fixed {target_len}-record shape", file=sys.stderr)
        out_path, stats = process_session(csv_path, meta, args, null_mask_holder, target_len)
        report.append(stats)
        print(f"{csv_path.stem}: {stats}")
        if out_path is not None:
            npz = np.load(out_path, allow_pickle=True)
            if len(npz["windows"]):
                all_windows.append(npz["windows"])
                all_labels.append(npz["labels"])
                all_sessions.append(npz["session_id"])

    if null_mask_holder["mask"] is None:
        print(f"WARNING: no session with --empty-label '{args.empty_label}' was found; "
              f"null-subcarrier masking was skipped for every session.", file=sys.stderr)

    if all_windows:
        np.savez_compressed(args.out_dir / "all_sessions.npz",
                            windows=np.concatenate(all_windows),
                            labels=np.concatenate(all_labels),
                            session_id=np.concatenate(all_sessions))

    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))

    per_label = {}
    for s in report:
        if s.get("n_windows", 0) and "label" in s:
            c = per_label.setdefault(s["label"], {"sessions": 0, "windows": 0})
            c["sessions"] += 1
            c["windows"] += s["n_windows"]
    print("\nper label (sessions with >=1 window):")
    for lbl, c in sorted(per_label.items()):
        flag = "" if c["sessions"] >= args.min_sessions else \
            f"  <- fewer than {args.min_sessions} sessions, record more"
        print(f"  {lbl:<14} {c['sessions']:3d} sessions {c['windows']:5d} windows{flag}")

    total_windows = sum(s.get("n_windows", 0) for s in report)
    print(f"\nDone. {total_windows} windows across {len(report)} sessions -> {args.out_dir}/")


if __name__ == "__main__":
    main()
