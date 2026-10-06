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

--scale-shift: the fixed-point exponent from AN14281's CSI format table for
the header/format you used in wlan-set-csi-param-header. Confirm this value
against the note before trusting absolute amplitude scale; it does not affect
per-record normalization (Phase 5), only cross-session absolute comparisons.

--empty-label: which --label value in your sessions represents an empty room.
Used once, on the longest such session, to find null/pilot subcarriers.
"""
import argparse
import glob
import json
import pathlib
import sys

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

def make_windows(amp_matrix, tsf, is_stall, window_sec, overlap, min_fill=0.7):
    tsf_sec = (tsf.astype(np.float64) - tsf[0]) / 1e6  # assumes TSF in microseconds; see note below
    windows, starts = [], []
    t = 0.0
    step = window_sec * (1 - overlap)
    total = tsf_sec[-1] if len(tsf_sec) else 0.0
    expected_n = None
    while t + window_sec <= total + 1e-9:
        mask = (tsf_sec >= t) & (tsf_sec < t + window_sec)
        idx = np.where(mask)[0]
        if expected_n is None and t == 0.0:
            expected_n = None  # will be set from the first full window below
        if len(idx) > 0 and not is_stall[idx].any():
            windows.append(amp_matrix[idx])
            starts.append(t)
        t += step
    return windows, starts


def pad_or_trim(windows, target_len):
    out = []
    for w in windows:
        if len(w) >= target_len:
            out.append(w[:target_len])
        else:
            pad = np.repeat(w[-1:], target_len - len(w), axis=0)
            out.append(np.concatenate([w, pad], axis=0))
    return np.stack(out) if out else np.empty((0, target_len, windows[0].shape[1] if windows else 0))


# ---------- Main per-session pipeline ----------------------------------------

def process_session(csv_path, meta, args, null_mask_holder):
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

    windows, starts = make_windows(amp, tsf, is_stall, args.window_sec, args.overlap)
    target_len = int(round(args.window_sec * meta.get("rate_hz", 10)))
    windows_arr = pad_or_trim(windows, max(target_len, 1))

    labels = np.array([meta["label"]] * len(windows_arr))
    session_ids = np.array([csv_path.stem] * len(windows_arr))

    out_path = args.out_dir / f"{csv_path.stem}.npz"
    np.savez_compressed(out_path, windows=windows_arr, labels=labels, session_id=session_ids,
                        window_start_s=np.array(starts))

    stats = {
        "session": csv_path.stem, "label": meta["label"], "raw": n_raw, "bad": n_bad,
        "inconsistent": n_inconsistent, "kept_records": len(df), "outlier_points": n_outliers,
        "stall_count": int(is_stall.sum()), "n_windows": len(windows_arr),
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
    ap.add_argument("--stall-factor", type=float, default=3.0)
    ap.add_argument("--hampel-window", type=int, default=7)
    ap.add_argument("--hampel-sigmas", type=float, default=3.0)
    args = ap.parse_args()

    args.data_dir = pathlib.Path(args.data_dir)
    args.out_dir = pathlib.Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

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

    null_mask_holder = {"mask": None}
    all_windows, all_labels, all_sessions = [], [], []
    report = []

    for csv_path in csv_paths:
        meta_path = csv_path.with_suffix(".json")
        if not meta_path.exists():
            print(f"skip {csv_path.name}: no matching .json", file=sys.stderr)
            continue
        meta = json.loads(meta_path.read_text())
        out_path, stats = process_session(csv_path, meta, args, null_mask_holder)
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
    total_windows = sum(s.get("n_windows", 0) for s in report)
    print(f"\nDone. {total_windows} windows across {len(report)} sessions -> {args.out_dir}/")


if __name__ == "__main__":
    main()
