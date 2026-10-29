#!/usr/bin/env python3
"""csi_verify_trials.py - check whether a recorded trial actually captured the
event, by finding where "activity" (per-record change in the CSI amplitude
pattern) rises above the session's own quiet baseline, and flagging trials
where that activity is missing, too close to the start/end, or too short.

This does NOT know your protocol's intended timing (e.g. "walk starts at
t=5s") - it only reports where something happened. You compare that against
what you expected (from --start-delay and your own notes) to judge accuracy.

Decoding matches csi_clean.py Phase 1/3 (no masking/Hampel - this is a raw
diagnostic view). Activity metric: for each record, the L2 distance between
its normalized amplitude vector and a rolling median "baseline" vector built
from the quietest portion of the same session. A sustained run above a
threshold is reported as one "event".

Needs: pip install numpy pandas matplotlib

Usage:
  # One trial, with a plot showing the detected event region
  python csi_verify_trials.py --files data/approaching_1.csv --plot

  # A whole folder of trials for one class: prints a flagged summary table
  python csi_verify_trials.py --files data/approaching_*.csv --expect-start 5 --expect-end 10

  # Batch-check everything and write a CSV report
  python csi_verify_trials.py --files data/approaching_*.csv data/leaving_*.csv \
      --out-csv trial_report.csv

  # If the default/auto baseline needs k-sigma too low to be trustworthy on
  # short trials, use the known still-edges from your protocol instead:
  python csi_verify_trials.py --files data/approaching_*.csv --baseline-edge-s 1.0
"""
import argparse
import pathlib

import numpy as np
import pandas as pd

HDR, TAIL = 48, 4


def decode_session(csv_path, scale_shift):
    df = pd.read_csv(csv_path)
    amps, tsfs = [], []
    mode_bytes = df["data_bytes"].mode().iloc[0]
    for _, row in df.iterrows():
        if row["data_bytes"] != mode_bytes:
            continue
        try:
            b = bytes.fromhex(row["record_hex"])
        except ValueError:
            continue
        if len(b) < HDR + TAIL or int.from_bytes(b[2:4], "little") != 0xABCD:
            continue
        if int.from_bytes(b[0:2], "little") * 4 != len(b):
            continue
        data_bytes = (int.from_bytes(b[44:46], "little") - 1) * 4
        if HDR + data_bytes + TAIL != len(b):
            continue
        payload = np.frombuffer(b[HDR:HDR + data_bytes], dtype=np.int8).astype(np.float32)
        payload /= (2 ** scale_shift)
        I, Q = payload[0::2], payload[1::2]
        amps.append(np.sqrt(I**2 + Q**2))
        tsfs.append(row["tsf"])
    if not amps:
        return None, None
    n_sc = {len(a) for a in amps}
    if len(n_sc) != 1:
        return None, None
    return np.stack(amps), np.array(tsfs, dtype=np.float64)


def normalize_rows(amp):
    return amp / (amp.mean(axis=1, keepdims=True) + 1e-9)


def activity_score(amp_norm, t=None, baseline_frac=0.2, edge_s=None):
    """Per-record distance from a baseline vector.

    Two baseline strategies:
    - edges (edge_s given, needs `t`): baseline = median of records in the
      first and last `edge_s` seconds. Use this for short event trials with a
      known still-start/still-end protocol - it can't be contaminated by the
      event itself, unlike the statistical approach below.
    - auto (default): median of the `baseline_frac` fraction of records with
      the smallest distance to the overall median. Works without knowing
      where the quiet part is, but can be skewed if the event occupies a
      large share of a short file.
    Returns a 1D activity array, one value per record.
    """
    if edge_s is not None and t is not None:
        edge_idx = np.where((t - t[0] <= edge_s) | (t[-1] - t <= edge_s))[0]
        if len(edge_idx) >= 5:
            baseline = np.median(amp_norm[edge_idx], axis=0)
            return np.linalg.norm(amp_norm - baseline, axis=1)
        # fall through to auto if the edges are too short to have enough records

    overall_med = np.median(amp_norm, axis=0)
    d0 = np.linalg.norm(amp_norm - overall_med, axis=1)
    n_quiet = max(5, int(len(amp_norm) * baseline_frac))
    quiet_idx = np.argsort(d0)[:n_quiet]
    baseline = np.median(amp_norm[quiet_idx], axis=0)
    return np.linalg.norm(amp_norm - baseline, axis=1)


def find_event(t, activity, k_sigma=3.0, min_run_s=0.3):
    """Flag records where activity exceeds median + k_sigma*MAD, then return
    the [start_s, end_s] span of the longest contiguous run at least
    `min_run_s` long. Returns (None, None, threshold) if nothing qualifies."""
    med = np.median(activity)
    mad = np.median(np.abs(activity - med)) * 1.4826 + 1e-9
    threshold = med + k_sigma * mad
    above = activity > threshold

    best = (None, None, 0.0)
    i = 0
    n = len(above)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            span = t[j - 1] - t[i]
            if span >= min_run_s and span > best[2]:
                best = (t[i], t[j - 1], span)
            i = j
        else:
            i += 1
    return best[0], best[1], threshold


def check_trial(path, scale_shift, k_sigma, min_run_s, expect_start, expect_end, edge_margin_s,
                baseline_edge_s):
    amp, tsf = decode_session(path, scale_shift)
    if amp is None:
        return {"file": str(path), "status": "DECODE_FAILED", "n_records": 0}

    t = (tsf - tsf[0]) / 1e6  # assumes TSF in microseconds
    duration = t[-1] - t[0] if len(t) > 1 else 0.0
    amp_n = normalize_rows(amp)
    act = activity_score(amp_n, t=t, edge_s=baseline_edge_s)
    ev_start, ev_end, threshold = find_event(t, act, k_sigma, min_run_s)

    flags = []
    if ev_start is None:
        flags.append("NO_EVENT_DETECTED")
    else:
        if ev_start < edge_margin_s:
            flags.append("EVENT_NEAR_START")
        if (duration - ev_end) < edge_margin_s:
            flags.append("EVENT_NEAR_END")
        if expect_start is not None and expect_end is not None:
            if not (expect_start - edge_margin_s <= ev_start and ev_end <= expect_end + edge_margin_s):
                flags.append("OUTSIDE_EXPECTED_WINDOW")

    return {
        "file": str(path), "status": "OK" if not flags else "CHECK",
        "n_records": len(t), "duration_s": round(duration, 2),
        "event_start_s": None if ev_start is None else round(ev_start, 2),
        "event_end_s": None if ev_end is None else round(ev_end, 2),
        "event_span_s": None if ev_start is None else round(ev_end - ev_start, 2),
        "flags": ";".join(flags) if flags else "",
    }


def plot_trial(path, scale_shift, k_sigma, min_run_s, out, baseline_edge_s):
    import matplotlib.pyplot as plt

    amp, tsf = decode_session(path, scale_shift)
    if amp is None:
        print(f"{path}: could not decode any valid records")
        return
    t = (tsf - tsf[0]) / 1e6
    amp_n = normalize_rows(amp)
    act = activity_score(amp_n, t=t, edge_s=baseline_edge_s)
    ev_start, ev_end, threshold = find_event(t, act, k_sigma, min_run_s)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    im = ax1.imshow(amp_n.T, aspect="auto", origin="lower", extent=[t[0], t[-1], 0, amp_n.shape[1]],
                    cmap="viridis")
    ax1.set_ylabel("subcarrier index")
    ax1.set_title(f"{pathlib.Path(path).name}  (n={len(t)} records, {t[-1]-t[0]:.1f}s)")
    fig.colorbar(im, ax=ax1, label="amplitude (row-normalized)")

    ax2.plot(t, act, color="tab:blue", label="activity (distance from baseline)")
    ax2.axhline(threshold, color="gray", linestyle="--", label="detection threshold")
    if ev_start is not None:
        ax2.axvspan(ev_start, ev_end, color="orange", alpha=0.3, label="detected event")
        ax1.axvspan(ev_start, ev_end, color="orange", alpha=0.15)
    ax2.set_xlabel("time (s, from TSF)")
    ax2.set_ylabel("activity score")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}"
         + (f"  | detected event {ev_start:.1f}-{ev_end:.1f}s" if ev_start is not None
            else "  | NO EVENT DETECTED"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--scale-shift", type=int, default=0)
    ap.add_argument("--k-sigma", type=float, default=3.0,
                    help="how many MADs above the quiet baseline counts as 'active' (lower = more sensitive)")
    ap.add_argument("--min-run-s", type=float, default=0.3,
                    help="shortest sustained run of activity that counts as a real event, not a blip")
    ap.add_argument("--edge-margin-s", type=float, default=1.0,
                    help="an event starting/ending within this many seconds of the recording's "
                         "edges, or outside --expect-start/--expect-end, is flagged")
    ap.add_argument("--expect-start", type=float, default=None,
                    help="where you expected the event to start, in seconds from the start of "
                         "the recorded window (i.e. after --start-delay already elapsed)")
    ap.add_argument("--expect-end", type=float, default=None,
                    help="where you expected the event to end, same reference as --expect-start")
    ap.add_argument("--baseline-edge-s", type=float, default=None,
                    help="build the baseline from the first/last N seconds of each file instead "
                         "of a statistical 'quietest fraction' guess. Use this for short event "
                         "trials with a known still-start/still-end protocol (e.g. --start-delay "
                         "already covers settling, and you stand still ~1-2s before/after walking) "
                         "- it can't be contaminated by the event itself, which the default 'auto' "
                         "baseline can be on short, event-heavy files. Try e.g. 1.0 or 1.5.")
    ap.add_argument("--plot", action="store_true", help="save one PNG per file instead of a table")
    ap.add_argument("--out-csv", default=None, help="write the summary table to this CSV")
    args = ap.parse_args()

    files = sorted(set(pathlib.Path(f) for f in args.files))
    for f in files:
        if not f.exists():
            raise SystemExit(f"not found: {f}")

    if args.plot:
        for f in files:
            out = f.with_suffix(".verify.png")
            plot_trial(f, args.scale_shift, args.k_sigma, args.min_run_s, out, args.baseline_edge_s)
        return

    rows = [check_trial(f, args.scale_shift, args.k_sigma, args.min_run_s,
                        args.expect_start, args.expect_end, args.edge_margin_s,
                        args.baseline_edge_s)
           for f in files]
    report = pd.DataFrame(rows)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", 40)
    print(report.to_string(index=False))

    n_flagged = (report["status"] == "CHECK").sum()
    print(f"\n{n_flagged} of {len(report)} trial(s) flagged for manual review.")
    if n_flagged:
        print("Flags: NO_EVENT_DETECTED (walk may have been missed or too subtle), "
              "EVENT_NEAR_START/END (event may be clipped by the recording window), "
              "OUTSIDE_EXPECTED_WINDOW (detected motion doesn't match --expect-start/--expect-end).")

    if args.out_csv:
        report.to_csv(args.out_csv, index=False)
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()