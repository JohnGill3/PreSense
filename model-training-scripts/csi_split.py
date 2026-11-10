#!/usr/bin/env python3
"""csi_split.py - split clean/all_sessions.npz into train/val/test BY SESSION.

Splitting by session_id (not by window) matters because windows from the same
session are correlated - same room, same AP position, same RF conditions at
that time of day. Splitting by window lets near-duplicate windows land on
both sides of the split, which inflates validation/test accuracy without
the model actually generalizing to a new session.

Needs: pip install numpy

Usage:
    python csi_split.py --input clean/all_sessions.npz --out-dir clean \
        --val-frac 0.15 --test-frac 0.15 --seed 42

Strategy: group sessions by label, shuffle within each label, then assign
whole sessions to test, then val, then train - so every label is represented
in every split (assuming you have at least a few sessions per label).

With --assignment <split_manifest.json> (written by csi_clean.py
--window-mode records) the split is NOT recomputed: the predetermined
session assignment is used as-is, so the preprocessing fitted on training
sessions only stays valid, and extra per-window arrays (window_start) are kept.
"""
import argparse
import json
import pathlib
import random
from collections import defaultdict

import numpy as np


def assign_sessions(sessions_by_label, val_frac, test_frac, seed, test_days=(), verbose=True):
    """Return {session_id: "train"/"val"/"test"}. Sessions whose id contains one of
    test_days (e.g. "20261026") go to test; the rest are split per label."""
    rng = random.Random(seed)
    split_of_session = {}

    for lbl, label_sessions in sessions_by_label.items():
        sessions = sorted(label_sessions)  # sort first for reproducibility, then shuffle
        forced = [s for s in sessions if any(d in s for d in test_days)]
        for s in forced:
            split_of_session[s] = "test"
        sessions = [s for s in sessions if s not in forced]
        rng.shuffle(sessions)
        n = len(sessions)
        if test_days:
            n_test = 0
        else:
            n_test = max(1, round(n * test_frac)) if n >= 3 else (1 if n > 1 else 0)
        n_val = max(1, round(n * val_frac)) if n >= 3 else (1 if n > 2 else 0)
        n_test = min(n_test, n - 1) if n > 1 else 0
        n_val = min(n_val, n - n_test - 1) if n - n_test > 1 else 0

        test_sessions = sessions[:n_test]
        val_sessions = sessions[n_test:n_test + n_val]
        train_sessions = sessions[n_test + n_val:]

        if not train_sessions:
            print(f"WARNING: label '{lbl}' has only {n} session(s) - nothing left for "
                  f"train after val/test. Record more sessions for this label.")

        for s in test_sessions:
            split_of_session[s] = "test"
        for s in val_sessions:
            split_of_session[s] = "val"
        for s in train_sessions:
            split_of_session[s] = "train"

        if verbose:
            print(f"label '{lbl}': {n + len(forced)} sessions -> train={len(train_sessions)} "
                  f"val={len(val_sessions)} test={len(test_sessions) + len(forced)}")
    return split_of_session


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="clean/all_sessions.npz")
    ap.add_argument("--out-dir", default="clean")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--assignment", default=None,
                    help="split_manifest.json from csi_clean.py --window-mode records; "
                         "use its session assignment instead of a new random split")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    windows, labels, session_id = data["windows"], data["labels"], data["session_id"]
    extra = {k: data[k] for k in data.files
             if k not in ("windows", "labels", "session_id", "split") and len(data[k]) == len(labels)}

    if args.assignment:
        manifest = json.loads(pathlib.Path(args.assignment).read_text())
        split_of_session = manifest["session_assignment"]
        missing = sorted(set(map(str, session_id)) - set(split_of_session))
        if missing:
            raise SystemExit(f"sessions missing from {args.assignment}: {missing[:5]}...")
        print(f"using predetermined session assignment from {args.assignment}")
    else:
        # unique sessions per label
        sessions_by_label = defaultdict(set)
        for lbl, sid in zip(labels, session_id):
            sessions_by_label[str(lbl)].add(str(sid))
        split_of_session = assign_sessions(sessions_by_label, args.val_frac, args.test_frac, args.seed)

    split_arr = np.array([split_of_session.get(str(sid), "train") for sid in session_id])

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "val", "test"):
        idx = split_arr == split
        out_path = out_dir / f"{split}.npz"
        np.savez_compressed(out_path, windows=windows[idx], labels=labels[idx],
                            session_id=session_id[idx], **{k: v[idx] for k, v in extra.items()})
        counts = {str(l): int((labels[idx] == l).sum()) for l in np.unique(labels)}
        summary[split] = {"n_windows": int(idx.sum()), "per_label": counts,
                          "n_sessions": len(set(session_id[idx].tolist()))}
        print(f"{split}: {idx.sum()} windows from {summary[split]['n_sessions']} sessions "
              f"-> {out_path}")
        if args.assignment:
            empty = [l for l, c in counts.items() if c == 0]
            if empty:
                print(f"WARNING: {split} has no windows for classes {empty}")

    used = {str(s): split_of_session[str(s)] for s in np.unique(session_id)}
    (out_dir / "split_summary.json").write_text(json.dumps(
        {"session_assignment": used, "counts": summary,
         "assignment_source": args.assignment or "random"}, indent=2))

    # sanity check: no session should appear in more than one split
    session_splits = defaultdict(set)
    for sid, sp in zip(session_id, split_arr):
        session_splits[str(sid)].add(sp)
    leaks = {s: v for s, v in session_splits.items() if len(v) > 1}
    if leaks:
        print(f"ERROR: sessions appearing in multiple splits: {leaks}")
    else:
        print("OK: every session is entirely within one split.")


if __name__ == "__main__":
    main()
