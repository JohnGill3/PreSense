#!/usr/bin/env python3
"""Reproduce the session/day grouped follow-up experiments from the repo root."""
import csv
import hashlib
import json
import logging
import pathlib
import sys
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model-training-scripts"))
import csi_clean as clean
import csi_train_eval as train

OUT = pathlib.Path(__file__).resolve().parent
PARAMS = {"model": "rf", "trees": 100, "depth": 8, "leaf": 1, "two_stage": True}
STATIC = {"empty", "idle-nearby"}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_recordings():
    recordings, audit, manifest = {}, [], []
    for path in sorted((ROOT / "data-collection-scripts/data").glob("*.csv")):
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text())
        frame = pd.read_csv(path)
        valid, bad = clean.phase1_validate(frame)
        valid, inconsistent = clean.phase2_filter(valid, None)
        if valid.empty:
            raise ValueError(f"No valid rows in {path}")
        amp, _ = clean.phase3_decode(valid, 0)
        tsf = valid["tsf"].to_numpy()
        stalls = clean.find_stalls(tsf)
        label = meta["label"]
        if not path.stem.startswith(label + "_"):
            raise ValueError(f"Filename/label mismatch: {path.name}")
        date = path.stem.rsplit("_", 2)[1]
        recordings[path.stem] = {
            "amp": amp, "tsf": tsf, "stalls": stalls, "label": label,
            "date": date, "records": meta["records"],
        }
        windows, _, underfilled = clean.make_windows(amp, tsf, stalls, 6, 0.75, 60, 0.7)
        audit.append({
            "session": path.stem, "date": date, "label": label,
            "raw_records": len(frame), "valid_records": len(valid),
            "bad_records": bad, "inconsistent_records": inconsistent,
            "metadata_records_match": len(frame) == meta["records"],
            "rate_hz": meta["rate_hz"], "bad_lines": meta.get("bad_lines", 0),
            "stall_count": int(stalls.sum()), "windows": len(windows),
            "underfilled_windows": underfilled,
            "channels": "|".join(map(str, sorted(frame["channel"].unique()))),
            "sources": "|".join(sorted(frame["src"].unique())),
            "src_mac_filter": meta.get("src_mac_filter"),
            "csi_config": "|".join(command for command in meta.get("commands", [])
                                 if command.startswith("wlan-set-csi-param-header ")),
            "firmware_warning_count": len(meta.get("firmware_warnings", [])),
            "median_rssi_a": float(valid["rssi_a"].median()),
            "mean_csi_amplitude": float(amp.mean()),
            "median_sinr": float(valid["sinr"].median()),
        })
        for source in (path, meta_path):
            manifest.append({"path": str(source.relative_to(ROOT)),
                             "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    if not recordings:
        raise ValueError("No recordings found")
    write_csv(OUT / "recording_audit.csv", audit)
    write_json(OUT / "input_manifest.json", manifest)
    summaries = []
    for date, label in sorted({(row["date"], row["label"]) for row in audit}):
        rows = [row for row in audit if (row["date"], row["label"]) == (date, label)]
        summaries.append({
            "date": date, "label": label, "sessions": len(rows),
            "usable_sessions": sum(row["windows"] > 0 for row in rows),
            "windows": sum(row["windows"] for row in rows),
            "median_session_rssi_a": float(np.median([r["median_rssi_a"] for r in rows])),
            "mean_session_csi_amplitude": float(np.mean([r["mean_csi_amplitude"] for r in rows])),
            "rate_hz_min": min(r["rate_hz"] for r in rows),
            "rate_hz_max": max(r["rate_hz"] for r in rows),
            "stall_count": sum(r["stall_count"] for r in rows),
        })
    write_csv(OUT / "date_label_summary.csv", summaries)
    write_json(OUT / "setup_audit.json", {
        key: sorted({str(row[key]) for row in audit})
        for key in ("channels", "sources", "src_mac_filter", "csi_config")
    })
    return recordings


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(OUT / "experiments.log", mode="w"),
                                  logging.StreamHandler()])
    recordings = load_recordings()
    cache = {}

    def reference(sessions):
        candidates = [s for s in sessions if recordings[s]["label"] == "empty"]
        if not candidates:
            raise ValueError("No training-only empty reference available")
        return min(candidates, key=lambda s: (-recordings[s]["records"], s))

    def dataset(ref):
        if ref not in cache:
            mask = clean.build_null_mask(recordings[ref]["amp"])
            all_windows, labels, sessions, starts = [], [], [], []
            for sid, recording in recordings.items():
                amp = clean.normalize_rows(recording["amp"][:, ~mask])
                windows, times, _ = clean.make_windows(
                    amp, recording["tsf"], recording["stalls"], 6, 0.75, 60, 0.7)
                if not windows:
                    continue
                windows = clean.pad_or_trim(windows, 60)
                all_windows.append(windows)
                labels.extend([recording["label"]] * len(windows))
                sessions.extend([sid] * len(windows))
                starts.extend(times)
            windows = np.concatenate(all_windows)
            cache[ref] = {
                "windows": windows, "labels": np.array(labels),
                "session_id": np.array(sessions), "window_start_s": np.array(starts),
                "X": train.featurize(windows, "full")[0], "mask": mask,
            }
        return cache[ref]

    summary = {}
    for exclude_day in (False, True):
        selected = [s for s, r in recordings.items()
                    if not exclude_day or r["date"] != "20261005"]
        ref = reference(selected)
        base = dataset(ref)
        keep = np.isin(base["session_id"], selected)
        sessions = base["session_id"][keep]
        original = base["labels"][keep]
        dates = np.array([recordings[s]["date"] for s in sessions])
        for merge in (False, True):
            name = ("no1005" if exclude_day else "all_dates") + ("_static" if merge else "_five_class")
            directory = OUT / name
            directory.mkdir(exist_ok=True)
            labels = np.where(np.isin(original, list(STATIC)), "static", original) if merge else original
            classes = sorted(np.unique(labels).tolist())
            ids = {c: i for i, c in enumerate(classes)}
            y = np.array([ids[c] for c in labels])
            static_ids = [ids["static"]] if merge else [ids[c] for c in sorted(STATIC)]
            np.savez_compressed(directory / "all_sessions.npz",
                                windows=base["windows"][keep], labels=labels,
                                original_labels=original, session_id=sessions,
                                window_start_s=base["window_start_s"][keep])
            np.save(directory / "null_subcarrier_mask.npy", base["mask"])
            report = {
                "classes": classes, "params": PARAMS, "seed": 42,
                "window_sec": 6, "overlap": 0.75, "vote": 3,
                "excluded_dates": ["20261005"] if exclude_day else [],
                "label_mapping": {c: "static" for c in sorted(STATIC)} if merge else {},
                "n_sessions": len(np.unique(sessions)), "n_windows": len(y),
                "class_windows": dict(Counter(labels.tolist())),
                "class_sessions": {c: len(np.unique(sessions[labels == c])) for c in classes},
                "derived_dataset_reference": ref,
                "preprocessing": "Training-only longest empty reference refitted in every fold",
            }
            for mode, groups in (("loso", sessions), ("leave_day_out", dates)):
                predictions = np.full(len(y), -1, dtype=int)
                folds, prediction_rows = [], []
                for fit, test in LeaveOneGroupOut().split(y, y, groups):
                    fit_sessions = set(sessions[fit])
                    test_sessions = set(sessions[test])
                    assert not fit_sessions & test_sessions
                    assert set(y[fit]) == set(y)
                    fold_ref = reference(fit_sessions)
                    assert fold_ref not in test_sessions
                    fold_data = dataset(fold_ref)
                    assert np.array_equal(fold_data["session_id"][keep], sessions)
                    features = fold_data["X"][keep]
                    model = train.make_model(PARAMS, 42, int((~fold_data["mask"]).sum()), static_ids)
                    model.fit(features[fit], y[fit])
                    predictions[test] = model.predict(features[test])
                    group = str(groups[test[0]])
                    fold = {"group": group, "training_reference": fold_ref,
                            "train_sessions": sorted(fit_sessions), "test_sessions": sorted(test_sessions)}
                    if mode == "leave_day_out":
                        fold["metrics"] = train.summarize(
                            f"{name}: day {group}", y[test], predictions[test], sessions[test], classes, 3)
                    folds.append(fold)
                    for i in test:
                        prediction_rows.append({
                            "session": sessions[i], "date": dates[i],
                            "window_start_s": float(base["window_start_s"][keep][i]),
                            "original_label": original[i], "label": labels[i],
                            "prediction": classes[predictions[i]], "held_out_group": group,
                            "training_reference": fold_ref,
                        })
                assert (predictions >= 0).all()
                report[mode] = train.summarize(name + ": " + mode, y, predictions, sessions, classes, 3)
                report[mode]["folds"] = folds
                write_csv(directory / f"{mode}_predictions.csv", prediction_rows)
                write_json(directory / "report.json", report)
            if merge:
                model = train.make_model(PARAMS, 42, int((~base["mask"]).sum()), static_ids)
                model.fit(base["X"][keep], y)
                joblib.dump({
                    "model": model, "classes": classes, "features": "full",
                    "two_stage": True, "static_classes": ["static"], "window_len": 60,
                    "null_subcarrier_mask": base["mask"], "label_mapping": report["label_mapping"],
                    "training_reference": ref,
                }, directory / "model.joblib")
            summary[name] = {key: value for key, value in report.items()
                             if key not in ("loso", "leave_day_out")}
            for mode in ("loso", "leave_day_out"):
                summary[name][mode] = {key: value for key, value in report[mode].items()
                                       if key not in ("folds", "per_session")}
            write_json(OUT / "summary.json", summary)


if __name__ == "__main__":
    main()
