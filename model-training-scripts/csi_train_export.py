#!/usr/bin/env python3
"""csi_train_export.py - train the random forest, evaluate it, export firmware files.

Needs: pip install numpy scikit-learn emlearn
Run AFTER csi_clean.py and csi_split.py.

Usage:
  python csi_train_export.py --clean-dir clean --out-dir firmware_export \
      --window-sec 3 --overlap 0.5 --ap-mac aa:bb:cc:dd:ee:ff

Use the SAME --window-sec / --overlap / --scale-shift / --stall-factor you gave
csi_clean.py, otherwise the firmware constants will not match the training data.

Outputs in --out-dir (copy the .h files into the project's source/ folder):
  csi_model.h         emlearn forest (function csi_model_predict)
  csi_model_config.h  constants the C featurizer needs
  csi_selftest.h      one cleaned window per class + expected features/class
  train_report.json   accuracy, confusion matrices, per-session accuracy
"""
import argparse
import json
import pathlib

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def featurize(w):
    """w: [n, T, n_sc] -> [n, 2*n_sc] = [mean over time | variance over time].
    Population variance (ddof=0). The C code in csi_presence.c must match this."""
    return np.concatenate([w.mean(axis=1), w.var(axis=1)], axis=1).astype(np.float32)


def load(path):
    d = np.load(path, allow_pickle=True)
    return d["windows"].astype(np.float32), d["labels"].astype(str), d["session_id"].astype(str)


def c_float_array(name, arr):
    flat = np.asarray(arr, dtype=np.float32).ravel()
    rows = ["    " + ", ".join(f"{v:.9e}f" for v in flat[i:i + 6]) for i in range(0, len(flat), 6)]
    return f"static const float {name}[{len(flat)}] = {{\n" + ",\n".join(rows) + "\n};\n"


def c_int_array(ctype, name, arr):
    vals = [str(int(v)) for v in arr]
    rows = ["    " + ", ".join(vals[i:i + 16]) for i in range(0, len(vals), 16)]
    return f"static const {ctype} {name}[{len(vals)}] = {{\n" + ",\n".join(rows) + "\n};\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean-dir", default="clean")
    ap.add_argument("--out-dir", default="firmware_export")
    ap.add_argument("--trees", type=int, default=10)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--window-sec", type=float, default=3.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--scale-shift", type=int, default=0)
    ap.add_argument("--stall-factor", type=float, default=3.0)
    ap.add_argument("--ap-mac", default=None, help="only accept records from this MAC on the MCU")
    ap.add_argument("--empty-label", default="empty",
                    help="class that means 'vacant'; every other class reports detected=true")
    a = ap.parse_args()

    clean, out = pathlib.Path(a.clean_dir), pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tr_w, tr_y, tr_s = load(clean / "train.npz")
    va_w, va_y, va_s = load(clean / "val.npz")
    te_w, te_y, te_s = load(clean / "test.npz")

    classes = sorted(set(tr_y.tolist()))
    idx = {c: i for i, c in enumerate(classes)}

    def enc(y):
        unknown = set(y.tolist()) - set(idx)
        if unknown:
            raise SystemExit(f"labels missing from train set: {unknown}")
        return np.array([idx[c] for c in y])

    ytr, yva, yte = enc(tr_y), enc(va_y), enc(te_y)
    Xtr, Xva, Xte = featurize(tr_w), featurize(va_w), featurize(te_w)

    # ---- train -------------------------------------------------------------
    clf = RandomForestClassifier(n_estimators=a.trees, max_depth=a.depth,
                                 class_weight="balanced", random_state=42)
    clf.fit(Xtr, ytr)

    # ---- evaluate ----------------------------------------------------------
    labels = list(range(len(classes)))
    report = {"classes": classes}
    for name, X, y, s in (("val", Xva, yva, va_s), ("test", Xte, yte, te_s)):
        pred = clf.predict(X)
        acc = accuracy_score(y, pred)
        cm = confusion_matrix(y, pred, labels=labels)
        print(f"\n== {name}: accuracy {acc:.3f}")
        print(classification_report(y, pred, labels=labels, target_names=classes, zero_division=0))
        print("confusion matrix (rows=true, cols=predicted):\n", cm)
        per_session = {sid: float(accuracy_score(y[s == sid], pred[s == sid])) for sid in np.unique(s)}
        for sid, v in per_session.items():
            print(f"  {sid}: {v:.3f}")
        report[name] = {"accuracy": float(acc), "confusion": cm.tolist(), "per_session": per_session}

    # ---- export forest to C ------------------------------------------------
    import emlearn
    cmodel = emlearn.convert(clf, method="inline")
    cmodel.save(file=str(out / "csi_model.h"), name="csi_model")
    try:  # the Python-side check API differs between emlearn versions
        c_pred = np.asarray(cmodel.predict(Xte)).astype(int).ravel()
        agree = float(np.mean(c_pred == clf.predict(Xte)))
        print(f"\nPython vs emlearn C agreement on test set: {agree:.4f}")
        report["c_agreement"] = agree
    except Exception as e:  # noqa: BLE001
        print(f"\nemlearn Python-side check unavailable ({e}); rely on csi_selftest.h on the board.")

    # ---- constants for the C featurizer -----------------------------------
    T, n_sc = tr_w.shape[1], tr_w.shape[2]
    mask_path = clean / "null_subcarrier_mask.npy"
    if mask_path.exists():
        mask = np.load(mask_path)
        kept = np.where(~mask)[0]
        if len(kept) != n_sc:
            raise SystemExit(f"mask keeps {len(kept)} subcarriers but windows have {n_sc}")
        raw_n_sc = len(mask)
    else:
        kept, raw_n_sc = np.arange(n_sc), n_sc

    hop = max(1, int(round(T * (1.0 - a.overlap))))
    empty_idx = idx.get(a.empty_label, 255)  # 255 = no such class, detected is then always true
    if empty_idx == 255:
        print(f"WARNING: no class named '{a.empty_label}'; firmware will always report detected=true")
    stall_us = int(a.stall_factor * a.window_sec / T * 1e6)
    mac = [int(x, 16) for x in a.ap_mac.split(":")] if a.ap_mac else None
    if mac and len(mac) != 6:
        raise SystemExit("--ap-mac must have 6 bytes")

    names = ", ".join(f'"{c}"' for c in classes)
    cfg = f"""/* Generated by csi_train_export.py - do not edit. Must match the trained model. */
#ifndef CSI_MODEL_CONFIG_H_
#define CSI_MODEL_CONFIG_H_
#include <stdint.h>

#define CSI_RAW_N_SC     {raw_n_sc}U   /* subcarriers in one raw record */
#define CSI_N_SC         {n_sc}U   /* subcarriers kept after null-subcarrier mask */
#define CSI_WIN_LEN      {T}U   /* records per window */
#define CSI_HOP          {hop}U   /* new records between two classifications */
#define CSI_N_FEATURES   {2 * n_sc}U   /* [mean x N_SC | variance x N_SC] */
#define CSI_N_CLASSES    {len(classes)}U
#define CSI_EMPTY_CLASS  {empty_idx}U   /* class index meaning "vacant" (255 = none) */
#define CSI_STALL_US     {stall_us}ULL   /* TSF gap that discards the window (assumes TSF in us) */
#define CSI_IQ_SCALE     {1.0 / (2 ** a.scale_shift):.9e}f
#define CSI_HAVE_AP_MAC  {1 if mac else 0}

static const char *const CSI_CLASS_NAME[CSI_N_CLASSES] = {{ {names} }};
{c_int_array("uint8_t", "CSI_AP_MAC", mac) if mac else ""}
/* index of each kept subcarrier inside the raw record */
{c_int_array("uint16_t", "CSI_KEPT_SC", kept)}
#endif
"""
    (out / "csi_model_config.h").write_text(cfg)

    # ---- self-test vectors: first test window of each class ---------------
    picks = [int(np.where(yte == c)[0][0]) for c in labels if (yte == c).any()]
    st_win = te_w[picks]
    st_feat = featurize(st_win)
    st_cls = clf.predict(st_feat)
    st = ("/* Generated by csi_train_export.py - do not edit. */\n#ifndef CSI_SELFTEST_H_\n"
          "#define CSI_SELFTEST_H_\n#include <stdint.h>\n"
          f"#define CSI_SELFTEST_N {len(picks)}\n"
          + c_float_array("CSI_SELFTEST_WIN", st_win)
          + c_float_array("CSI_SELFTEST_FEAT", st_feat)
          + c_int_array("uint8_t", "CSI_SELFTEST_CLASS", st_cls)
          + "#endif\n")
    (out / "csi_selftest.h").write_text(st)

    report["config"] = {"window_len": T, "hop": hop, "n_sc": n_sc, "raw_n_sc": raw_n_sc,
                        "n_features": 2 * n_sc, "stall_us": stall_us}
    (out / "train_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote csi_model.h, csi_model_config.h, csi_selftest.h, train_report.json to {out}/")


if __name__ == "__main__":
    main()
