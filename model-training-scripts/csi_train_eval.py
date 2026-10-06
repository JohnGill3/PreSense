#!/usr/bin/env python3
"""csi_train_eval.py - train a random forest on windowed CSI and evaluate it.
No firmware export: no emlearn, no C headers. Just training, metrics and logs.

Needs: pip install numpy scikit-learn joblib
Run AFTER csi_clean.py and csi_split.py.

Usage:
  python csi_train_eval.py --clean-dir clean --out-dir results
  python csi_train_eval.py --clean-dir clean --out-dir results --trees 20 --depth 12 --save-model

Outputs in --out-dir:
  train_eval.log      full log (also printed to the console)
  train_report.json   accuracy, per-class metrics, confusion matrices, per-session accuracy
  model.joblib        only with --save-model (fitted model + class list)
"""
import argparse
import json
import logging
import pathlib
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

log = logging.getLogger("csi_train_eval")


def featurize(w):
    """w: [n, T, n_sc] -> [n, 2*n_sc] = [mean over time | variance over time]."""
    return np.concatenate([w.mean(axis=1), w.var(axis=1)], axis=1).astype(np.float32)


def load(path):
    d = np.load(path, allow_pickle=True)
    return d["windows"].astype(np.float32), d["labels"].astype(str), d["session_id"].astype(str)


def describe(name, y, s, classes):
    counts = {c: int((y == c).sum()) for c in classes}
    log.info("%-5s %6d windows, %2d sessions | per class: %s",
             name, len(y), len(set(s.tolist())), counts)
    return counts


def evaluate(name, clf, X, y_int, sessions, classes):
    labels = list(range(len(classes)))
    t0 = time.perf_counter()
    pred = clf.predict(X)
    ms = (time.perf_counter() - t0) * 1000.0

    acc = accuracy_score(y_int, pred)
    cm = confusion_matrix(y_int, pred, labels=labels)
    rep = classification_report(y_int, pred, labels=labels, target_names=classes,
                                zero_division=0, output_dict=True)

    log.info("---- %s ----", name.upper())
    log.info("accuracy %.4f on %d windows (predict took %.1f ms)", acc, len(y_int), ms)
    log.info("per-class report:\n%s",
             classification_report(y_int, pred, labels=labels, target_names=classes, zero_division=0))
    log.info("confusion matrix (rows = true, cols = predicted; order: %s):\n%s", classes, cm)

    per_session = {}
    for sid in np.unique(sessions):
        m = sessions == sid
        per_session[str(sid)] = float(accuracy_score(y_int[m], pred[m]))
        log.info("  session %-40s accuracy %.3f (%d windows)", sid, per_session[str(sid)], int(m.sum()))
    worst = min(per_session, key=per_session.get)
    log.info("worst session: %s (%.3f)", worst, per_session[worst])

    return {"accuracy": float(acc), "n_windows": int(len(y_int)), "confusion": cm.tolist(),
            "per_class": {c: rep[c] for c in classes}, "per_session": per_session}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean-dir", default="clean")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--trees", type=int, default=10)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-model", action="store_true", help="also write model.joblib")
    ap.add_argument("--quiet", action="store_true", help="console shows warnings only (file log stays full)")
    a = ap.parse_args()

    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(out / "train_eval.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING if a.quiet else logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(fh)
    log.addHandler(ch)

    clean = pathlib.Path(a.clean_dir)
    log.info("loading data from %s", clean)
    tr_w, tr_y, tr_s = load(clean / "train.npz")
    va_w, va_y, va_s = load(clean / "val.npz")
    te_w, te_y, te_s = load(clean / "test.npz")

    classes = sorted(set(tr_y.tolist()))
    idx = {c: i for i, c in enumerate(classes)}
    log.info("classes (index order): %s", classes)
    log.info("window shape: %d records x %d subcarriers", tr_w.shape[1], tr_w.shape[2])

    counts = {"train": describe("train", tr_y, tr_s, classes),
              "val": describe("val", va_y, va_s, classes),
              "test": describe("test", te_y, te_s, classes)}

    for name, y in (("val", va_y), ("test", te_y)):
        unknown = set(y.tolist()) - set(idx)
        if unknown:
            raise SystemExit(f"{name} set has labels that are not in the train set: {unknown}")
    for c in classes:
        for name in ("train", "val", "test"):
            if counts[name][c] == 0:
                log.warning("class '%s' has no windows in %s; its metrics will be 0", c, name)

    y_tr = np.array([idx[c] for c in tr_y])
    y_va = np.array([idx[c] for c in va_y])
    y_te = np.array([idx[c] for c in te_y])

    t0 = time.perf_counter()
    X_tr, X_va, X_te = featurize(tr_w), featurize(va_w), featurize(te_w)
    log.info("features: %d per window (built in %.2f s)", X_tr.shape[1], time.perf_counter() - t0)

    log.info("training RandomForest: trees=%d depth=%d class_weight=balanced seed=%d",
             a.trees, a.depth, a.seed)
    t0 = time.perf_counter()
    clf = RandomForestClassifier(n_estimators=a.trees, max_depth=a.depth,
                                 class_weight="balanced", random_state=a.seed)
    clf.fit(X_tr, y_tr)
    log.info("trained in %.2f s", time.perf_counter() - t0)

    train_acc = float(accuracy_score(y_tr, clf.predict(X_tr)))
    log.info("train accuracy %.4f (a large gap to val/test means overfitting)", train_acc)

    report = {"classes": classes, "counts": counts, "train_accuracy": train_acc,
              "params": {"trees": a.trees, "depth": a.depth, "seed": a.seed},
              "val": evaluate("val", clf, X_va, y_va, va_s, classes),
              "test": evaluate("test", clf, X_te, y_te, te_s, classes)}

    gap = train_acc - report["test"]["accuracy"]
    log.info("train - test accuracy gap: %.3f", gap)
    if gap > 0.15:
        log.warning("gap above 0.15: likely overfitting or too few sessions per class")

    imp = clf.feature_importances_
    n_sc = X_tr.shape[1] // 2
    log.info("importance: mean features %.3f | variance features %.3f", imp[:n_sc].sum(), imp[n_sc:].sum())
    top = np.argsort(imp)[::-1][:10]
    log.info("top features: %s",
             [f"{'mean' if i < n_sc else 'var'}[{i % n_sc}]={imp[i]:.3f}" for i in top])
    report["importance"] = {"mean_total": float(imp[:n_sc].sum()), "var_total": float(imp[n_sc:].sum())}

    (out / "train_report.json").write_text(json.dumps(report, indent=2))
    if a.save_model:
        import joblib
        joblib.dump({"model": clf, "classes": classes}, out / "model.joblib")
        log.info("saved model to %s", out / "model.joblib")
    log.info("done. report: %s, log: %s", out / "train_report.json", out / "train_eval.log")


if __name__ == "__main__":
    main()
