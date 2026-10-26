#!/usr/bin/env python3
"""csi_train_eval.py - train a tree ensemble on windowed CSI and evaluate it.
No firmware export: no emlearn, no C headers. Just training, metrics and logs.

Needs: pip install numpy scikit-learn joblib
Run AFTER csi_clean.py and csi_split.py. See TRAINING_IMPROVEMENTS.md.

Usage:
  # quick train/val/test run (defaults: dynamic features, random forest)
  python csi_train_eval.py --clean-dir clean --out-dir results

  # most reliable accuracy estimate: leave-one-session-out over ALL sessions
  python csi_train_eval.py --clean-dir clean --out-dir results --cv

  # best option found so far: two-stage model on 6 s windows (see the guide)
  python csi_train_eval.py --clean-dir clean --out-dir results --two-stage --cv

  # pick hyper-parameters by leave-one-session-out on train+val, refit on
  # train+val, score once on the held-out test set, save the model
  python csi_train_eval.py --clean-dir clean --out-dir results \
      --grid --fit-on trainval --vote 3 --save-model

Feature sets (--features):
  meanvar  [mean | variance] per subcarrier. What csi_presence.c computes today;
           use it for anything you will export with csi_train_export.py.
  dynamic  [variance | slope | mean |first difference|] per subcarrier. Drops the
           absolute mean level (it identifies the session/day, not the activity)
           and adds the time trend that separates approaching from leaving.
           Not yet implemented in firmware.
  full     [mean | variance | slope | mean |first difference|]; used by --two-stage.

Outputs in --out-dir:
  train_eval.log      full log (also printed to the console)
  train_report.json   accuracy, per-class metrics, confusion matrices, per-session accuracy
  model.joblib        only with --save-model (fitted model + class list + feature set)
"""
import argparse
import itertools
import json
import logging
import pathlib
import time
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

log = logging.getLogger("csi_train_eval")


def feat_meanvar(w):
    return [("mean", w.mean(axis=1)), ("var", w.var(axis=1))]


def feat_dynamic(w):
    T = w.shape[1]
    t = np.arange(T, dtype=np.float32) - (T - 1) / 2.0
    centered = w - w.mean(axis=1, keepdims=True)
    slope = (centered * t[None, :, None]).sum(axis=1) / float((t ** 2).sum())
    absdiff = np.abs(np.diff(w, axis=1)).mean(axis=1)
    return [("var", w.var(axis=1)), ("slope", slope), ("absdiff", absdiff)]


def feat_full(w):
    return feat_meanvar(w)[:1] + feat_dynamic(w)


FEATURE_SETS = {"meanvar": feat_meanvar, "dynamic": feat_dynamic, "full": feat_full}


def featurize(w, kind):
    """w: [n, T, n_sc] -> ([n, k*n_sc] float32, [block names])."""
    blocks = FEATURE_SETS[kind](w)
    return (np.concatenate([b for _, b in blocks], axis=1).astype(np.float32),
            [n for n, _ in blocks])


def make_forest(kind, trees, depth, leaf, seed):
    cls = {"rf": RandomForestClassifier, "et": ExtraTreesClassifier}[kind]
    return cls(n_estimators=trees, max_depth=depth, min_samples_leaf=leaf,
               class_weight="balanced", random_state=seed, n_jobs=-1)


class TwoStage(ClassifierMixin, BaseEstimator):
    """Gate -> specialist. Expects 'full' features [mean | var | slope | absdiff].

    gate    : static (e.g. empty, idle-nearby) vs moving, on [var|slope|absdiff]
    static  : which static class, on [var|slope|absdiff] - the absolute mean level
              mostly identifies the session/day, so it is left out here
    moving  : which moving class, on all features - where the person ends up
              (mean level) helps separate approaching from leaving
    """

    def __init__(self, n_sc=1, static_classes=(), model="rf", trees=100, depth=8, leaf=1, seed=42):
        self.n_sc, self.static_classes = n_sc, static_classes
        self.model, self.trees, self.depth, self.leaf, self.seed = model, trees, depth, leaf, seed

    def _forest(self):
        return make_forest(self.model, self.trees, self.depth, self.leaf, self.seed)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        dyn = X[:, self.n_sc:]
        static = np.isin(y, self.static_classes)
        if static.all() or not static.any():
            raise ValueError("two-stage needs both static and moving classes in the training data")
        self.gate_ = self._forest().fit(dyn, static)
        self.static_ = self._forest().fit(dyn[static], y[static])
        self.moving_ = self._forest().fit(X[~static], y[~static])
        return self

    def predict(self, X):
        dyn = X[:, self.n_sc:]
        gate = self.gate_.predict(dyn).astype(bool)
        out = np.empty(len(X), dtype=self.classes_.dtype)
        if gate.any():
            out[gate] = self.static_.predict(dyn[gate])
        if (~gate).any():
            out[~gate] = self.moving_.predict(X[~gate])
        return out

    @property
    def feature_importances_(self):
        dyn_imp = (self.gate_.feature_importances_ + self.static_.feature_importances_) / 2
        full = np.concatenate([np.zeros(self.n_sc), dyn_imp])
        return (full + self.moving_.feature_importances_) / 2


def make_model(p, seed, n_sc, static_ids):
    if p.get("two_stage"):
        return TwoStage(n_sc=n_sc, static_classes=tuple(static_ids), model=p["model"],
                        trees=p["trees"], depth=p["depth"], leaf=p["leaf"], seed=seed)
    return make_forest(p["model"], p["trees"], p["depth"], p["leaf"], seed)


def load(path):
    d = np.load(path, allow_pickle=True)
    return d["windows"].astype(np.float32), d["labels"].astype(str), d["session_id"].astype(str)


def describe(name, y, s, classes):
    counts = {c: int((y == c).sum()) for c in classes}
    log.info("%-5s %6d windows, %2d sessions | per class: %s",
             name, len(y), len(set(s.tolist())), counts)
    return counts


def vote(pred, sessions, k):
    """Causal majority vote over the last k window predictions of each session,
    the way the firmware would smooth its output. Windows keep file order."""
    if k <= 1:
        return pred
    out = pred.copy()
    for sid in np.unique(sessions):
        idx = np.where(sessions == sid)[0]
        for j, i in enumerate(idx):
            recent = pred[idx[max(0, j - k + 1):j + 1]]
            out[i] = Counter(recent.tolist()).most_common(1)[0][0]
    return out


def session_vote_accuracy(y, pred, sessions):
    hits = [Counter(pred[sessions == s].tolist()).most_common(1)[0][0] == y[sessions == s][0]
            for s in np.unique(sessions)]
    return float(np.mean(hits))


def summarize(name, y_int, pred, sessions, classes, k_vote):
    labels = list(range(len(classes)))
    acc = accuracy_score(y_int, pred)
    bal = balanced_accuracy_score(y_int, pred)
    cm = confusion_matrix(y_int, pred, labels=labels)
    rep = classification_report(y_int, pred, labels=labels, target_names=classes,
                                zero_division=0, output_dict=True)
    log.info("---- %s ----", name.upper())
    log.info("accuracy %.4f | balanced accuracy %.4f on %d windows", acc, bal, len(y_int))
    log.info("per-class report:\n%s",
             classification_report(y_int, pred, labels=labels, target_names=classes, zero_division=0))
    log.info("confusion matrix (rows = true, cols = predicted; order: %s):\n%s", classes, cm)

    res = {"accuracy": float(acc), "balanced_accuracy": float(bal), "n_windows": int(len(y_int)),
           "confusion": cm.tolist(), "per_class": {c: rep[c] for c in classes}}
    if k_vote > 1:
        pv = vote(pred, sessions, k_vote)
        res["vote_k"] = k_vote
        res["vote_accuracy"] = float(accuracy_score(y_int, pv))
        res["vote_balanced_accuracy"] = float(balanced_accuracy_score(y_int, pv))
        log.info("with %d-window majority vote: accuracy %.4f | balanced %.4f",
                 k_vote, res["vote_accuracy"], res["vote_balanced_accuracy"])
    res["session_vote_accuracy"] = session_vote_accuracy(y_int, pred, sessions)
    log.info("whole-session majority vote: %.4f of %d sessions correct",
             res["session_vote_accuracy"], len(np.unique(sessions)))

    per_session = {}
    for sid in np.unique(sessions):
        m = sessions == sid
        per_session[str(sid)] = float(accuracy_score(y_int[m], pred[m]))
        log.debug("  session %-40s accuracy %.3f (%d windows)", sid, per_session[str(sid)], int(m.sum()))
    worst = sorted(per_session, key=per_session.get)[:5]
    log.info("worst sessions: %s", ", ".join(f"{s} ({per_session[s]:.2f})" for s in worst))
    res["per_session"] = per_session
    return res


def loso_predict(X, y, groups, p, seed, n_sc, static_ids):
    m = make_model(p, seed, n_sc, static_ids)
    return cross_val_predict(m, X, y, groups=groups, cv=LeaveOneGroupOut())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean-dir", default="clean")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--features", choices=sorted(FEATURE_SETS), default="dynamic")
    ap.add_argument("--model", choices=["rf", "et"], default="rf",
                    help="rf = RandomForest, et = ExtraTrees (both exportable by emlearn)")
    ap.add_argument("--two-stage", action="store_true",
                    help="gate static vs moving first, then a specialist per group "
                         "(forces --features full); usually the most accurate option")
    ap.add_argument("--static-classes", default="empty,idle-nearby",
                    help="comma-separated labels the --two-stage gate treats as 'static'")
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--depth", type=int, default=8, help="0 = unlimited")
    ap.add_argument("--min-samples-leaf", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vote", type=int, default=1,
                    help="also report accuracy after a causal majority vote over this many windows")
    ap.add_argument("--cv", action="store_true",
                    help="leave-one-session-out over train+val+test; report and exit")
    ap.add_argument("--grid", action="store_true",
                    help="choose model/trees/depth/leaf by leave-one-session-out on train+val "
                         "(balanced accuracy), then train the final model with the winner")
    ap.add_argument("--fit-on", choices=["train", "trainval"], default="train",
                    help="trainval = fit the final model on train+val and score only test")
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

    if a.two_stage and a.features != "full":
        log.info("--two-stage: using --features full (was '%s')", a.features)
        a.features = "full"
    static_names = [c.strip() for c in a.static_classes.split(",") if c.strip()]
    if a.two_stage:
        missing = set(static_names) - set(idx)
        if missing:
            raise SystemExit(f"--static-classes not found in train labels: {missing}")
    static_ids = [idx[c] for c in static_names if c in idx]
    n_sc = tr_w.shape[2]

    t0 = time.perf_counter()
    (X_tr, blocks), (X_va, _), (X_te, _) = (featurize(w, a.features) for w in (tr_w, va_w, te_w))
    log.info("features '%s': %d per window = %s x %d subcarriers (built in %.2f s)",
             a.features, X_tr.shape[1], blocks, tr_w.shape[2], time.perf_counter() - t0)
    if a.features != "meanvar":
        log.info("note: csi_train_export.py / csi_presence.c only implement 'meanvar' features")

    params = {"model": a.model, "trees": a.trees, "depth": a.depth or None,
              "leaf": a.min_samples_leaf, "two_stage": a.two_stage}
    report = {"classes": classes, "counts": counts, "features": a.features}

    if a.cv:
        X = np.concatenate([X_tr, X_va, X_te])
        y = np.concatenate([y_tr, y_va, y_te])
        s = np.concatenate([tr_s, va_s, te_s])
        log.info("leave-one-session-out over %d sessions with %s", len(np.unique(s)), params)
        t0 = time.perf_counter()
        pred = loso_predict(X, y, s, params, a.seed, n_sc, static_ids)
        log.info("cross-validation took %.1f s", time.perf_counter() - t0)
        report["params"] = params
        report["loso"] = summarize("leave-one-session-out", y, pred, s, classes, a.vote)
        (out / "train_report.json").write_text(json.dumps(report, indent=2))
        log.info("done. report: %s, log: %s", out / "train_report.json", out / "train_eval.log")
        return

    if a.grid:
        X = np.concatenate([X_tr, X_va])
        y = np.concatenate([y_tr, y_va])
        s = np.concatenate([tr_s, va_s])
        grid = [{"model": m, "trees": t, "depth": d, "leaf": l, "two_stage": a.two_stage}
                for m, t, d, l in itertools.product(["rf", "et"], [100], [6, 8, 12, None], [1, 3])]
        log.info("grid search: %d combinations, leave-one-session-out on train+val (%d sessions)",
                 len(grid), len(np.unique(s)))
        results = []
        for p in grid:
            pred = loso_predict(X, y, s, p, a.seed, n_sc, static_ids)
            bal = balanced_accuracy_score(y, pred)
            results.append({**p, "balanced_accuracy": float(bal),
                            "accuracy": float(accuracy_score(y, pred))})
            log.info("  %-60s balanced %.4f", p, bal)
        best = max(results, key=lambda r: r["balanced_accuracy"])
        params = {k: best[k] for k in ("model", "trees", "depth", "leaf", "two_stage")}
        log.info("best: %s (balanced accuracy %.4f)", params, best["balanced_accuracy"])
        report["grid"] = sorted(results, key=lambda r: -r["balanced_accuracy"])

    if a.fit_on == "trainval":
        X_fit, y_fit = np.concatenate([X_tr, X_va]), np.concatenate([y_tr, y_va])
    else:
        X_fit, y_fit = X_tr, y_tr

    log.info("training %s%s on %s (%d windows): trees=%d depth=%s min_samples_leaf=%d "
             "class_weight=balanced seed=%d", params["model"], " two-stage" if a.two_stage else "",
             a.fit_on, len(y_fit), params["trees"], params["depth"], params["leaf"], a.seed)
    t0 = time.perf_counter()
    clf = make_model(params, a.seed, n_sc, static_ids)
    clf.fit(X_fit, y_fit)
    log.info("trained in %.2f s", time.perf_counter() - t0)

    train_acc = float(accuracy_score(y_fit, clf.predict(X_fit)))
    log.info("train accuracy %.4f (a large gap to val/test means overfitting)", train_acc)

    report.update({"train_accuracy": train_acc, "fit_on": a.fit_on,
                   "params": {**params, "seed": a.seed}})
    if a.fit_on == "train":
        report["val"] = summarize("val", y_va, clf.predict(X_va), va_s, classes, a.vote)
    report["test"] = summarize("test", y_te, clf.predict(X_te), te_s, classes, a.vote)

    gap = train_acc - report["test"]["accuracy"]
    log.info("train - test accuracy gap: %.3f", gap)
    if gap > 0.15:
        log.warning("gap above 0.15: likely overfitting or too few sessions per class")

    imp = clf.feature_importances_
    totals = {b: float(imp[i * n_sc:(i + 1) * n_sc].sum()) for i, b in enumerate(blocks)}
    log.info("importance per feature block: %s", {b: round(v, 3) for b, v in totals.items()})
    top = np.argsort(imp)[::-1][:10]
    log.info("top features: %s", [f"{blocks[i // n_sc]}[{i % n_sc}]={imp[i]:.3f}" for i in top])
    report["importance"] = totals

    (out / "train_report.json").write_text(json.dumps(report, indent=2))
    if a.save_model:
        import joblib
        joblib.dump({"model": clf, "classes": classes, "features": a.features,
                     "two_stage": a.two_stage, "static_classes": static_names,
                     "window_len": int(tr_w.shape[1])}, out / "model.joblib")
        log.info("saved model to %s", out / "model.joblib")
    log.info("done. report: %s, log: %s", out / "train_report.json", out / "train_eval.log")


if __name__ == "__main__":
    main()
