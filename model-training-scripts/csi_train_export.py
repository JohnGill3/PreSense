#!/usr/bin/env python3
"""csi_train_export.py - export a SAVED model (csi_train_eval.py --save-model) to firmware.

Needs: pip install numpy pandas scikit-learn joblib ; gcc on PATH
Run AFTER csi_clean.py --window-mode records, csi_split.py --assignment and
csi_train_eval.py --save-model. Nothing is retrained: the exact fitted model in
model.joblib is converted, and every firmware constant comes from the
preprocess_config.json saved with it (no manual --window-sec/--overlap).

Usage:
  python csi_train_export.py --model results/model.joblib --clean-dir clean \
      --out-dir firmware_export --on-classes approaching --off-classes leaving,empty

Supported models: single RandomForest/ExtraTrees on meanvar, dynamic or full
features, and the --two-stage model (gate + static + moving forests).

Checks (any failure exits non-zero and nothing is marked deployable):
  * model/preprocessing/clean-dir compatibility (classes, shapes, kept subcarriers)
  * the generated C, compiled on this machine, must give the same class and
    confidence as the Python reference on every val/test window
  * raw-stream replay (csi_replay_parity.py) of the held-out sessions through
    the real csi_presence.c must reproduce the Python windows and classes

Outputs in --out-dir (copy the .h files into the project's source/ folder):
  csi_model.h          forest table(s) + csi_infer() (class + confidence), bit-exact with
                     the Python reference (sklearn decisions on float32 features)
  csi_model_config.h   window/subcarrier/radio/class/action constants
  csi_selftest.h       cleaned windows + expected features/class for the boot self-test
  presense-control.v1.schema.json  control-UART schema with this model's class enum
  export_report.json   metrics of the deployed model, parity results, provenance
"""
import argparse
import hashlib
import json
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix)

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import csi_train_eval  # noqa: E402  (model.joblib references csi_train_eval.TwoStage)

FEATURE_SET_ID = {"meanvar": 0, "dynamic": 1, "full": 2}
BLOCKS = {"meanvar": 2, "dynamic": 3, "full": 4}
ACTION_HOLD, ACTION_ON, ACTION_OFF = 0, 1, 2


def featurize(w, kind):
    return csi_train_eval.featurize(w, kind)[0]


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


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# ---------- forest -> C tables, and the exact Python reference of csi_infer() ----
# The forests are written as node tables instead of emlearn's inline C, whose
# thresholds are printed with 6 decimals (variance/slope features are ~1e-3..1e-5).
# sklearn sends a float32 feature left when x <= threshold (a double); for float32
# x that equals x <= t32, with t32 = the largest float32 <= threshold. Leaves hold
# each tree's class fractions (float32); trees are soft-voted with a double
# accumulator in estimator order, as RandomForest/ExtraTrees.predict_proba do.

def f32_floor(t):
    t32 = np.float32(t)
    if float(t32) > t:
        t32 = np.nextafter(t32, np.float32(-np.inf))
    return t32


class Forest:
    def __init__(self, forest, name):
        self.name, self.n_classes = name, len(forest.classes_)
        self.n_features = int(forest.n_features_in_)
        feat, thr, left, right, leaves, roots = [], [], [], [], [], []
        for est in forest.estimators_:
            t = est.tree_
            base = len(feat)
            roots.append(base)
            for i in range(t.node_count):
                if t.children_left[i] == -1:
                    v = np.asarray(t.value[i][0], dtype=np.float64)
                    feat.append(-1)
                    thr.append(np.float32(0))
                    left.append(len(leaves))
                    right.append(0)
                    leaves.append((v / v.sum()).astype(np.float32))
                else:
                    feat.append(int(t.feature[i]))
                    thr.append(f32_floor(float(t.threshold[i])))
                    left.append(base + int(t.children_left[i]))
                    right.append(base + int(t.children_right[i]))
        self.feat = np.array(feat, dtype=np.int32)
        self.thr = np.array(thr, dtype=np.float32)
        self.left = np.array(left, dtype=np.int64)
        self.right = np.array(right, dtype=np.int64)
        self.leaf = np.stack(leaves).astype(np.float32)
        self.roots = np.array(roots, dtype=np.int64)
        if max(len(feat), len(leaves)) >= 65535 or self.n_features > 32767:
            raise SystemExit(f"forest {name} too large for 16-bit node tables")

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        r = np.arange(len(X))
        acc = np.zeros((len(X), self.n_classes), dtype=np.float64)
        for root in self.roots:
            node = np.full(len(X), root, dtype=np.int64)
            while True:
                internal = self.feat[node] >= 0
                if not internal.any():
                    break
                f = np.where(internal, self.feat[node], 0)
                go_left = X[r, f] <= self.thr[node]
                node = np.where(internal, np.where(go_left, self.left[node], self.right[node]), node)
            acc += self.leaf[self.left[node]].astype(np.float64)
        return acc / float(len(self.roots))

    def c_code(self):
        n, p = self.name, self.name.upper()
        return (f"\n/* forest '{n}': {len(self.roots)} trees, {len(self.feat)} nodes, {len(self.leaf)} leaves, "
                f"{self.n_classes} classes, {self.n_features} features */\n"
                + c_int_array("int16_t", f"{p}_FEAT", self.feat)
                + c_float_array(f"{p}_THR", self.thr)
                + c_int_array("uint16_t", f"{p}_LEFT", self.left)
                + c_int_array("uint16_t", f"{p}_RIGHT", self.right)
                + c_float_array(f"{p}_LEAF", self.leaf)
                + c_int_array("uint16_t", f"{p}_ROOT", self.roots)
                + f"static const csi_forest_t {p} = {{ {p}_FEAT, {p}_THR, {p}_LEFT, {p}_RIGHT, {p}_LEAF, "
                  f"{p}_ROOT, {len(self.roots)}U, {self.n_classes}U, {self.n_features}U }};\n")

    def flash_bytes(self):
        return int(len(self.feat) * 8 + self.leaf.size * 4 + len(self.roots) * 2)


FOREST_C = r"""
typedef struct
{
    const int16_t *feat;   /* feature index, -1 = leaf */
    const float *thr;      /* go left when x <= thr */
    const uint16_t *left;  /* left child (node), or leaf row for a leaf */
    const uint16_t *right; /* right child (node) */
    const float *leaf;     /* [leaves][n_classes] per-tree class fractions */
    const uint16_t *root;  /* first node of each tree */
    uint16_t n_trees;
    uint8_t n_classes;
    uint16_t n_features;
} csi_forest_t;

/* Soft vote: p[c] = mean over trees of the leaf fraction for class c.
 * Returns the first class with the highest p (like numpy argmax), or -1. */
static int32_t csi_forest_predict(const csi_forest_t *m, const float *x, int32_t n, double *p)
{
    if ((x == NULL) || (n != (int32_t)m->n_features)) { return -1; }
    for (uint32_t c = 0U; c < m->n_classes; c++) { p[c] = 0.0; }
    for (uint32_t t = 0U; t < m->n_trees; t++) {
        uint32_t node = m->root[t];
        uint32_t guard = 0U;
        while (m->feat[node] >= 0) {
            node = (x[m->feat[node]] <= m->thr[node]) ? m->left[node] : m->right[node];
            if (++guard > 1024U) { return -1; }
        }
        const float *lf = &m->leaf[(uint32_t)m->left[node] * m->n_classes];
        for (uint32_t c = 0U; c < m->n_classes; c++) { p[c] += (double)lf[c]; }
    }
    int32_t best = 0;
    for (uint32_t c = 0U; c < m->n_classes; c++) {
        p[c] /= (double)m->n_trees;
        if (p[c] > p[best]) { best = (int32_t)c; }
    }
    return best;
}

static uint8_t csi_pct(double p)
{
    double v = (p * 100.0) + 0.5;
    return (uint8_t)((v < 0.0) ? 0.0 : ((v > 100.0) ? 100.0 : v));
}
"""


def argmax_first(p):
    return np.argmax(p, axis=1)  # first maximum, like the C loop


def conf_pct(p):
    return np.floor(np.clip(np.asarray(p, dtype=np.float64) * 100.0 + 0.5, 0.0, 100.0)).astype(int)


class Deployed:
    """Bit-exact Python reference of the generated csi_infer()."""

    def __init__(self, bundle, n_sc, n_classes):
        self.two_stage = bool(bundle["two_stage"])
        self.n_sc, self.n_classes = n_sc, n_classes
        m = bundle["model"]
        if self.two_stage:
            gate_classes = [bool(c) for c in m.gate_.classes_]
            if gate_classes != [False, True]:
                raise SystemExit(f"two-stage gate classes must be [False, True], got {gate_classes}")
            self.gate_static_idx = 1
            self.gate, self.static, self.moving = (Forest(m.gate_, "csi_gate"), Forest(m.static_, "csi_static"),
                                                   Forest(m.moving_, "csi_moving"))
            self.static_map = [int(c) for c in m.static_.classes_]
            self.moving_map = [int(c) for c in m.moving_.classes_]
            self.forests = [self.gate, self.static, self.moving]
        else:
            if [int(c) for c in m.classes_] != list(range(n_classes)):
                raise SystemExit(f"model classes {m.classes_} are not 0..{n_classes - 1}")
            self.single = Forest(m, "csi_forest")
            self.forests = [self.single]

    def infer(self, X):
        X = np.asarray(X, dtype=np.float32)
        r = np.arange(len(X))
        if not self.two_stage:
            p = self.single.predict_proba(X)
            cls = argmax_first(p)
            return cls, conf_pct(p[r, cls])
        dyn = X[:, self.n_sc:]
        pg = self.gate.predict_proba(dyn)
        g = argmax_first(pg)
        ps, pm = self.static.predict_proba(dyn), self.moving.predict_proba(X)
        ls, lm = argmax_first(ps), argmax_first(pm)
        st = g == self.gate_static_idx
        cls = np.where(st, np.array(self.static_map)[ls], np.array(self.moving_map)[lm])
        conf = pg[r, g] * np.where(st, ps[r, ls], pm[r, lm])
        return cls, conf_pct(conf)

    def flash_bytes(self):
        return sum(f.flash_bytes() for f in self.forests)

    def header(self):
        n = self.n_classes
        code = "".join(f.c_code() for f in self.forests)
        if not self.two_stage:
            body = f"""
/* Returns the global class index (or -1); *conf_pct = mean tree probability, %. */
static int32_t csi_infer(const float *f, int32_t n, uint8_t *conf_pct)
{{
    double p[{n}];
    int32_t best = csi_forest_predict(&CSI_FOREST, f, n, p);
    if (best < 0) {{ return -1; }}
    *conf_pct = csi_pct(p[best]);
    return best;
}}
"""
        else:
            ns, nm = len(self.static_map), len(self.moving_map)
            body = f"""
/* Two-stage: gate (static vs moving) on [var|slope|absdiff], then a specialist.
 * Returns the global class index (or -1); *conf_pct = gate p x specialist p, %. */
static const uint8_t CSI_STATIC_MAP[{ns}] = {{ {", ".join(map(str, self.static_map))} }};
static const uint8_t CSI_MOVING_MAP[{nm}] = {{ {", ".join(map(str, self.moving_map))} }};
#define CSI_GATE_STATIC_IDX {self.gate_static_idx}

static int32_t csi_infer(const float *f, int32_t n, uint8_t *conf_pct)
{{
    double pg[2], ps[{ns}], pm[{nm}];
    const float *dyn = f + CSI_N_SC;
    int32_t ndyn     = n - (int32_t)CSI_N_SC;
    int32_t g        = csi_forest_predict(&CSI_GATE, dyn, ndyn, pg);
    if (g < 0) {{ return -1; }}
    if (g == CSI_GATE_STATIC_IDX) {{
        int32_t k = csi_forest_predict(&CSI_STATIC, dyn, ndyn, ps);
        if (k < 0) {{ return -1; }}
        *conf_pct = csi_pct(pg[g] * ps[k]);
        return CSI_STATIC_MAP[k];
    }}
    int32_t k = csi_forest_predict(&CSI_MOVING, f, n, pm);
    if (k < 0) {{ return -1; }}
    *conf_pct = csi_pct(pg[g] * pm[k]);
    return CSI_MOVING_MAP[k];
}}
"""
        return ("/* Generated by csi_train_export.py - do not edit. Include from ONE .c file only. */\n"
                "#ifndef CSI_MODEL_H_\n#define CSI_MODEL_H_\n#include <stddef.h>\n#include <stdint.h>\n"
                "#include \"csi_model_config.h\"\n" + FOREST_C + code + body + "#endif\n")


# ---------- host C check -------------------------------------------------------

HARNESS = r"""
#include <stdio.h>
#include "csi_model_config.h"
#include "csi_model.h"
int main(int argc, char **argv)
{
    FILE *f = fopen(argv[1], "rb");
    float x[CSI_N_FEATURES];
    if (f == NULL) { return 2; }
    while (fread(x, sizeof(float), CSI_N_FEATURES, f) == CSI_N_FEATURES) {
        uint8_t c = 0;
        int32_t k = csi_infer(x, (int32_t)CSI_N_FEATURES, &c);
        printf("%d %u\n", (int)k, (unsigned)c);
    }
    fclose(f);
    return 0;
}
"""


def host_check(out, X, ref_cls, ref_conf):
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        raise SystemExit("no host C compiler (gcc/cc) found: cannot verify the generated C")
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        (td / "harness.c").write_text(HARNESS)
        exe = td / "harness"
        r = subprocess.run([cc, "-std=c99", "-O2", "-Wall", "-Werror=implicit-function-declaration",
                            "-Wno-unused-function", "-Wno-unused-variable", "-Wno-unused-parameter",
                            "-Wno-unused-but-set-variable", f"-I{out}", str(td / "harness.c"),
                            "-o", str(exe), "-lm"], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"generated C does not compile:\n{r.stderr[-3000:]}")
        X.astype(np.float32).tofile(td / "x.bin")
        r = subprocess.run([str(exe), str(td / "x.bin")], capture_output=True, text=True, check=True)
    got = np.array([[int(v) for v in line.split()] for line in r.stdout.splitlines()]).reshape(-1, 2)
    if len(got) != len(X):
        raise SystemExit(f"host check returned {len(got)} results for {len(X)} vectors")
    cls_ok = got[:, 0] == ref_cls
    conf_ok = np.abs(got[:, 1] - ref_conf) <= 1
    return {"n": int(len(X)), "class_agreement": float(cls_ok.mean()),
            "conf_agreement": float(conf_ok.mean()), "class_mismatches": int((~cls_ok).sum())}


def action_eval(classes, action, y, pred, sessions):
    """Replays the ON/OFF policy per held-out session (start OFF, windows in order).
    approaching should end ON; leaving/empty should end OFF; a session that never
    should switch ON (empty, idle-nearby, passing-by starting from OFF) counts a false ON."""
    names = ["hold", "on", "off"]
    want = np.array(action)[y]
    got = np.array(action)[pred]
    conf = {f"{names[a]}->{names[b]}": int(np.sum((want == a) & (got == b))) for a in range(3) for b in range(3)}
    per_class = {}
    for sid in dict.fromkeys(sessions):
        m = sessions == sid
        label = classes[int(y[m][0])]
        state, ever_on = 0, False
        for a in got[m]:
            state = 1 if a == ACTION_ON else 0 if a == ACTION_OFF else state
            ever_on |= state == 1
        st = per_class.setdefault(label, {"sessions": 0, "end_on": 0, "ever_on": 0})
        st["sessions"] += 1
        st["end_on"] += int(state == 1)
        st["ever_on"] += int(ever_on)
    return {"window_action_confusion": conf, "per_class_sessions": per_class}


# ---------- main -----------------------------------------------------------------

def parse_classes(text):
    return [c.strip() for c in (text or "").split(",") if c.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model.joblib from csi_train_eval.py --save-model")
    ap.add_argument("--clean-dir", required=True, help="clean dir the model was trained from")
    ap.add_argument("--out-dir", default="firmware_export")
    ap.add_argument("--on-classes", default="approaching", help="classes that switch the output ON")
    ap.add_argument("--off-classes", default="leaving,empty", help="classes that switch the output OFF")
    ap.add_argument("--empty-label", default=None,
                    help="class meaning 'vacant' for the diagnostic 'detected' flag "
                         "(default: the empty label used by csi_clean.py)")
    ap.add_argument("--no-ap-filter", action="store_true",
                    help="do not restrict the firmware to the training AP MAC (not recommended)")
    ap.add_argument("--selftest-per-class", type=int, default=2)
    ap.add_argument("--skip-replay", action="store_true",
                    help="skip the raw-stream replay through csi_presence.c (export is then "
                         "marked NOT deployable)")
    a = ap.parse_args()

    import joblib
    import sklearn

    clean, out = pathlib.Path(a.clean_dir), pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(a.model)

    # ---- compatibility --------------------------------------------------------
    pre = bundle.get("preprocess")
    pre_path = clean / "preprocess_config.json"
    if pre is None or pre.get("mode") != "records":
        raise SystemExit("model was not trained on csi_clean.py --window-mode records data; "
                         "time-window models cannot be reproduced by the firmware")
    if not pre_path.exists() or json.loads(pre_path.read_text()) != pre:
        raise SystemExit(f"{pre_path} differs from the preprocessing saved in the model")
    classes = list(bundle["classes"])
    kind = bundle["features"]
    n_sc, win_len = int(pre["n_sc"]), int(pre["win_len"])
    if bundle.get("window_len") != win_len or bundle.get("n_sc", n_sc) != n_sc:
        raise SystemExit("model window/subcarrier shape does not match preprocess_config.json")
    if classes != sorted(pre["classes"]):
        raise SystemExit(f"model classes {classes} != dataset classes {pre['classes']}")
    if bundle["two_stage"] and kind != "full":
        raise SystemExit("two-stage export requires 'full' features")
    if kind not in FEATURE_SET_ID:
        raise SystemExit(f"unsupported feature set {kind}")
    if len(pre["kept_sc"]) != n_sc or max(pre["kept_sc"]) >= pre["raw_n_sc"]:
        raise SystemExit("kept subcarrier list is inconsistent")
    if len(classes) > 255 or any(not c.replace("-", "").replace("_", "").isalnum() or len(c) > 32
                                 for c in classes):
        raise SystemExit("class names must be 1-32 chars of [A-Za-z0-9_-] (sent unescaped in JSON)")

    on_c, off_c = parse_classes(a.on_classes), parse_classes(a.off_classes)
    unknown = (set(on_c) | set(off_c)) - set(classes)
    if unknown or set(on_c) & set(off_c):
        raise SystemExit(f"bad --on/--off classes (unknown={unknown}, overlap={set(on_c) & set(off_c)})")
    action = [ACTION_ON if c in on_c else ACTION_OFF if c in off_c else ACTION_HOLD for c in classes]
    empty_label = a.empty_label or pre.get("empty_label", "empty")
    empty_idx = classes.index(empty_label) if empty_label in classes else 255

    va_w, va_y, va_s = load(clean / "val.npz")
    te_w, te_y, te_s = load(clean / "test.npz")
    for w in (va_w, te_w):
        if w.shape[1:] != (win_len, n_sc):
            raise SystemExit(f"window shape {w.shape[1:]} != ({win_len}, {n_sc})")
    idx = {c: i for i, c in enumerate(classes)}
    y_va, y_te = np.array([idx[c] for c in va_y]), np.array([idx[c] for c in te_y])
    X_va, X_te = featurize(va_w, kind), featurize(te_w, kind)
    n_feat = X_te.shape[1]
    if n_feat != BLOCKS[kind] * n_sc:
        raise SystemExit("feature count mismatch")

    # ---- convert ---------------------------------------------------------------
    dep = Deployed(bundle, n_sc, len(classes))
    report = {"classes": classes, "features": kind, "two_stage": dep.two_stage,
              "params": bundle.get("params"), "fit_on": bundle.get("fit_on"),
              "actions": {c: ["hold", "on", "off"][x] for c, x in zip(classes, action)},
              "model_table_bytes": dep.flash_bytes()}
    labels = list(range(len(classes)))
    for name, X, y, s in (("val", X_va, y_va, va_s), ("test", X_te, y_te, te_s)):
        pred, _ = dep.infer(X)
        sk = bundle["model"].predict(X)
        rep = classification_report(y, pred, labels=labels, target_names=classes, zero_division=0,
                                    output_dict=True)
        report[name] = {"accuracy": float(accuracy_score(y, pred)),
                        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                        "confusion": confusion_matrix(y, pred, labels=labels).tolist(),
                        "per_class": {c: rep[c] for c in classes},
                        "sklearn_agreement": float(np.mean(sk == pred)),
                        "actions": action_eval(classes, action, y, pred, s)}
        print(f"{name}: deployed accuracy {report[name]['accuracy']:.3f}, balanced "
              f"{report[name]['balanced_accuracy']:.3f}, agreement with sklearn "
              f"{report[name]['sklearn_agreement']:.3f}")

    # ---- headers -----------------------------------------------------------------
    mac = [int(x, 16) for x in pre["src_mac"].split(":")]
    names = ", ".join(f'"{c}"' for c in classes)
    cfg = f"""/* Generated by csi_train_export.py - do not edit. Must match the trained model. */
#ifndef CSI_MODEL_CONFIG_H_
#define CSI_MODEL_CONFIG_H_
#include <stdint.h>

#define CSI_RAW_N_SC     {pre['raw_n_sc']}U   /* subcarriers in one raw record */
#define CSI_N_SC         {n_sc}U   /* subcarriers kept after null-subcarrier mask */
#define CSI_WIN_LEN      {win_len}U   /* records per window */
#define CSI_HOP          {pre['hop']}U   /* new records between two classifications */
#define CSI_FEATURE_SET  {FEATURE_SET_ID[kind]}   /* 0 [mean|var], 1 [var|slope|absdiff], 2 [mean|var|slope|absdiff] */
#define CSI_N_FEATURES   {n_feat}U
#define CSI_TWO_STAGE    {1 if dep.two_stage else 0}
#define CSI_N_CLASSES    {len(classes)}U
#define CSI_EMPTY_CLASS  {empty_idx}U   /* class index meaning "vacant" (255 = none) */
#define CSI_STALL_US     {int(pre['stall_us'])}ULL   /* TSF jump (us) that restarts the window */
#define CSI_IQ_SCALE     {pre['iq_scale']:.9e}f
#define CSI_CHANNEL      {int(pre['channel'])}U   /* header byte 37 of accepted records */
#define CSI_HAVE_AP_MAC  {0 if a.no_ap_filter else 1}

/* Control-UART policy per class: 0 hold current state, 1 ON, 2 OFF */
#define CSI_ACTION_HOLD 0U
#define CSI_ACTION_ON   1U
#define CSI_ACTION_OFF  2U

static const char *const CSI_CLASS_NAME[CSI_N_CLASSES] = {{ {names} }};
{c_int_array("uint8_t", "CSI_CLASS_ACTION", action)}{c_int_array("uint8_t", "CSI_AP_MAC", mac)}
/* index of each kept subcarrier inside the raw record */
{c_int_array("uint16_t", "CSI_KEPT_SC", pre['kept_sc'])}
#endif
"""
    (out / "csi_model_config.h").write_text(cfg)
    (out / "csi_model.h").write_text(dep.header())

    # ---- self-test vectors: first windows of each class in the test set --------
    picks = []
    for c in labels:
        picks += [int(i) for i in np.where(y_te == c)[0][:a.selftest_per_class]]
    st_win = te_w[picks]
    st_feat = featurize(st_win, kind)
    st_cls, st_conf = dep.infer(st_feat)
    st = ("/* Generated by csi_train_export.py - do not edit. Windows are chronological. */\n"
          "#ifndef CSI_SELFTEST_H_\n#define CSI_SELFTEST_H_\n#include <stdint.h>\n"
          f"#define CSI_SELFTEST_N {len(picks)}\n"
          + c_float_array("CSI_SELFTEST_WIN", st_win)
          + c_float_array("CSI_SELFTEST_FEAT", st_feat)
          + c_int_array("uint8_t", "CSI_SELFTEST_CLASS", st_cls)
          + "#endif\n")
    (out / "csi_selftest.h").write_text(st)

    # ---- parity: compiled C vs Python reference ----------------------------------
    X_all = np.concatenate([X_va, X_te])
    ref_cls, ref_conf = dep.infer(X_all)
    hc = host_check(out, X_all, ref_cls, ref_conf)
    report["host_c_check"] = hc
    print(f"host C check: {hc['n']} vectors, class agreement {hc['class_agreement']:.4f}, "
          f"confidence agreement {hc['conf_agreement']:.4f}")
    if hc["class_mismatches"] or hc["conf_agreement"] < 1.0:
        raise SystemExit("generated C disagrees with the Python reference: NOT deployable")

    # ---- control schema with this model's classes ----------------------------------
    schema = json.loads((REPO / "schemas" / "presense-control.v1.schema.json").read_text())
    schema["properties"]["class"] = {"enum": classes + [None],
                                     "description": schema["properties"]["class"]["description"]}
    (out / "presense-control.v1.schema.json").write_text(json.dumps(schema, indent=2) + "\n")

    report["config"] = {k: pre[k] for k in ("win_len", "hop", "n_sc", "raw_n_sc", "stall_us", "channel",
                                            "src_mac", "measured_rate_hz", "window_span_s")}
    report["config"]["n_features"] = n_feat
    report["provenance"] = {
        "model": str(a.model), "model_sha256": sha256(a.model),
        "preprocess_sha256": sha256(pre_path), "generator": "csi_forest tables v1",
        "sklearn": sklearn.__version__, "numpy": np.__version__, "python": platform.python_version(),
        "headers": {f: sha256(out / f) for f in ("csi_model.h", "csi_model_config.h", "csi_selftest.h")},
    }

    deployable = True
    if a.skip_replay:
        print("raw-stream replay skipped: export is NOT deployable")
        deployable = False
    else:
        import csi_replay_parity
        rp = csi_replay_parity.run(out, clean, bundle, dep, split="test")
        report["replay_parity"] = rp
        if not rp["ok"]:
            (out / "export_report.json").write_text(json.dumps({**report, "deployable": False}, indent=2))
            raise SystemExit("raw-stream replay through csi_presence.c disagrees with Python: NOT deployable")
    report["deployable"] = deployable
    (out / "export_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote csi_model.h, csi_model_config.h, csi_selftest.h, "
          f"presense-control.v1.schema.json, export_report.json to {out}/ (deployable={deployable})")


if __name__ == "__main__":
    main()
