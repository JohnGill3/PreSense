#!/usr/bin/env python3
"""csi_plot.py - plot raw CSI signals from csi_logger.py CSVs for visual comparison.

Decoding matches csi_clean.py (Phases 1/3/5, scale_shift default 0; no masking
or Hampel filtering here - this is a raw/diagnostic view, not the training
pipeline). Compare e.g. idle_nearby vs empty before committing to full
collection.

Needs: pip install numpy pandas matplotlib

Usage examples:
  # Heatmap (time x subcarrier) + mean-amplitude-over-time, one file each
  python csi_plot.py --files data/empty_1.csv data/idle_nearby_1.csv

  # Overlay mean amplitude over time for several sessions of each class
  python csi_plot.py --files data/empty_*.csv data/idle_nearby_*.csv \
      --mode overlay --labels empty,empty,idle_nearby,idle_nearby

  # Per-subcarrier variance comparison (bar chart) between two classes
  python csi_plot.py --files data/empty_1.csv data/idle_nearby_1.csv --mode variance

  # RSSI over time, read straight from the CSV columns (no decoding needed)
  python csi_plot.py --files data/empty_1.csv data/idle_nearby_1.csv --mode rssi
"""
import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HDR, TAIL = 48, 4


def decode_session(csv_path, scale_shift):
    """Return (amp[n_records, n_sc], tsf[n_records], rssi[n_records]) for one CSV,
    decoding only rows that pass the same header checks as csi_clean.py Phase 1-2."""
    df = pd.read_csv(csv_path)
    amps, tsfs, rssis = [], [], []
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
        rssis.append(row["rssi_a"])
    if not amps:
        raise SystemExit(f"{csv_path}: no valid records decoded (check --scale-shift, or that "
                         f"the file has the expected header columns)")
    n_sc = {len(a) for a in amps}
    if len(n_sc) != 1:
        raise SystemExit(f"{csv_path}: inconsistent subcarrier counts {n_sc}; "
                         f"this session mixed bandwidths/packet types")
    return np.stack(amps), np.array(tsfs, dtype=np.float64), np.array(rssis, dtype=np.float64)


def time_axis(tsf):
    t = (tsf - tsf[0]) / 1e6  # assumes TSF in microseconds, consistent with csi_clean.py
    return t


def normalize_rows(amp):
    return amp / (amp.mean(axis=1, keepdims=True) + 1e-9)


def mode_heatmap(files, labels, scale_shift, normalize, out):
    n = len(files)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), squeeze=False)
    for ax, f, lbl in zip(axes[:, 0], files, labels):
        amp, tsf, _ = decode_session(f, scale_shift)
        if normalize:
            amp = normalize_rows(amp)
        t = time_axis(tsf)
        im = ax.imshow(amp.T, aspect="auto", origin="lower",
                       extent=[t[0], t[-1], 0, amp.shape[1]], cmap="viridis")
        ax.set_title(f"{lbl}  ({pathlib.Path(f).name})  n={amp.shape[0]} records")
        ax.set_xlabel("time (s, from TSF)")
        ax.set_ylabel("subcarrier index")
        fig.colorbar(im, ax=ax, label="amplitude" + (" (row-normalized)" if normalize else ""))
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


def mode_overlay(files, labels, scale_shift, normalize, out):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    label_color = {}
    for f, lbl in zip(files, labels):
        amp, tsf, _ = decode_session(f, scale_shift)
        if normalize:
            amp = normalize_rows(amp)
        t = time_axis(tsf)
        mean_amp = amp.mean(axis=1)
        if lbl not in label_color:
            label_color[lbl] = colors[len(label_color) % len(colors)]
        ax.plot(t, mean_amp, color=label_color[lbl], alpha=0.7,
               label=lbl if lbl not in ax.get_legend_handles_labels()[1] else None)
    ax.set_xlabel("time (s, from TSF)")
    ax.set_ylabel("mean amplitude across subcarriers" + (" (row-normalized)" if normalize else ""))
    ax.set_title("Mean CSI amplitude over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


def mode_variance(files, labels, scale_shift, normalize, out):
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.8 / max(len(files), 1)
    for i, (f, lbl) in enumerate(zip(files, labels)):
        amp, _, _ = decode_session(f, scale_shift)
        if normalize:
            amp = normalize_rows(amp)
        var = amp.var(axis=0)
        x = np.arange(len(var)) + i * width
        ax.bar(x, var, width=width, label=f"{lbl} ({pathlib.Path(f).name})", alpha=0.8)
    ax.set_xlabel("subcarrier index")
    ax.set_ylabel("variance over time" + (" (row-normalized)" if normalize else ""))
    ax.set_title("Per-subcarrier variance (low + flat across files = likely null/pilot subcarrier)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


def mode_rssi(files, labels, scale_shift, normalize, out):
    fig, ax = plt.subplots(figsize=(10, 4))
    for f, lbl in zip(files, labels):
        df = pd.read_csv(f)
        t = time_axis(df["tsf"].to_numpy(dtype=np.float64))
        ax.plot(t, df["rssi_a"], marker=".", linestyle="-", alpha=0.7,
               label=f"{lbl} ({pathlib.Path(f).name})")
    ax.set_xlabel("time (s, from TSF)")
    ax.set_ylabel("RSSI chain A (dBm)")
    ax.set_title("RSSI over time (straight from CSV columns, no decoding)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True, help="CSV files from csi_logger.py")
    ap.add_argument("--labels", default=None,
                    help="comma-separated label per file (for legend grouping in overlay/variance); "
                         "defaults to each file's own name")
    ap.add_argument("--mode", choices=["heatmap", "overlay", "variance", "rssi"], default="heatmap",
                    help="heatmap: time x subcarrier per file. overlay: mean amplitude over time, "
                         "all files on one plot, colored by label. variance: per-subcarrier variance "
                         "bars, grouped by file. rssi: RSSI over time from the CSV columns directly.")
    ap.add_argument("--scale-shift", type=int, default=0, help="see csi_clean.py; default 0")
    ap.add_argument("--normalize", action="store_true",
                    help="apply per-record normalization (divide by row mean) before plotting, "
                         "matching csi_clean.py Phase 5 - use this to see shape, not absolute level")
    ap.add_argument("--out", default="csi_plot.png")
    args = ap.parse_args()

    files = [pathlib.Path(f) for f in args.files]
    for f in files:
        if not f.exists():
            raise SystemExit(f"not found: {f}")
    labels = args.labels.split(",") if args.labels else [f.stem for f in files]
    if len(labels) != len(files):
        raise SystemExit(f"--labels has {len(labels)} entries but {len(files)} files were given")

    fn = {"heatmap": mode_heatmap, "overlay": mode_overlay,
         "variance": mode_variance, "rssi": mode_rssi}[args.mode]
    fn(files, labels, args.scale_shift, args.normalize, args.out)


if __name__ == "__main__":
    main()
