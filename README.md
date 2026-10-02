# PreSense

Wi-Fi CSI presence/motion sensing on the NXP FRDM-RW612. The board captures
raw Channel State Information (CSI) from Wi-Fi frames, a Python pipeline turns
recorded sessions into a trained classifier, and the classifier runs on the
RW612's Cortex-M33 (via [emlearn](https://github.com/emlearn/emlearn)),
reporting results over the console and/or a UART pin as NDJSON.

This README has not been validated end to end on hardware. Known gaps are
flagged inline — check them before trusting a step blindly.

## Repository layout

```
PreSense/
├── application/         # Firmware: NXP's dm-motion-detection-using-wifi-csi-on-rw61x
│                         # demo, revised with the CSI logger and presence classifier
│   └── source/
│       ├── csi_raw_log.c/.h       # Driver callback -> queue -> logger task
│       ├── csi_presence.c/.h      # Preprocessing, featurize, emlearn inference
│       ├── csi_uart_out.c/.h      # NDJSON formatting + UART/console dispatch
│       ├── csi_model.h            # Generated: emlearn forest (not committed, see below)
│       ├── csi_model_config.h     # Generated: window/subcarrier/class constants
│       └── csi_selftest.h         # Generated: known-good vectors for on-device self-test
├── sdk/                  # MCUXpresso SDK v25.09.00 for FRDM-RW612
├── scripts/               # Python pipeline (this repo's own code)
│   ├── csi_logger.py       # Step 2: record one labelled session from the board
│   ├── csi_clean.py        # Step 3: validate, decode, window raw sessions
│   ├── csi_split.py        # Step 4: split windows into train/val/test BY SESSION
│   ├── csi_train_eval.py   # Step 5a: train + evaluate only, with logging
│   └── csi_train_export.py # Step 5b: train + evaluate + export C headers
├── data/                  # Recorded sessions (csi_logger.py output) — gitignore raw data
├── clean/                 # csi_clean.py / csi_split.py output — gitignore
├── firmware_export/       # csi_train_export.py output — copy into application/source/
└── README.md
```

**Suggested `.gitignore` entries:** `data/`, `clean/`, `firmware_export/`,
`results/` — these are regenerated, not hand-authored, and recorded CSI
sessions can be large. Commit a small sample session if you want the pipeline
runnable out of the box.

## How the pieces fit together

```
  [FRDM-RW612 + AP]                    [PC / dev machine]
        |                                     |
  csi_logger.py  --serial-->  data/*.csv + *.json   (one pair per recorded session)
                                     |
                               csi_clean.py           (validate, decode, window)
                                     |
                              clean/*.npz, all_sessions.npz
                                     |
                               csi_split.py            (split BY SESSION)
                                     |
                       clean/{train,val,test}.npz
                                     |
                     +---------------+----------------+
                     |                                |
            csi_train_eval.py                 csi_train_export.py
         (metrics + logs only)         (metrics + firmware_export/*.h)
                                                        |
                                    copy csi_model.h, csi_model_config.h,
                                    csi_selftest.h into application/source/
                                                        |
                                           build + flash application/
                                                        |
                                    board classifies live CSI, reports NDJSON
                                      over console and/or a UART pin
```

## Prerequisites

- **Hardware:** FRDM-RW612, USB-C cable, an access point you control (fixed
  SSID/password, fixed 5 GHz channel — channel hopping or "Auto" mode makes
  session data inconsistent).
- **Firmware toolchain:** MCUXpresso IDE (or MCUXpresso for VS Code) with the
  MCUXpresso Installer's Python/west, GNU Arm toolchain and J-Link/LinkServer
  components installed.
- **Python:** 3.9+ recommended.
  ```bash
  pip install pyserial numpy pandas scipy scikit-learn emlearn joblib
  ```

## 1. Build and flash the data-collection firmware

Use `application/`, built from NXP's
[dm-motion-detection-using-wifi-csi-on-rw61x](https://github.com/nxp-appcodehub/dm-motion-detection-using-wifi-csi-on-rw61x)
demo with `csi_raw_log.c/.h` added and registered via
`wlan_register_csi_user_callback()`.

For this stage, build with:
- `CSI_LOG_RAW=1` (print raw CSI hex — this is what `csi_logger.py` parses)
- `CSI_PRESENCE_ENABLE=0` (classifier not built yet)

Flash it (Debug or GUI Flash Tool in MCUXpresso IDE), then bring up the CSI
capture over the serial console (115200 8-N-1) once per boot — the Wi-Fi
profile does not survive a reset:

```
wlan-add test ssid <SSID> wpa2 psk <PASSWORD>
wlan-connect test
wlan-set-csi-param-header sta 1 66051 66051 170 1 40 0 0
wlan-set-csi-filter add <AP_MAC> 255 08 0
wlan-csi-cfg
```

**Unverified:** the `channel` field's exact encoding in
`wlan-set-csi-param-header`, and whether `csi_monitor_enable`/`ra4us` monitor
mode works while also connected via `wlan-connect`. Check AN14281 §4.1 / §7
and `wlan_tests.c` before relying on either.

## 2. Record labelled sessions — `csi_logger.py`

```bash
python scripts/csi_logger.py \
  --port COM5 --label empty --duration 180 --start-delay 15 \
  --cmd "wlan-add test ssid <SSID> wpa2 psk <PASSWORD>" \
  --cmd "wlan-connect test" \
  --cmd "wlan-set-csi-param-header sta 1 66051 66051 170 1 40 0 0" \
  --cmd "wlan-set-csi-filter add <AP_MAC> 255 08 0" \
  --cmd "wlan-csi-cfg" \
  --src-mac <AP_MAC> \
  --stop-cmd "wlan-set-csi-param-header sta 2 66051 66051 170 1 40 0 0" \
  --stop-cmd "wlan-csi-cfg" \
  --out data
```

Close any terminal program first — the port can only be open in one place.
Record multiple sessions per label (`empty`, `walking`, `seated`, etc.), on
different days where possible, so `csi_split.py` has independent sessions to
hold out. Each run writes `data/<label>_<timestamp>.csv` (raw records, kept in
full as hex) and a matching `.json` (metadata: duration, record count, rate,
firmware warnings).

Key flags: `--port`, `--baud` (default 115200, must match the board),
`--label`, `--duration`, `--start-delay` (time to leave the room),
`--src-mac` (keep only this AP's records). Run `--help` for the rest.

## 3. Clean and window — `csi_clean.py`

```bash
python scripts/csi_clean.py \
  --data-dir data --out-dir clean \
  --empty-label empty --src-mac <AP_MAC> \
  --window-sec 3 --overlap 0.5 \
  --hampel-window 1
```

Reads every `data/*.csv` with a matching `.json` (mixed labels in one
directory is expected — the label comes from each session's own `.json`, not
from a command-line flag). Per session: validates record headers, filters to
a consistent MAC/bandwidth/channel, decodes I/Q into per-subcarrier amplitude,
masks null/pilot subcarriers (using the longest `--empty-label` session as the
reference), normalizes each record, flags TSF stalls, optionally runs a
Hampel outlier filter, and slices into overlapping windows.

**`--hampel-window 1` disables the Hampel filter.** The firmware's
`csi_presence.c` does not implement it, so leave it off for data that will
train a model you intend to deploy — otherwise training and on-device
preprocessing diverge.

Outputs: `clean/<session>.npz` per session, `clean/all_sessions.npz`
combined, `clean/null_subcarrier_mask.npy`, and `clean/report.json` with
per-phase drop counts — check this after every run.

`--scale-shift` defaults to 0 (see script docstring: AN14281 does not
document a fixed-point exponent for this field the way earlier drafts of this
project assumed; per-record normalization absorbs a constant scale anyway).

## 4. Split by session — `csi_split.py`

```bash
python scripts/csi_split.py \
  --input clean/all_sessions.npz --out-dir clean \
  --val-frac 0.15 --test-frac 0.15 --seed 42
```

Assigns **whole sessions** to train/val/test, never individual windows —
windows from the same session are correlated (same room, same AP position,
same RF conditions), so a per-window split would leak information and
overstate accuracy. Writes `clean/{train,val,test}.npz` and
`clean/split_summary.json`, and prints a warning if any session leaked across
splits (should never happen) or if a class has too few sessions to populate
all three splits — record more sessions for that class if so.

## 5a. Train and evaluate only — `csi_train_eval.py`

```bash
python scripts/csi_train_eval.py --clean-dir clean --out-dir results
```

Trains a `RandomForestClassifier` on `[mean | variance]` per-subcarrier
features, no firmware export. Logs to both the console and
`results/train_eval.log`: per-split class counts, training time, train
accuracy, per-class precision/recall/F1 and confusion matrix for val and
test, per-session accuracy (worst session called out), the train/test
accuracy gap (warns above 0.15 — likely overfitting or too few sessions), and
feature importance. Writes `results/train_report.json`. Use `--save-model` to
also write `results/model.joblib` for later export without retraining; use
this stage to iterate on `--trees`/`--depth` before committing to export.

## 5b. Train, evaluate and export for firmware — `csi_train_export.py`

```bash
python scripts/csi_train_export.py \
  --clean-dir clean --out-dir firmware_export \
  --window-sec 3 --overlap 0.5 \
  --ap-mac <AP_MAC> --empty-label empty
```

**`--window-sec` and `--overlap` must match the values given to
`csi_clean.py`.** The script does not read them back from `clean/`; a mismatch
produces firmware constants that silently disagree with the trained model.

Same training/evaluation as `csi_train_eval.py`, plus:
- Converts the forest to C via `emlearn.convert(..., method='inline')`.
- Checks Python-vs-generated-C agreement on the test set where the installed
  emlearn version supports it.
- Emits the window length, hop, kept-subcarrier list, I/Q scale, stall
  threshold, class names and the "empty" class index as `#define`s / arrays —
  values the firmware needs but has no other way to know.
- Picks one test window per class and exports it (raw window, expected
  features, expected class) for an on-device self-test.

Outputs in `firmware_export/`:

| File | Purpose |
| --- | --- |
| `csi_model.h` | The trained forest, as C (`csi_model_predict()`) |
| `csi_model_config.h` | Window/subcarrier/class constants matching the trained model |
| `csi_selftest.h` | Known-input/known-output vectors for `CSI_PRESENCE_SELFTEST` |
| `train_report.json` | Same metrics as `csi_train_eval.py`, plus the config and C-agreement check |

## 6. Integrate the exported model into the firmware

1. Copy `firmware_export/csi_model.h`, `csi_model_config.h` and
   `csi_selftest.h` into `application/source/`, alongside the existing
   `csi_presence.c/.h`, `csi_uart_out.c/.h` and `csi_raw_log.c/.h`.
2. In **Project Properties > C/C++ Build > Settings > Preprocessor**, set:
   - `CSI_PRESENCE_ENABLE=1` — feed records into the classifier
   - `CSI_PRESENCE_SELFTEST=1` — run the self-test at boot (first build only)
   - `CSI_LOG_RAW=0` — stop hex-dumping raw CSI once the classifier is live,
     or it floods the console/UART
   - Optionally `CSI_PRES_UART_BASE=USARTn` and `CSI_PRES_UART_CLK_HZ=...` to
     also send NDJSON out a physical UART pin, not just the console
3. Check `csi_model.h`'s generated `csi_model_predict()` signature. If it
   differs from `const float *, size_t` (emlearn's generated signature has
   varied by version), adjust the call in `csi_presence.c`'s `predict()`.
4. Build and flash. At boot, with `CSI_PRESENCE_SELFTEST=1`, the console
   should print `PRES_SELFTEST,i,OK,...` for each class — this confirms the
   C `featurize()` and the generated model agree with Python on identical
   inputs. It does **not** confirm I/Q decoding from a live record; verify
   that separately by replaying a logged session's records through the board
   and comparing its classifications to `csi_train_export.py`'s predictions
   for the same session.
5. Live output is one NDJSON line per classification:
   ```json
   {"version":1,"event":"presence","detected":true,"confidence":0.67,
    "model_latency_ms":0,"model_latency_us":180,"rssi":-58,"seq":1042,
    "up_ms":123456,"class":"walking","crc":41234}
   ```
   plus periodic `heartbeat` frames and `error` frames (`no_csi`, `model`) —
   see `csi_uart_out.h` for the full set of `CSI_OUT_*` build options
   (queue depth, heartbeat interval, CRC on/off, console mirroring).

## Firmware module reference

| Module | Role |
| --- | --- |
| `csi_raw_log.c/.h` | Registered as the Wi-Fi driver's CSI callback. Validates and copies each record into a pool, then a low-priority task prints raw hex (`CSI_LOG_RAW`) and/or feeds `csi_presence_feed()` (`CSI_PRESENCE_ENABLE`). Keeps the driver callback itself fast. |
| `csi_presence.c/.h` | Reimplements `csi_clean.py`'s Phases 1, 3–6 and 8 in C: validate, decode I/Q → amplitude, mask non-data subcarriers, per-record normalize, drop windows spanning a TSF stall, slide a window and classify every `CSI_HOP` records, majority-vote over `CSI_VOTE_N` results. **Does not implement Phase 7 (Hampel filtering)** — train with `--hampel-window 1` to match. |
| `csi_uart_out.c/.h` | Formats classification results, heartbeats and errors as NDJSON and sends them to the console and/or a UART pin from a dedicated low-priority task, so formatting/transmit time never blocks CSI processing. |

## Known gaps / things to verify before relying on this project

- **CSI fixed-point scale:** no confirmed AN14281 field for `--scale-shift`;
  currently defaults to 0 and relies on per-record normalization.
- **TSF unit:** assumed microseconds throughout (`csi_clean.py` and
  `csi_presence.c`'s `CSI_STALL_US`) — confirm against AN14281.
- **`wlan-set-csi-param-header` channel field encoding:** not confirmed.
- **RW612 as SoftAP generating CSI:** AN14281 §7.7 suggests it's possible;
  untested here. Start with STA mode (documented, used throughout this repo).
- **emlearn `csi_model_predict()` signature:** varies by emlearn version —
  check the generated header.
- **UART pin / FLEXCOMM mapping, driver header names, DWT cycle counter
  availability:** check the FRDM-RW612 schematic/user manual and SDK headers
  for your exact toolchain version.
- **Header I/O voltage:** confirm 3.3 V on the pins you wire to an external
  host before connecting a different board.

## License / attribution

`application/` is derived from NXP's
[dm-motion-detection-using-wifi-csi-on-rw61x](https://github.com/nxp-appcodehub/dm-motion-detection-using-wifi-csi-on-rw61x)
(NXP Application Code Hub). Check that repo's license before redistributing.
`sdk/` is NXP's MCUXpresso SDK — redistribute per its license, or `.gitignore`
it and document the exact SDK version/build-ID instead of committing it.
