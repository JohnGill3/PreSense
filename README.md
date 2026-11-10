# PreSense

Wi-Fi CSI presence/motion sensing on the NXP FRDM-RW612. The board captures
raw Channel State Information (CSI) from Wi-Fi frames, a Python pipeline turns
recorded sessions into a trained classifier, and the classifier runs on the
RW612's Cortex-M33 as exported C forest tables, driving an ON/OFF
`control` JSON stream on a dedicated UART (diagnostic NDJSON stays on the console).

This README has not been validated end to end on hardware. Known gaps are
flagged inline — check them before trusting a step blindly.

## Repository layout

```
PreSense/
├── csi_raw_log.c/.h        # Firmware: driver callback -> record pool -> worker task
├── csi_presence.c/.h       # Firmware: preprocessing, features, exported forest inference
├── csi_control.c/.h        # Firmware: ON/OFF policy, control JSON formatter, CRC-16
├── csi_uart_out.c/.h       # Firmware: output task (control UART + debug console NDJSON)
├── data-collection-scripts/
│   ├── csi_logger.py       # Step 2: record one labelled session from the board
│   └── data/               # Recorded sessions: <label>_<date>_<time>.csv + .json (gitignored)
├── model-training-scripts/
│   ├── csi_clean.py        # Step 3: validate, decode, window (time or records mode)
│   ├── csi_split.py        # Step 4: split windows into train/val/test BY SESSION
│   ├── csi_train_eval.py   # Step 5a: train + evaluate (+ --save-model)
│   ├── csi_train_export.py # Step 5b: export a saved model to C headers, with parity checks
│   ├── csi_replay_parity.py# Raw CSV -> real csi_presence.c replay check (host gcc)
│   └── host_stubs/         # Minimal FreeRTOS/SDK headers for host compilation
├── schemas/presense-control.v1.schema.json  # UART control message schema
├── receiver/presense_control_rx.py          # Reference receiver / parser specification
├── tests/                  # pytest: control policy, UART task, receiver, records mode
├── artifacts/              # Pipeline outputs (clean/, results/, firmware_export/) — gitignored
└── README.md
```

The firmware sources are fragments for the NXP
dm-motion-detection-using-wifi-csi-on-rw61x demo (SDK 25.12, MCUXpresso IDE);
the demo project and SDK themselves are not in this repository.

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
  pip install pyserial numpy pandas scipy scikit-learn joblib jsonschema pytest
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
python data-collection-scripts/csi_logger.py \
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

For higher-rate capture, `data-collection-scripts/csi_logger.py` supports
optional Linux `--ping-target <RW612_IP> --ping-rate 100` traffic and
`--target-csi-rate 100` diagnostics. See
[HIGH_RATE_CAPTURE.md](data-collection-scripts/HIGH_RATE_CAPTURE.md) for
data-frame filtering, serial bandwidth limits, manual iperf options, and
training-rate compatibility. Generating 100 packets/s does not guarantee
100 CSI records/s; 115200-baud hex logging cannot sustain that target.

## 3. Clean and window — `csi_clean.py`

```bash
python model-training-scripts/csi_clean.py \
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

Windows holding fewer than `--min-fill` (default 0.7) of the expected
records are dropped instead of padded, and a per-label session summary warns
about labels with fewer than `--min-sessions` (default 8) sessions.

Outputs: `clean/<session>.npz` per session, `clean/all_sessions.npz`
combined, `clean/null_subcarrier_mask.npy`, and `clean/report.json` with
per-phase drop counts — check this after every run.

`--scale-shift` defaults to 0 (see script docstring: AN14281 does not
document a fixed-point exponent for this field the way earlier drafts of this
project assumed; per-record normalization absorbs a constant scale anyway).

## 4. Split by session — `csi_split.py`

```bash
python model-training-scripts/csi_split.py \
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
python model-training-scripts/csi_train_eval.py --clean-dir clean --out-dir results
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

For the 5-class motion model, `--two-stage` (static-vs-moving gate plus two
specialists), `--cv` (leave-one-session-out), `--grid` and `--vote` give the
best and most trustworthy accuracy; the default feature set is now `dynamic`
(`--features meanvar` reproduces the firmware-compatible features). See
[TRAINING_IMPROVEMENTS.md](TRAINING_IMPROVEMENTS.md) for measured results,
the recommended commands, and what the firmware needs before deploying them.

## 5b. Deployment pipeline: records-mode clean → train → exact export

The firmware classifies a sliding window of `CSI_WIN_LEN` accepted records
every `CSI_HOP` records. A deployable model must therefore be trained on
**records-mode** windows built with the firmware's own rules. Time-window
mode (sections 3–5a) is still available for experiments, but it cannot be
exported. All scripts are in `model-training-scripts/`. Use a separate virtual
environment:

```bash
python3 -m venv .venv-deploy
.venv-deploy/bin/pip install -r requirements.txt   # numpy pandas scikit-learn joblib jsonschema pyserial pytest
```

**Baseline:** mean/variance features, 30-record windows (about 3.1 s at the
measured ~9.76 records/s), hop 15.

```bash
P=.venv-deploy/bin/python; S=model-training-scripts; B=artifacts/presense/baseline
$P $S/csi_clean.py --data-dir data-collection-scripts/data --out-dir $B/clean \
  --window-mode records --win-len 30 --hop 15 --hampel-window 1 \
  --empty-label empty --val-frac 0.15 --test-frac 0.15 --seed 42
$P $S/csi_split.py --input $B/clean/all_sessions.npz --out-dir $B/clean \
  --assignment $B/clean/split_manifest.json
$P $S/csi_train_eval.py --clean-dir $B/clean --out-dir $B/results \
  --features meanvar --model rf --trees 10 --depth 10 --seed 42 --vote 1 --save-model
$P $S/csi_train_export.py --model $B/results/model.joblib \
  --clean-dir $B/clean --out-dir $B/firmware_export
```

**Improved:** `[mean | variance | slope | mean absolute adjacent difference]`
features, 60-record windows (about 6.1 s), hop 15. It uses the **same session
split** as the baseline. The model settings were chosen on validation only.

```bash
I=artifacts/presense/improved
$P $S/csi_clean.py --data-dir data-collection-scripts/data --out-dir $I/clean \
  --window-mode records --win-len 60 --hop 15 --hampel-window 1 \
  --empty-label empty --split-manifest $B/clean/split_manifest.json
$P $S/csi_split.py --input $I/clean/all_sessions.npz --out-dir $I/clean \
  --assignment $I/clean/split_manifest.json
$P $S/csi_train_eval.py --clean-dir $I/clean --out-dir $I/results \
  --features full --model rf --trees 30 --depth 8 --seed 42 --vote 1 --save-model
$P $S/csi_train_export.py --model $I/results/model.joblib \
  --clean-dir $I/clean --out-dir $I/firmware_export
```

What records mode guarantees (`csi_clean.py --window-mode records`):

- **Same record rules as the firmware:**
  - Records are validated, then filtered by AP MAC, channel (header byte 37)
    and payload size.
  - Amplitudes are normalized per record.
  - A TSF gap larger than `stall_us`, or a TSF reset, restarts the window.
  - The first window is full; after that, one window every `hop` records.
  - There is no padding.
- **The split is fixed before fitting:**
  - Sessions are assigned to train/val/test first.
  - The null-subcarrier mask is fitted only on the longest **training**
    `empty` session.
  - `preprocess_config.json` records everything the exporter needs (mask,
    filters, `stall_us`, window layout, raw-file hashes, split).
  - The exporter refuses missing or conflicting metadata instead of guessing.

`csi_train_export.py` exports the **saved** model; it never retrains. It
supports `meanvar`, `dynamic` and `full` features, and single or two-stage
forests. Its own table-driven C forest makes the same decisions as sklearn
(thresholds are rounded down to float32), so emlearn is no longer needed.

The export is marked `deployable` only if all of these blocking checks pass:

1. The generated C forest, compiled on the host, agrees with Python on every
   val and test vector (class and confidence).
2. **Raw replay** (`csi_replay_parity.py`): the raw CSV records of each test
   session are streamed through the real `csi_presence.c`. Window boundaries,
   features (relative error ≤ 1e-4), classes, actions and the boot self-test
   must all match Python.

Replay can also be run on its own for any split:

```bash
$P $S/csi_replay_parity.py --model $I/results/model.joblib --clean-dir $I/clean \
  --export-dir $I/firmware_export --split train
```

Outputs in `firmware_export/`:

| File | Purpose |
| --- | --- |
| `csi_model.h` | Forest node tables plus `csi_infer()`. Include it only from `csi_presence.c`. |
| `csi_model_config.h` | Window, hop, feature set, mask, filters, stall, class names and `CSI_CLASS_ACTION[]` |
| `csi_selftest.h` | Feature vectors and expected outputs for `CSI_PRESENCE_SELFTEST` |
| `presense-control.v1.schema.json` | Control-message JSON Schema, with this model's class names |
| `export_report.json` | Metrics, action evaluation, parity results, table size and provenance (no Wi-Fi credentials) |

`--on-classes` and `--off-classes` change the action policy. The defaults are:
`approaching` → ON; `leaving` and `empty` → OFF; anything else → hold.

### Results on held-out sessions (same split for both models)

| | Baseline (meanvar, 30 rec) | Improved (full, 60 rec) |
| --- | --- | --- |
| Test accuracy / balanced accuracy | 0.685 / 0.648 | 0.799 / 0.744 |
| Validation balanced accuracy | 0.471 | 0.572 |
| Forest table size (flash) | ~26 KB | ~53 KB |
| Ring buffer (RAM) | 30×47×4 = 5.6 KB | 60×47×4 = 11.3 KB |
| Test approaching sessions ending ON | 4/5 | 3/4 |
| Test leaving sessions wrongly ending ON | 4/5 | 2/4 |
| Test passing-by / empty / idle sessions ever ON | 0 / 0 / 0 | 0 / 0 / 0 |

Both models are **weak**. There are only 14–16 test sessions, with a single
`empty` session and a single `idle-nearby` session, so these figures are
noisy. The improved model is better on every held-out measure, but it needs
6 s of data before it can decide. An approach or departure lasts only about
76 records, so it gets just 1–2 decisions per pass.

Leaving often looks like approaching, which leaves control latched ON.
This is the main risk: a departure that is misclassified keeps the output ON
until a later `empty` or `leaving` decision, or until a stale/fault timeout.
Treat both exports as integration test builds, not production models. Agree
acceptance targets first and measure them live before relying on either.

## 6. Integrate the exported model into the firmware (SDK 25.12, MCUXpresso)

1. Copy these files into the demo project's `source/` folder:
   - From the repository root: `csi_raw_log.c/.h`, `csi_presence.c/.h`,
     `csi_uart_out.c/.h`, `csi_control.c/.h`.
   - From one `firmware_export/` directory: `csi_model.h`,
     `csi_model_config.h`, `csi_selftest.h`.
2. In **Project Properties > C/C++ Build > Settings > Preprocessor**, define:

   | Define | Value | Meaning |
   | --- | --- | --- |
   | `CSI_PRESENCE_ENABLE` | `1` | Feed records to the classifier (default 0 = logger only) |
   | `CSI_LOG_RAW` | `1` (default) | Also print raw `CSI,` hex lines for `csi_logger.py`; set `0` for a quieter console |
   | `CSI_PRESENCE_SELFTEST` | `1` (default) | Boot self-test. A failure latches control OFF. |
   | `CSI_VOTE_N` | `1` (default) | No smoothing; the agreed initial policy |
   | `CSI_PRES_UART_BASE` | e.g. `USART2` | Spare FLEXCOMM used as the control UART |
   | `CSI_PRES_UART_FLEXCOMM` | e.g. `0U` | FLEXCOMM index; the clock is read with `CLOCK_GetFlexCommClkFreq()` |
   | `CSI_PRES_UART_CLK_HZ` | clock in Hz | Alternative to `CSI_PRES_UART_FLEXCOMM` (one of them is required) |
   | `CSI_OUT_UART_PROTOCOL` | `CSI_OUT_PROTO_CONTROL` (default) | Control UART sends only `control` frames |
   | `CSI_CTRL_PERIOD_MS` / `CSI_CTRL_STALE_MS` | `1000` / `5000` (defaults) | Resend period / OFF after no prediction |
   | `CSI_MEASURE_LATENCY` | `1` (default) | DWT cycle counter; set `0` if DWT is unavailable |

3. Configure that FLEXCOMM's clock and its TX pin in **MCUXpresso Config
   Tools**; the macros alone do not set up pins. Pick a genuinely unused
   instance and check the header pin and I/O voltage (3.3 V) in the
   FRDM-RW612 board manual. Do not reuse the debug-console UART.
4. Call `csi_raw_log_init()` once FreeRTOS objects can be created, and check
   that it returns 0. Only then register `csi_raw_log_cb` with
   `wlan_register_csi_user_callback()`. `csi_raw_log_init()` itself calls
   `csi_presence_init()`, which calls `csi_out_init()`. The output task sends
   OFF immediately, and control stays OFF until a valid prediction arrives.
5. Build, then check the linker's flash/RAM report and the task stacks. Flash
   at boot should show `PRES_SELFTEST,...,OK` lines on the console. Live CSI
   must come from the training AP (`c2:95:da:01:6e:c9`, channel 149);
   records from any other source are dropped by design. The channel filter
   uses header byte 37, as recorded during data collection; confirm this
   field on the live AP.
6. Debug console (USB) is unchanged: v1 `presence`, `heartbeat` and `error`
   NDJSON, plus mirrored `control` lines (`CSI_CTRL_MIRROR_CONSOLE`).
7. If the board can enter PM3 (power-down), the control UART loses its state.
   Restore its pins/clock and call `csi_out_uart_reinit()` in the PM3-exit
   hook, after the debug console is restored.

Avoid passing a function-like value such as
`CSI_PRES_UART_CLK_HZ=CLOCK_GetFlexCommClkFreq(0U)` through the IDE's
preprocessor settings: the generated makefile does not quote it. Use
`CSI_PRES_UART_FLEXCOMM` instead.

### 6a. Reference integration: `dm-motion-detection-using-wifi-csi-on-rw61x`

The FRDM-RW612 Wi-Fi sensing demo has been integrated as follows (improved model):

| File | Change |
| --- | --- |
| `source/csi_*.c/.h`, `csi_model*.h`, `csi_selftest.h` | Copied from here and `artifacts/presense/improved/firmware_export/` |
| `.cproject` (Debug and Release), `flags.cmake` | `CSI_PRESENCE_ENABLE=1`, `CSI_LOG_RAW=1`, `CSI_PRES_UART_BASE=USART0`, `CSI_PRES_UART_FLEXCOMM=0U` |
| `CMakeLists.txt` | Registers the `csi_*` sources (the IDE build compiles all of `source/` anyway) |
| `board/board.c/.h` | `BOARD_InitPresenseUart()`: `IO_MUX_FC0_USART_DATA` (mikroBUS GPIO_3 = TX, GPIO_2 = RX unused), FRG0 from PLL, attached to FLEXCOMM0 |
| `board/lpm.c` | PM3 exit re-runs `BOARD_InitPresenseUart()` and `csi_out_uart_reinit()` |
| `source/main.c` | Calls `BOARD_InitPresenseUart()` after the debug console. Runs `csi_raw_log_init()` **before** the Wi-Fi connect loop, so OFF is sent from boot even if the AP is unreachable, and prints the free heap. Registers `csi_raw_log_cb` only if init succeeded. |

Notes:

- FLEXCOMM0 was unused: FC1 drives the LCD over SPI, FC2 is the touch I2C, and FC3 is the
  debug console. No other code muxes GPIO_2/GPIO_3. **TX is mikroBUS GPIO_3** (confirmed on the board);
  wire it to the receiver's RX, plus a common ground, at 3.3 V. GPIO_2 (FC0 RXD) is muxed but not enabled.
- `BOARD_InitSleepPinConfig()` drives non-AON pins low in sleep, so the TX line goes low (a UART
  break) during PM2/PM3. The receiver drops partial lines and its 3 s watchdog turns it OFF.
- Added cost (ARM GCC 13.2, `-Os`): roughly 177 KB flash for the improved model, of which
  113 KB is the self-test table (set `CSI_PRESENCE_SELFTEST=0` to drop it). About 21 KB of `.bss`
  (16 x 512 B record pool plus a 60 x 47 float ring). Two 4 KB task stacks plus queues come from the
  100 KB FreeRTOS heap. Worst static stack frame is 120 B. Check the `Free heap after CSI init` line at boot.
- The forest sums class probabilities in `double` for exact sklearn parity. On the
  single-precision FPU this is a few dozen soft-float adds per inference.
- Raw logging stays on (`CSI_LOG_RAW=1`), so `csi_logger.py` can keep collecting data while the
  classifier runs. The worker feeds the classifier first, then prints the hex line. `csi_logger.py`
  only parses lines starting with `CSI,`, so the interleaved NDJSON and `control` lines are ignored.
  The console uses FreeRTOS-synchronised `PRINTF` (`SDK_OS_FREE_RTOS`), so lines from different tasks
  don't split each other. At ~10 records/s the hex is ~3.3 KB/s, well within the 115200-baud
  console. Set `CSI_LOG_RAW=0` for a readable console. Set `CSI_PRESENCE_ENABLE=0` for the original logger only.
- `wlan_tests.c` can still register `csi_raw_log_cb` from the CLI. The callback ignores data
  until `csi_raw_log_init()` has succeeded.

## 7. UART control output and receiver contract

The control UART runs at 115200 baud, 8N1, with no flow control, and is
one-way. Wire RW612 TX to the receiver's RX and connect a common ground.
Use 3.3 V logic only: never connect it to RS-232 levels, or to a 5 V input
without level shifting.

Each message is one compact JSON line ending in `\n`, with a fixed key order:

```json
{"version":1,"event":"control","state":"on","reason":"prediction","class":"approaching","confidence":0.87,"seq":42,"up_ms":123456,"crc":19763}
{"version":1,"event":"control","state":"off","reason":"startup","class":null,"confidence":null,"seq":0,"up_ms":12,"crc":64277}
```

- **`state`:** `on` or `off`. This is the policy output, not the
  classifier's `detected` flag.
  - `approaching` sets ON.
  - `leaving` and `empty` set OFF.
  - `idle-nearby` and `passing-by` keep the current state.
- **`reason`:**
  - `prediction`: `class` and `confidence` are filled in.
  - `startup`: the board has just booted.
  - `stale`: no successful prediction for `CSI_CTRL_STALE_MS`, which forces OFF.
  - `fault`: self-test or initialization failure (latched until reboot), or
    an invalid model output. Forces OFF.
- **When frames are sent:** immediately on every state change, and the
  current state is repeated every 1 s.
- **`seq`:** a per-UART uint32 counter that wraps around.
- **`crc`:** CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection,
  check value `"123456789"` → `0x29B1`). It covers every byte before
  `,"crc":`, exactly as received.
- **Schema:** `schemas/presense-control.v1.schema.json` (draft 2020-12). Each
  export also writes a copy with that model's class names.

Receiver rules (implemented by `receiver/presense_control_rx.py`, the
reference parser):

1. Accept only complete `\n`-terminated lines of at most 256 bytes. After an
   over-long line, discard data up to the next newline.
2. Check the CRC, then the schema. Ignore other `event` values and unknown
   `version` values.
3. Act only on `state`.
4. Start OFF at boot, and turn OFF after **3 s** without a valid frame.
   Because the link is one-way, this watchdog is what makes a broken cable
   or a crashed board fail safe.
5. Track `seq`: count gaps and duplicates, and treat a jump back to a small
   value as a sender reboot.

```bash
.venv-deploy/bin/python receiver/presense_control_rx.py --port /dev/ttyUSB0   # live, via a USB-UART adapter
.venv-deploy/bin/python receiver/presense_control_rx.py --file capture.txt    # offline check
```

Before connecting the real receiver, check the following with a USB-UART
adapter (and ideally a logic analyser):

- baud rate and framing;
- OFF at startup;
- the 1 s resend;
- ON on an approach;
- OFF on stale data or when the AP is switched off.

## Tests

```bash
.venv-deploy/bin/python -m pytest tests -q
```

- `tests/test_control.py`:
  - C control policy and CRC vector.
  - Golden frames and rejected frames.
  - Receiver watchdog, `seq` wrap, resync and reboot detection.
  - Output-task timing under a scripted FreeRTOS stub: startup OFF, 1 s
    resend, stale at exactly the deadline, fault latch, outdated results
    ignored.
- `tests/test_records_mode.py`:
  - Window/hop schedule and TSF resets.
  - Mask fitted on training sessions only.
  - Applying a fixed `csi_split --assignment`.

## Firmware module reference

| Module | Role |
| --- | --- |
| `csi_raw_log.c/.h` | Wi-Fi CSI callback. Validates each record, copies it into a pool and never blocks. A low-priority worker then prints hex (`CSI_LOG_RAW`) and/or calls `csi_presence_feed()` (`CSI_PRESENCE_ENABLE`). |
| `csi_presence.c/.h` | Firmware version of records-mode `csi_clean.py`: validate, MAC/channel/size filter, I/Q → amplitude, mask, normalize, TSF stall/reset handling. Keeps a chronological ring buffer and computes `meanvar`/`dynamic`/`full` features (selected by `CSI_FEATURE_SET`), then calls `csi_infer()`. Also runs the boot self-test. No Hampel filter; train with `--hampel-window 1`. |
| `csi_control.c/.h` | Pure-C ON/OFF policy, `control` JSON formatter (bounded buffer; on overflow sends an OFF fallback frame) and CRC-16. Has no RTOS dependency, so it can be tested on the host. |
| `csi_uart_out.c/.h` | Low-priority output task driven by deadlines. Sends `control` frames on the control UART, and v1 diagnostics plus mirrored control lines on the console. Handles stale and fault rules. |

## Known gaps / things to verify before relying on this project

- **Accuracy:** both models are weak and were scored on few held-out
  sessions. No day-holdout result exists yet, and the session labels have no
  transition timestamps, so live action latency and false-ON rate must be
  measured on hardware.
- **Hardware not yet verified:**
  - The reference integration (section 6a) compiles with the ARM toolchain
    against the project's SDK headers, but it has not been linked in the IDE or
    run, and runtime heap, stack and inference time have not been measured.
  - Live replay of records on the board has not been run.
  - The receiver hardware has not been chosen.
- **CSI fixed-point scale:** no confirmed AN14281 field for `--scale-shift`;
  it defaults to 0 and relies on per-record normalization.
- **TSF unit:** assumed to be microseconds; confirm against AN14281.
- **Channel field (header byte 37) encoding:** it matches the data collected
  here (149), but has not been confirmed against AN14281.
- **RW612 as a SoftAP generating CSI:** untested. Use STA mode.
- **Header I/O voltage:** confirm 3.3 V on the pins you wire to another board.

## License / attribution

`application/` is derived from NXP's
[dm-motion-detection-using-wifi-csi-on-rw61x](https://github.com/nxp-appcodehub/dm-motion-detection-using-wifi-csi-on-rw61x)
(NXP Application Code Hub). Check that repo's license before redistributing.
`sdk/` is NXP's MCUXpresso SDK — redistribute per its license, or `.gitignore`
it and document the exact SDK version/build-ID instead of committing it.
