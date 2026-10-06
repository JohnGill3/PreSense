#!/usr/bin/env python3
"""Record one labelled CSI session from the RW612 (firmware: csi_raw_log.c).

Needs:  pip install pyserial
The COM port can only be open in one program, so close PuTTY/Tera Term first.
The script can send the wifi_cli commands itself with --cmd / --stop-cmd.

Example (replace the <...> values):
  python csi_logger.py --port COM5 --label empty --duration 180 --start-delay 15 \
    --cmd "wlan-add test ssid <SSID> wpa2 psk <PASSWORD>" \
    --cmd "wlan-connect test" \
    --cmd "wlan-set-csi-param-header sta 1 66051 66051 170 1 40 0 0" \
    --cmd "wlan-set-csi-filter add <AP_MAC> 255 08 0" \
    --cmd "wlan-csi-cfg" \
    --src-mac <AP_MAC> \
    --stop-cmd "wlan-set-csi-param-header sta 2 66051 66051 170 1 40 0 0" \
    --stop-cmd "wlan-csi-cfg"

Each --cmd (including "wlan-connect") is followed by a plain --cmd-delay
second wait, during which the board's response is shown (unless --quiet) via
send(). There is no separate wait specifically for a connect confirmation -
make sure --cmd-delay is long enough for your AP/board to associate, or
split connect and CSI-enable commands into two separate script runs if
association time varies a lot.

Output (in --out): <label>_<timestamp>.csv  one row per valid record, with the
                   complete raw record as hex (lossless; decode I/Q later)
                   <label>_<timestamp>.json session metadata and counters
"""
import argparse
import csv
import json
import time
from pathlib import Path

import serial

HDR, TAIL = 48, 4
FIELDS = ["host_time", "seq", "tick_ms", "tsf", "src", "channel", "rssi_a",
          "nf_a", "sinr", "ap_type", "data_bytes", "record_hex"]


def s8(x):
    return x - 256 if x > 127 else x


def parse_record(hexstr, nbytes):
    """Return header fields if the record is well formed, else None."""
    try:
        b = bytes.fromhex(hexstr)
    except ValueError:
        return None
    if len(b) != nbytes or len(b) < HDR + TAIL:
        return None
    if int.from_bytes(b[2:4], "little") != 0xABCD:
        return None
    if int.from_bytes(b[0:2], "little") * 4 != len(b):
        return None
    data_bytes = (int.from_bytes(b[44:46], "little") - 1) * 4
    if HDR + data_bytes + TAIL != len(b):
        return None
    return {
        "tsf": int.from_bytes(b[12:20], "little"),
        "src": ":".join(f"{x:02x}" for x in b[26:32]),
        "channel": b[37],
        "rssi_a": s8(b[32]),
        "nf_a": s8(b[34]),
        "sinr": s8(b[36]),
        "ap_type": b[38],
        "data_bytes": data_bytes,
    }


def send(ser, cmd, delay, quiet=False):
    """Send one command and, unless `quiet`, print every line the board sends
    back for `delay` seconds, so command errors (bad syntax, invalid args,
    etc.) are visible instead of being silently swallowed. In quiet mode the
    board's output is still drained (so it doesn't pile up for a later read),
    just not printed - except a line containing "error"/"fail", which always
    prints, since a quiet run shouldn't hide a real failure."""
    print(f">> {cmd}")
    ser.write((cmd + "\r\n").encode())
    deadline = time.time() + delay
    while time.time() < deadline:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            is_err = ("error" in line.lower()) or ("fail" in line.lower())
            if not quiet or is_err:
                print(f"   {line}")
            if is_err:
                print(f"   ^^^ possible error after: {cmd}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="e.g. COM5 or /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--label", required=True, help="e.g. empty, walking, seated")
    ap.add_argument("--duration", type=float, default=120, help="recording seconds")
    ap.add_argument("--start-delay", type=float, default=0,
                    help="seconds to wait before recording (time to leave the room)")
    ap.add_argument("--src-mac", default=None, help="keep only records from this MAC")
    ap.add_argument("--out", default="data")
    ap.add_argument("--notes", default="")
    ap.add_argument("--cmd", action="append", default=[], help="sent before recording")
    ap.add_argument("--stop-cmd", action="append", default=[], help="sent after recording")
    ap.add_argument("--cmd-delay", type=float, default=2.0, help="seconds after each command")
    ap.add_argument("--boot-wait", type=float, default=5.0,
                    help="seconds to wait right after opening the port, printing anything the "
                         "board sends (e.g. a reboot/boot banner), before the first --cmd. "
                         "Opening the port can reset some boards; this makes that visible "
                         "instead of silently losing the first commands. Set 0 to skip.")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the board's line-by-line command responses (boot-wait "
                         "banner, command echo, connect-wait scan). Lines containing "
                         "'error'/'fail', and a connect timeout, still print.")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = out / f"{a.label}_{stamp}"
    want_mac = a.src_mac.lower() if a.src_mac else None

    ser = serial.Serial(a.port, a.baud, timeout=1)

    if a.boot_wait > 0:
        if not a.quiet:
            print(f"Waiting {a.boot_wait:.0f}s after opening the port, in case it reset the "
                  f"board (printing anything received; a boot banner here confirms a reset) ...")
        deadline = time.time() + a.boot_wait
        saw_anything = False
        while time.time() < deadline:
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                saw_anything = True
                if not a.quiet:
                    print(f"   {line}")
        if not saw_anything and not a.quiet:
            print("   (nothing received - port open likely did NOT reset the board)")

    for c in a.cmd:
        send(ser, c, a.cmd_delay, quiet=a.quiet)

    if a.start_delay > 0:
        print(f"Recording starts in {a.start_delay:.0f} s ...")
        time.sleep(a.start_delay)
    ser.reset_input_buffer()

    good = bad = skipped = seq_gaps = 0
    fw_msgs, last_seq = [], None
    t0 = time.time()
    next_report = t0 + 5
    print(f"Recording '{a.label}' for {a.duration:.0f} s (Ctrl-C to stop early)")

    try:
        with open(f"{base}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            while time.time() - t0 < a.duration:
                line = ser.readline().decode(errors="ignore").strip()
                if line.startswith("CSI,"):
                    p = line.split(",", 4)
                    try:
                        seq, tick, nbytes = int(p[1]), int(p[2]), int(p[3])
                        rec = parse_record(p[4], nbytes)
                    except (ValueError, IndexError):
                        rec = None
                    if rec is None:
                        bad += 1
                        continue
                    if want_mac and rec["src"] != want_mac:
                        skipped += 1
                        continue
                    if last_seq is not None and seq != last_seq + 1:
                        seq_gaps += 1
                    last_seq = seq
                    w.writerow([f"{time.time():.6f}", seq, tick, rec["tsf"], rec["src"],
                                rec["channel"], rec["rssi_a"], rec["nf_a"], rec["sinr"],
                                rec["ap_type"], rec["data_bytes"], p[4]])
                    good += 1
                elif line.startswith(("CSI_DROP", "CSI_BAD")):
                    fw_msgs.append(line)
                    print(line)
                if time.time() >= next_report:
                    next_report += 5
                    print(f"  {good} records ({good / (time.time() - t0):.1f}/s), "
                          f"{bad} bad, {len(fw_msgs)} firmware warnings")
    except KeyboardInterrupt:
        print("Stopped early.")

    t1 = time.time()
    for c in a.stop_cmd:
        send(ser, c, a.cmd_delay, quiet=a.quiet)
    ser.close()

    meta = {
        "label": a.label, "notes": a.notes, "port": a.port, "baud": a.baud,
        "start_epoch": t0, "end_epoch": t1, "duration_s": round(t1 - t0, 2),
        "records": good, "rate_hz": round(good / max(t1 - t0, 1e-9), 2),
        "bad_lines": bad, "other_mac_skipped": skipped, "seq_gaps": seq_gaps,
        "firmware_warnings": fw_msgs[-20:], "src_mac_filter": want_mac,
        "commands": a.cmd,
    }
    Path(f"{base}.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {base}.csv ({good} records) and {base}.json")


if __name__ == "__main__":
    main()