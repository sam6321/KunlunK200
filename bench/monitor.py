#!/usr/bin/env python3
"""Sample xpu_smi -m while a GEMM bench runs. SIGTERM/SIGINT → write summary."""
from __future__ import annotations

import argparse
import csv
import os
import signal
import statistics
import subprocess
import sys
import time

FIELDS = [
    "t_unix",
    "elapsed_s",
    "pci",
    "dev_id",
    "die_c",
    "hbm_c",
    "power_w",
    "freq_mhz",
    "l3_used_mb",
    "hbm_used_mb",
    "use_pct",
    "fw",
    "model",
]


def parse_m_line(line: str) -> dict | None:
    p = line.split()
    if len(p) < 22:
        return None
    try:
        return {
            "pci": p[0],
            "dev_id": int(p[2]),
            "die_c": float(p[4]),
            "hbm_c": float(p[6]),
            "power_w": float(p[8]) / 1000.0,
            "freq_mhz": float(p[9]),
            "l3_used_mb": float(p[15]),
            "hbm_used_mb": float(p[17]),
            "use_pct": float(p[19]),
            "fw": p[20],
            "model": p[21],
        }
    except (ValueError, IndexError):
        return None


def smi_m(smi: str) -> list[dict]:
    r = subprocess.run([smi, "-m"], capture_output=True, text=True, timeout=5)
    if r.returncode != 0:
        return []
    out = []
    for line in r.stdout.splitlines():
        row = parse_m_line(line.strip())
        if row:
            out.append(row)
    return out


def summarize(rows: list[dict], path: str) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("no samples\n")
        return
    by_dev: dict[int, list[dict]] = {}
    for r in rows:
        by_dev.setdefault(r["dev_id"], []).append(r)

    def stats(vals: list[float]) -> str:
        return "min=%.1f mean=%.1f max=%.1f" % (
            min(vals),
            statistics.mean(vals),
            max(vals),
        )

    dur = max(rows[-1]["elapsed_s"], 1e-6)
    lines = ["smi_samples=%d duration_s=%.2f" % (len(rows), rows[-1]["elapsed_s"])]
    # Board power is duplicated on both PD lines — use dev 0.
    p0 = by_dev.get(0) or next(iter(by_dev.values()))
    pwr = [x["power_w"] for x in p0]
    lines.append("power_w " + stats(pwr))
    lines.append("energy_j=%.1f" % (statistics.mean(pwr) * dur))
    lines.append("freq_mhz " + stats([x["freq_mhz"] for x in p0]))
    for dev, rs in sorted(by_dev.items()):
        lines.append(
            "pd%d die_c %s hbm_c %s use_pct %s"
            % (
                dev,
                stats([x["die_c"] for x in rs]),
                stats([x["hbm_c"] for x in rs]),
                stats([x["use_pct"] for x in rs]),
            )
        )
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    sys.stdout.write(text)
    sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smi", default=os.environ.get("XPU_SMI", "xpu_smi"))
    ap.add_argument("--interval", type=float, default=0.25)
    ap.add_argument("--out", required=True, help="CSV path")
    ap.add_argument("--summary", default="", help="text summary path")
    args = ap.parse_args()
    summary = args.summary or (os.path.splitext(args.out)[0] + ".summary.txt")

    stop = {"n": False}

    def halt(signum, _frame):
        stop["n"] = True

    signal.signal(signal.SIGINT, halt)
    signal.signal(signal.SIGTERM, halt)

    t0 = time.time()
    rows: list[dict] = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        f.flush()
        while not stop["n"]:
            now = time.time()
            for rec in smi_m(args.smi):
                rec["t_unix"] = "%.3f" % now
                rec["elapsed_s"] = "%.3f" % (now - t0)
                w.writerow({k: rec.get(k, "") for k in FIELDS})
                rows.append(
                    {
                        **rec,
                        "elapsed_s": now - t0,
                        "die_c": rec["die_c"],
                        "hbm_c": rec["hbm_c"],
                        "power_w": rec["power_w"],
                        "freq_mhz": rec["freq_mhz"],
                        "use_pct": rec["use_pct"],
                        "dev_id": rec["dev_id"],
                    }
                )
            f.flush()
            time.sleep(max(0.05, args.interval))
    summarize(rows, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
