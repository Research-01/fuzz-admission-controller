#!/usr/bin/env python3
import csv
import time
from collections import deque
from datetime import datetime


INTERVAL_SEC = 1
HISTORY_LEN = 300
OUTPUT_CSV = "cpu_psi_metrics.csv"


def read_cpu_times():
    with open("/proc/stat", "r") as f:
        parts = f.readline().strip().split()
    if parts[0] != "cpu":
        raise RuntimeError("Unexpected /proc/stat format")
    values = [int(v) for v in parts[1:8]]
    user, nice, system, idle, iowait, irq, softirq = values
    return user, nice, system, idle, iowait, irq, softirq


def cpu_percent(prev, curr):
    prev_total = sum(prev)
    curr_total = sum(curr)
    total_delta = curr_total - prev_total
    if total_delta <= 0:
        return 0.0, 0.0, 0.0
    user_delta = (curr[0] + curr[1]) - (prev[0] + prev[1])
    system_delta = (curr[2] + curr[5] + curr[6]) - (prev[2] + prev[5] + prev[6])
    iowait_delta = curr[4] - prev[4]
    return (
        100.0 * user_delta / total_delta,
        100.0 * system_delta / total_delta,
        100.0 * iowait_delta / total_delta,
    )


def get_psi(resource):
    data = {}
    with open(f"/proc/pressure/{resource}", "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            key = parts[0]
            vals = {}
            for part in parts[1:]:
                k, v = part.split("=")
                vals[k] = float(v)
            data[key] = vals
    return data


def main():
    print(f"--- Monitoring CPU + PSI (Interval: {INTERVAL_SEC}s) ---")
    print(f"Writing CSV to {OUTPUT_CSV}")
    print("Press Ctrl+C to stop.")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Time",
            "CPU_User",
            "CPU_Sys",
            "CPU_Wait",
            "PSI_CPU_Some",
            "PSI_Mem_Full",
            "PSI_IO_Full",
        ])

    cpu_hist = deque(maxlen=HISTORY_LEN)
    psi_hist = deque(maxlen=HISTORY_LEN)

    prev = read_cpu_times()

    try:
        while True:
            time.sleep(INTERVAL_SEC)

            curr = read_cpu_times()
            usr, sys, wait = cpu_percent(prev, curr)
            prev = curr

            psi_c = get_psi("cpu")
            psi_m = get_psi("memory")
            psi_i = get_psi("io")

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row = [
                ts,
                f"{usr:.2f}",
                f"{sys:.2f}",
                f"{wait:.2f}",
                f"{psi_c.get('some', {}).get('avg10', 0.0):.2f}",
                f"{psi_m.get('full', {}).get('avg10', 0.0):.2f}",
                f"{psi_i.get('full', {}).get('avg10', 0.0):.2f}",
            ]

            cpu_hist.append((usr, sys, wait))
            psi_hist.append((row[4], row[5], row[6]))

            with open(OUTPUT_CSV, "a", newline="") as f:
                csv.writer(f).writerow(row)
    except KeyboardInterrupt:
        print("\nStopping. CSV saved.")


if __name__ == "__main__":
    main()
