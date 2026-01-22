#!/usr/bin/env python3
import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def _read_csv_tail(path, max_points):
    if not Path(path).exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows[-max_points:]


def _parse_rows(rows):
    ts = []
    fric = []
    eng = []
    cpu = []
    psi = []
    score = []
    for row in rows:
        try:
            ts.append(datetime.strptime(row["Time"], "%Y-%m-%d %H:%M:%S"))
        except (KeyError, ValueError):
            continue
        try:
            fric.append(float(row["FrictionSigned"]))
        except (KeyError, ValueError):
            fric.append(0.0)
        try:
            eng.append(float(row["Energy"]))
        except (KeyError, ValueError):
            eng.append(0.0)
        try:
            cpu.append(float(row["CPUUtil"]))
        except (KeyError, ValueError):
            cpu.append(0.0)
        try:
            psi.append(float(row["PSI"]))
        except (KeyError, ValueError):
            psi.append(0.0)
        try:
            score.append(float(row["Score"]))
        except (KeyError, ValueError):
            score.append(0.0)
    return ts, fric, eng, cpu, psi, score


def main():
    parser = argparse.ArgumentParser(description="Real-time fuzzy monitoring plots.")
    parser.add_argument("--inputs", default="/tmp/ksense/fuzzy_monitor.csv", help="1s inputs CSV")
    parser.add_argument("--scores", default="/tmp/ksense/fuzzy_score.csv", help="Score CSV")
    parser.add_argument("--window", type=int, default=300, help="Points to display")
    parser.add_argument("--interval", type=int, default=1000, help="Update interval ms")
    args = parser.parse_args()

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    (line_fric,) = ax1.plot([], [], label="FrictionSigned")
    (line_eng,) = ax1.plot([], [], label="Energy")
    (line_cpu,) = ax1.plot([], [], label="CPUUtil")
    (line_psi,) = ax1.plot([], [], label="PSI")
    ax1.set_title("Inputs (1s)")
    ax1.set_xlabel("Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    (line_score,) = ax2.plot([], [], label="Score")
    (line_fric2,) = ax2.plot([], [], label="FrictionSigned")
    (line_eng2,) = ax2.plot([], [], label="Energy")
    (line_cpu2,) = ax2.plot([], [], label="CPUUtil")
    (line_psi2,) = ax2.plot([], [], label="PSI")
    ax2.set_title("Score + Inputs")
    ax2.set_xlabel("Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    def update_inputs(_):
        rows = _read_csv_tail(args.inputs, args.window)
        ts, fric, eng, cpu, psi, _score = _parse_rows(rows)
        if not ts:
            return
        line_fric.set_data(ts, fric)
        line_eng.set_data(ts, eng)
        line_cpu.set_data(ts, cpu)
        line_psi.set_data(ts, psi)
        ax1.relim()
        ax1.autoscale_view()

    def update_scores(_):
        rows = _read_csv_tail(args.scores, args.window)
        ts, fric, eng, cpu, psi, score = _parse_rows(rows)
        if not ts:
            return
        line_score.set_data(ts, score)
        line_fric2.set_data(ts, fric)
        line_eng2.set_data(ts, eng)
        line_cpu2.set_data(ts, cpu)
        line_psi2.set_data(ts, psi)
        ax2.relim()
        ax2.autoscale_view()

    anim1 = FuncAnimation(fig1, update_inputs, interval=args.interval, cache_frame_data=False)
    anim2 = FuncAnimation(fig2, update_scores, interval=args.interval, cache_frame_data=False)

    plt.show()
    _ = (anim1, anim2)


if __name__ == "__main__":
    main()
