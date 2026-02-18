#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
from datetime import datetime
from typing import List, Optional


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ksense.fuzzy_controller import FuzzyConfig, FuzzyController
from ksense.config import ENERGY_CALIB_WIN_S, GRID_STEP_S


def _f(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_time(s: str):
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _read_rows(path: str, tail: int = 0) -> List[dict]:
    if tail and tail > 0:
        buf = []
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                buf.append(row)
        return buf[-tail:]

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def _load_initial_rows(path: str, tail: int = 0):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tail and tail > 0:
            buf = []
            for row in reader:
                buf.append(row)
            rows = buf[-tail:]
        else:
            rows = [row for row in reader]
        pos = f.tell()
        headers = reader.fieldnames or []
    return rows, headers, pos


def _parse_row_from_line(headers: List[str], line: str) -> Optional[dict]:
    if not line.strip():
        return None
    row_vals = next(csv.reader([line]))
    if len(row_vals) < len(headers):
        return None
    return dict(zip(headers, row_vals))


def _smooth(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values
    out: List[float] = []
    buf: List[float] = []
    for v in values:
        buf.append(v)
        if len(buf) > window:
            buf.pop(0)
        out.append(sum(buf) / len(buf))
    return out


def _interp_linear(x: List[float], y: List[float], factor: int) -> List[float]:
    if factor <= 1 or len(x) <= 1:
        return y
    out: List[float] = []
    for i in range(len(x) - 1):
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]
        out.append(y0)
        for k in range(1, factor):
            t = k / factor
            out.append(y0 + (y1 - y0) * t)
    out.append(y[-1])
    return out


def _interp_time(x: List[float], factor: int) -> List[float]:
    if factor <= 1 or len(x) <= 1:
        return x
    out: List[float] = []
    for i in range(len(x) - 1):
        x0, x1 = x[i], x[i + 1]
        out.append(x0)
        for k in range(1, factor):
            t = k / factor
            out.append(x0 + (x1 - x0) * t)
    out.append(x[-1])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Plot controller behavior on kernel_metrics.csv")
    p.add_argument(
        "--input",
        default=os.getenv("KSENSE_METRICS_CSV", "/tmp/ksense/kernel_metrics.csv"),
        help="Input kernel_metrics.csv path (default: KSENSE_METRICS_CSV or /tmp/ksense/kernel_metrics.csv)",
    )
    p.add_argument("--tail", type=int, default=0, help="Only plot the last N rows")
    p.add_argument("--output", default="fuzzy_controller_plot.png", help="Output PNG path")
    p.add_argument("--show", action="store_true", help="Show the plot window")
    p.add_argument(
        "--realtime",
        action="store_true",
        help="Replay in real time using CSV timestamps",
    )
    p.add_argument(
        "--interval-s",
        type=float,
        default=1.0,
        help="Fallback interval when timestamps are missing (default: 1.0s)",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay speed multiplier (1.0 = real time, 2.0 = 2x faster)",
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Moving-average window for smoother plots (default: 1 = no smoothing)",
    )
    p.add_argument(
        "--interp-factor",
        type=int,
        default=1,
        help="Linear interpolation factor for smoother lines (default: 1 = no interpolation)",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="Follow the CSV as it grows and update the plot in real time",
    )
    p.add_argument(
        "--refresh-s",
        type=float,
        default=1.0,
        help="Live mode refresh interval (default: 1.0s)",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=3600,
        help="Max points to keep in live mode (default: 3600)",
    )
    args = p.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is required. Install with: python3 -m pip install matplotlib")
        return 1

    cfg = FuzzyConfig(csv_path=args.input, rules_enabled=False)
    controller = FuzzyController(cfg=cfg)

    if args.live:
        rows, headers, pos = _load_initial_rows(args.input, tail=args.tail)
    else:
        rows = _read_rows(args.input, tail=args.tail)
        headers = []
        pos = 0

    if not rows:
        print(f"No rows found in {args.input}")
        return 1

    times = []
    friction = []
    energy = []
    cpu = []
    psi = []
    score = []
    decision = []

    prev_ts = None
    fcal = None
    calib_fric = []
    fcal_min_samples = max(30, int(ENERGY_CALIB_WIN_S / GRID_STEP_S))

    def _consume_row(row):
        nonlocal fcal, calib_fric, prev_ts
        ts = (row.get("Time") or "").strip()
        t_parsed = _parse_time(ts)
        t = t_parsed or ts

        fric = _f(row.get("Friction"))
        direction = _f(row.get("Direction"))
        eng = _f(row.get("Energy"))
        cpu_v = _f(row.get("CPUUtil"))
        psi_v = _f(row.get("PSI"))
        baseline_mode = (row.get("BaselineMode") or "").strip().upper()

        if fric is not None:
            calib_fric.append(fric)
            if fcal is None and baseline_mode == "FROZEN" and len(calib_fric) >= fcal_min_samples:
                fcal = float(sum(calib_fric) / len(calib_fric))

        # New direction logic: sign(Ft - Fcal) once Fcal is available
        if fcal is not None and fric is not None:
            if fric > fcal:
                direction = 1.0
            elif fric < fcal:
                direction = -1.0
            else:
                direction = 0.0

        fric_signed = None
        if fric is not None:
            fric_signed = fric * (direction if direction is not None else 1.0)

        def _inject_collect_metrics():
            return {
                "fric": fric_signed,
                "eng": eng,
                "cpu": cpu_v,
                "psi": psi_v,
                "last_direction": direction,
                "fric_short_p99": None,
                "fric_long_p99": None,
                "eng_short_p99": None,
                "eng_long_p99": None,
            }

        controller._collect_metrics = _inject_collect_metrics  # type: ignore[attr-defined]
        report = controller.decide()

        times.append(t)
        friction.append(fric_signed if fric_signed is not None else float("nan"))
        energy.append(eng if eng is not None else float("nan"))
        cpu.append(cpu_v if cpu_v is not None else float("nan"))
        psi.append(psi_v if psi_v is not None else float("nan"))
        score.append(float(report.get("score", 0.0)))
        decision.append(report.get("decision", ""))

        if args.realtime:
            if t_parsed and prev_ts:
                delta = (t_parsed - prev_ts).total_seconds()
                if delta < 0:
                    delta = 0
                if args.speed and args.speed > 0:
                    delta = delta / args.speed
                if delta > 0:
                    time.sleep(delta)
            else:
                time.sleep(max(0.0, args.interval_s))
            prev_ts = t_parsed

        if args.live and len(times) > args.max_points:
            times.pop(0)
            friction.pop(0)
            energy.pop(0)
            cpu.pop(0)
            psi.pop(0)
            score.pop(0)
            decision.pop(0)

        return t_parsed

    for row in rows:
        _consume_row(row)

    if args.live:
        plt.ion()
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        line_score, = axes[0].plot([], [], color="#1f77b4", label="Score")
        axes[0].axhline(45, color="#999999", linestyle="--", linewidth=1)
        axes[0].axhline(70, color="#999999", linestyle="--", linewidth=1)
        axes[0].set_ylabel("Score")
        axes[0].set_title("Controller Score")
        axes[0].legend(loc="upper right")

        line_fric, = axes[1].plot([], [], color="#ff7f0e", label="Friction (signed)")
        line_eng, = axes[1].plot([], [], color="#9467bd", label="Energy")
        axes[1].set_ylabel("Friction / Energy")
        axes[1].set_title("Friction & Energy")
        axes[1].legend(loc="upper right")

        line_cpu, = axes[2].plot([], [], color="#17becf", label="CPUUtil")
        line_psi, = axes[2].plot([], [], color="#8c564b", label="PSI")
        axes[2].set_ylabel("CPU / PSI")
        axes[2].set_title("CPUUtil & PSI")
        axes[2].legend(loc="upper right")

        fig.tight_layout()

        def _update_plot():
            times_num = []
            for t in times:
                if isinstance(t, datetime):
                    import matplotlib.dates as mdates
                    times_num.append(mdates.date2num(t))
                else:
                    times_num.append(len(times_num))

            y_score = _smooth(score, args.smooth_window)
            y_fric = _smooth(friction, args.smooth_window)
            y_eng = _smooth(energy, args.smooth_window)
            y_cpu = _smooth(cpu, args.smooth_window)
            y_psi = _smooth(psi, args.smooth_window)

            if args.interp_factor and args.interp_factor > 1:
                x_plot = _interp_time(times_num, args.interp_factor)
                y_score = _interp_linear(times_num, y_score, args.interp_factor)
                y_fric = _interp_linear(times_num, y_fric, args.interp_factor)
                y_eng = _interp_linear(times_num, y_eng, args.interp_factor)
                y_cpu = _interp_linear(times_num, y_cpu, args.interp_factor)
                y_psi = _interp_linear(times_num, y_psi, args.interp_factor)
            else:
                x_plot = times_num

            line_score.set_data(x_plot, y_score)
            line_fric.set_data(x_plot, y_fric)
            line_eng.set_data(x_plot, y_eng)
            line_cpu.set_data(x_plot, y_cpu)
            line_psi.set_data(x_plot, y_psi)

            for ax in axes:
                ax.relim()
                ax.autoscale_view()

            fig.canvas.draw()
            fig.canvas.flush_events()

        _update_plot()

        with open(args.input, "r", encoding="utf-8") as f:
            if headers:
                f.seek(pos)
            else:
                headers_line = f.readline()
                headers = [h.strip() for h in headers_line.split(",")] if headers_line else []
            while True:
                line = f.readline()
                if not line:
                    time.sleep(args.refresh_s)
                    continue
                row = _parse_row_from_line(headers, line)
                if not row:
                    continue
                _consume_row(row)
                _update_plot()

        return 0

    # Convert times to numeric axis for interpolation
    times_num = []
    for t in times:
        if isinstance(t, datetime):
            # matplotlib date number
            import matplotlib.dates as mdates
            times_num.append(mdates.date2num(t))
        else:
            times_num.append(len(times_num))

    # Smooth signals if requested
    if args.smooth_window and args.smooth_window > 1:
        score = _smooth(score, args.smooth_window)
        friction = _smooth(friction, args.smooth_window)
        energy = _smooth(energy, args.smooth_window)
        cpu = _smooth(cpu, args.smooth_window)
        psi = _smooth(psi, args.smooth_window)

    # Interpolate for smoother lines if requested
    if args.interp_factor and args.interp_factor > 1:
        times_plot = _interp_time(times_num, args.interp_factor)
        score_plot = _interp_linear(times_num, score, args.interp_factor)
        friction_plot = _interp_linear(times_num, friction, args.interp_factor)
        energy_plot = _interp_linear(times_num, energy, args.interp_factor)
        cpu_plot = _interp_linear(times_num, cpu, args.interp_factor)
        psi_plot = _interp_linear(times_num, psi, args.interp_factor)
    else:
        times_plot = times_num
        score_plot = score
        friction_plot = friction
        energy_plot = energy
        cpu_plot = cpu
        psi_plot = psi

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Score + decision
    ax = axes[0]
    ax.plot(times_plot, score_plot, label="Score", color="#1f77b4")
    ax.axhline(45, color="#999999", linestyle="--", linewidth=1)
    ax.axhline(70, color="#999999", linestyle="--", linewidth=1)
    allow_idx = [i for i, d in enumerate(decision) if d == "allow"]
    deny_idx = [i for i, d in enumerate(decision) if d == "deny"]
    if allow_idx:
        ax.scatter([times_num[i] for i in allow_idx], [score[i] for i in allow_idx], s=10, c="#2ca02c", label="allow")
    if deny_idx:
        ax.scatter([times_num[i] for i in deny_idx], [score[i] for i in deny_idx], s=10, c="#d62728", label="deny")
    ax.set_ylabel("Score")
    ax.set_title("Controller Score & Decision")
    ax.legend(loc="upper right")

    # Friction & Energy
    ax = axes[1]
    ax.plot(times_plot, friction_plot, label="Friction (signed)", color="#ff7f0e")
    ax.plot(times_plot, energy_plot, label="Energy", color="#9467bd")
    ax.set_ylabel("Friction / Energy")
    ax.set_title("Friction & Energy")
    ax.legend(loc="upper right")

    # CPU & PSI
    ax = axes[2]
    ax.plot(times_plot, cpu_plot, label="CPUUtil", color="#17becf")
    ax.plot(times_plot, psi_plot, label="PSI", color="#8c564b")
    ax.set_ylabel("CPU / PSI")
    ax.set_title("CPUUtil & PSI")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
