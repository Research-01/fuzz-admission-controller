#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from collections import Counter, deque
from typing import Dict, List, Optional


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ksense.fuzzy_controller import FuzzyConfig, FuzzyController
from ksense.helpers import ensure_csv


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


def _metrics_from_row(row: Dict[str, str]) -> dict:
    fric = _f(row.get("Friction"))
    direction = _f(row.get("Direction"))
    eng = _f(row.get("Energy"))
    cpu = _f(row.get("CPUUtil"))
    psi = _f(row.get("PSI"))

    fric_signed = None
    if fric is not None:
        fric_signed = fric * (direction if direction is not None else 1.0)

    return {
        "fric": fric_signed,
        "eng": eng,
        "cpu": cpu,
        "psi": psi,
        "last_direction": direction,
        "fric_short_p99": None,
        "fric_long_p99": None,
        "eng_short_p99": None,
        "eng_long_p99": None,
        "baseline_mode": (row.get("BaselineMode") or "").strip(),
        "baseline_samples": (row.get("BaselineSamples") or "").strip(),
    }


def _read_rows(path: str, tail: int = 0) -> List[Dict[str, str]]:
    if tail and tail > 0:
        buf = deque(maxlen=tail)
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                buf.append(row)
        return list(buf)

    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Replay fuzzy controller decisions over a kernel_metrics.csv file.")
    p.add_argument(
        "--input",
        default=os.getenv("KSENSE_METRICS_CSV", "/tmp/ksense/kernel_metrics.csv"),
        help="Input kernel_metrics.csv path (default: KSENSE_METRICS_CSV or /tmp/ksense/kernel_metrics.csv)",
    )
    p.add_argument(
        "--tail",
        type=int,
        default=0,
        help="Only process the last N rows (default: 0 = all)",
    )
    p.add_argument(
        "--output",
        default=os.getenv("FUZZY_REPLAY_OUT", "/tmp/ksense/fuzzy_replay.csv"),
        help="Output CSV path (default: FUZZY_REPLAY_OUT or /tmp/ksense/fuzzy_replay.csv)",
    )
    p.add_argument(
        "--print-every",
        type=int,
        default=0,
        help="Print every Nth decision (default: 0 = no printing)",
    )
    args = p.parse_args()

    rows = _read_rows(args.input, tail=args.tail)
    if not rows:
        print(f"[replay] no rows in {args.input}")
        return 1

    cfg = FuzzyConfig(csv_path=args.input, rules_enabled=False)
    controller = FuzzyController(cfg=cfg)

    # Avoid writing /tmp/ksense/fuzzy_score.csv during replay.
    controller._write_score_sample = lambda *_a, **_kw: None

    ensure_csv(
        args.output,
        [
            "InputTime",
            "BaselineMode",
            "BaselineSamples",
            "Friction",
            "Direction",
            "Energy",
            "CPUUtil",
            "PSI",
            "Score",
            "Level",
            "Decision",
            "BadCount",
            "GoodCount",
            "MediumBand",
            "Reason",
            "Missing",
        ],
    )

    decision_counts = Counter()

    for i, row in enumerate(rows, start=1):
        m = _metrics_from_row(row)

        def _inject_collect_metrics():
            return {
                "fric": m["fric"],
                "eng": m["eng"],
                "cpu": m["cpu"],
                "psi": m["psi"],
                "last_direction": m["last_direction"],
                "fric_short_p99": None,
                "fric_long_p99": None,
                "eng_short_p99": None,
                "eng_long_p99": None,
            }

        controller._collect_metrics = _inject_collect_metrics  # type: ignore[attr-defined]
        report = controller.decide()

        metrics = report.get("metrics") or {}
        missing = report.get("missing") or []

        out_row = [
            (row.get("Time") or "").strip(),
            m["baseline_mode"],
            m["baseline_samples"],
            "" if _f(row.get("Friction")) is None else f"{float(row['Friction']):.6f}",
            "" if _f(row.get("Direction")) is None else f"{float(row['Direction']):.1f}",
            "" if _f(row.get("Energy")) is None else f"{float(row['Energy']):.6f}",
            "" if _f(row.get("CPUUtil")) is None else f"{float(row['CPUUtil']):.6f}",
            "" if _f(row.get("PSI")) is None else f"{float(row['PSI']):.6f}",
            f"{float(report.get('score', 0.0)):.6f}",
            str(report.get("level", "")),
            str(report.get("decision", "")),
            str(report.get("bad_count", "")),
            str(report.get("good_count", "")),
            str(report.get("medium_band", "")),
            str(report.get("reason", "")),
            ",".join([str(x) for x in missing]),
        ]

        with open(args.output, "a", encoding="utf-8") as f:
            f.write(",".join(out_row) + "\n")

        decision_counts[out_row[10]] += 1

        if args.print_every and (i % args.print_every == 0):
            print(
                f"[replay] {i}/{len(rows)} t={out_row[0]} decision={out_row[10]} score={out_row[8]} level={out_row[9]}"
            )

    print(f"[replay] wrote {len(rows)} rows to {args.output}")
    print(f"[replay] decisions: {dict(decision_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

