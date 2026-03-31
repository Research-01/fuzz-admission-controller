#!/usr/bin/env python3
import argparse
import csv
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def _parse_time(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _f(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    v = str(s).strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _load_rows(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def _load_offer_rows(path: str) -> Tuple[List[datetime], Dict[str, List[float]], List[str]]:
    rows = _load_rows(path)
    times: List[datetime] = []
    data: Dict[str, List[float]] = {
        "score": [],
        "decision_v": [],
        "cpu": [],
        "ram": [],
        "storage": [],
    }
    decisions: List[str] = []

    for row in rows:
        ts = _parse_time(row.get("Time", ""))
        if ts is None:
            continue
        times.append(ts)
        d = (row.get("Decision") or "").strip().lower()
        decisions.append(d)
        data["decision_v"].append(1.0 if d == "allow" else 0.0)
        data["score"].append(_f(row.get("Score")) or 0.0)
        data["cpu"].append(_f(row.get("SellableCPU")) or 0.0)
        data["ram"].append(_f(row.get("SellableRAM_GB")) or 0.0)
        data["storage"].append(_f(row.get("SellableStorage_GB")) or 0.0)

    return times, data, decisions


def _load_fuzzy_score_rows(path: str) -> Tuple[List[datetime], Dict[str, List[float]]]:
    rows = _load_rows(path)
    times: List[datetime] = []
    data: Dict[str, List[float]] = {
        "friction": [],
        "energy": [],
        "cpu": [],
        "psi": [],
        "score": [],
    }
    for row in rows:
        ts = _parse_time(row.get("Time", ""))
        if ts is None:
            continue
        times.append(ts)
        data["friction"].append(_f(row.get("FrictionSigned")) or 0.0)
        data["energy"].append(_f(row.get("Energy")) or 0.0)
        data["cpu"].append(_f(row.get("CPUUtil")) or 0.0)
        data["psi"].append(_f(row.get("PSI")) or 0.0)
        data["score"].append(_f(row.get("Score")) or 0.0)
    return times, data


def main() -> int:
    p = argparse.ArgumentParser(description="Plot resource-offer behavior and decision changes.")
    p.add_argument("--input", default="/tmp/ksense/resource_offer.csv", help="Input resource_offer.csv")
    p.add_argument(
        "--fuzzy-score",
        default="",
        help="Optional fuzzy_score.csv path to plot friction/energy/cpu/psi as well",
    )
    p.add_argument("--output", default="resource_offer_behavior.png", help="Output PNG")
    p.add_argument("--show", action="store_true", help="Show interactive plot")
    args = p.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is required. Install with: python3 -m pip install matplotlib")
        return 1

    t_offer, offer, decisions = _load_offer_rows(args.input)
    if not t_offer:
        print(f"No valid timestamped rows found in {args.input}")
        return 1

    c = Counter(decisions)
    print("Decision counts:", dict(c))
    if decisions:
        allow_ratio = 100.0 * (sum(1 for x in decisions if x == "allow") / len(decisions))
        print(f"Allow ratio: {allow_ratio:.2f}%")

    transitions = 0
    for i in range(1, len(decisions)):
        if decisions[i] != decisions[i - 1]:
            transitions += 1
            t0 = t_offer[i - 1].strftime("%H:%M:%S")
            t1 = t_offer[i].strftime("%H:%M:%S")
            print(f"Transition #{transitions}: {t0} -> {t1} ({decisions[i-1]} -> {decisions[i]})")
    if transitions == 0:
        print("No decision transitions detected in this file.")

    fuzzy_enabled = bool(args.fuzzy_score.strip())
    t_fuzzy: List[datetime] = []
    fuzzy: Dict[str, List[float]] = {}
    if fuzzy_enabled:
        try:
            t_fuzzy, fuzzy = _load_fuzzy_score_rows(args.fuzzy_score)
        except FileNotFoundError:
            print(f"Warning: fuzzy score file not found: {args.fuzzy_score} (plotting without friction/energy)")
            fuzzy_enabled = False

    rows = 4 if fuzzy_enabled and t_fuzzy else 3
    fig, axes = plt.subplots(rows, 1, figsize=(14, 12 if rows == 4 else 10), sharex=True)

    row_i = 0
    if rows == 4:
        axes[row_i].plot(t_fuzzy, fuzzy["friction"], color="#ff7f0e", linewidth=1.3, label="Friction (signed)")
        axes[row_i].plot(t_fuzzy, fuzzy["energy"], color="#9467bd", linewidth=1.3, label="Energy")
        axes[row_i].set_ylabel("Friction/Energy")
        axes[row_i].legend(loc="upper right")
        axes[row_i].grid(alpha=0.3)
        row_i += 1

    axes[row_i].plot(t_offer, offer["score"], color="#1f77b4", linewidth=1.5, label="Fuzzy Score")
    axes[row_i].axhline(45.0, color="#999999", linestyle="--", linewidth=1, label="Medium threshold")
    axes[row_i].axhline(70.0, color="#cc3333", linestyle="--", linewidth=1, label="High threshold")
    axes[row_i].set_ylabel("Score")
    axes[row_i].legend(loc="upper right")
    axes[row_i].grid(alpha=0.3)
    row_i += 1

    axes[row_i].step(
        t_offer,
        offer["decision_v"],
        where="post",
        color="#2ca02c",
        linewidth=1.5,
        label="Decision (allow=1, deny=0)",
    )
    axes[row_i].set_ylim(-0.1, 1.1)
    axes[row_i].set_ylabel("Decision")
    axes[row_i].legend(loc="upper right")
    axes[row_i].grid(alpha=0.3)
    row_i += 1

    axes[row_i].plot(t_offer, offer["cpu"], color="#d62728", linewidth=1.5, label="Sellable CPU")
    axes[row_i].plot(t_offer, offer["ram"], color="#17becf", linewidth=1.5, label="Sellable RAM (GB)")
    ax_storage = axes[row_i].twinx()
    ax_storage.plot(
        t_offer,
        offer["storage"],
        color="#8c564b",
        linewidth=1.4,
        linestyle="--",
        label="Sellable Storage (GB)",
    )
    axes[row_i].set_ylabel("CPU / RAM")
    ax_storage.set_ylabel("Storage")
    axes[row_i].set_xlabel("Time")
    axes[row_i].grid(alpha=0.3)
    lines_left, labels_left = axes[row_i].get_legend_handles_labels()
    lines_right, labels_right = ax_storage.get_legend_handles_labels()
    axes[row_i].legend(lines_left + lines_right, labels_left + labels_right, loc="upper right")

    fig.suptitle("Resource Offer + Controller Behavior", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
