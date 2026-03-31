#!/usr/bin/env python3
import argparse
import csv
import os
from datetime import datetime
from typing import Dict, List, Optional


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


def _f(s: str) -> Optional[float]:
    v = (s or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _series(rows: List[dict], key: str, default: float = 0.0) -> List[float]:
    out: List[float] = []
    for r in rows:
        v = _f(r.get(key, ""))
        out.append(default if v is None else v)
    return out


def _decision_series(rows: List[dict]) -> List[float]:
    out: List[float] = []
    for r in rows:
        d = (r.get("decision") or "").strip().lower()
        out.append(1.0 if d == "allow" else 0.0)
    return out


def _load(path: str) -> Dict[str, List]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    times: List[datetime] = []
    kept_rows: List[dict] = []
    for row in rows:
        ts = _parse_time(row.get("capture_time", ""))
        if ts is None:
            continue
        times.append(ts)
        kept_rows.append(row)
    return {"times": times, "rows": kept_rows}


def _save_usage_plot(out_dir: str, times: List[datetime], rows: List[dict]) -> str:
    import matplotlib.pyplot as plt

    cpu = _series(rows, "current_cpu_pct")
    psi = _series(rows, "current_psi_pct")
    disk = _series(rows, "current_disk_pct")
    ram = _series(rows, "current_ram_gb")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(times, cpu, label="Current CPU %", color="#d62728")
    axes[0].plot(times, psi, label="Current PSI %", color="#1f77b4")
    axes[0].plot(times, disk, label="Current Storage Used %", color="#8c564b", linestyle="--")
    axes[0].set_ylabel("Percent")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(times, ram, label="Current RAM (GB)", color="#17becf")
    axes[1].set_ylabel("RAM GB")
    axes[1].set_xlabel("Time")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")
    fig.suptitle("Current Resource Usage (CPU/PSI/RAM/Storage)")
    fig.tight_layout()
    out = os.path.join(out_dir, "graph1_current_usage.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _save_decision_sellable_plot(out_dir: str, times: List[datetime], rows: List[dict]) -> str:
    import matplotlib.pyplot as plt

    decision = _decision_series(rows)
    sell_cpu = _series(rows, "sellable_cpu")
    sell_ram = _series(rows, "sellable_ram_gb")
    sell_storage = _series(rows, "sellable_storage_gb")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].step(times, decision, where="post", color="#2ca02c", label="Decision (allow=1, deny=0)")
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].set_ylabel("Decision")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(times, sell_cpu, color="#d62728", label="Sellable CPU")
    axes[1].plot(times, sell_ram, color="#17becf", label="Sellable RAM GB")
    ax2 = axes[1].twinx()
    ax2.plot(times, sell_storage, color="#8c564b", linestyle="--", label="Sellable Storage GB")
    axes[1].set_ylabel("CPU / RAM")
    ax2.set_ylabel("Storage")
    axes[1].set_xlabel("Time")
    axes[1].grid(alpha=0.3)
    left_lines, left_labels = axes[1].get_legend_handles_labels()
    right_lines, right_labels = ax2.get_legend_handles_labels()
    axes[1].legend(left_lines + right_lines, left_labels + right_labels, loc="upper right")
    fig.suptitle("Controller Decision vs Sellable Resources")
    fig.tight_layout()
    out = os.path.join(out_dir, "graph2_decision_sellable.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _save_fuzzy_plot(out_dir: str, times: List[datetime], rows: List[dict]) -> str:
    import matplotlib.pyplot as plt

    friction = _series(rows, "friction_signed")
    energy = _series(rows, "energy_scaled")
    cpu = _series(rows, "fuzzy_cpu_pct")
    psi = _series(rows, "fuzzy_psi_pct")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(times, friction, color="#ff7f0e", label="Friction (signed)")
    axes[0].plot(times, energy, color="#9467bd", label="Energy")
    axes[0].set_ylabel("Friction / Energy")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(times, cpu, color="#d62728", label="Fuzzy CPU % input")
    axes[1].plot(times, psi, color="#1f77b4", label="Fuzzy PSI % input")
    axes[1].set_ylabel("Percent")
    axes[1].set_xlabel("Time")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")
    fig.suptitle("Fuzzy Inputs Over Time")
    fig.tight_layout()
    out = os.path.join(out_dir, "graph3_friction_cpu_psi.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _save_combined_plot(out_dir: str, times: List[datetime], rows: List[dict], total_ram_gb: float, total_storage_gb: float) -> str:
    import matplotlib.pyplot as plt

    friction = _series(rows, "friction_signed")
    cpu = _series(rows, "current_cpu_pct")
    psi = _series(rows, "current_psi_pct")
    disk = _series(rows, "current_disk_pct")
    ram_gb = _series(rows, "current_ram_gb")
    ram_pct = [(x / total_ram_gb) * 100.0 for x in ram_gb]

    decision = _decision_series(rows)
    score = _series(rows, "score")
    sell_cpu = _series(rows, "sellable_cpu")
    sell_ram = _series(rows, "sellable_ram_gb")
    sell_storage = _series(rows, "sellable_storage_gb")
    sell_ram_pct = [(x / total_ram_gb) * 100.0 for x in sell_ram]
    sell_storage_pct = [(x / total_storage_gb) * 100.0 for x in sell_storage]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(times, friction, color="#ff7f0e", label="Friction")
    axes[0].plot(times, cpu, color="#d62728", label="CPU %")
    axes[0].plot(times, psi, color="#1f77b4", label="PSI %")
    axes[0].plot(times, disk, color="#8c564b", linestyle="--", label="Storage Used %")
    axes[0].plot(times, ram_pct, color="#17becf", label="RAM Used %")
    axes[0].set_ylabel("Input Signals")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].step(times, decision, where="post", color="#2ca02c", label="Decision (allow=1)")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_ylabel("Decision")
    axes[1].grid(alpha=0.3)
    ax_score = axes[1].twinx()
    ax_score.plot(times, score, color="#1f77b4", label="Fuzzy Score", linewidth=1.2)
    ax_score.axhline(45.0, color="#999999", linestyle="--", linewidth=1)
    ax_score.axhline(70.0, color="#cc3333", linestyle="--", linewidth=1)
    ax_score.set_ylabel("Score")
    d_lines, d_labels = axes[1].get_legend_handles_labels()
    s_lines, s_labels = ax_score.get_legend_handles_labels()
    axes[1].legend(d_lines + s_lines, d_labels + s_labels, loc="upper right")

    axes[2].plot(times, sell_cpu, color="#d62728", label="Sellable CPU")
    axes[2].plot(times, sell_ram_pct, color="#17becf", label="Sellable RAM %")
    axes[2].plot(times, sell_storage_pct, color="#8c564b", linestyle="--", label="Sellable Storage %")
    axes[2].set_ylabel("Sellable")
    axes[2].set_xlabel("Time")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper right")
    fig.suptitle("Friction/CPU/PSI + Score/Decision vs Sellable Resources")
    fig.tight_layout()
    out = os.path.join(out_dir, "graph4_combined.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Plot debug trace captured from /resource_offer_debug.")
    p.add_argument("--input", required=True, help="Input CSV from capture_resource_offer_debug.py")
    p.add_argument("--out-dir", default="/tmp/ksense/plots", help="Output directory")
    p.add_argument("--total-ram-gb", type=float, default=2048.0, help="Node total RAM in GB")
    p.add_argument("--total-storage-gb", type=float, default=80078.0, help="Node total storage in GB")
    args = p.parse_args()

    try:
        import matplotlib.pyplot  # noqa: F401
    except Exception:
        print("matplotlib is required. Install with: python3 -m pip install matplotlib")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    loaded = _load(args.input)
    times: List[datetime] = loaded["times"]
    rows: List[dict] = loaded["rows"]
    if not times:
        print(f"No valid rows found in {args.input}")
        return 1

    out1 = _save_usage_plot(args.out_dir, times, rows)
    out2 = _save_decision_sellable_plot(args.out_dir, times, rows)
    out3 = _save_fuzzy_plot(args.out_dir, times, rows)
    out4 = _save_combined_plot(args.out_dir, times, rows, args.total_ram_gb, args.total_storage_gb)

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")
    print(f"Saved: {out3}")
    print(f"Saved: {out4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
