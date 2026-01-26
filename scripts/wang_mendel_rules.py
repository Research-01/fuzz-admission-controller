#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict


def _trimf(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b != a else 0.0
    return (c - x) / (c - b) if c != b else 0.0


def _trapmf(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a) if b != a else 0.0
    return (d - x) / (d - c) if d != c else 0.0


def _gaussmf(x, mean, sigma):
    if sigma <= 0:
        return 0.0
    return math.exp(-((x - mean) ** 2) / (2.0 * sigma * sigma))


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _auto_pct(v):
    """
    If v looks like 0..1, convert to 0..100. If it already looks like percent, keep it.
    """
    if v <= 1.5:  # heuristic
        return _clamp(v * 100.0, 0.0, 100.0)
    return _clamp(v, 0.0, 100.0)


def _t_norm(values, mode):
    if mode == "min":
        return min(values)
    if mode == "prod":
        p = 1.0
        for v in values:
            p *= v
        return p
    raise ValueError(f"Unknown t-norm: {mode}")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Rule triggering check for an existing fuzzy rule base (72 rules)."
    )
    p.add_argument("--input", required=True, help="Input CSV file.")
    p.add_argument("--usage", default="rule_usage.csv", help="Output per-rule usage summary CSV.")
    p.add_argument("--firing", default="rule_firing.csv", help="Output per-sample triggered rules CSV.")
    p.add_argument("--eps", type=float, default=0.01, help="Activation threshold epsilon.")
    p.add_argument("--t_norm", choices=["min", "prod"], default="min", help="AND operator for rule firing.")
    p.add_argument(
        "--log_all",
        action="store_true",
        help="If set, logs all 72 rules each step (large file). Otherwise logs only triggered rules.",
    )
    return p.parse_args()


def build_rulebase():
    """
    Encodes your provided tables as:
      (CPU, PSI, Friction, Energy) -> Output
    Labels are standardized to:
      CPU: Normal, High
      PSI: Low, Medium, High
      Friction: UnderUtilized, Short, Moderate, Long
      Energy: Short, Moderate, Long
      Output: UnderUtilized, Short, Moderate, Long
    """
    F = ["UnderUtilized", "Short", "Moderate", "Long"]
    E = ["Short", "Moderate", "Long"]

    def tbl(rows):
        m = {}
        for fi, f_lab in enumerate(F):
            for ei, e_lab in enumerate(E):
                m[(f_lab, e_lab)] = rows[fi][ei]
        return m

    n_l = tbl(
        [
            ["UnderUtilized", "Short", "Short"],
            ["Short", "Short", "Moderate"],
            ["Short", "Moderate", "Moderate"],
            ["Moderate", "Long", "Long"],
        ]
    )

    n_m = tbl(
        [
            ["Short", "Short", "Moderate"],
            ["Short", "Moderate", "Moderate"],
            ["Moderate", "Moderate", "Long"],
            ["Moderate", "Long", "Long"],
        ]
    )

    n_h = tbl(
        [
            ["Short", "Moderate", "Moderate"],
            ["Moderate", "Moderate", "Long"],
            ["Moderate", "Long", "Long"],
            ["Long", "Long", "Long"],
        ]
    )

    h_l = tbl(
        [
            ["Short", "Short", "Short"],
            ["Short", "Moderate", "Moderate"],
            ["Moderate", "Long", "Long"],
            ["Long", "Long", "Long"],
        ]
    )

    h_m = tbl(
        [
            ["Short", "Short", "Moderate"],
            ["Moderate", "Moderate", "Moderate"],
            ["Moderate", "Long", "Long"],
            ["Long", "Long", "Long"],
        ]
    )

    h_h = tbl(
        [
            ["Moderate", "Moderate", "Long"],
            ["Moderate", "Moderate", "Long"],
            ["Moderate", "Long", "Long"],
            ["Long", "Long", "Long"],
        ]
    )

    big = {
        ("Normal", "Low"): n_l,
        ("Normal", "Medium"): n_m,
        ("Normal", "High"): n_h,
        ("High", "Low"): h_l,
        ("High", "Medium"): h_m,
        ("High", "High"): h_h,
    }

    rulebase = {}
    for cpu in ["Normal", "High"]:
        for psi in ["Low", "Medium", "High"]:
            m = big[(cpu, psi)]
            for fric in ["UnderUtilized", "Short", "Moderate", "Long"]:
                for eng in ["Short", "Moderate", "Long"]:
                    out = m[(fric, eng)]
                    rulebase[(cpu, psi, fric, eng)] = out

    return rulebase


def memberships(cpu_pct, psi_pct, fric_val, eng_val):
    """
    Define membership functions. Adjust parameters to match your system if needed.
    """
    cpu_states = {
        "Normal": _trapmf(cpu_pct, 0, 0, 60, 80),
        "High": _trapmf(cpu_pct, 60, 80, 100, 100),
    }
    psi_states = {
        "Low": _trapmf(psi_pct, 0, 0, 20, 40),
        "Medium": _trimf(psi_pct, 20, 50, 80),
        "High": _trapmf(psi_pct, 60, 80, 100, 100),
    }
    fric_states = {
        "UnderUtilized": _trapmf(fric_val, -300, -300, 0, 50),
        "Short": _gaussmf(fric_val, 50, 25),
        "Moderate": _gaussmf(fric_val, 125, 40),
        "Long": _trapmf(fric_val, 150, 250, 300, 300),
    }
    eng_states = {
        "Short": _trapmf(eng_val, 0, 0, 90, 150),
        "Moderate": _gaussmf(eng_val, 150, 45),
        "Long": _trapmf(eng_val, 150, 210, 300, 300),
    }
    return cpu_states, psi_states, fric_states, eng_states


def main():
    args = _parse_args()
    rulebase = build_rulebase()

    stats = defaultdict(lambda: {"count": 0, "sum_alpha": 0.0, "max_alpha": 0.0})
    total_steps = 0
    firing_rows = []

    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_steps += 1
            ts = row.get("Time", "")

            try:
                fric = float(row["FrictionSigned"])
                eng = float(row["Energy"])
                cpu = float(row["CPUUtil"])
                psi = float(row["PSI"])
            except (KeyError, ValueError):
                continue

            latency = row.get("Latency", "")
            decision = row.get("Decision", "")

            cpu_pct = _auto_pct(cpu)
            psi_pct = _auto_pct(psi)
            fric_val = _clamp(fric, -300.0, 300.0)
            eng_val = _clamp(eng, 0.0, 300.0)

            cpu_m, psi_m, fric_m, eng_m = memberships(cpu_pct, psi_pct, fric_val, eng_val)

            for (cpu_lab, psi_lab, fric_lab, eng_lab), out_lab in rulebase.items():
                alpha = _t_norm(
                    [cpu_m[cpu_lab], psi_m[psi_lab], fric_m[fric_lab], eng_m[eng_lab]],
                    args.t_norm,
                )

                st = stats[(cpu_lab, psi_lab, fric_lab, eng_lab, out_lab)]
                if alpha > 0.0:
                    st["sum_alpha"] += alpha
                    st["max_alpha"] = max(st["max_alpha"], alpha)
                if alpha >= args.eps:
                    st["count"] += 1

                if args.log_all or alpha >= args.eps:
                    firing_rows.append(
                        {
                            "Time": ts,
                            "CPU": cpu_lab,
                            "PSI": psi_lab,
                            "Friction": fric_lab,
                            "Energy": eng_lab,
                            "Output": out_lab,
                            "Alpha": f"{alpha:.6f}",
                            "CPUUtil": f"{cpu_pct:.3f}",
                            "PSIVal": f"{psi_pct:.3f}",
                            "FrictionSigned": f"{fric_val:.3f}",
                            "EnergyVal": f"{eng_val:.3f}",
                            "Latency": latency,
                            "Decision": decision,
                        }
                    )

    with open(args.usage, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "CPU",
                "PSI",
                "Friction",
                "Energy",
                "Output",
                "TriggeredCount",
                "TriggeredRate",
                "AvgAlphaAllSteps",
                "MaxAlpha",
            ]
        )

        items = sorted(
            stats.items(),
            key=lambda kv: (kv[1]["count"], kv[1]["max_alpha"]),
            reverse=True,
        )

        for (cpu_lab, psi_lab, fric_lab, eng_lab, out_lab), st in items:
            triggered_count = st["count"]
            triggered_rate = (triggered_count / total_steps) if total_steps else 0.0
            avg_alpha = (st["sum_alpha"] / total_steps) if total_steps else 0.0
            writer.writerow(
                [
                    cpu_lab,
                    psi_lab,
                    fric_lab,
                    eng_lab,
                    out_lab,
                    triggered_count,
                    f"{triggered_rate:.6f}",
                    f"{avg_alpha:.6f}",
                    f"{st['max_alpha']:.6f}",
                ]
            )

    with open(args.firing, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Time",
            "CPU",
            "PSI",
            "Friction",
            "Energy",
            "Output",
            "Alpha",
            "CPUUtil",
            "PSIVal",
            "FrictionSigned",
            "EnergyVal",
            "Latency",
            "Decision",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(firing_rows)


if __name__ == "__main__":
    main()
