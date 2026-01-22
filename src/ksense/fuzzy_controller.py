import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from .helpers import ensure_csv

@dataclass
class FuzzyConfig:
    csv_path: str = os.getenv("KSENSE_METRICS_CSV", "/tmp/ksense/kernel_metrics.csv")
    short_win_s: int = int(os.getenv("FUZZY_SHORT_WIN_S", "10"))
    long_win_s: int = int(os.getenv("FUZZY_LONG_WIN_S", "60"))
    max_csv_lines: int = int(os.getenv("FUZZY_MAX_CSV_LINES", "2000"))

    bad_threshold: int = int(os.getenv("FUZZY_BAD_THRESHOLD", "3"))
    good_threshold: int = int(os.getenv("FUZZY_GOOD_THRESHOLD", "2"))

    allow_on_missing: bool = os.getenv("FUZZY_ALLOW_ON_MISSING", "false").lower() == "true"
    monitor_csv: str = os.getenv("FUZZY_MONITOR_CSV", "/tmp/ksense/fuzzy_monitor.csv")



def _percentile(values, p):
    if not values:
        return None
    vals = sorted(values)
    k = int(round((p / 100.0) * (len(vals) - 1)))
    return vals[max(0, min(k, len(vals) - 1))]


def _trimf(x, params):
    a, b, c = params
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def _trapmf(x, params):
    a, b, c, d = params
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b != a else 0.0
    return (d - x) / (d - c) if d != c else 0.0


def _gaussmf(x, mean, sigma):
    return math.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


def _read_last_lines(path, max_lines, max_bytes=1024 * 1024):
    try:
        size = os.stat(path).st_size
    except FileNotFoundError:
        return []
    with open(path, "r", encoding="utf-8") as f:
        if size > max_bytes:
            f.seek(max(0, size - max_bytes))
        data = f.read().splitlines()
    return data[-max_lines:]


def _parse_recent_metrics(cfg: FuzzyConfig):
    lines = _read_last_lines(cfg.csv_path, cfg.max_csv_lines)
    if not lines:
        return []
    headers = lines[0].split(",")
    try:
        time_idx = headers.index("Time")
        fric_idx = headers.index("Friction")
        eng_idx = headers.index("Energy")
        dir_idx = headers.index("Direction")
    except ValueError:
        return []

    now = datetime.now()
    long_cutoff = now - timedelta(seconds=cfg.long_win_s)
    entries = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) <= max(time_idx, fric_idx, eng_idx):
            continue
        ts = parts[time_idx]
        try:
            ts_dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if ts_dt < long_cutoff:
            continue
        fric = parts[fric_idx]
        eng = parts[eng_idx]
        direction = parts[dir_idx]
        try:
            fric_val = float(fric) if fric else None
        except ValueError:
            fric_val = None
        try:
            eng_val = float(eng) if eng else None
        except ValueError:
            eng_val = None
        try:
            dir_val = float(direction) if direction else None
        except ValueError:
            dir_val = None
        entries.append((ts_dt, fric_val, eng_val, dir_val))
    return entries


class ResourceSampler:
    def __init__(self):
        self._prev_total = None
        self._prev_idle = None

    def cpu_util(self):
        try:
            with open("/proc/stat", "r", encoding="utf-8") as f:
                line = f.readline()
        except FileNotFoundError:
            return None
        parts = line.split()
        if len(parts) < 5:
            return None
        values = [int(v) for v in parts[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values)
        if self._prev_total is None:
            self._prev_total = total
            self._prev_idle = idle
            return None
        total_delta = total - self._prev_total
        idle_delta = idle - self._prev_idle
        self._prev_total = total
        self._prev_idle = idle
        if total_delta <= 0:
            return None
        return max(0.0, min(1.0, 1.0 - (idle_delta / total_delta)))

    def mem_util(self):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                data = f.read().splitlines()
        except FileNotFoundError:
            return None
        mem_total = None
        mem_avail = None
        for line in data:
            if line.startswith("MemTotal:"):
                mem_total = float(line.split()[1])
            elif line.startswith("MemAvailable:"):
                mem_avail = float(line.split()[1])
        if not mem_total or mem_avail is None:
            return None
        used = max(0.0, mem_total - mem_avail)
        return max(0.0, min(1.0, used / mem_total))

    def psi(self):
        psi_vals = []
        for path in ("/proc/pressure/cpu", "/proc/pressure/memory", "/proc/pressure/io"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
            except FileNotFoundError:
                continue
            for line in lines:
                if line.startswith("some"):
                    for part in line.split():
                        if part.startswith("avg10="):
                            try:
                                psi_vals.append(float(part.split("=")[1]))
                            except ValueError:
                                pass
        if not psi_vals:
            return None
        psi = max(psi_vals)
        # Kernel PSI avg is expressed as a percentage. Normalize to 0-1 if needed.
        if psi > 1.0:
            psi = psi / 100.0
        return psi


class FuzzyController:
    def __init__(self, cfg: Optional[FuzzyConfig] = None):
        self.cfg = cfg or FuzzyConfig()
        self._lock = threading.Lock()
        self._bad = 0
        self._good = 0
        self._last_decision = "allow"
        self._sampler = ResourceSampler()
        self._last_monitor_ts = 0.0
        ensure_csv(self.cfg.monitor_csv, ["Time", "FrictionSigned", "Energy", "CPUUtil", "PSI"])

    def _fuzzy_score(self, fric, eng, cpu, psi):
        # Fixed input axes:
        # CPU: 0-100, PSI: 0-100, Friction: -300..300, Energy: 0..300
        cpu_pct = max(0.0, min(100.0, cpu * 100.0))
        psi_pct = max(0.0, min(100.0, psi * 100.0))
        fric_val = max(-300.0, min(300.0, fric))
        eng_val = max(0.0, min(300.0, eng))

        # CPU membership
        cpu_normal = _trapmf(cpu_pct, [0, 0, 60, 80])
        cpu_high = _trapmf(cpu_pct, [60, 80, 100, 100])

        # PSI membership
        psi_low = _trapmf(psi_pct, [0, 0, 20, 40])
        psi_med = _trimf(psi_pct, [20, 50, 80])
        psi_high = _trapmf(psi_pct, [60, 80, 100, 100])

        # Friction membership (-300..300)
        fric_under = _trapmf(fric_val, [-300, -300, 0, 50])
        fric_short = _gaussmf(fric_val, 50, 25)
        fric_mod = _gaussmf(fric_val, 125, 40)
        fric_long = _trapmf(fric_val, [150, 250, 300, 300])

        # Energy membership (0..300)
        eng_short = _trapmf(eng_val, [0, 0, 90, 150])
        eng_mod = _gaussmf(eng_val, 150, 45)
        eng_long = _trapmf(eng_val, [150, 210, 300, 300])

        # Output memberships (0..100)
        out_under = [0, 0, 15, 30]
        out_short = [15, 40, 65]
        out_mod = [40, 65, 90]
        out_long = [65, 90, 100, 100]

        cpu_states = {"normal": cpu_normal, "high": cpu_high}
        psi_states = {"low": psi_low, "med": psi_med, "high": psi_high}
        fric_states = {
            "under": fric_under,
            "short": fric_short,
            "mod": fric_mod,
            "long": fric_long,
        }
        eng_states = {"short": eng_short, "mod": eng_mod, "long": eng_long}

        rule_table = {
            ("normal", "low"): {
                "under": {"short": "under", "mod": "short", "long": "short"},
                "short": {"short": "short", "mod": "short", "long": "mod"},
                "mod": {"short": "short", "mod": "mod", "long": "mod"},
                "long": {"short": "mod", "mod": "long", "long": "long"},
            },
            ("normal", "med"): {
                "under": {"short": "short", "mod": "short", "long": "mod"},
                "short": {"short": "short", "mod": "mod", "long": "mod"},
                "mod": {"short": "mod", "mod": "mod", "long": "long"},
                "long": {"short": "mod", "mod": "long", "long": "long"},
            },
            ("normal", "high"): {
                "under": {"short": "short", "mod": "mod", "long": "mod"},
                "short": {"short": "mod", "mod": "mod", "long": "long"},
                "mod": {"short": "mod", "mod": "long", "long": "long"},
                "long": {"short": "long", "mod": "long", "long": "long"},
            },
            ("high", "low"): {
                "under": {"short": "short", "mod": "short", "long": "short"},
                "short": {"short": "short", "mod": "mod", "long": "mod"},
                "mod": {"short": "mod", "mod": "long", "long": "long"},
                "long": {"short": "long", "mod": "long", "long": "long"},
            },
            ("high", "med"): {
                "under": {"short": "short", "mod": "short", "long": "mod"},
                "short": {"short": "mod", "mod": "mod", "long": "mod"},
                "mod": {"short": "mod", "mod": "long", "long": "long"},
                "long": {"short": "long", "mod": "long", "long": "long"},
            },
            ("high", "high"): {
                "under": {"short": "mod", "mod": "mod", "long": "long"},
                "short": {"short": "mod", "mod": "mod", "long": "long"},
                "mod": {"short": "mod", "mod": "long", "long": "long"},
                "long": {"short": "long", "mod": "long", "long": "long"},
            },
        }

        out_strength = {"under": 0.0, "short": 0.0, "mod": 0.0, "long": 0.0}
        for cpu_key, cpu_mu in cpu_states.items():
            for psi_key, psi_mu in psi_states.items():
                table = rule_table[(cpu_key, psi_key)]
                for fric_key, fric_mu in fric_states.items():
                    for eng_key, eng_mu in eng_states.items():
                        out_key = table[fric_key][eng_key]
                        strength = min(cpu_mu, psi_mu, fric_mu, eng_mu)
                        if strength > out_strength[out_key]:
                            out_strength[out_key] = strength

        # Aggregate output membership and defuzzify via centroid
        num = 0.0
        den = 0.0
        for x in range(0, 101):
            mu_under = min(out_strength["under"], _trapmf(x, out_under))
            mu_short = min(out_strength["short"], _trimf(x, out_short))
            mu_mod = min(out_strength["mod"], _trimf(x, out_mod))
            mu_long = min(out_strength["long"], _trapmf(x, out_long))
            mu = max(mu_under, mu_short, mu_mod, mu_long)
            num += x * mu
            den += mu
        score = (num / den) if den > 0 else 0.0

        if score >= 70.0:
            level = "high"
        elif score >= 45.0:
            level = "medium"
        else:
            level = "low"
        return score, level, {
            "cpu_pct": cpu_pct,
            "psi_pct": psi_pct,
            "friction_scaled": fric_val,
            "energy_scaled": eng_val,
        }

    def evaluate(self):
        cfg = self.cfg
        entries = _parse_recent_metrics(cfg)
        now = datetime.now()
        short_cutoff = now - timedelta(seconds=cfg.short_win_s)

        fric_vals_short = []
        eng_vals_short = []
        fric_vals_long = []
        eng_vals_long = []
        last_direction = None

        for ts, fric, eng, direction in entries:
            if fric is not None:
                fric_vals_long.append(fric)
                if ts >= short_cutoff:
                    fric_vals_short.append(fric)
            if eng is not None:
                eng_vals_long.append(eng)
                if ts >= short_cutoff:
                    eng_vals_short.append(eng)
            if direction is not None:
                last_direction = direction

        fric_abs_long = [abs(v) for v in fric_vals_long]
        fric_abs_short = [abs(v) for v in fric_vals_short]
        fric_long_p99 = _percentile(fric_abs_long, 99) if fric_abs_long else None
        eng_long_p99 = _percentile(eng_vals_long, 99) if eng_vals_long else None
        fric_short_p99 = _percentile(fric_abs_short, 99) if fric_abs_short else None
        eng_short_p99 = _percentile(eng_vals_short, 99) if eng_vals_short else None

        fric_candidates = [v for v in [fric_short_p99, fric_long_p99] if v is not None]
        eng_candidates = [v for v in [eng_short_p99, eng_long_p99] if v is not None]
        fric = max(fric_candidates) if fric_candidates else None
        eng = max(eng_candidates) if eng_candidates else None
        if fric is not None and last_direction is not None:
            fric = fric * last_direction

        cpu = self._sampler.cpu_util()
        psi = self._sampler.psi()

        missing = [name for name, val in [("friction", fric), ("energy", eng), ("cpu", cpu), ("psi", psi)] if val is None]
        critical_missing = [name for name in missing if name in ("friction", "energy")]
        if critical_missing and not cfg.allow_on_missing:
            self._write_monitor_sample(
                (fric or 0.0) * (last_direction or 1.0),
                eng or 0.0,
                cpu or 0.0,
                psi or 0.0,
            )
            return {
                "decision": "deny",
                "reason": f"missing metrics: {', '.join(missing)}",
                "score": 1.0,
                "level": "high",
                "metrics": {},
                "missing": missing,
            }

        fric_val = fric or 0.0
        eng_val = eng or 0.0
        cpu_val = cpu or 0.0
        psi_val = psi or 0.0
        score, level, scaled = self._fuzzy_score(fric_val, eng_val, cpu_val, psi_val)

        report = {
            "decision": "allow",
            "score": score,
            "level": level,
            "metrics": {
                "direction": last_direction,
                "friction_signed": fric_val,
                "friction_p99_short": fric_short_p99,
                "friction_p99_long": fric_long_p99,
                "energy_p99_short": eng_short_p99,
                "energy_p99_long": eng_long_p99,
                "cpu_util": cpu,
                "psi_avg10": psi,
                "cpu_pct": scaled["cpu_pct"],
                "psi_pct": scaled["psi_pct"],
                "friction_scaled": scaled["friction_scaled"],
                "energy_scaled": scaled["energy_scaled"],
            },
            "missing": missing,
        }
        self._write_monitor_sample(fric_val, eng_val, cpu_val, psi_val)
        return report

    def _write_monitor_sample(self, friction_signed, energy, cpu, psi):
        now = time.time()
        if now - self._last_monitor_ts < 1.0:
            return
        self._last_monitor_ts = now
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.cfg.monitor_csv, "a", newline="") as f:
            f.write(f"{ts_str},{friction_signed:.6f},{energy:.6f},{cpu:.6f},{psi:.6f}\n")

    def decide(self):
        with self._lock:
            report = self.evaluate()
            if report["decision"] == "deny":
                self._bad += 1
                self._good = 0
                decision = "deny" if self._bad >= self.cfg.bad_threshold else self._last_decision
            else:
                if report["level"] == "high":
                    self._bad += 1
                    self._good = 0
                else:
                    self._good += 1
                    self._bad = 0
                if report["level"] == "high" and self._bad >= self.cfg.bad_threshold:
                    decision = "deny"
                elif report["level"] != "high" and self._good >= self.cfg.good_threshold:
                    decision = "allow"
                else:
                    decision = self._last_decision

            self._last_decision = decision
            report["decision"] = decision
            report["bad_count"] = self._bad
            report["good_count"] = self._good
            return report
