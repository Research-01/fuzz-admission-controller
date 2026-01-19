#!/usr/bin/env python3
#
# step_response_prober_3apps_persec.py
#
# Step-response workload prober for up to 3 applications in parallel.
#
# What this version guarantees:
# - Writes ONE CSV ROW PER SECOND (per-second P95/P99, success, timeouts, achieved RPS).
# - Applies a step-response ramp: RPS increases step-by-step; each step lasts --step-hold seconds.
# - Optional knee detection + auto-backoff controller (does NOT exit on knee unless --stop-on-knee is set).
# - Optional saturation stop (success rate / timeouts).
# - HARD stop after --max-runtime-s seconds (default 600 = 10 minutes), even mid-step.
#
# Notes:
# - requests.Session is not thread-safe: uses per-thread Sessions via thread-local storage.
# - Achieved RPS may be lower than target under saturation.

import time
import random
import string
import csv
import json
import argparse
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future, as_completed


# ---------------- CONFIG ----------------
HOSTNODE_IP = "172.22.174.42"

# App 1: Social Media
SOCIAL_PORT = 30080
SOCIAL_ENDPOINT = "/wrk2-api/post/compose"
SOCIAL_URL = f"http://{HOSTNODE_IP}:{SOCIAL_PORT}{SOCIAL_ENDPOINT}"

# App 2: Hotel Reservation (mixed workload)
HOTEL_PORT = 32192
HOTEL_BASE = f"http://{HOSTNODE_IP}:{HOTEL_PORT}"

# App 3: CustomerFeedback
FEEDBACK_URL = f"http://{HOSTNODE_IP}:32001/feedback/analyse"
FEEDBACK_INPUT_FILE = "/home/abd/Desktop/CustomerFeedback/file_processing/feedback_input_text_unique_variations.json"

REQUEST_TIMEOUT_SEC_DEFAULT = 10.0

ENABLE_SOCIAL_DEFAULT = True
ENABLE_HOTEL_DEFAULT = True
ENABLE_FEEDBACK_DEFAULT = True

# Hotel mixed ratios
SEARCH_RATIO = 0.60
RECOMMEND_RATIO = 0.39
USER_RATIO = 0.005

# ---------------- Thread-local sessions ----------------
_tls = threading.local()


def get_session() -> requests.Session:
    sess = getattr(_tls, "session", None)
    if sess is None:
        sess = requests.Session()
        _tls.session = sess
    return sess


# ---------------- SOCIAL PAYLOAD ----------------
def generate_payload_social() -> Dict[str, str]:
    uid = random.randint(0, 962)
    text = "".join(random.choices(string.ascii_letters, k=256))
    return {
        "username": f"user_{uid}",
        "user_id": str(uid),
        "text": text,
        "media_ids": "[]",
        "media_types": "[]",
        "post_type": "0",
    }


# ---------------- HOTEL MIXED REQUESTS ----------------
def hotel_get_user() -> Tuple[str, str]:
    uid = random.randint(0, 500)
    user_name = f"Cornell_{uid}"
    password = str(uid) * 10
    return user_name, password


def hotel_random_lat_lon() -> Tuple[float, float]:
    lat = 38.0235 + (random.randint(0, 481) - 240.5) / 1000.0
    lon = -122.095 + (random.randint(0, 325) - 157.0) / 1000.0
    return lat, lon


def hotel_random_dates_search() -> Tuple[str, str]:
    in_date = random.randint(9, 23)
    out_date = random.randint(in_date + 1, 24)
    return f"2015-04-{in_date:02d}", f"2015-04-{out_date:02d}"


def hotel_req_search() -> Tuple[str, str]:
    in_date, out_date = hotel_random_dates_search()
    lat, lon = hotel_random_lat_lon()
    url = f"{HOTEL_BASE}/hotels?inDate={in_date}&outDate={out_date}&lat={lat}&lon={lon}"
    return "GET", url


def hotel_req_recommend() -> Tuple[str, str]:
    coin = random.random()
    if coin < 0.33:
        req_param = "dis"
    elif coin < 0.66:
        req_param = "rate"
    else:
        req_param = "price"
    lat, lon = hotel_random_lat_lon()
    url = f"{HOTEL_BASE}/recommendations?require={req_param}&lat={lat}&lon={lon}"
    return "GET", url


def hotel_req_user_login() -> Tuple[str, str]:
    user_name, password = hotel_get_user()
    url = f"{HOTEL_BASE}/user?username={user_name}&password={password}"
    return "POST", url


def hotel_req_reserve() -> Tuple[str, str]:
    in_date = random.randint(9, 23)
    out_date = in_date + random.randint(1, 5)
    in_date_str = f"2015-04-{in_date:02d}"
    out_date_str = f"2015-04-{out_date:02d}"

    lat, lon = hotel_random_lat_lon()
    hotel_id = random.randint(1, 80)
    user_name, password = hotel_get_user()
    cust_name = user_name
    num_room = 1

    url = (
        f"{HOTEL_BASE}/reservation?"
        f"inDate={in_date_str}&outDate={out_date_str}&lat={lat}&lon={lon}"
        f"&hotelId={hotel_id}&customerName={cust_name}&username={user_name}"
        f"&password={password}&number={num_room}"
    )
    return "POST", url


def hotel_generate_mixed_request() -> Tuple[str, str]:
    coin = random.random()
    if coin < SEARCH_RATIO:
        return hotel_req_search()
    elif coin < SEARCH_RATIO + RECOMMEND_RATIO:
        return hotel_req_recommend()
    elif coin < SEARCH_RATIO + RECOMMEND_RATIO + USER_RATIO:
        return hotel_req_user_login()
    else:
        return hotel_req_reserve()


# ---------------- FEEDBACK DATA ----------------
def load_feedback_data(path: str) -> List[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Feedback input file must be a non-empty JSON list: {path}")
    return data


# ---------------- Low-level single request functions ----------------
# Return: (ok, latency_ms, timed_out)
def do_social_one(timeout_s: float) -> Tuple[bool, Optional[float], bool]:
    sess = get_session()
    try:
        start = time.perf_counter()
        resp = sess.post(SOCIAL_URL, data=generate_payload_social(), timeout=timeout_s)
        dur_ms = (time.perf_counter() - start) * 1000.0
        ok = (resp.status_code == 200)
        return ok, (dur_ms if ok else None), False
    except requests.exceptions.Timeout:
        return False, None, True
    except requests.exceptions.RequestException:
        return False, None, False


def do_hotel_one(timeout_s: float) -> Tuple[bool, Optional[float], bool]:
    sess = get_session()
    method, url = hotel_generate_mixed_request()
    try:
        start = time.perf_counter()
        if method == "GET":
            resp = sess.get(url, timeout=timeout_s)
        else:
            resp = sess.post(url, timeout=timeout_s)
        dur_ms = (time.perf_counter() - start) * 1000.0
        ok = (resp.status_code == 200)
        return ok, (dur_ms if ok else None), False
    except requests.exceptions.Timeout:
        return False, None, True
    except requests.exceptions.RequestException:
        return False, None, False


def do_feedback_one(timeout_s: float, feedback_data: List[dict]) -> Tuple[bool, Optional[float], bool]:
    sess = get_session()
    payload = random.choice(feedback_data)
    try:
        start = time.perf_counter()
        resp = sess.post(FEEDBACK_URL, json=payload, timeout=timeout_s)
        dur_ms = (time.perf_counter() - start) * 1000.0
        ok = (resp.status_code == 200)
        return ok, (dur_ms if ok else None), False
    except requests.exceptions.Timeout:
        return False, None, True
    except requests.exceptions.RequestException:
        return False, None, False


# ---------------- Metrics ----------------
def percentile(samples: List[float], p: float) -> Optional[float]:
    if not samples:
        return None
    return float(np.percentile(samples, p))


@dataclass
class WindowResult:
    target_rps: float
    duration_s: float
    sent: int
    ok: int
    timeouts: int
    latencies_ms: List[float]

    @property
    def achieved_rps(self) -> float:
        return self.sent / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def success_rate(self) -> float:
        return (self.ok / self.sent) if self.sent > 0 else 0.0

    @property
    def p99_ms(self) -> Optional[float]:
        return percentile(self.latencies_ms, 99.0)

    @property
    def p95_ms(self) -> Optional[float]:
        return percentile(self.latencies_ms, 95.0)


# ---------------- Rate-controlled 1-second window runner ----------------
def run_window_rate_controlled(
    fn,
    fn_args: Tuple,
    target_rps: float,
    duration_s: float,
    timeout_s: float,
    max_workers: int,
    max_pending: int,
    deadline_mono: Optional[float],
) -> WindowResult:
    """
    Schedule requests at ~target_rps for duration_s, but ALSO stop if deadline_mono is reached.
    """
    if target_rps <= 0.0:
        return WindowResult(target_rps, max(duration_s, 1e-6), sent=0, ok=0, timeouts=0, latencies_ms=[])

    latencies: List[float] = []
    sent = ok = timeouts = 0

    start_t = time.monotonic()
    end_t = start_t + duration_s
    interval = 1.0 / target_rps
    next_send = start_t

    pending: List[Future] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while True:
            now = time.monotonic()

            if deadline_mono is not None and now >= deadline_mono:
                break

            # Drain completed futures
            if pending:
                still_pending = []
                for fut in pending:
                    if fut.done():
                        try:
                            _ok, lat_ms, _to = fut.result()
                            sent += 1
                            if _to:
                                timeouts += 1
                            if _ok and lat_ms is not None:
                                ok += 1
                                latencies.append(lat_ms)
                        except Exception:
                            sent += 1
                    else:
                        still_pending.append(fut)
                pending = still_pending

            if now >= end_t:
                break

            # Backpressure: avoid unbounded queueing on the load generator itself
            if len(pending) >= max_pending:
                time.sleep(0.001)
                continue

            if now >= next_send:
                pending.append(pool.submit(fn, timeout_s, *fn_args))
                next_send += interval
                continue

            sleep_for = min(0.01, max(0.0, next_send - now))
            time.sleep(sleep_for)

        # Final drain (bounded by deadline)
        for fut in as_completed(pending):
            if deadline_mono is not None and time.monotonic() >= deadline_mono:
                break
            try:
                _ok, lat_ms, _to = fut.result()
                sent += 1
                if _to:
                    timeouts += 1
                if _ok and lat_ms is not None:
                    ok += 1
                    latencies.append(lat_ms)
            except Exception:
                sent += 1

    actual_dur = max(0.001, time.monotonic() - start_t)
    return WindowResult(target_rps=target_rps, duration_s=actual_dur, sent=sent, ok=ok, timeouts=timeouts, latencies_ms=latencies)


# ---------------- Knee/saturation logic ----------------
def knee_detect_p99(
    current_p99: Optional[float],
    history_p99: List[Optional[float]],
    min_history: int,
    jump_factor: float,
) -> bool:
    """
    Knee trigger if current_p99 jumps by jump_factor over median of last min_history valid points.
    """
    if current_p99 is None:
        return True
    vals = [v for v in history_p99 if v is not None]
    if len(vals) < min_history:
        return False
    ref = float(np.median(vals[-min_history:]))
    if ref <= 0:
        return False
    return current_p99 >= jump_factor * ref


def saturation_detect(success_rate: float, timeouts: int, max_timeouts: int, min_success_rate: float) -> bool:
    return (success_rate < min_success_rate) or (timeouts > max_timeouts)


# ---------------- CLI ----------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Step-response prober for 3 apps (parallel), per-second CSV.")
    ap.add_argument("--no-social", action="store_true", help="Disable Social app probing")
    ap.add_argument("--no-hotel", action="store_true", help="Disable Hotel app probing")
    ap.add_argument("--no-feedback", action="store_true", help="Disable Feedback app probing")

    # Step sweep
    ap.add_argument("--step-hold", type=float, default=60.0, help="Seconds per step (default: 60)")
    ap.add_argument("--rps-start", type=float, default=10.0, help="Starting RPS (default: 10)")
    ap.add_argument("--rps-max", type=float, default=200.0, help="Max RPS to try (default: 200)")
    ap.add_argument("--ramp", choices=["linear", "mult"], default="linear", help="Ramp type (default: linear)")
    ap.add_argument("--rps-step", type=float, default=5.0, help="Linear step size (default: 5)")
    ap.add_argument("--rps-mult", type=float, default=1.30, help="Multiplicative factor per step (default: 1.30)")

    # Distribution
    ap.add_argument("--mode", choices=["per-app", "total"], default="per-app",
                    help="per-app: each app gets RPS; total: split across apps (default: per-app)")

    # Per-second window duration
    ap.add_argument("--window-s", type=float, default=1.0, help="Measurement window seconds (default: 1.0)")

    # Request settings
    ap.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT_SEC_DEFAULT, help="HTTP timeout seconds (default: 10)")
    ap.add_argument("--workers", type=int, default=200, help="Worker threads per app (default: 200)")
    ap.add_argument("--max-pending", type=int, default=2000, help="Max outstanding futures per app (default: 2000)")

    # Control
    ap.add_argument("--auto-stop", action="store_true",
                    help="Stop on saturation. By default knee does NOT stop unless --stop-on-knee.")
    ap.add_argument("--stop-on-knee", action="store_true",
                    help="If set, stop on knee (legacy behavior).")
    ap.add_argument("--auto-backoff", action="store_true",
                    help="If set, knee triggers backoff (reduce RPS + hold), and continues.")

    ap.add_argument("--min-success", type=float, default=0.95, help="Saturation if success < this (default: 0.95)")
    ap.add_argument("--max-timeouts", type=int, default=10, help="Saturation if timeouts > this per window (default: 10)")
    ap.add_argument("--knee-jump-factor", type=float, default=1.50, help="Knee if p99 jumps by factor (default: 1.50)")
    ap.add_argument("--knee-min-history", type=int, default=30,
                    help="Min history points (seconds) for knee reference (default: 30)")
    ap.add_argument("--knee-consecutive", type=int, default=5,
                    help="Require knee condition N consecutive windows (default: 5)")

    # Backoff behavior
    ap.add_argument("--backoff-factor", type=float, default=0.70, help="Multiply RPS by this on knee (default: 0.70)")
    ap.add_argument("--backoff-steps", type=float, default=0.0,
                    help="Alternatively subtract RPS by this amount on knee (default: 0 disabled)")
    ap.add_argument("--backoff-hold-steps", type=int, default=60,
                    help="Hold this many windows after backoff (default: 60 windows ~= 60s)")
    ap.add_argument("--resume-stable-steps", type=int, default=30,
                    help="Need this many stable windows before ramp resumes (default: 30)")
    ap.add_argument("--min-rps", type=float, default=1.0, help="Minimum RPS after backoff (default: 1.0)")
    ap.add_argument("--lock-after-first-knee", action="store_true",
                    help="After first knee+backoff, stay at backed-off RPS (no ramp).")

    # Hard stop
    ap.add_argument("--max-runtime-s", type=float, default=600.0,
                    help="Hard stop after this many seconds wall-clock (default: 600 = 10 minutes)")

    ap.add_argument("--out", type=str, default="client_step_response_3apps_persec.csv",
                    help="CSV output filename")
    return ap.parse_args()


# ---------------- MAIN ----------------
def main():
    args = parse_args()

    enable_social = ENABLE_SOCIAL_DEFAULT and (not args.no_social)
    enable_hotel = ENABLE_HOTEL_DEFAULT and (not args.no_hotel)
    enable_feedback = ENABLE_FEEDBACK_DEFAULT and (not args.no_feedback)

    enabled_apps: List[str] = []
    if enable_social:
        enabled_apps.append("social")
    if enable_hotel:
        enabled_apps.append("hotel")
    if enable_feedback:
        enabled_apps.append("feedback")
    if not enabled_apps:
        raise SystemExit("No apps enabled. Remove --no-* flags or set ENABLE_*_DEFAULT=True.")

    feedback_data: List[dict] = []
    if enable_feedback:
        feedback_data = load_feedback_data(FEEDBACK_INPUT_FILE)

    print("--- Step-Response Prober (3 apps parallel, per-second CSV) ---")
    print(f"Enabled apps: {enabled_apps}")
    print(f"Mode: {args.mode} | Window: {args.window_s}s | Step hold: {args.step_hold}s")
    print(f"Ramp: {args.ramp} | start={args.rps_start} max={args.rps_max} "
          f"{'+step='+str(args.rps_step) if args.ramp=='linear' else 'xmult='+str(args.rps_mult)}")
    print(f"Timeout: {args.timeout}s | Workers/app: {args.workers} | Max pending/app: {args.max_pending}")
    print(f"Auto-stop: {args.auto_stop} (stop_on_knee={args.stop_on_knee}) | Auto-backoff: {args.auto_backoff}")
    print(f"Knee: factor={args.knee_jump_factor} hist={args.knee_min_history}s consec={args.knee_consecutive}")
    print(f"Hard stop after: {args.max_runtime_s:.0f}s")
    print(f"CSV output: {args.out}\n")

    # CSV header (PER SECOND)
    header = [
        "WallTime",
        "Elapsed_s",
        "Step",
        "SecInStep",
        "EnabledApps",
        "ControllerState",
        "TargetRPS_Input",
        "Social_TargetRPS", "Social_AchievedRPS", "Social_SuccessRate", "Social_Timeouts", "Social_P95_ms", "Social_P99_ms",
        "Hotel_TargetRPS",  "Hotel_AchievedRPS",  "Hotel_SuccessRate",  "Hotel_Timeouts",  "Hotel_P95_ms",  "Hotel_P99_ms",
        "Feedback_TargetRPS","Feedback_AchievedRPS","Feedback_SuccessRate","Feedback_Timeouts","Feedback_P95_ms","Feedback_P99_ms",
        "AGG_P99_ms",
        "KneeNow",
        "KneeConsecutive",
        "StopReason",
    ]
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerow(header)

    # Controller state
    state = "RAMPING"  # RAMPING / BACKOFF_HOLD / WAIT_STABLE / LOCKED
    backoff_hold_left = 0
    stable_needed = 0
    saw_first_knee = False

    # Step state
    step_idx = 1
    sec_in_step = 0
    rps_input = float(args.rps_start)

    knee_hits_consecutive = 0
    agg_p99_history: List[Optional[float]] = []

    t_global_start = time.monotonic()
    deadline = t_global_start + float(args.max_runtime_s)

    def compute_targets(rps_in: float) -> Dict[str, float]:
        if args.mode == "per-app":
            return {app: rps_in for app in enabled_apps}
        per = rps_in / float(len(enabled_apps))
        return {app: per for app in enabled_apps}

    def step_ramp_next(rps_in: float) -> float:
        if args.ramp == "linear":
            return rps_in + float(args.rps_step)
        return rps_in * float(args.rps_mult)

    def apply_backoff(cur: float) -> float:
        if float(args.backoff_steps) > 0.0:
            new = cur - float(args.backoff_steps)
        else:
            new = cur * float(args.backoff_factor)
        return max(float(args.min_rps), new)

    def fmt(v: Optional[float]) -> str:
        return "NA" if v is None else f"{v:.2f}"

    try:
        while True:
            now_m = time.monotonic()
            if now_m >= deadline:
                print("\nHard stop reached (max-runtime-s).")
                break

            # step transition based on step-hold seconds
            if sec_in_step >= int(max(1.0, float(args.step_hold))):
                sec_in_step = 0
                step_idx += 1
                if state == "RAMPING":
                    rps_input = step_ramp_next(rps_input)
                    if rps_input > float(args.rps_max):
                        print("\nReached rps-max.")
                        break

            sec_in_step += 1

            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            elapsed_s = time.monotonic() - t_global_start

            targets = compute_targets(rps_input)

            # Run a 1-second window for each enabled app in parallel
            window_dur = float(args.window_s)
            with ThreadPoolExecutor(max_workers=3) as outer:
                futs: Dict[str, Future] = {}
                if enable_social:
                    futs["social"] = outer.submit(
                        run_window_rate_controlled,
                        do_social_one,
                        tuple(),
                        float(targets.get("social", 0.0)),
                        window_dur,
                        float(args.timeout),
                        int(args.workers),
                        int(args.max_pending),
                        deadline,
                    )
                if enable_hotel:
                    futs["hotel"] = outer.submit(
                        run_window_rate_controlled,
                        do_hotel_one,
                        tuple(),
                        float(targets.get("hotel", 0.0)),
                        window_dur,
                        float(args.timeout),
                        int(args.workers),
                        int(args.max_pending),
                        deadline,
                    )
                if enable_feedback:
                    futs["feedback"] = outer.submit(
                        run_window_rate_controlled,
                        do_feedback_one,
                        (feedback_data,),
                        float(targets.get("feedback", 0.0)),
                        window_dur,
                        float(args.timeout),
                        int(args.workers),
                        int(args.max_pending),
                        deadline,
                    )
                results: Dict[str, WindowResult] = {k: v.result() for k, v in futs.items()}

            def app_metrics(app: str) -> Tuple[float, float, float, int, Optional[float], Optional[float]]:
                r = results.get(app)
                if r is None:
                    return 0.0, 0.0, 0.0, 0, None, None
                return r.target_rps, r.achieved_rps, r.success_rate, r.timeouts, r.p95_ms, r.p99_ms

            s_t, s_a, s_succ, s_to, s_p95, s_p99 = app_metrics("social")
            h_t, h_a, h_succ, h_to, h_p95, h_p99 = app_metrics("hotel")
            f_t, f_a, f_succ, f_to, f_p95, f_p99 = app_metrics("feedback")

            # Aggregate p99 = max of p99s across enabled apps
            p99_candidates: List[float] = []
            for app in enabled_apps:
                rr = results.get(app)
                if rr is not None and rr.p99_ms is not None:
                    p99_candidates.append(rr.p99_ms)
            agg_p99 = max(p99_candidates) if p99_candidates else None
            agg_p99_history.append(agg_p99)

            knee_now = knee_detect_p99(
                current_p99=agg_p99,
                history_p99=agg_p99_history[:-1],
                min_history=int(args.knee_min_history),
                jump_factor=float(args.knee_jump_factor),
            )
            if knee_now:
                knee_hits_consecutive += 1
            else:
                knee_hits_consecutive = 0

            stop_reason = ""

            # Saturation stop (optional)
            if args.auto_stop:
                for app in enabled_apps:
                    rr = results.get(app)
                    if rr is None:
                        continue
                    if saturation_detect(rr.success_rate, rr.timeouts, int(args.max_timeouts), float(args.min_success)):
                        stop_reason = f"SATURATION_{app.upper()}(succ={rr.success_rate:.3f},to={rr.timeouts})"
                        break

            # Knee action: backoff or stop-on-knee
            if not stop_reason and knee_hits_consecutive >= int(args.knee_consecutive):
                if args.auto_backoff:
                    saw_first_knee = True
                    new_rps = apply_backoff(rps_input)
                    if new_rps < rps_input:
                        rps_input = new_rps
                    state = "BACKOFF_HOLD"
                    backoff_hold_left = int(args.backoff_hold_steps)
                    stable_needed = int(args.resume_stable_steps)
                    knee_hits_consecutive = 0
                    print(
                        f"\n[{ts}] KNEE -> backoff to InputRPS={rps_input:.2f} | "
                        f"hold={backoff_hold_left} windows | need stable={stable_needed} windows\n"
                    )
                elif args.auto_stop and args.stop_on_knee:
                    stop_reason = f"KNEE_AGG_P99(jump_factor={args.knee_jump_factor})"

            # Advance controller state (per-second)
            if args.lock_after_first_knee and saw_first_knee:
                state = "LOCKED"
            else:
                if state == "BACKOFF_HOLD":
                    backoff_hold_left -= 1
                    if backoff_hold_left <= 0:
                        state = "WAIT_STABLE"

                elif state == "WAIT_STABLE":
                    if knee_now:
                        rps_input = apply_backoff(rps_input)
                        state = "BACKOFF_HOLD"
                        backoff_hold_left = int(args.backoff_hold_steps)
                        stable_needed = int(args.resume_stable_steps)
                        print(f"\n[{ts}] Still unstable -> backoff again to InputRPS={rps_input:.2f}\n")
                    else:
                        stable_needed -= 1
                        if stable_needed <= 0:
                            state = "RAMPING"

            # Print one-line status each second
            print(
                f"[{ts}] step={step_idx:03d} sec={sec_in_step:03d} state={state:11s} "
                f"InputRPS={rps_input:7.2f} AGG_P99={fmt(agg_p99):>8s} knee={int(knee_now)} kc={knee_hits_consecutive}",
                end="\r"
            )

            # CSV row
            def cell_float(v: Optional[float]) -> str:
                return "" if v is None else f"{v:.4f}"

            with open(args.out, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    ts,
                    f"{elapsed_s:.3f}",
                    step_idx,
                    sec_in_step,
                    "|".join(enabled_apps),
                    state,
                    f"{rps_input:.4f}",
                    f"{s_t:.4f}", f"{s_a:.4f}", f"{s_succ:.6f}", s_to, cell_float(s_p95), cell_float(s_p99),
                    f"{h_t:.4f}", f"{h_a:.4f}", f"{h_succ:.6f}", h_to, cell_float(h_p95), cell_float(h_p99),
                    f"{f_t:.4f}", f"{f_a:.4f}", f"{f_succ:.6f}", f_to, cell_float(f_p95), cell_float(f_p99),
                    cell_float(agg_p99),
                    int(bool(knee_now)),
                    int(knee_hits_consecutive),
                    stop_reason,
                ])

            if stop_reason:
                print(f"\n\nStopping: {stop_reason}\n")
                break

            # If deadline is very close, exit cleanly next loop
            if time.monotonic() >= deadline:
                print("\nHard stop reached (max-runtime-s).")
                break

            # Align to ~1s loop wall-clock (best-effort)
            # (run_window_rate_controlled already consumes ~window_s, so this is usually small)
            # No extra sleep needed.

    except KeyboardInterrupt:
        print("\nStopping prober (Ctrl+C).")

    # Ensure a clean newline after carriage-return printing
    print(f"\nSaved CSV: {args.out}")


if __name__ == "__main__":
    main()
