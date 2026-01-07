#!/usr/bin/env python3
#
# client_prober_3apps_p99_fast_toggle.py
# Measures P99 latency for up to 3 applications in parallel and writes one CSV row per interval.
# Adds easy per-app enable/disable toggles via:
#   1) CONFIG booleans (ENABLE_*)
#   2) CLI flags: --no-social / --no-hotel / --no-feedback
#

import time
import random
import string
import csv
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np

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

# Sampling
INTERVAL_SEC = 1.0
BURST_SIZE = 10

# IMPORTANT: feedback calls can be slow. Set timeout accordingly.
REQUEST_TIMEOUT_SEC = 10.0

OUTPUT_FILE = "client_p99_3apps.csv"

# Concurrency settings
MAX_WORKERS_PER_APP = BURST_SIZE

# Hotel mixed ratios (from your Lua)
SEARCH_RATIO = 0.60
RECOMMEND_RATIO = 0.39
USER_RATIO = 0.005
# remaining probability goes to reserve()

# ---- Easy toggles (defaults) ----
# Set these to False if you want to hard-disable an app without using CLI flags.
ENABLE_SOCIAL_DEFAULT = True
ENABLE_HOTEL_DEFAULT = True
ENABLE_FEEDBACK_DEFAULT = True


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


# ---------------- LOW-LEVEL REQUEST FUNCTIONS (single request each) ----------------
def do_social_one(session: requests.Session) -> Optional[float]:
    try:
        start = time.perf_counter()
        resp = session.post(SOCIAL_URL, data=generate_payload_social(), timeout=REQUEST_TIMEOUT_SEC)
        dur_ms = (time.perf_counter() - start) * 1000.0
        return dur_ms if resp.status_code == 200 else None
    except requests.exceptions.RequestException:
        return None


def do_hotel_one(session: requests.Session) -> Optional[float]:
    method, url = hotel_generate_mixed_request()
    try:
        start = time.perf_counter()
        if method == "GET":
            resp = session.get(url, timeout=REQUEST_TIMEOUT_SEC)
        else:
            resp = session.post(url, timeout=REQUEST_TIMEOUT_SEC)
        dur_ms = (time.perf_counter() - start) * 1000.0
        return dur_ms if resp.status_code == 200 else None
    except requests.exceptions.RequestException:
        return None


def do_feedback_one(session: requests.Session, feedback_data: List[dict]) -> Optional[float]:
    payload = random.choice(feedback_data)
    try:
        start = time.perf_counter()
        resp = session.post(FEEDBACK_URL, json=payload, timeout=REQUEST_TIMEOUT_SEC)
        dur_ms = (time.perf_counter() - start) * 1000.0
        return dur_ms if resp.status_code == 200 else None
    except requests.exceptions.RequestException:
        return None


# ---------------- BURST (CONCURRENT) ----------------
def fire_burst_concurrent(fn, args_tuple, n: int, max_workers: int) -> List[float]:
    samples: List[float] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(fn, *args_tuple) for _ in range(n)]
        for fut in as_completed(futures):
            v = fut.result()
            if v is not None:
                samples.append(v)
    return samples


def p99(samples: List[float]) -> Optional[float]:
    if not samples:
        return None
    return float(np.percentile(samples, 99))


def fmt_msg(enabled: bool, value: Optional[float]) -> str:
    if not enabled:
        return "DISABLED"
    return "FAILED" if value is None else f"{value:.2f} ms"


def csv_cell(enabled: bool, value: Optional[float]) -> str:
    if not enabled or value is None:
        return ""
    return f"{value:.4f}"


# ---------------- MAIN ----------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="3-app concurrent-burst P99 prober with per-app toggles.")
    ap.add_argument("--no-social", action="store_true", help="Disable Social app probing")
    ap.add_argument("--no-hotel", action="store_true", help="Disable Hotel app probing")
    ap.add_argument("--no-feedback", action="store_true", help="Disable Feedback app probing")

    ap.add_argument("--interval", type=float, default=INTERVAL_SEC, help="Sampling interval seconds (default: 1.0)")
    ap.add_argument("--burst", type=int, default=BURST_SIZE, help="Requests per app per interval (default: 10)")
    ap.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT_SEC, help="HTTP timeout seconds (default: 10)")
    ap.add_argument("--out", type=str, default=OUTPUT_FILE, help="CSV output filename (default: client_p99_3apps.csv)")
    return ap.parse_args()


def main():
    global INTERVAL_SEC, BURST_SIZE, REQUEST_TIMEOUT_SEC, OUTPUT_FILE, MAX_WORKERS_PER_APP

    args = parse_args()

    # Apply runtime overrides
    INTERVAL_SEC = float(args.interval)
    BURST_SIZE = int(args.burst)
    REQUEST_TIMEOUT_SEC = float(args.timeout)
    OUTPUT_FILE = str(args.out)
    MAX_WORKERS_PER_APP = BURST_SIZE

    # Compute enable flags (defaults can be hard-set in config, then overridden by CLI --no-*)
    enable_social = ENABLE_SOCIAL_DEFAULT and (not args.no_social)
    enable_hotel = ENABLE_HOTEL_DEFAULT and (not args.no_hotel)
    enable_feedback = ENABLE_FEEDBACK_DEFAULT and (not args.no_feedback)

    # Load feedback data only if feedback is enabled
    feedback_data: List[dict] = []
    if enable_feedback:
        feedback_data = load_feedback_data(FEEDBACK_INPUT_FILE)

    print("--- 3-App P99 Prober (Concurrent Bursts, Toggleable) ---")
    print(f"Enabled: social={enable_social} | hotel={enable_hotel} | feedback={enable_feedback}")
    print(f"Social  : POST  {SOCIAL_URL}")
    print(f"Hotel   : MIXED {HOTEL_BASE} (/hotels, /recommendations, /user, /reservation)")
    print(f"Feedback: POST  {FEEDBACK_URL}")
    if enable_feedback:
        print(f"Input   : {FEEDBACK_INPUT_FILE}")
    print(f"Interval target: {INTERVAL_SEC}s | Samples per app per interval: {BURST_SIZE}")
    print(f"Timeout : {REQUEST_TIMEOUT_SEC}s")
    print(f"CSV     : {OUTPUT_FILE} (written in current directory)")

    # Initialize CSV
    with open(OUTPUT_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Social_P99_ms", "Hotel_P99_ms", "Feedback_P99_ms"])

    social_sess = requests.Session()
    hotel_sess = requests.Session()
    feedback_sess = requests.Session()

    try:
        while True:
            loop_start = time.time()
            ts = time.strftime("%Y-%m-%d %H:%M:%S")

            s_p99: Optional[float] = None
            h_p99: Optional[float] = None
            f_p99: Optional[float] = None

            # Build enabled tasks only
            tasks = []
            with ThreadPoolExecutor(max_workers=3) as outer:
                if enable_social:
                    tasks.append(("social", outer.submit(
                        fire_burst_concurrent, do_social_one, (social_sess,), BURST_SIZE, MAX_WORKERS_PER_APP
                    )))
                if enable_hotel:
                    tasks.append(("hotel", outer.submit(
                        fire_burst_concurrent, do_hotel_one, (hotel_sess,), BURST_SIZE, MAX_WORKERS_PER_APP
                    )))
                if enable_feedback:
                    tasks.append(("feedback", outer.submit(
                        fire_burst_concurrent, do_feedback_one, (feedback_sess, feedback_data), BURST_SIZE, MAX_WORKERS_PER_APP
                    )))

                # Collect results
                results = {}
                for name, fut in tasks:
                    results[name] = fut.result()

            if enable_social:
                s_p99 = p99(results.get("social", []))
            if enable_hotel:
                h_p99 = p99(results.get("hotel", []))
            if enable_feedback:
                f_p99 = p99(results.get("feedback", []))

            print(
                f"[{ts}] "
                f"Social P99: {fmt_msg(enable_social, s_p99)} | "
                f"Hotel P99: {fmt_msg(enable_hotel, h_p99)} | "
                f"Feedback P99: {fmt_msg(enable_feedback, f_p99)}"
            )

            with open(OUTPUT_FILE, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ts,
                    csv_cell(enable_social, s_p99),
                    csv_cell(enable_hotel, h_p99),
                    csv_cell(enable_feedback, f_p99),
                ])

            # Wall-clock aligned sleep
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, INTERVAL_SEC - elapsed))

    except KeyboardInterrupt:
        print("\nStopping prober.")


if __name__ == "__main__":
    main()

