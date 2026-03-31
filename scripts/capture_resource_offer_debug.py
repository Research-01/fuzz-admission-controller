#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Dict, Optional


def _f(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_nested(payload: Dict[str, object], key: str, nested: str):
    block = payload.get(key)
    if not isinstance(block, dict):
        return None
    return block.get(nested)


def _fetch_json(url: str) -> Optional[Dict[str, object]]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=3.0) as r:
            data = json.loads(r.read().decode("utf-8"))
            if isinstance(data, dict):
                return data
            return None
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _data_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        n = sum(1 for _ in f)
    return max(0, n - 1)


def _ensure_csv(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(path):
        return
    headers = [
        "capture_time",
        "snapshot_time",
        "decision",
        "decision_reason",
        "fuzzy_decision",
        "level",
        "score",
        "current_cpu_pct",
        "current_psi_pct",
        "current_ram_gb",
        "current_disk_pct",
        "friction_signed",
        "energy_scaled",
        "fuzzy_cpu_pct",
        "fuzzy_psi_pct",
        "sellable_cpu",
        "sellable_ram_gb",
        "sellable_storage_gb",
        "pred_cpu_pct",
        "pred_psi_pct",
        "pred_ram_gb",
        "pred_disk_pct",
        "safety_multiplier",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(headers)


def _append_row(path: str, payload: Dict[str, object]) -> None:
    now_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [
        now_s,
        payload.get("timestamp"),
        payload.get("decision"),
        payload.get("decision_reason"),
        payload.get("fuzzy_decision"),
        payload.get("level"),
        _f(payload.get("score")),
        _f(_get_nested(payload, "usage_latest", "cpu_usage_pct")),
        _f(_get_nested(payload, "usage_latest", "cpu_psi_some_pct")),
        _f(_get_nested(payload, "usage_latest", "ram_usage_gb")),
        _f(_get_nested(payload, "usage_latest", "disk_used_pct")),
        _f(_get_nested(payload, "fuzzy_inputs", "friction_signed")),
        _f(_get_nested(payload, "fuzzy_inputs", "energy_scaled")),
        _f(_get_nested(payload, "fuzzy_inputs", "cpu_util")),
        _f(_get_nested(payload, "fuzzy_inputs", "psi_1s")),
        _f(_get_nested(payload, "offer", "cpu")),
        _f(_get_nested(payload, "offer", "ram")),
        _f(_get_nested(payload, "offer", "storage")),
        _f(_get_nested(payload, "predicted", "cpu_pct")),
        _f(_get_nested(payload, "predicted", "psi_pct")),
        _f(_get_nested(payload, "predicted", "ram_gb")),
        _f(_get_nested(payload, "predicted", "disk_pct")),
        _f(payload.get("safety_multiplier")),
    ]
    with open(path, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(row)


def main() -> int:
    p = argparse.ArgumentParser(description="Capture /resource_offer_debug snapshots into CSV.")
    p.add_argument("--url", default="http://127.0.0.1:8080/resource_offer_debug", help="Debug endpoint URL")
    p.add_argument("--output", default="/tmp/ksense/resource_offer_debug_trace.csv", help="Output CSV path")
    p.add_argument("--interval-s", type=float, default=0.5, help="Polling interval in seconds")
    p.add_argument("--max-seconds", type=float, default=0.0, help="Stop after this many seconds (0=disabled)")
    p.add_argument(
        "--until-usage-csv",
        default="",
        help="Optional usage mirror CSV path; stop when rows reach --until-usage-rows",
    )
    p.add_argument("--until-usage-rows", type=int, default=0, help="Target data rows in usage CSV before stop")
    args = p.parse_args()

    _ensure_csv(args.output)

    start = time.monotonic()
    last_snapshot_time = ""
    captured = 0
    usage_goal_seen = 0

    while True:
        payload = _fetch_json(args.url)
        if payload is not None:
            snap_ts = str(payload.get("timestamp") or "")
            if snap_ts and snap_ts != last_snapshot_time:
                _append_row(args.output, payload)
                last_snapshot_time = snap_ts
                captured += 1

        if args.max_seconds > 0 and (time.monotonic() - start) >= args.max_seconds:
            break

        if args.until_usage_csv and args.until_usage_rows > 0:
            rows = _data_rows(args.until_usage_csv)
            if rows >= args.until_usage_rows:
                usage_goal_seen += 1
            else:
                usage_goal_seen = 0
            if usage_goal_seen >= 3:
                break

        time.sleep(max(0.05, args.interval_s))

    print(f"Captured snapshots: {captured}")
    print(f"Output CSV: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
