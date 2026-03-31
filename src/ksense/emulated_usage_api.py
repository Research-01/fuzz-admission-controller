#!/usr/bin/env python3
import csv
import glob
import os
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

USAGE_FIELDS = [
    "timestamp",
    "cpu_usage_pct",
    "cpu_psi_some_pct",
    "ram_usage_pct",
    "ram_usage_mi",
    "disk_used_pct",
    "sched_total_ms",
    "dstate_total_ms",
    "softirq_total_ms",
]


def _parse_ts(value: str) -> Optional[datetime]:
    s = (value or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _row_to_usage(row: Dict[str, str]) -> Optional[Dict[str, object]]:
    ts = row.get("timestamp") or row.get("Time") or ""
    if _parse_ts(ts) is None:
        return None

    out: Dict[str, object] = {"timestamp": ts}
    for key in USAGE_FIELDS[1:]:
        val = row.get(key)
        if val is None:
            out[key] = None
            continue
        s = str(val).strip()
        if not s:
            out[key] = None
            continue
        try:
            out[key] = float(s)
        except ValueError:
            out[key] = None
    return out


class HotelPlayback:
    def __init__(self, csv_paths: List[str], loop: bool):
        self._streams = self._load_streams(csv_paths)
        self._loop = loop
        self._lock = threading.Lock()
        self._file_idx = 0
        self._row_idx = 0
        self._latest: Optional[Dict[str, object]] = None

    def _load_streams(self, csv_paths: List[str]) -> List[List[Dict[str, object]]]:
        streams: List[List[Dict[str, object]]] = []
        for path in csv_paths:
            if not os.path.exists(path):
                continue
            rows: List[Dict[str, object]] = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    parsed = _row_to_usage(row)
                    if parsed is not None:
                        rows.append(parsed)
            if rows:
                streams.append(rows)
        return streams

    def step(self) -> None:
        if not self._streams:
            return
        with self._lock:
            stream = self._streams[self._file_idx]
            self._latest = dict(stream[self._row_idx])

            if self._row_idx < len(stream) - 1:
                self._row_idx += 1
                return

            # End of current file -> move to next file.
            self._row_idx = 0
            if self._file_idx < len(self._streams) - 1:
                self._file_idx += 1
            elif self._loop:
                self._file_idx = 0
            else:
                # Stay at final row if not looping.
                self._file_idx = len(self._streams) - 1
                self._row_idx = len(self._streams[self._file_idx]) - 1

    def latest(self) -> Optional[Dict[str, object]]:
        with self._lock:
            if self._latest is None:
                return None
            return dict(self._latest)

    @property
    def size(self) -> int:
        return sum(len(s) for s in self._streams)


def _create_app(playback: HotelPlayback) -> FastAPI:
    app = FastAPI(title="ksense-emulated-usage-api")

    @app.get("/healthz", response_class=PlainTextResponse)
    def healthz() -> str:
        return "ok\n"

    @app.get("/usage/latest")
    def usage_latest() -> Dict[str, object]:
        latest = playback.latest()
        if latest is None:
            raise HTTPException(status_code=503, detail="no usage data loaded")
        return latest

    return app


def _default_hotel_csvs() -> List[str]:
    return sorted(glob.glob(os.path.join(os.getcwd(), "hotel*.csv")))


def run_server(host: str = "127.0.0.1", port: int = 8090) -> None:
    csvs_env = os.getenv("MZ_EMULATED_USAGE_CSVS", "").strip()
    if csvs_env:
        csv_paths = [x.strip() for x in csvs_env.split(",") if x.strip()]
    else:
        csv_paths = _default_hotel_csvs()

    tick_s = float(os.getenv("MZ_EMULATED_USAGE_TICK_S", "5"))
    loop = os.getenv("MZ_EMULATED_USAGE_LOOP", "true").lower() == "true"

    playback = HotelPlayback(csv_paths=csv_paths, loop=loop)
    if playback.size > 0:
        playback.step()

    def _loop() -> None:
        next_t = time.monotonic() + tick_s
        while True:
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            playback.step()
            next_t += tick_s

    threading.Thread(target=_loop, daemon=True).start()

    print(f"[emulated-usage-api] listening on http://{host}:{port}")
    print(f"[emulated-usage-api] source CSVs: {csv_paths}")
    print(f"[emulated-usage-api] tick interval: {tick_s}s")
    print(f"[emulated-usage-api] loaded rows: {playback.size}")

    app = _create_app(playback=playback)
    uvicorn.run(app, host=host, port=port, log_level="info")
