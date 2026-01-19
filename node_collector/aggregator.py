import os
import time
from collections import defaultdict
from datetime import datetime


class TimeWindowAggregator:
    def __init__(self, window_seconds=10):
        self.window = window_seconds
        self.data = defaultdict(lambda: defaultdict(lambda: {
            "count": 0, "latency_sum": 0, "latency_max": 0,
            "latencies": [], "bytes_sum": 0, "errors": 0,
        }))
        self.buffer = []

    def add_event(self, event_name, latency_us, ret, bytes_val):
        ts_sec = time.time()
        window_start = int(ts_sec / self.window) * self.window
        key = (window_start, event_name)
        stats = self.data[window_start][key]
        stats["count"] += 1
        stats["latency_sum"] += latency_us
        stats["latency_max"] = max(stats["latency_max"], latency_us)
        stats["latencies"].append(latency_us)
        if bytes_val != 0:
            stats["bytes_sum"] += bytes_val
        if ret < 0:
            stats["errors"] += 1

    def flush_windows_before(self, ts_sec):
        current_window = int(ts_sec / self.window) * self.window
        for window in list(self.data.keys()):
            if window < current_window - self.window:
                self._flush_window(window)
                del self.data[window]

    def _flush_window(self, window):
        for (_, event_name), stats in self.data[window].items():
            if stats["count"] == 0:
                continue
            latencies = sorted(stats["latencies"])
            n = len(latencies)
            record = {
                "timestamp": int(window),
                "scope": "node",
                "event": event_name,
                "count": stats["count"],
                "latency_avg_us": stats["latency_sum"] / stats["count"],
                "latency_max_us": stats["latency_max"],
                "latency_p50_us": latencies[int(n * 0.5)] if n > 0 else 0,
                "latency_p95_us": latencies[int(n * 0.95)] if n > 0 else 0,
                "latency_p99_us": latencies[int(n * 0.99)] if n > 0 else 0,
                "total_latency_us": stats["latency_sum"],
                "bytes_total": stats["bytes_sum"],
                "error_count": stats["errors"],
                "error_rate": stats["errors"] / stats["count"],
            }
            self.buffer.append(record)

    def write_to_disk(self):
        if not self.buffer:
            return
        os.makedirs("./aggregated", exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"./aggregated/node_syscalls_{date_str}.csv"
        write_header = not os.path.exists(filename)
        with open(filename, "a") as f:
            if write_header:
                f.write(
                    "timestamp,scope,event,count,latency_avg_us,latency_max_us,latency_p50_us,"
                    "latency_p95_us,latency_p99_us,total_latency_us,bytes_total,error_count,error_rate\n"
                )
            for rec in self.buffer:
                f.write(
                    f"{rec['timestamp']},{rec['scope']},{rec['event']},{rec['count']},"
                    f"{rec['latency_avg_us']:.2f},{rec['latency_max_us']},"
                    f"{rec['latency_p50_us']},{rec['latency_p95_us']},{rec['latency_p99_us']},"
                    f"{rec['total_latency_us']},"
                    f"{rec['bytes_total']},{rec['error_count']},{rec['error_rate']:.4f}\n"
                )
        print(f"[WRITE] Wrote {len(self.buffer)} records to {filename}")
        self.buffer.clear()
