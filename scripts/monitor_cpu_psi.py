#!/usr/bin/env python3
import time
import os
import sys
import csv
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
INTERVAL_SEC = 1.0
OUTPUT_CSV = "cpu_psi_metrics.csv"
HISTORY_LEN = 60  # Show last 60 seconds on graph

# ============================================================
# METRIC READERS (Lightweight /proc parsing)
# ============================================================

def get_cpu_times():
    """Reads /proc/stat to get raw CPU ticks."""
    try:
        with open("/proc/stat", "r") as f:
            line = f.readline()  # 'cpu  ...'
            if not line.startswith("cpu "):
                return None
            parts = [int(x) for x in line.split()[1:]]
            # user(0)+nice(1), system(2)+irq(5)+softirq(6), idle(3), iowait(4)
            idle_all = parts[3] + parts[4]
            system_all = parts[2] + parts[5] + parts[6]
            user_all = parts[0] + parts[1]
            total = sum(parts)
            return {"total": total, "idle": idle_all, "user": user_all, "sys": system_all, "io": parts[4]}
    except FileNotFoundError:
        return None

def get_psi(resource):
    """Reads /proc/pressure/{cpu, memory, io}."""
    metrics = {"some": 0.0, "full": 0.0}
    path = f"/proc/pressure/{resource}"
    if not os.path.exists(path): return metrics
    try:
        with open(path, "r") as f:
            for line in f:
                parts = line.split()
                key = parts[0]
                if key in metrics:
                    # avg10=X.XX is usually the 2nd element (index 1)
                    val_str = next((x for x in parts if x.startswith("avg10=")), None)
                    if val_str:
                        metrics[key] = float(val_str.split("=")[1])
    except: pass
    return metrics

# ============================================================
# MAIN
# ============================================================
def main():
    print(f"--- Monitoring & Plotting (Interval: {INTERVAL_SEC}s) ---")
    print(f"Logging to: {OUTPUT_CSV}")
    print("Close the plot window to stop.")

    # 1. Setup CSV
    file_exists = os.path.exists(OUTPUT_CSV)
    csv_file = open(OUTPUT_CSV, "a", newline="")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(["Time", "CPU_User", "CPU_Sys", "CPU_Wait", "PSI_CPU", "PSI_Mem_Full", "PSI_IO_Full"])
    
    # 2. Setup Plotting Buffers
    x_time = deque(maxlen=HISTORY_LEN)
    y_cpu_usr = deque(maxlen=HISTORY_LEN)
    y_cpu_sys = deque(maxlen=HISTORY_LEN)
    y_cpu_wait = deque(maxlen=HISTORY_LEN)
    
    y_psi_cpu = deque(maxlen=HISTORY_LEN)
    y_psi_mem = deque(maxlen=HISTORY_LEN)
    y_psi_io = deque(maxlen=HISTORY_LEN)

    # 3. Setup Matplotlib Figure
    plt.ion() # Interactive mode on
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.canvas.manager.set_window_title("System Friction Monitor")

    # Subplot 1: CPU Stack
    ax1.set_title("CPU Usage (%)")
    ax1.set_ylabel("Utilization %")
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    
    # Lines for CPU
    l_usr, = ax1.plot([], [], label="User", color="#3498db")
    l_sys, = ax1.plot([], [], label="System", color="#f1c40f")
    l_wait, = ax1.plot([], [], label="IO Wait", color="#e74c3c")
    ax1.legend(loc="upper left")

    # Subplot 2: PSI
    ax2.set_title("Pressure Stall Information (PSI avg10)")
    ax2.set_ylabel("Stall %")
    ax2.set_ylim(0, 50) # PSI usually low, but can spike to 100
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)

    l_p_cpu, = ax2.plot([], [], label="CPU (Some)", color="#3498db", linestyle="--")
    l_p_mem, = ax2.plot([], [], label="Mem (Full)", color="#9b59b6", linewidth=2)
    l_p_io, = ax2.plot([], [], label="IO (Full)", color="#e74c3c", linewidth=2)
    ax2.legend(loc="upper left")

    prev_cpu = get_cpu_times()
    start_time = time.time()

    try:
        while plt.fignum_exists(fig.number):
            loop_start = time.time()
            ts_str = datetime.now().strftime("%H:%M:%S")

            # --- READ METRICS ---
            curr_cpu = get_cpu_times()
            psi_c = get_psi("cpu")
            psi_m = get_psi("memory")
            psi_i = get_psi("io")

            # CPU Calc
            if prev_cpu and curr_cpu:
                d_tot = curr_cpu["total"] - prev_cpu["total"]
                d_usr = curr_cpu["user"] - prev_cpu["user"]
                d_sys = curr_cpu["sys"] - prev_cpu["sys"]
                d_io = curr_cpu["io"] - prev_cpu["io"]
                
                if d_tot > 0:
                    p_usr = 100 * d_usr / d_tot
                    p_sys = 100 * d_sys / d_tot
                    p_io  = 100 * d_io / d_tot
                else:
                    p_usr = p_sys = p_io = 0
            else:
                p_usr = p_sys = p_io = 0
            
            prev_cpu = curr_cpu

            # --- UPDATE BUFFERS ---
            # Relative time for x-axis (0 to -60) or just incrementing seconds
            # Using simple incrementing counter for stability
            now_sec = time.time() - start_time
            x_time.append(now_sec)
            
            y_cpu_usr.append(p_usr)
            y_cpu_sys.append(p_sys)
            y_cpu_wait.append(p_io)
            
            y_psi_cpu.append(psi_c["some"])
            y_psi_mem.append(psi_m["full"])
            y_psi_io.append(psi_i["full"])

            # --- WRITE CSV ---
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                f"{p_usr:.1f}", f"{p_sys:.1f}", f"{p_io:.1f}",
                f"{psi_c['some']:.2f}", f"{psi_m['full']:.2f}", f"{psi_i['full']:.2f}"
            ])
            csv_file.flush()

            # --- UPDATE PLOT ---
            # CPU Lines
            l_usr.set_data(x_time, y_cpu_usr)
            l_sys.set_data(x_time, y_cpu_sys)
            l_wait.set_data(x_time, y_cpu_wait)
            
            # PSI Lines
            l_p_cpu.set_data(x_time, y_psi_cpu)
            l_p_mem.set_data(x_time, y_psi_mem)
            l_p_io.set_data(x_time, y_psi_io)

            # Rescale axes
            ax1.set_xlim(min(x_time), max(x_time) + 1)
            ax2.set_xlim(min(x_time), max(x_time) + 1)
            
            # Auto-scale Y if PSI spikes high
            if max(y_psi_mem, default=0) > 45 or max(y_psi_io, default=0) > 45:
                ax2.set_ylim(0, 100)
            else:
                ax2.set_ylim(0, 50)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            # --- SLEEP ---
            elapsed = time.time() - loop_start
            pause_time = max(0.01, INTERVAL_SEC - elapsed)
            time.sleep(pause_time)

    except KeyboardInterrupt:
        pass
    finally:
        csv_file.close()
        print("\nMonitor stopped.")

if __name__ == "__main__":
    main()
