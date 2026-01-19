import os

# -------------------------
# Collector cadence
# -------------------------
WINDOW_SEC = 1.0
GRID_STEP_S = 1  # should match WINDOW_SEC

# -------------------------
# Output files
# -------------------------
OUT_CSV = "kernel_metrics.csv"
LIVE_PNG = "kernel_live.png"  # updated continuously in headless mode

# -------------------------
# Energy Settings (ADAPTIVE WINDOW ON |ΔF|)
# -------------------------
W_MIN = 5
W_MAX = 40
VOLATILITY_ALPHA = 0.10
VOL_EPS = 1e-6

ENERGY_CALIBRATE_AFTER_FREEZE = True
ENERGY_CALIB_WIN_S = 60  # seconds of friction history to calibrate from
ENERGY_TARGET_W = (W_MIN + W_MAX) / 2.0

# -------------------------
# Mahalanobis friction config
# -------------------------
MAHAL_MIN_SAMPLES = 20
MAHAL_REG_REL = 1e-3
MAHAL_REG_ABS = 1e-9

# -------------------------
# Calibration / baseline
# -------------------------
WARMUP_S = 600
BASELINE_WIN_S = 5 * 60
FREEZE_BASELINE_AFTER_WARMUP = True
MIN_SCHED_CNT_FOR_BASELINE = 50
BASELINE_GATE_DIST = 3.08  # not used if baseline frozen

# -------------------------
# Plotting controls
# -------------------------
ENABLE_PLOT = True
PLOT_WINDOW_MIN = 60
SAVE_PNG_EVERY_S = 10

# -------------------------
# Matplotlib backend (GUI vs headless)
# -------------------------
HEADLESS = (os.environ.get("DISPLAY", "") == "")
