from collections import deque
from typing import Tuple

import numpy as np

from .config import ENERGY_TARGET_W, VOL_EPS, VOLATILITY_ALPHA, W_MAX, W_MIN


class AdaptiveVolatilityEnergy:
    def __init__(self, w_min=W_MIN, w_max=W_MAX, alpha=VOLATILITY_ALPHA):
        self.w_min = int(w_min)
        self.w_max = int(w_max)
        self.alpha = float(alpha)
        self.vol = 0.0
        self.buf = deque(maxlen=self.w_max + 5)
        self.current_w = self.w_max
        self.k_factor = 1.0
        self._calibrated = False

    def calibrate(self, abs_deltas: np.ndarray):
        abs_deltas = np.asarray(abs_deltas, dtype=float)
        abs_deltas = abs_deltas[np.isfinite(abs_deltas)]
        if abs_deltas.size < 10:
            self.k_factor = 1.0
            self._calibrated = True
            return
        med = float(np.median(abs_deltas))
        self.k_factor = max(1e-6, med * float(ENERGY_TARGET_W))
        self._calibrated = True

    def update(self, friction: float) -> Tuple[float, int]:
        if not np.isfinite(friction):
            return float("nan"), self.current_w

        self.buf.append(float(friction))

        if len(self.buf) >= 2:
            grad = abs(self.buf[-1] - self.buf[-2])
            self.vol = self.alpha * grad + (1.0 - self.alpha) * self.vol

        raw_w = int(self.k_factor / (self.vol + VOL_EPS))
        self.current_w = max(self.w_min, min(self.w_max, raw_w))

        if len(self.buf) < 2:
            return 0.0, self.current_w

        lookback = min(len(self.buf), self.current_w)
        window = np.array(list(self.buf)[-lookback:], dtype=float)
        energy = float(np.mean(np.abs(np.diff(window)))) if window.size >= 2 else 0.0
        return energy, self.current_w
