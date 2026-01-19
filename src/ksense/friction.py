import numpy as np

from .config import MAHAL_MIN_SAMPLES, MAHAL_REG_ABS, MAHAL_REG_REL


def mahalanobis_distance(x: np.ndarray, X: np.ndarray) -> float:
    try:
        x = np.asarray(x, dtype=float).reshape(-1)
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or x.ndim != 1:
            return float("nan")
        if X.shape[1] != x.shape[0]:
            return float("nan")

        X = X[np.all(np.isfinite(X), axis=1)]
        if not np.all(np.isfinite(x)):
            return float("nan")

        d = x.shape[0]
        if X.shape[0] < max(MAHAL_MIN_SAMPLES, d + 2):
            return float("nan")

        mu = np.mean(X, axis=0)
        Sigma = np.cov(X, rowvar=False, bias=False)
        if Sigma.shape != (d, d) or not np.all(np.isfinite(Sigma)):
            return float("nan")

        diag = np.diag(Sigma)
        diag_mean = float(np.mean(diag)) if np.all(np.isfinite(diag)) else 0.0
        reg = MAHAL_REG_ABS + MAHAL_REG_REL * max(diag_mean, 0.0)
        Sigma_reg = Sigma + reg * np.eye(d)

        L = np.linalg.cholesky(Sigma_reg)
        diff = (x - mu).reshape(-1, 1)
        y = np.linalg.solve(L, diff)
        dist2 = float(np.dot(y[:, 0], y[:, 0]))
        return float(np.sqrt(max(dist2, 0.0)))
    except Exception:
        return float("nan")


def mahalanobis_distance_and_direction(x: np.ndarray, X: np.ndarray):
    try:
        x = np.asarray(x, dtype=float).reshape(-1)
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or x.ndim != 1:
            return float("nan"), float("nan")
        if X.shape[1] != x.shape[0]:
            return float("nan"), float("nan")

        X = X[np.all(np.isfinite(X), axis=1)]
        if not np.all(np.isfinite(x)):
            return float("nan"), float("nan")

        d = x.shape[0]
        if X.shape[0] < max(MAHAL_MIN_SAMPLES, d + 2):
            return float("nan"), float("nan")

        mu = np.mean(X, axis=0)
        Sigma = np.cov(X, rowvar=False, bias=False)
        if Sigma.shape != (d, d) or not np.all(np.isfinite(Sigma)):
            return float("nan"), float("nan")

        diag = np.diag(Sigma)
        diag_mean = float(np.mean(diag)) if np.all(np.isfinite(diag)) else 0.0
        reg = MAHAL_REG_ABS + MAHAL_REG_REL * max(diag_mean, 0.0)
        Sigma_reg = Sigma + reg * np.eye(d)

        L = np.linalg.cholesky(Sigma_reg)
        diff = (x - mu).reshape(-1, 1)
        y = np.linalg.solve(L, diff)

        dist2 = float(np.dot(y[:, 0], y[:, 0]))
        friction = float(np.sqrt(max(dist2, 0.0)))

        raw_direction = float(np.sum(y[:, 0]))
        direction = 1.0 if raw_direction >= 0 else -1.0

        return friction, direction
    except Exception:
        return float("nan"), float("nan")
