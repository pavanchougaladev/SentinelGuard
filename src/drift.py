"""Population Stability Index (PSI) drift check - no heavy dependencies."""
import numpy as np
import pandas as pd


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / max(len(ref), 1)
    c = np.histogram(cur, edges)[0] / max(len(cur), 1)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, cols) -> dict:
    out = {}
    for col in cols:
        v = psi(ref[col].values, cur[col].values)
        out[col] = {"psi": round(v, 4),
                    "status": "ALERT" if v > 0.25 else "WARN" if v > 0.1 else "ok"}
    return out
