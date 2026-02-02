from typing import List, Optional, Dict
import numpy as np


def summarize_scores(scores: List[Optional[float]]) -> Dict[str, float]:
    """Summary stats for a list of possibly-missing scores."""
    valid = [s for s in scores if s is not None]
    n = len(scores)
    nv = len(valid)
    out: Dict[str, float] = {
        "n": float(n),
        "valid_n": float(nv),
        "valid_rate": (nv / n) if n > 0 else 0.0,
    }
    if nv == 0:
        out.update({"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "unique": 0.0})
        return out

    arr = np.array(valid, dtype=float)
    out.update({
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "unique": float(len(set(valid))),
    })
    return out


def aligned_delta_stats(scores_pref: List[Optional[float]], scores_non: List[Optional[float]]) -> Dict[str, float]:
    """
    Compute delta stats using aligned samples (i-th sample of pref minus i-th sample of non),
    skipping indices where either is None.
    """
    deltas = []
    for a, b in zip(scores_pref, scores_non):
        if a is None or b is None:
            continue
        deltas.append(a - b)

    if not deltas:
        return {"n": 0.0, "p_gt": np.nan, "p_eq": np.nan, "p_lt": np.nan, "mean": np.nan, "std": np.nan}

    arr = np.array(deltas, dtype=float)
    return {
        "n": float(len(deltas)),
        "p_gt": float((arr > 0).mean()),
        "p_eq": float((arr == 0).mean()),
        "p_lt": float((arr < 0).mean()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def mean_or_nan(xs: List[Optional[float]]) -> float:
    """Mean over valid samples, NaN if none."""
    vals = [v for v in xs if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def score_range(xs: List[Optional[float]]) -> float:
    """Range (max-min) over valid samples, NaN if none."""
    vals = [v for v in xs if v is not None]
    if not vals:
        return float("nan")
    return float(max(vals) - min(vals))


def range_ge(xs: List[Optional[float]], thr: float) -> bool:
    """True if range(xs) >= thr over valid samples."""
    r = score_range(xs)
    return (not np.isnan(r)) and (r >= thr)
