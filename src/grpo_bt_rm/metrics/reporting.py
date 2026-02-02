from collections import Counter
from typing import List, Optional
import numpy as np


def uncertain_extreme(xs: List[float], low: float, high: float) -> bool:
    """True if xs contains both <=low and >=high."""
    if not xs:
        return False
    return (min(xs) <= low) and (max(xs) >= high)


def bucket_counts(values: List[float], low: float, high: float):
    """Return (low_count, mid_count, high_count) based on thresholds."""
    lowc = sum(1 for v in values if v <= low)
    highc = sum(1 for v in values if v >= high)
    mid = len(values) - lowc - highc
    return lowc, mid, highc


def histogram(values: List[float], bin_size: Optional[float] = None) -> Counter:
    """
    If bin_size is None: integer bins ("85", "92", ...)
    Else: bins by rounding to nearest multiple of bin_size ("3.5", "4.0", ...)
    Returns Counter[str].
    """
    c = Counter()
    if not values:
        return c

    if bin_size is None:
        for v in values:
            c[str(int(round(float(v))))] += 1
        return c

    for v in values:
        vv = round(float(v) / bin_size) * bin_size
        c[f"{vv:.1f}"] += 1
    return c


def summarize_array(name: str, xs: List[float]):
    """Print robust summary stats for an array."""
    if not xs:
        print(f"{name}: <empty>")
        return
    arr = np.array(xs, dtype=float)
    pcts = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
    print(
        f"{name}: n={len(arr)} mean={arr.mean():.4f} std={arr.std():.4f} "
        f"min={arr.min():.4f} p5={pcts[1]:.4f} p25={pcts[2]:.4f} "
        f"p50={pcts[3]:.4f} p75={pcts[4]:.4f} p95={pcts[5]:.4f} max={arr.max():.4f}"
    )
