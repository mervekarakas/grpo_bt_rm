import math

def softplus(x: float) -> float:
    """Numerically stable softplus."""
    if x > 0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))

def log_sigmoid(x: float) -> float:
    """log(sigmoid(x)) in a numerically stable way."""
    return -softplus(-x)
    

def scaled_delta(delta: float, temp: float = 1.0, clip: float = 0.0, neg_clip: float = 0.0) -> float:
    if temp <= 0:
        raise ValueError("temp must be > 0")
    x = delta / temp
    if clip and clip > 0 and x > clip:
        x = clip
    if neg_clip and neg_clip > 0 and x < -neg_clip:
        x = -neg_clip
    return x


def bt_logloss(d_pref_minus_non: float, temp: float = 1.0) -> float:
    """
    Bradley–Terry log-loss for a preferred-minus-nonpreferred score difference:
      loss = softplus(-(d/temp))
    Lower is better.
    """
    if temp <= 0:
        raise ValueError("temp must be > 0")
    return softplus(-(d_pref_minus_non / temp))

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
