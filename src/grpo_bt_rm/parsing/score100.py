import re
from typing import Optional

_SCORE_RE = re.compile(r"<s>\s*(\d{1,3})\s*</s>", re.IGNORECASE)

def parse_score100_first(output_text: str) -> Optional[float]:
    m = _SCORE_RE.findall(output_text or "")
    if not m:
        return None
    raw = m[0].strip()  # score-first
    try:
        s = int(raw)
    except ValueError:
        return None
    if not (0 <= s <= 100):
        return None
    return float(s)

def parse_score100_last(output_text: str) -> Optional[float]:
    m = _SCORE_RE.findall(output_text or "")
    if not m:
        return None
    raw = m[-1].strip()  # score-last
    try:
        s = int(raw)
    except ValueError:
        return None
    if not (0 <= s <= 100):
        return None
    return float(s)
