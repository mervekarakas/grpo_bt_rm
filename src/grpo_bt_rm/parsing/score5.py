import re
from typing import Optional

_SCORE_RE = re.compile(r"<s>\s*([1-5](?:\.\d)?)\s*</s>", re.IGNORECASE)

def _validate(raw: str) -> Optional[float]:
    raw = raw.strip()
    try:
        score = float(raw)
    except ValueError:
        return None
    if not (1.0 <= score <= 5.0):
        return None
    # integer or exactly one decimal; reject 2+ decimals
    if "." in raw and len(raw.split(".")[-1]) != 1:
        return None
    return score

def parse_score5_first(output_text: str) -> Optional[float]:
    """Parse 1–5 score from the FIRST <s>...</s> tag (for score-first prompts)."""
    matches = _SCORE_RE.findall(output_text or "")
    if not matches:
        return None
    return _validate(matches[0])

def parse_score5_last(output_text: str) -> Optional[float]:
    """Parse 1–5 score from the LAST <s>...</s> tag (for score-last prompts)."""
    matches = _SCORE_RE.findall(output_text or "")
    if not matches:
        return None
    return _validate(matches[-1])
