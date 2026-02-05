from dataclasses import dataclass
from typing import Callable, Dict

from .score100 import score100_v1
from .score5 import score5_v1, score5_v2, score5_v3, score5_v4
from .hh import hh_score100_v1, hh_score100_v2, hh_score100_v3, hh_score100_v4, hh_score5_v1, hh_score5_v2, hh_score5_v3, hh_score5_v4

PromptFn = Callable[[str, str], str]

@dataclass(frozen=True)
class PromptSpec:
    fn: PromptFn
    default_parser: str   # e.g. "score5_last"
    desc: str = ""

PROMPT_REGISTRY: Dict[str, PromptSpec] = {
    "score100_v1": PromptSpec(score100_v1, default_parser="score100_first", desc="0–100 score-first"),
    "score5_v1":   PromptSpec(score5_v1,   default_parser="score5_last",    desc="1–5 score-last"),
    "score5_v2":   PromptSpec(score5_v2,   default_parser="score5_first",   desc="1–5 score-first + aspect"),
    "score5_v3":   PromptSpec(score5_v3,   default_parser="score5_first",   desc="1–5 score-first no criteria"),
    "score5_v4":   PromptSpec(score5_v4,   default_parser="score5_first",   desc="1–5 score-first rubric"),
    "hh_score100_v1": PromptSpec(hh_score100_v1, default_parser="score100_first", desc="HH 0–100 score-first"),
    "hh_score100_v2": PromptSpec(hh_score100_v2, default_parser="score100_first", desc="HH 0–100 concise"),
    "hh_score5_v1":   PromptSpec(hh_score5_v1,   default_parser="score5_first",   desc="HH 1–5 score-first"),
    "hh_score5_v2":   PromptSpec(hh_score5_v2,   default_parser="score5_first",   desc="HH 1–5 minimal"),
    "hh_score100_v3": PromptSpec(hh_score100_v3, default_parser="score100_last",  desc="HH 0–100 reason-first"),
    "hh_score5_v3":   PromptSpec(hh_score5_v3,   default_parser="score5_last",    desc="HH 1–5 reason-first"),
    "hh_score100_v4": PromptSpec(hh_score100_v4, default_parser="score100_last",  desc="HH 0–100 pros-cons-first"),
    "hh_score5_v4":   PromptSpec(hh_score5_v4,   default_parser="score5_last",    desc="HH 1–5 pros-cons-first"),
}

def get_prompt(name: str) -> PromptSpec:
    if name not in PROMPT_REGISTRY:
        raise KeyError(f"Unknown prompt '{name}'. Available: {sorted(PROMPT_REGISTRY.keys())}")
    return PROMPT_REGISTRY[name]
