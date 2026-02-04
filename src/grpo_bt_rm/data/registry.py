from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from . import summarize_from_feedback as sff
from . import anthropic_hh as hh

# Type aliases matching the adapter interface
LoadSplitFn = Callable[[str], Any]  # (split) -> datasets.Dataset
ExtractPairFn = Callable[[Dict[str, Any]], Tuple[str, str, str, int]]


@dataclass(frozen=True)
class DatasetAdapter:
    load_split: LoadSplitFn
    extract_pair: ExtractPairFn
    name: str                # e.g. "Anthropic/hh-rlhf"
    default_eval_split: str  # "test" for HH, "validation" for summarize


DATASET_REGISTRY: Dict[str, DatasetAdapter] = {
    "summarize_from_feedback": DatasetAdapter(
        load_split=sff.load_comparisons_split,
        extract_pair=sff.extract_pair,
        name="openai/summarize_from_feedback:comparisons",
        default_eval_split="validation",
    ),
    "anthropic_hh": DatasetAdapter(
        load_split=hh.load_split,
        extract_pair=hh.extract_pair,
        name="Anthropic/hh-rlhf",
        default_eval_split="test",
    ),
}


def get_dataset(name: str) -> DatasetAdapter:
    if name not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {sorted(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name]
