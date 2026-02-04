from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from datasets import load_dataset

_DATASET_NAME = "openai/summarize_from_feedback"
_CONFIG = "comparisons"


def load_comparisons_split(split: str = "validation"):
    """
    Load a split from openai/summarize_from_feedback (comparisons).
    Returns a HuggingFace datasets Split object.
    """
    ds = load_dataset(_DATASET_NAME, _CONFIG)
    return ds[split]


def extract_pair(example: Dict[str, Any]) -> Tuple[str, str, str, int]:
    """
    Return (post, summary0, summary1, label) from one comparisons example.

    label = 0 means summary0 preferred
    label = 1 means summary1 preferred
    """
    post = example["info"]["post"]
    summaries = example["summaries"]
    s0 = summaries[0]["text"]
    s1 = summaries[1]["text"]
    label = int(example["choice"])
    return post, s0, s1, label


def sample_indices(n: int, split: str = "validation", seed: int = 0) -> List[int]:
    """
    Sample n indices from the given comparisons split (without replacement).
    """
    data = load_comparisons_split(split)
    idxs = list(range(len(data)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    return idxs[:n]
