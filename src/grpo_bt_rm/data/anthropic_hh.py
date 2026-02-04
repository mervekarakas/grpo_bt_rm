from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from datasets import load_dataset

_DATASET_NAME = "Anthropic/hh-rlhf"
_TURN_SEP_HUMAN = "\n\nHuman:"
_TURN_SEP_ASSISTANT = "\n\nAssistant:"


def load_split(split: str = "test"):
    """
    Load a split from Anthropic/hh-rlhf.
    Available splits: train, test (no validation).
    """
    ds = load_dataset(_DATASET_NAME)
    return ds[split]


def _parse_conversation(text: str) -> List[Tuple[str, str]]:
    """
    Parse an Anthropic-HH conversation string into a list of (role, content) turns.
    The format is: \n\nHuman: ... \n\nAssistant: ...
    Returns list of ("Human", content) / ("Assistant", content) tuples.
    """
    turns: List[Tuple[str, str]] = []
    # Split on \n\nHuman: or \n\nAssistant: while keeping the delimiter
    parts: List[str] = []
    remaining = text
    while remaining:
        # Find the next turn separator
        h_pos = remaining.find(_TURN_SEP_HUMAN)
        a_pos = remaining.find(_TURN_SEP_ASSISTANT)

        positions = []
        if h_pos != -1:
            positions.append(h_pos)
        if a_pos != -1:
            positions.append(a_pos)

        if not positions:
            # No more separators; remaining is trailing text
            if remaining.strip():
                parts.append(remaining)
            break

        earliest = min(positions)
        if earliest > 0:
            # Text before the first separator (usually empty)
            parts.append(remaining[:earliest])
        if earliest == h_pos:
            sep = _TURN_SEP_HUMAN
        else:
            sep = _TURN_SEP_ASSISTANT
        remaining = remaining[earliest + len(sep):]

        # Find the end of this turn (next separator)
        h2 = remaining.find(_TURN_SEP_HUMAN)
        a2 = remaining.find(_TURN_SEP_ASSISTANT)
        ends = []
        if h2 != -1:
            ends.append(h2)
        if a2 != -1:
            ends.append(a2)

        if ends:
            end = min(ends)
            content = remaining[:end]
            remaining = remaining[end:]
        else:
            content = remaining
            remaining = ""

        role = "Human" if sep == _TURN_SEP_HUMAN else "Assistant"
        turns.append((role, content.strip()))

    return turns


def extract_pair(example: Dict[str, Any]) -> Tuple[str, str, str, int]:
    """
    Return (prompt, response0, response1, label) from one HH example.

    - prompt: shared conversation prefix (all turns up to and including
      the last Human turn)
    - response0: final Assistant turn from chosen (= preferred)
    - response1: final Assistant turn from rejected
    - label: always 0 (response0 is always the chosen/preferred one)
    """
    chosen_turns = _parse_conversation(example["chosen"])
    rejected_turns = _parse_conversation(example["rejected"])

    # Extract final assistant response from each
    response0 = ""
    for role, content in reversed(chosen_turns):
        if role == "Assistant":
            response0 = content
            break

    response1 = ""
    for role, content in reversed(rejected_turns):
        if role == "Assistant":
            response1 = content
            break

    # Build shared prompt: all turns before the last assistant response
    # Use chosen conversation; strip the final assistant turn
    prompt_turns = []
    for role, content in chosen_turns:
        if role == "Assistant" and content == response0:
            break
        prompt_turns.append(f"{role}: {content}")

    prompt = "\n\n".join(prompt_turns)

    # label = 0 means response0 (chosen) is preferred
    return prompt, response0, response1, 0


def sample_indices(n: int, split: str = "test", seed: int = 0) -> List[int]:
    """
    Sample n indices from the given HH split (without replacement).
    """
    data = load_split(split)
    idxs = list(range(len(data)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    return idxs[:n]
