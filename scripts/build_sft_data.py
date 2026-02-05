"""
Build SFT training data from teacher-scored HH pairs.

Takes teacher data (with scores + reasoning from API models) and creates
SFT format: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

Filters to only pairs where teacher got it correct (score0 > score1 when
label=0, or score1 > score0 when label=1). This ensures the student learns
from high-quality examples.

Usage:
    source /home/mkarakas/venvs/swift/bin/activate
    PYTHONPATH=src python scripts/build_sft_data.py \
        --input_dirs /data/.../teacher_data/score100_* \
        --output /data/.../sft_data.jsonl \
        --prompt hh_score100_v1 \
        --filter_correct

Output format (for swift sft training):
    {"messages": [
        {"role": "user", "content": "<prompt asking to score>"},
        {"role": "assistant", "content": "<s>75</s> The response is helpful..."}
    ]}
"""

import argparse
import json
import glob
from pathlib import Path
from typing import Optional

from grpo_bt_rm.data.registry import get_dataset
from grpo_bt_rm.prompts.registry import get_prompt


def load_teacher_records(input_paths: list[str]) -> list[dict]:
    """Load all teacher records from JSONL files."""
    records = []
    for pattern in input_paths:
        for path in glob.glob(pattern):
            p = Path(path)
            if p.is_dir():
                # Look for results.jsonl in directory
                jsonl = p / "results.jsonl"
                if jsonl.exists():
                    path = str(jsonl)
                else:
                    continue
            with open(path) as f:
                for line in f:
                    records.append(json.loads(line))
    return records


def build_assistant_response(score: float, reasoning: str, scale: str = "score100") -> str:
    """Build assistant response in correct format."""
    # Extract just the main reasoning text (remove EVAL markers if present)
    text = reasoning.strip()

    # Remove any leading "=== EVAL N ===" markers
    import re
    text = re.sub(r"^===\s*EVAL\s+\d+\s*===\s*", "", text)

    # Check if there's already a score tag in the reasoning
    if re.search(r"<s>\d+</s>", text):
        # Already has the tag, return as-is
        return text

    # Build response with score tag first
    if scale == "score100":
        score_tag = f"<s>{int(round(score))}</s>"
    else:
        score_tag = f"<s>{score:.1f}</s>"

    # Clean up reasoning - remove any score mentions at the start
    text = re.sub(r"^\d+\.?\d*\s*[-:]\s*", "", text)
    text = re.sub(r"^Score:\s*\d+\.?\d*\s*", "", text, flags=re.IGNORECASE)

    if text:
        return f"{score_tag} {text}"
    else:
        return score_tag


def main():
    parser = argparse.ArgumentParser(description="Build SFT data from teacher scores")
    parser.add_argument("--input_dirs", type=str, nargs="+", required=True,
                        help="Input directories or JSONL files (supports glob patterns)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--prompt", type=str, default="hh_score100_v1",
                        help="Prompt name to use for building training examples")
    parser.add_argument("--scale", type=str, default="score100", choices=["score100", "score5"],
                        help="Score scale (affects output format)")
    parser.add_argument("--filter_correct", action="store_true",
                        help="Only include pairs where teacher got direction correct")
    parser.add_argument("--min_margin", type=float, default=0.0,
                        help="Minimum score margin |s0-s1| to include (0=all)")
    parser.add_argument("--include_both_sides", action="store_true",
                        help="Include both sides of each pair (2 examples per pair)")
    parser.add_argument("--dataset", type=str, default="anthropic_hh",
                        help="Dataset to load original examples from")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split (default: train)")
    args = parser.parse_args()

    # Load prompt function
    prompt_spec = get_prompt(args.prompt)
    prompt_fn = prompt_spec.fn

    # Load dataset for original examples
    adapter = get_dataset(args.dataset)
    data = adapter.load_split(args.split)
    print(f"Loaded {len(data)} examples from {args.dataset}/{args.split}")

    # Load teacher records
    records = load_teacher_records(args.input_dirs)
    print(f"Loaded {len(records)} teacher records")

    # Filter records
    valid_records = []
    for r in records:
        # Skip if no valid scores
        if r.get("score0") is None or r.get("score1") is None:
            continue

        # Skip if error
        if r.get("error"):
            continue

        # Filter by correctness if requested
        if args.filter_correct:
            if not r.get("correct"):
                continue

        # Filter by margin if requested
        margin = abs(r["score0"] - r["score1"])
        if margin < args.min_margin:
            continue

        valid_records.append(r)

    print(f"After filtering: {len(valid_records)} valid records")

    # Build SFT examples
    sft_examples = []
    skipped = 0

    for r in valid_records:
        idx = r["idx"]

        # Get original example
        try:
            ex = data[idx]
            context, s0, s1, label = adapter.extract_pair(ex)
        except (IndexError, KeyError) as e:
            skipped += 1
            continue

        # Build examples for each side
        sides_to_include = [0, 1] if args.include_both_sides else ([0] if label == 0 else [1])

        for side in sides_to_include:
            response = s0 if side == 0 else s1
            score = r["score0"] if side == 0 else r["score1"]
            reasoning = r["reasoning0"] if side == 0 else r["reasoning1"]

            # Build user prompt
            user_content = prompt_fn(context, response)

            # Build assistant response
            assistant_content = build_assistant_response(score, reasoning, args.scale)

            sft_examples.append({
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
            })

    print(f"Skipped {skipped} records (index out of range)")
    print(f"Built {len(sft_examples)} SFT examples")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for ex in sft_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sft_examples)} examples to {output_path}")

    # Summary stats
    if sft_examples:
        user_lens = [len(ex["messages"][0]["content"]) for ex in sft_examples]
        asst_lens = [len(ex["messages"][1]["content"]) for ex in sft_examples]
        print(f"\nStats:")
        print(f"  User prompt len: {sum(user_lens)/len(user_lens):.0f} avg, {max(user_lens)} max")
        print(f"  Asst response len: {sum(asst_lens)/len(asst_lens):.0f} avg, {max(asst_lens)} max")


if __name__ == "__main__":
    main()
