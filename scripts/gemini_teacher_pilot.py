"""
Pilot: Use Gemini as a teacher model to score HH pairs in batches.
Measures accuracy to see if Gemini is a strong enough teacher for SFT.

Usage:
    source /home/mkarakas/venvs/swift/bin/activate
    PYTHONPATH=src python scripts/gemini_teacher_pilot.py \
        --n_pairs 200 --batch_size 10 --model gemini-2.5-flash
"""

import argparse
import json
import os
import re
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import google.generativeai as genai

from grpo_bt_rm.data.registry import get_dataset


def build_batch_prompt(pairs: list[dict], scale: str = "score5") -> str:
    """Build a single prompt that asks Gemini to evaluate multiple responses."""

    if scale == "score5":
        score_instruction = (
            "Score each response 1.0-5.0 (1=harmful/useless, 2=poor, 3=okay, 4=good, 5=excellent).\n"
            "Use the EXACT format: <s>x.y</s>"
        )
    else:  # score100
        score_instruction = (
            "Score each response 0-100 (0=harmful, 50=mixed, 100=excellent).\n"
            "Use the EXACT format: <s>NN</s>"
        )

    header = f"""You are an impartial judge evaluating AI assistant responses.
You will see {len(pairs)} evaluations below. For EACH one:
1. Write a brief analysis (2-3 sentences): Is it helpful? Safe? Relevant? Clear?
2. Output the score in EXACT format: <s>score</s>

{score_instruction}

Separate each evaluation with a line: === EVAL N ===
"""

    blocks = []
    for i, pair in enumerate(pairs):
        block = f"""=== EVAL {i+1} ===
CONVERSATION:
{pair['context']}

RESPONSE:
{pair['response']}
"""
        blocks.append(block)

    return header + "\n".join(blocks)


def parse_batch_response(text: str, n_expected: int, scale: str = "score5"):
    """Parse multiple evaluations from a single Gemini response."""
    # Split on === EVAL N === markers
    parts = re.split(r"===\s*EVAL\s+\d+\s*===", text)
    # First part is usually empty or header
    parts = [p.strip() for p in parts if p.strip()]

    results = []
    for i, part in enumerate(parts):
        # Extract score
        if scale == "score5":
            matches = re.findall(r"<s>([\d.]+)</s>", part)
        else:
            matches = re.findall(r"<s>(\d+)</s>", part)

        score = None
        if matches:
            try:
                score = float(matches[-1])  # take last match (reason-first)
            except ValueError:
                pass

        results.append({
            "reasoning": part,
            "score": score,
            "raw_match": matches[-1] if matches else None,
        })

    # Pad if we got fewer results than expected
    while len(results) < n_expected:
        results.append({"reasoning": "", "score": None, "raw_match": None})

    return results[:n_expected]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pairs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of evaluations per API call")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--scale", type=str, default="score5", choices=["score5", "score100"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str,
                        default="/data/mkarakas/experiments/grpo_bt_rm/gemini_teacher_pilot")
    parser.add_argument("--api_key", type=str,
                        default=os.environ.get("GEMINI_API_KEY", ""))
    parser.add_argument("--rpm_limit", type=int, default=5,
                        help="Max requests per minute (free tier: 5 for pro, 10 for flash)")
    args = parser.parse_args()

    # Setup
    genai.configure(api_key=args.api_key)
    model = genai.GenerativeModel(args.model)
    adapter = get_dataset("anthropic_hh")
    data = adapter.load_split(args.split)

    import random
    rng = random.Random(args.seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    indices = indices[: args.n_pairs]

    out_dir = Path(args.output_dir) / f"{args.scale}_{args.model.replace('/', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.jsonl"

    # Each pair has 2 responses -> 2 evaluations
    # We batch `batch_size` evaluations per API call
    # So we process batch_size//2 pairs per call (each pair = 2 evals)
    pairs_per_call = max(1, args.batch_size // 2)
    evals_per_call = pairs_per_call * 2

    print(f"Model: {args.model}")
    print(f"Scale: {args.scale}")
    print(f"Pairs: {args.n_pairs}, batch_size: {evals_per_call} evals/call")
    print(f"Total API calls needed: {(args.n_pairs + pairs_per_call - 1) // pairs_per_call}")
    print(f"RPM limit: {args.rpm_limit}")
    print(f"Output: {out_file}")
    print()

    correct = 0
    total = 0
    valid = 0
    total_scores = 0
    total_api_calls = 0
    request_times = []

    with open(out_file, "w") as fout:
        for chunk_start in range(0, len(indices), pairs_per_call):
            chunk_indices = indices[chunk_start : chunk_start + pairs_per_call]

            # Build evaluation items for this batch
            eval_items = []
            pair_data = []
            for idx in chunk_indices:
                ex = data[idx]
                context, s0, s1, label = adapter.extract_pair(ex)
                pair_data.append({
                    "idx": idx, "context": context,
                    "s0": s0, "s1": s1, "label": label,
                })
                # Eval 0: response0 (chosen), Eval 1: response1 (rejected)
                eval_items.append({"context": context, "response": s0})
                eval_items.append({"context": context, "response": s1})

            prompt = build_batch_prompt(eval_items, scale=args.scale)

            # Rate limiting
            now = time.time()
            request_times = [t for t in request_times if now - t < 60]
            if len(request_times) >= args.rpm_limit:
                wait = 60 - (now - request_times[0]) + 1
                print(f"  Rate limit: waiting {wait:.0f}s...")
                time.sleep(wait)

            # Call Gemini
            try:
                response = model.generate_content(prompt)
                request_times.append(time.time())
                total_api_calls += 1

                results = parse_batch_response(
                    response.text, n_expected=len(eval_items), scale=args.scale
                )

                # Process pairs
                for i, pd in enumerate(pair_data):
                    r0 = results[i * 2]       # chosen response score
                    r1 = results[i * 2 + 1]   # rejected response score

                    score0 = r0["score"]
                    score1 = r1["score"]

                    total += 1

                    if score0 is not None:
                        valid += 1
                        total_scores += 1
                    if score1 is not None:
                        total_scores += 1

                    pair_correct = None
                    if score0 is not None and score1 is not None:
                        if pd["label"] == 0:
                            # response0 is preferred -> score0 should be higher
                            pair_correct = score0 > score1
                        else:
                            pair_correct = score1 > score0
                        if pair_correct:
                            correct += 1

                    record = {
                        "idx": pd["idx"],
                        "label": pd["label"],
                        "score0": score0,
                        "score1": score1,
                        "reasoning0": r0["reasoning"],
                        "reasoning1": r1["reasoning"],
                        "correct": pair_correct,
                    }
                    fout.write(json.dumps(record) + "\n")

                acc = correct / total if total > 0 else 0
                valid_rate = total_scores / (total * 2) if total > 0 else 0
                print(
                    f"[{total}/{args.n_pairs}] "
                    f"acc={acc:.3f} valid={valid_rate:.3f} "
                    f"api_calls={total_api_calls} "
                    f"(tokens: {response.usage_metadata.prompt_token_count}+{response.usage_metadata.candidates_token_count})"
                )

            except Exception as e:
                print(f"  ERROR on batch starting at pair {chunk_start}: {e}")
                # Write empty records for failed batch
                for pd in pair_data:
                    total += 1
                    record = {
                        "idx": pd["idx"], "label": pd["label"],
                        "score0": None, "score1": None,
                        "reasoning0": "", "reasoning1": "",
                        "correct": None, "error": str(e),
                    }
                    fout.write(json.dumps(record) + "\n")
                time.sleep(10)  # back off on error

    # Final summary
    scored_pairs = sum(1 for _ in open(out_file)
                       if json.loads(_).get("correct") is not None)
    correct_pairs = sum(1 for _ in open(out_file)
                        if json.loads(_).get("correct") is True)

    print(f"\n=== Final Results ===")
    print(f"Total pairs: {total}")
    print(f"Scored pairs (both valid): {scored_pairs}")
    print(f"Correct: {correct_pairs}/{scored_pairs} = {correct_pairs/scored_pairs:.3f}" if scored_pairs > 0 else "No scored pairs")
    print(f"Valid score rate: {total_scores}/{total*2} = {total_scores/(total*2):.3f}" if total > 0 else "")
    print(f"Total API calls: {total_api_calls}")
    print(f"Output: {out_file}")


if __name__ == "__main__":
    main()
