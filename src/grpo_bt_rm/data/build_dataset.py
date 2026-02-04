import argparse
import json
import os

from grpo_bt_rm.data.summarize_from_feedback import load_comparisons_split, extract_pair
from grpo_bt_rm.prompts.registry import get_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--split", default="train", help="HF split: train/validation/test (if present)")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only write this many pairs (smoke test)")
    ap.add_argument("--prompt", default="score100_v1", help="Prompt name from registry")
    ap.add_argument("--add_meta", action="store_true", help="Add prompt/dataset metadata to each row")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    spec = get_prompt(args.prompt)
    prompt_fn = spec.fn  # <-- IMPORTANT

    data = load_comparisons_split(args.split)

    wrote = 0
    pair_id = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in data:
            post, s0, s1, pref = extract_pair(ex)

            rec0 = {
                "messages": [{"role": "user", "content": prompt_fn(post, s0)}],
                "pair_id": pair_id,
                "side": 0,
                "preferred_side": pref,
            }
            rec1 = {
                "messages": [{"role": "user", "content": prompt_fn(post, s1)}],
                "pair_id": pair_id,
                "side": 1,
                "preferred_side": pref,
            }

            if args.add_meta:
                meta = {
                    "prompt_name": args.prompt,
                    "parser_name": spec.default_parser,
                    "dataset": "openai/summarize_from_feedback:comparisons",
                    "split": args.split,
                }
                rec0.update(meta)
                rec1.update(meta)

            f.write(json.dumps(rec0, ensure_ascii=False) + "\n")
            f.write(json.dumps(rec1, ensure_ascii=False) + "\n")

            wrote += 2
            pair_id += 1
            if args.limit > 0 and pair_id >= args.limit:
                break

    print(f"Wrote {wrote} rows ({pair_id} pairs) to {args.out}")
    with open(args.out, "r", encoding="utf-8") as fin:
        print("First two lines:")
        print(fin.readline().strip())
        print(fin.readline().strip())


if __name__ == "__main__":
    main()
