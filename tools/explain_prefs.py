from __future__ import annotations

import argparse
import json
import os
import random
from typing import List, Optional

from datasets import load_dataset

from grpo_bt_rm.utils.model import load_qwen_instruct
from grpo_bt_rm.utils.generation import generate_batch


def build_explanation_prompt(post: str, summary_pref: str, summary_nonpref: str) -> str:
    return f"""You are an expert TL;DR quality analyst.

[Post]
{post}

[Preferred summary]
{summary_pref}

[Non-preferred summary]
{summary_nonpref}

Humans preferred the [Preferred summary] over the [Non-preferred summary].

In 3–5 bullet points, explain what might be the reasons humans preferred this summary over the other.
Then, in one final sentence, summarize the main reasons for the human preference.
"""


def _extract_pair(ex):
    post = ex["info"]["post"]
    s0 = ex["summaries"][0]["text"]
    s1 = ex["summaries"][1]["text"]
    label = int(ex["choice"])
    return post, s0, s1, label


def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--n_examples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, default="outputs/explanations/qwen_pref_explanations.jsonl")

    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)

    ap.add_argument("--progress_every", type=int, default=10)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)

    ds = load_dataset("openai/summarize_from_feedback", "comparisons")
    data = ds[args.split]

    idxs = list(range(len(data)))
    rng.shuffle(idxs)
    idxs = idxs[:args.n_examples]

    tok, model, device = load_qwen_instruct(args.model_name)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        for t, idx in enumerate(idxs, start=1):
            post, s0, s1, label = _extract_pair(data[idx])

            if label == 0:
                pref, non = s0, s1
            else:
                pref, non = s1, s0

            prompt = build_explanation_prompt(post, pref, non)

            # generate one explanation
            outs = generate_batch(
                tok, model, [prompt],
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            explanation = outs[0].strip()

            rec = {
                "idx": int(idx),
                "split": args.split,
                "post": post,
                "summary_pref": pref,
                "summary_nonpref": non,
                "explanation": explanation,
                "model_name": args.model_name,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if t % args.progress_every == 0:
                print(f"wrote {t}/{len(idxs)}")

    print(f"Done. Wrote {len(idxs)} records to {args.output}")


if __name__ == "__main__":
    main()
