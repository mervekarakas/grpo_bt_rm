import argparse
import json
import os
import random
from typing import List
import numpy as np

# NOTE: until models.py/utils.py are moved into src/,
# run with: PYTHONPATH=src:. python -m grpo_bt_rm.eval.variance ...
from grpo_bt_rm.utils.model import load_qwen_instruct
from grpo_bt_rm.data.registry import get_dataset

from grpo_bt_rm.prompts.registry import get_prompt
from grpo_bt_rm.parsing.registry import get_parser
from grpo_bt_rm.eval.sampling import sample_scores_for_prompts_batch
from grpo_bt_rm.metrics.stats import (
    summarize_scores,
    aligned_delta_stats,
    mean_or_nan,
    range_ge,
)
from grpo_bt_rm.metrics.reporting import (
    uncertain_extreme,
    histogram,
)


def _parse_thresholds(s: str) -> List[float]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    thr = [float(p) for p in parts]
    # de-dup + sort
    thr = sorted(set(thr))
    return thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="summarize_from_feedback", help="Dataset name from registry")
    ap.add_argument("--split", type=str, default="")
    ap.add_argument("--n_pairs", type=int, default=50)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--save_jsonl", type=str, default="", help="Optional: save per-pair stats JSONL.")
    ap.add_argument("--batch_pairs", type=int, default=16, help="Number of pairs per batch (2 prompts per pair).")

    # NEW: prompt/parser selection
    ap.add_argument("--prompt", type=str, default="score100_v1", help="Prompt name from registry")
    ap.add_argument("--parser", type=str, default="", help="Parser name; if empty use prompt default")

    # range thresholds
    ap.add_argument(
        "--range_thresholds",
        type=str,
        default="20,30,40",
        help="Comma-separated range thresholds for wrong-pair diagnostics, e.g. '10,20,40'",
    )

    ap.add_argument("--unc_low", type=float, default=None,
    help="Low threshold for extreme uncertainty. If omitted, defaults depend on parser.")
    ap.add_argument("--unc_high", type=float, default=None,
    help="High threshold for extreme uncertainty. If omitted, defaults depend on parser.")

    # Multi-GPU sharding
    ap.add_argument("--shard_id", type=int, default=0, help="Shard index (0-based)")
    ap.add_argument("--num_shards", type=int, default=1, help="Total number of shards")

    args = ap.parse_args()

    range_thresholds = _parse_thresholds(args.range_thresholds)
    if not range_thresholds:
        raise ValueError("--range_thresholds produced an empty list; provide like '20,30,40'")

    # Resolve prompt + parser
    spec = get_prompt(args.prompt)
    prompt_fn = spec.fn
    parser_name = args.parser or spec.default_parser
    parse_fn = get_parser(parser_name)

    if args.unc_low is None:
        args.unc_low = 20.0 if "score100" in parser_name else 2.0
    if args.unc_high is None:
        args.unc_high = 80.0 if "score100" in parser_name else 4.0


    # ----------------------------
    # Data
    # ----------------------------
    adapter = get_dataset(args.dataset)
    split = args.split or adapter.default_eval_split
    data = adapter.load_split(split)
    rng = random.Random(args.seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    indices = indices[:args.n_pairs]

    # Shard partitioning (for multi-GPU runs)
    if args.num_shards > 1:
        shard_size = len(indices) // args.num_shards
        start = args.shard_id * shard_size
        end = start + shard_size if args.shard_id < args.num_shards - 1 else len(indices)
        indices = indices[start:end]

    # ----------------------------
    # Model
    # ----------------------------
    tokenizer, model, device = load_qwen_instruct()

    shard_info = f" shard={args.shard_id}/{args.num_shards}" if args.num_shards > 1 else ""
    print(f"\n=== Sampling variance test (BATCHED){shard_info} ===")
    print(f"prompt={args.prompt} parser={parser_name}")
    print(f"dataset={args.dataset} split={split} n_pairs={args.n_pairs} n_samples={args.n_samples}")
    if args.num_shards > 1:
        print(f"shard_id={args.shard_id} num_shards={args.num_shards} shard_pairs={len(indices)}")
    print(f"temp={args.temperature} top_p={args.top_p} top_k={args.top_k} max_new_tokens={args.max_new_tokens}")
    print(f"use_chat_template={args.use_chat_template} batch_pairs={args.batch_pairs}")
    print(f"uncertainty thresholds: low={args.unc_low} high={args.unc_high}")
    print(f"range_thresholds={range_thresholds}\n")

    # ----------------------------
    # Output writer
    # ----------------------------
    fout = None
    if args.save_jsonl:
        out_dir = os.path.dirname(args.save_jsonl)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fout = open(args.save_jsonl, "w", encoding="utf-8")

    # ----------------------------
    # Aggregates across pairs
    # ----------------------------
    valid_rates = []
    within_std = []
    within_unique = []
    tie_rates = []
    exp_correct_ties05 = []
    exp_correct_strict = []

    wrong_pairs = 0
    wrong_uncertain_any = 0
    wrong_uncertain_both = 0

    # range-based counters, keyed by threshold
    wrong_range_any = {thr: 0 for thr in range_thresholds}
    wrong_range_both = {thr: 0 for thr in range_thresholds}

    wrong_score_values: List[float] = []
    correct_score_values: List[float] = []

    first_bad = 0
    printed_pair0 = False

    # ----------------------------
    # Main batched loop over pairs
    # ----------------------------
    for base in range(0, len(indices), args.batch_pairs):
        chunk = indices[base:base + args.batch_pairs]

        posts = []
        s0s = []
        s1s = []
        labels = []

        for idx in chunk:
            ex = data[idx]
            post, s0, s1, label = adapter.extract_pair(ex)
            posts.append(post)
            s0s.append(s0)
            s1s.append(s1)
            labels.append(label)

        prompts = []
        for post, s0, s1 in zip(posts, s0s, s1s):
            prompts.append(prompt_fn(post, s0))
            prompts.append(prompt_fn(post, s1))

        scores_list, texts_list = sample_scores_for_prompts_batch(
            tokenizer=tokenizer,
            model=model,
            device=device,
            prompts=prompts,
            parse_score=parse_fn,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            use_chat_template=args.use_chat_template,
        )

        for i_in_chunk, idx in enumerate(chunk):
            label = labels[i_in_chunk]
            scores0 = scores_list[2 * i_in_chunk]
            scores1 = scores_list[2 * i_in_chunk + 1]
            texts0  = texts_list[2 * i_in_chunk]
            texts1  = texts_list[2 * i_in_chunk + 1]

            # show up to 3 format failures
            if first_bad < 3:
                if any(s is None for s in scores0):
                    k = scores0.index(None)
                    print("\n--- Example format failure (summary0) ---")
                    print(texts0[k])
                    print("--- end ---\n")
                    first_bad += 1
                if any(s is None for s in scores1) and first_bad < 3:
                    k = scores1.index(None)
                    print("\n--- Example format failure (summary1) ---")
                    print(texts1[k])
                    print("--- end ---\n")
                    first_bad += 1

            sum0 = summarize_scores(scores0)
            sum1 = summarize_scores(scores1)

            # aligned delta stats (pref - non)
            if label == 0:
                scores_pref, scores_non = scores0, scores1
            else:
                scores_pref, scores_non = scores1, scores0

            dstat = aligned_delta_stats(scores_pref, scores_non)

            if np.isnan(dstat["p_gt"]):
                p_corr_t05 = np.nan
                p_corr_strict = np.nan
            else:
                p_corr_strict = float(dstat["p_gt"])
                p_corr_t05 = float(dstat["p_gt"] + 0.5 * dstat["p_eq"])

            valid_rates.append((sum0["valid_rate"] + sum1["valid_rate"]) / 2.0)

            stds = [sum0["std"], sum1["std"]]
            stds = [x for x in stds if not np.isnan(x)]
            within_std.append(float(np.mean(stds)) if stds else np.nan)

            uniqs = [sum0["unique"], sum1["unique"]]
            uniqs = [x for x in uniqs if not np.isnan(x)]
            within_unique.append(float(np.mean(uniqs)) if uniqs else np.nan)

            tie_rates.append(dstat["p_eq"] if not np.isnan(dstat["p_eq"]) else np.nan)
            exp_correct_ties05.append(p_corr_t05)
            exp_correct_strict.append(p_corr_strict)

            # mean-prediction wrong-pair diagnostics
            m0 = mean_or_nan(scores0)
            m1 = mean_or_nan(scores1)
            if not np.isnan(m0) and not np.isnan(m1):
                pred_mean = 0 if m0 > m1 else 1
                is_wrong_mean = (pred_mean != label)

                v0 = [v for v in scores0 if v is not None]
                v1 = [v for v in scores1 if v is not None]

                if is_wrong_mean:
                    wrong_pairs += 1
                    wrong_score_values.extend(v0)
                    wrong_score_values.extend(v1)

                    u0 = uncertain_extreme([v for v in scores0 if v is not None], low=args.unc_low, high=args.unc_high)
                    u1 = uncertain_extreme([v for v in scores1 if v is not None], low=args.unc_low, high=args.unc_high)


                    if u0 or u1:
                        wrong_uncertain_any += 1
                    if u0 and u1:
                        wrong_uncertain_both += 1

                    # range thresholds
                    for thr in range_thresholds:
                        r0 = range_ge(scores0, thr)
                        r1 = range_ge(scores1, thr)
                        if r0 or r1:
                            wrong_range_any[thr] += 1
                        if r0 and r1:
                            wrong_range_both[thr] += 1
                else:
                    correct_score_values.extend(v0)
                    correct_score_values.extend(v1)

            if fout is not None:
                rec = {
                    "idx": int(idx),
                    "label": int(label),
                    "prompt": args.prompt,
                    "parser": parser_name,
                    "summary0_stats": sum0,
                    "summary1_stats": sum1,
                    "delta_stats_pref_minus_non": dstat,
                    "expected_correctness_ties05": p_corr_t05,
                    "expected_correctness_strict": p_corr_strict,
                    "scores0": scores0,
                    "scores1": scores1,
                }
                fout.write(json.dumps(rec) + "\n")

            if not printed_pair0:
                printed_pair0 = True
                print("\n--- Pair 0 quick view ---")
                print("idx:", int(idx))
                print("label (0 means summary0 preferred):", label)
                print("scores0:", scores0)
                print("scores1:", scores1)
                print("summary0 stats:", sum0)
                print("summary1 stats:", sum1)
                print("delta stats (pref - non):", dstat)
                print("expected_correctness (ties=0.5):", p_corr_t05)
                print("expected_correctness (strict):", p_corr_strict)
                print("--- end ---\n")

        done = min(base + len(chunk), len(indices))
        shard_prefix = f"[shard {args.shard_id}] " if args.num_shards > 1 else ""
        if done % 50 == 0 or done == len(indices):
            print(
                f"{shard_prefix}[{done}/{len(indices)}] "
                f"avg_valid_rate={np.nanmean(valid_rates):.3f} "
                f"avg_within_std={np.nanmean(within_std):.3f} "
                f"avg_tie_rate={np.nanmean(tie_rates):.3f} "
                f"avg_acc_ties05={np.nanmean(exp_correct_ties05):.3f} "
                f"avg_acc_strict={np.nanmean(exp_correct_strict):.3f}"
            )

    if fout is not None:
        fout.close()
        print(f"\nSaved per-pair results to: {args.save_jsonl}")

    print("\n=== Final summary across pairs ===")
    print(f"avg valid score rate:         {np.nanmean(valid_rates):.3f}")
    print(f"avg within-summary std:       {np.nanmean(within_std):.3f}")
    print(f"avg unique scores/summary:    {np.nanmean(within_unique):.3f}")
    print(f"avg tie rate (delta==0):      {np.nanmean(tie_rates):.3f}")
    print(f"avg accuracy (ties=0.5):      {np.nanmean(exp_correct_ties05):.3f}")
    print(f"avg accuracy (strict ties=0): {np.nanmean(exp_correct_strict):.3f}")

    print("\n=== Wrong-pair uncertainty (mean-score prediction) ===")
    if wrong_pairs == 0:
        print("wrong_pairs: 0")
    else:
        print(f"wrong_pairs: {wrong_pairs}")
        print(f"uncertain_any among wrong:  {wrong_uncertain_any / wrong_pairs:.3f}")
        print(f"uncertain_both among wrong: {wrong_uncertain_both / wrong_pairs:.3f}")

        # score100 → integer bins, score5 → 0.1 bins
        bin_size = None if "score100" in parser_name else 0.1

        wh = histogram(wrong_score_values, bin_size=bin_size)
        ch = histogram(correct_score_values, bin_size=bin_size)

        print("\nWrong sampled-score histogram:")
        def _hist_sort_key(x: str):
            return float(x) if bin_size is not None else int(x)

        for k in sorted(wh.keys(), key=_hist_sort_key):
            print(f"  {k}: {wh[k]}")

        print("\nCorrect sampled-score histogram:")
        for k in sorted(ch.keys(), key=_hist_sort_key):
            print(f"  {k}: {ch[k]}")

        print("\n=== Range-based uncertainty on WRONG pairs (mean-score prediction) ===")
        print("Definition: range = max(samples) - min(samples) over valid samples, per summary.")
        for thr in range_thresholds:
            print(f"range_any>={int(thr)} among wrong:  {wrong_range_any[thr] / wrong_pairs:.3f}")
        for thr in range_thresholds:
            print(f"range_both>={int(thr)} among wrong: {wrong_range_both[thr] / wrong_pairs:.3f}")

    print("\nNotes:")
    print("- avg accuracy (ties=0.5) = P(delta>0) + 0.5*P(delta==0) under aligned valid samples.")
    print("- avg accuracy (strict)  = P(delta>0) under aligned valid samples (ties count as 0).")


if __name__ == "__main__":
    main()
