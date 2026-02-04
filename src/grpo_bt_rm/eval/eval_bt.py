#!/usr/bin/env python3
import argparse
import os
import random
from typing import List, Optional

import numpy as np
import torch

from grpo_bt_rm.data.registry import get_dataset
from grpo_bt_rm.prompts.registry import get_prompt
from grpo_bt_rm.parsing.registry import get_parser

from grpo_bt_rm.utils.model import load_base_model_name, load_ckpt_model
from grpo_bt_rm.utils.generation import generate_batch
from grpo_bt_rm.utils.math import bt_logloss
from grpo_bt_rm.metrics.reporting import (
    uncertain_extreme,
    bucket_counts,
    histogram,
    summarize_array,
)

def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser()

    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default="summarize_from_feedback", help="Dataset name from registry")

    ap.add_argument("--n_pairs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")

    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)

    ap.add_argument("--batch_pairs", type=int, default=8)
    ap.add_argument("--progress_every", type=int, default=50)

    ap.add_argument("--prompt", type=str, default="score5_v1")
    ap.add_argument("--parser", type=str, default="")

    ap.add_argument("--bt_temp", type=float, default=1.0)
    ap.add_argument("--report_uncertainty", action="store_true")
    ap.add_argument("--unc_low", type=float, default=2.0)
    ap.add_argument("--unc_high", type=float, default=4.0)

    ap.add_argument("--report_margins", action="store_true")

    args = ap.parse_args(argv)
    if args.bt_temp <= 0:
        raise ValueError("--bt_temp must be > 0")

    random.seed(args.seed)

    # Resolve prompt + parser
    prompt_spec = get_prompt(args.prompt)
    prompt_fn = prompt_spec.fn
    parser_name = args.parser or prompt_spec.default_parser
    parse_fn = get_parser(parser_name)

    base_model = load_base_model_name(args.run_dir)
    print("Base model:", base_model)
    print("Checkpoint:", args.checkpoint)
    print(f"Prompt: {args.prompt}  Parser: {parser_name}")
    print(f"bt_temp={args.bt_temp} unc_low={args.unc_low} unc_high={args.unc_high}")

    tok, model = load_ckpt_model(base_model, args.checkpoint, args.dtype)

    adapter = get_dataset(args.dataset)
    val = adapter.load_split(adapter.default_eval_split)
    idxs = list(range(len(val)))
    random.shuffle(idxs)
    idxs = idxs[:args.n_pairs]

    total = correct = ties = skipped = 0
    deltas: List[float] = []
    bt_losses_all: List[float] = []
    bt_losses_nontie: List[float] = []

    non_tie_total = 0
    non_tie_correct = 0

    # Optional diagnostics
    wrong_pairs = 0
    wrong_uncertain_any = 0
    wrong_uncertain_both = 0
    wrong_score_values: List[float] = []
    correct_score_values: List[float] = []

    deltas_nontie: List[float] = []
    deltas_correct: List[float] = []
    deltas_wrong: List[float] = []
    bt_correct: List[float] = []
    bt_wrong: List[float] = []

    for start in range(0, len(idxs), args.batch_pairs):
        batch = idxs[start:start + args.batch_pairs]

        prompts: List[str] = []
        labels: List[int] = []

        for idx in batch:
            ex = val[idx]
            post, s0, s1, label = adapter.extract_pair(ex)
            prompts.append(prompt_fn(post, s0))
            prompts.append(prompt_fn(post, s1))
            labels.append(label)

        # Sample raw completions n_samples times
        all_out_texts: List[List[str]] = []
        for _ in range(args.n_samples):
            outs = generate_batch(
                tok, model, prompts,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            all_out_texts.append(outs)

        # Parse scores per item and average
        avg_scores: List[Optional[float]] = []
        raw_scores: List[Optional[List[float]]] = []
        for j in range(len(prompts)):
            vals = [parse_fn(all_out_texts[k][j]) for k in range(len(all_out_texts))]
            if any(v is None for v in vals):
                avg_scores.append(None)
                raw_scores.append(None)
            else:
                vals_f = [float(v) for v in vals]  # type: ignore
                avg_scores.append(sum(vals_f) / len(vals_f))
                raw_scores.append(vals_f)

        # Consume pairs
        for b_idx, label in enumerate(labels):
            z0 = avg_scores[2*b_idx]
            z1 = avg_scores[2*b_idx + 1]
            s0_samples = raw_scores[2*b_idx]
            s1_samples = raw_scores[2*b_idx + 1]

            if z0 is None or z1 is None or s0_samples is None or s1_samples is None:
                skipped += 1
                continue

            total += 1

            if abs(z0 - z1) < 1e-9:
                ties += 1
                correct += 0.5
                deltas.append(0.0)
                bt_losses_all.append(bt_logloss(0.0, temp=args.bt_temp))
                continue

            pred = 0 if z0 > z1 else 1
            is_correct = (pred == label)
            correct += 1.0 if is_correct else 0.0

            non_tie_total += 1
            non_tie_correct += 1 if is_correct else 0

            d = (z0 - z1) if label == 0 else (z1 - z0)  # pref - non
            d = float(d)
            deltas.append(d)
            deltas_nontie.append(d)

            loss = bt_logloss(d, temp=args.bt_temp)
            bt_losses_all.append(loss)
            bt_losses_nontie.append(loss)

            if args.report_margins:
                if is_correct:
                    deltas_correct.append(d); bt_correct.append(loss)
                else:
                    deltas_wrong.append(d); bt_wrong.append(loss)

            if args.report_uncertainty:
                if is_correct:
                    correct_score_values.extend(s0_samples)
                    correct_score_values.extend(s1_samples)
                else:
                    wrong_pairs += 1
                    wrong_score_values.extend(s0_samples)
                    wrong_score_values.extend(s1_samples)

                    u0 = uncertain_extreme(s0_samples, low=args.unc_low, high=args.unc_high)
                    u1 = uncertain_extreme(s1_samples, low=args.unc_low, high=args.unc_high)
                    if u0 or u1:
                        wrong_uncertain_any += 1
                    if u0 and u1:
                        wrong_uncertain_both += 1

            if total % args.progress_every == 0:
                nt_acc = non_tie_correct / max(non_tie_total, 1)
                print(
                    f"progress: {total}/{args.n_pairs} (skipped {skipped}) "
                    f"acc={correct/total:.4f} ties={ties/total:.4f} non_tie_acc={nt_acc:.4f}",
                    flush=True
                )

    acc = correct / max(total, 1)
    tie_rate = ties / max(total, 1)
    non_tie_acc = non_tie_correct / max(non_tie_total, 1)

    print(f"\nEvaluated pairs: {total} (skipped {skipped})")
    print(f"Accuracy (ties=0.5): {acc:.4f}")
    print(f"Tie rate: {tie_rate:.4f}")
    print(f"Non-tie accuracy: {non_tie_acc:.4f} (non-tie total={non_tie_total})")

    if deltas:
        arr = np.array(deltas, dtype=float)
        print(f"Delta(pref-non): mean={arr.mean():.4f} std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f}")

    if bt_losses_all:
        arr = np.array(bt_losses_all, dtype=float)
        print(f"BT log-loss (all pairs): mean={arr.mean():.4f} std={arr.std():.4f} (lower is better)")

    if bt_losses_nontie:
        arr = np.array(bt_losses_nontie, dtype=float)
        print(f"BT log-loss (non-tie):   mean={arr.mean():.4f} std={arr.std():.4f} (lower is better)")

    if args.report_uncertainty:
        print("\n--- Uncertainty diagnostic (wrong non-tie pairs) ---")
        if wrong_pairs == 0:
            print("Wrong non-tie pairs: 0")
        else:
            print(f"Wrong non-tie pairs: {wrong_pairs}")
            print(f"Uncertain_any among wrong: {wrong_uncertain_any / wrong_pairs:.4f}")
            print(f"Uncertain_both among wrong: {wrong_uncertain_both / wrong_pairs:.4f}")

        bin_size = 2.0 if "score100" in parser_name else 0.1

        if wrong_score_values:
            wl, wm, wh = bucket_counts(wrong_score_values, low=args.unc_low, high=args.unc_high)
            print(f"\nWrong bucket counts (low<={args.unc_low}/mid/high>={args.unc_high}): {wl}/{wm}/{wh} n={len(wrong_score_values)}")
            h = histogram(wrong_score_values, bin_size=bin_size)
            print(f"Wrong histogram (bin_size={bin_size}):")
            def _hist_sort_key(x: str):
                return float(x) if bin_size is not None else int(x)

            for k in sorted(h.keys(), key=_hist_sort_key):
                print(f"  {k}: {h[k]}")

        if correct_score_values:
            cl, cm, ch = bucket_counts(correct_score_values, low=args.unc_low, high=args.unc_high)
            print(f"\nCorrect bucket counts (low<={args.unc_low}/mid/high>={args.unc_high}): {cl}/{cm}/{ch} n={len(correct_score_values)}")
            h = histogram(correct_score_values, bin_size=bin_size)
            print(f"Correct histogram (bin_size={bin_size}):")
            def _hist_sort_key(x: str):
                return float(x) if bin_size is not None else int(x)

            for k in sorted(h.keys(), key=_hist_sort_key):
                print(f"  {k}: {h[k]}")

    if args.report_margins:
        print("\n--- Margin diagnostics (non-tie only) ---")
        summarize_array("delta_pref (all non-tie)", deltas_nontie)
        summarize_array("delta_pref (correct)", deltas_correct)
        summarize_array("delta_pref (wrong)", deltas_wrong)

        print("\n--- BT loss diagnostics (uses bt_temp) ---")
        summarize_array("bt_loss (all non-tie)", bt_losses_nontie)
        summarize_array("bt_loss (correct)", bt_correct)
        summarize_array("bt_loss (wrong)", bt_wrong)

if __name__ == "__main__":
    main()
