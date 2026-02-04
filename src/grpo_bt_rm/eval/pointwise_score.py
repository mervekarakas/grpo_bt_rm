from __future__ import annotations

import argparse
import random
from typing import List, Optional

import numpy as np

from grpo_bt_rm.data.registry import get_dataset
from grpo_bt_rm.prompts.registry import get_prompt
from grpo_bt_rm.parsing.registry import get_parser
from grpo_bt_rm.utils.model import load_qwen_instruct
from grpo_bt_rm.utils.generation import generate_batch
from grpo_bt_rm.metrics.reporting import (
    uncertain_extreme,
    bucket_counts,
    histogram,
    summarize_array,
)


def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", default="summarize_from_feedback", help="Dataset name from registry")
    ap.add_argument("--split", type=str, default="")
    ap.add_argument("--n_pairs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")

    ap.add_argument("--prompt", type=str, default="score5_v1")
    ap.add_argument("--parser", type=str, default="",
                    help="Parser name; if empty uses prompt default parser")

    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--n_samples", type=int, default=1,
                    help="Number of samples per summary; scores are averaged.")
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)

    ap.add_argument("--batch_pairs", type=int, default=8,
                    help="Pairs per generation batch (2 prompts per pair).")
    ap.add_argument("--progress_every", type=int, default=50)

    # Optional: uncertainty reporting on WRONG non-tie pairs
    ap.add_argument("--report_uncertainty", action="store_true")
    ap.add_argument("--unc_low", type=float, default=None,
                    help="Low threshold for uncertainty; if omitted, defaults based on parser.")
    ap.add_argument("--unc_high", type=float, default=None,
                    help="High threshold for uncertainty; if omitted, defaults based on parser.")

    args = ap.parse_args(argv)

    rng = random.Random(args.seed)

    # prompt + parser
    spec = get_prompt(args.prompt)
    prompt_fn = spec.fn
    parser_name = args.parser or spec.default_parser
    parse_fn = get_parser(parser_name)

    # default uncertainty thresholds based on score scale
    if args.unc_low is None or args.unc_high is None:
        if "score100" in parser_name:
            args.unc_low = 20.0 if args.unc_low is None else args.unc_low
            args.unc_high = 80.0 if args.unc_high is None else args.unc_high
        else:
            args.unc_low = 2.0 if args.unc_low is None else args.unc_low
            args.unc_high = 4.0 if args.unc_high is None else args.unc_high

    # data
    adapter = get_dataset(args.dataset)
    split = args.split or adapter.default_eval_split
    data = adapter.load_split(split)
    idxs = list(range(len(data)))
    rng.shuffle(idxs)
    idxs = idxs[:args.n_pairs]

    # model
    tok, model, device = load_qwen_instruct(args.model_name)

    print("\n=== Pointwise baseline scoring ===")
    print(f"model={args.model_name}")
    print(f"prompt={args.prompt} parser={parser_name}")
    print(f"dataset={args.dataset} split={split} n_pairs={args.n_pairs} seed={args.seed}")
    print(f"do_sample={args.do_sample} n_samples={args.n_samples} temp={args.temperature} top_p={args.top_p} top_k={args.top_k}")
    print(f"max_new_tokens={args.max_new_tokens} batch_pairs={args.batch_pairs}")
    print(f"unc_low={args.unc_low} unc_high={args.unc_high} report_uncertainty={args.report_uncertainty}\n")

    # metrics
    total = 0
    skipped = 0
    ties = 0
    correct_ties05 = 0.0
    non_tie_total = 0
    non_tie_correct = 0

    all_scores: List[float] = []
    scores_pref: List[float] = []
    scores_non: List[float] = []
    deltas: List[float] = []

    # uncertainty diagnostics (wrong non-tie only)
    wrong_pairs = 0
    wrong_uncertain_any = 0
    wrong_uncertain_both = 0
    wrong_score_values: List[float] = []
    correct_score_values: List[float] = []

    # loop in batches
    for start in range(0, len(idxs), args.batch_pairs):
        batch = idxs[start:start + args.batch_pairs]

        prompts: List[str] = []
        labels: List[int] = []

        for idx in batch:
            post, s0, s1, label = adapter.extract_pair(data[idx])
            prompts.append(prompt_fn(post, s0))
            prompts.append(prompt_fn(post, s1))
            labels.append(label)

        # generate n_samples times
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

        # parse per item, average
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

        # consume pairs
        for b_idx, label in enumerate(labels):
            z0 = avg_scores[2*b_idx]
            z1 = avg_scores[2*b_idx + 1]
            s0_samples = raw_scores[2*b_idx]
            s1_samples = raw_scores[2*b_idx + 1]

            if z0 is None or z1 is None or s0_samples is None or s1_samples is None:
                skipped += 1
                continue

            total += 1

            all_scores.extend([float(z0), float(z1)])

            # preferred/non for stats
            if label == 0:
                z_pref, z_non = float(z0), float(z1)
            else:
                z_pref, z_non = float(z1), float(z0)
            scores_pref.append(z_pref)
            scores_non.append(z_non)
            deltas.append(z_pref - z_non)

            # decision + metrics
            if abs(z0 - z1) < 1e-9:
                ties += 1
                correct_ties05 += 0.5
                continue

            pred = 0 if z0 > z1 else 1
            is_correct = (pred == label)

            correct_ties05 += 1.0 if is_correct else 0.0
            non_tie_total += 1
            non_tie_correct += 1 if is_correct else 0

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

        if total > 0 and (total % args.progress_every == 0):
            acc = correct_ties05 / total
            tie_rate = ties / total
            nt_acc = non_tie_correct / max(non_tie_total, 1)
            print(f"progress: {total}/{args.n_pairs} skipped={skipped} acc(ties=0.5)={acc:.4f} tie={tie_rate:.4f} non_tie_acc={nt_acc:.4f}", flush=True)

    if total == 0:
        print("No valid pairs scored (everything skipped). Check prompt/parser compliance.")
        return

    acc_ties05 = correct_ties05 / total
    tie_rate = ties / total
    acc_strict = non_tie_correct / max(non_tie_total, 1)

    print("\n=== Results ===")
    print(f"Evaluated pairs: {total} (skipped {skipped})")
    print(f"Accuracy (ties=0.5): {acc_ties05:.4f}")
    print(f"Tie rate:            {tie_rate:.4f}")
    print(f"Non-tie accuracy:    {acc_strict:.4f} (non-tie total={non_tie_total})")

    # score distributions
    def _summ(name: str, xs: List[float]):
        if not xs:
            print(f"{name}: <empty>")
            return
        arr = np.array(xs, dtype=float)
        print(
            f"{name}: n={len(arr)} mean={arr.mean():.4f} std={arr.std():.4f} "
            f"min={arr.min():.4f} p25={np.percentile(arr,25):.4f} "
            f"p50={np.percentile(arr,50):.4f} p75={np.percentile(arr,75):.4f} max={arr.max():.4f}"
        )

    _summ("all_scores", all_scores)
    _summ("scores_pref", scores_pref)
    _summ("scores_non", scores_non)
    _summ("delta(pref-non)", deltas)

    if args.report_uncertainty:
        print("\n--- Uncertainty diagnostic (wrong non-tie pairs) ---")
        if wrong_pairs == 0:
            print("Wrong non-tie pairs: 0")
        else:
            print(f"Wrong non-tie pairs: {wrong_pairs}")
            print(f"Uncertain_any among wrong:  {wrong_uncertain_any / wrong_pairs:.4f}")
            print(f"Uncertain_both among wrong: {wrong_uncertain_both / wrong_pairs:.4f}")

        # histogram binning
        bin_size = None if "score100" in parser_name else 0.1

        if wrong_score_values:
            wl, wm, whi = bucket_counts(wrong_score_values, low=args.unc_low, high=args.unc_high)
            print(f"\nWrong bucket counts (low<={args.unc_low}/mid/high>={args.unc_high}): {wl}/{wm}/{whi} n={len(wrong_score_values)}")
            h = histogram(wrong_score_values, bin_size=bin_size)
            print("Wrong histogram:")
            keyfn = (lambda x: int(x)) if bin_size is None else (lambda x: float(x))
            for k in sorted(h.keys(), key=keyfn):
                print(f"  {k}: {h[k]}")

        if correct_score_values:
            cl, cm, chi = bucket_counts(correct_score_values, low=args.unc_low, high=args.unc_high)
            print(f"\nCorrect bucket counts (low<={args.unc_low}/mid/high>={args.unc_high}): {cl}/{cm}/{chi} n={len(correct_score_values)}")
            h = histogram(correct_score_values, bin_size=bin_size)
            print("Correct histogram:")
            keyfn = (lambda x: int(x)) if bin_size is None else (lambda x: float(x))
            for k in sorted(h.keys(), key=keyfn):
                print(f"  {k}: {h[k]}")

    # Optional: extra percentile summaries like eval_bt --report_margins
    # (kept here if you want more detail)
    # summarize_array("delta(pref-non)", deltas)

if __name__ == "__main__":
    main()
