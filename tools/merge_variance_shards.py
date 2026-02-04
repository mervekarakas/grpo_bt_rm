"""Merge variance shard JSONLs and print aggregated summary.

Usage:
    python tools/merge_variance_shards.py shard0.jsonl shard1.jsonl ... --out merged.jsonl
"""
import argparse
import json
import os
import sys
from collections import Counter
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helpers (mirror the logic in grpo_bt_rm.metrics)
# ---------------------------------------------------------------------------

def _mean_or_nan(xs: List[Optional[float]]) -> float:
    vals = [v for v in xs if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _uncertain_extreme(xs: List[float], low: float, high: float) -> bool:
    if not xs:
        return False
    return (min(xs) <= low) and (max(xs) >= high)


def _range_ge(xs: List[Optional[float]], thr: float) -> bool:
    vals = [v for v in xs if v is not None]
    if not vals:
        return False
    r = max(vals) - min(vals)
    return r >= thr


def _histogram(values: List[float], bin_size: Optional[float] = None) -> Counter:
    c: Counter = Counter()
    if not values:
        return c
    if bin_size is None:
        for v in values:
            c[str(int(round(float(v))))] += 1
        return c
    for v in values:
        vv = round(float(v) / bin_size) * bin_size
        c[f"{vv:.1f}"] += 1
    return c


def _parse_thresholds(s: str) -> List[float]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    return sorted(set(float(p) for p in parts))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Merge variance shard JSONL files")
    ap.add_argument("shards", nargs="+", help="Shard JSONL files")
    ap.add_argument("--out", required=True, help="Output merged JSONL path")
    ap.add_argument(
        "--range_thresholds",
        type=str,
        default="20,30,40",
        help="Comma-separated range thresholds (should match variance.py run)",
    )
    ap.add_argument("--unc_low", type=float, default=None,
                    help="Low threshold for extreme uncertainty")
    ap.add_argument("--unc_high", type=float, default=None,
                    help="High threshold for extreme uncertainty")
    args = ap.parse_args()

    range_thresholds = _parse_thresholds(args.range_thresholds)

    # ------------------------------------------------------------------
    # Read all shard records
    # ------------------------------------------------------------------
    records = []
    for path in args.shards:
        if not os.path.isfile(path):
            print(f"WARNING: shard file not found: {path}", file=sys.stderr)
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    if not records:
        print("ERROR: no records found in shard files", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(records)} records from {len(args.shards)} shard(s)")

    # Detect parser from first record
    parser_name = records[0].get("parser", "")

    # Set uncertainty defaults based on parser
    unc_low = args.unc_low
    unc_high = args.unc_high
    if unc_low is None:
        unc_low = 20.0 if "score100" in parser_name else 2.0
    if unc_high is None:
        unc_high = 80.0 if "score100" in parser_name else 4.0

    # ------------------------------------------------------------------
    # Write merged JSONL
    # ------------------------------------------------------------------
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote merged JSONL: {args.out} ({len(records)} records)")

    # ------------------------------------------------------------------
    # Aggregate metrics (same logic as variance.py final summary)
    # ------------------------------------------------------------------
    valid_rates = []
    within_std = []
    within_unique = []
    tie_rates = []
    exp_correct_ties05 = []
    exp_correct_strict = []

    wrong_pairs = 0
    wrong_uncertain_any = 0
    wrong_uncertain_both = 0

    wrong_range_any = {thr: 0 for thr in range_thresholds}
    wrong_range_both = {thr: 0 for thr in range_thresholds}

    wrong_score_values: List[float] = []
    correct_score_values: List[float] = []

    for rec in records:
        sum0 = rec["summary0_stats"]
        sum1 = rec["summary1_stats"]
        dstat = rec["delta_stats_pref_minus_non"]
        label = rec["label"]
        scores0 = rec["scores0"]
        scores1 = rec["scores1"]

        p_corr_t05 = rec["expected_correctness_ties05"]
        p_corr_strict = rec["expected_correctness_strict"]

        valid_rates.append((sum0["valid_rate"] + sum1["valid_rate"]) / 2.0)

        stds = [sum0["std"], sum1["std"]]
        stds = [x for x in stds if not (x is None or np.isnan(x))]
        within_std.append(float(np.mean(stds)) if stds else np.nan)

        uniqs = [sum0["unique"], sum1["unique"]]
        uniqs = [x for x in uniqs if not (x is None or np.isnan(x))]
        within_unique.append(float(np.mean(uniqs)) if uniqs else np.nan)

        p_eq = dstat["p_eq"]
        tie_rates.append(p_eq if not (p_eq is None or np.isnan(p_eq)) else np.nan)
        exp_correct_ties05.append(p_corr_t05 if p_corr_t05 is not None else np.nan)
        exp_correct_strict.append(p_corr_strict if p_corr_strict is not None else np.nan)

        # Wrong-pair diagnostics (mean-score prediction)
        m0 = _mean_or_nan(scores0)
        m1 = _mean_or_nan(scores1)
        if not np.isnan(m0) and not np.isnan(m1):
            pred_mean = 0 if m0 > m1 else 1
            is_wrong_mean = (pred_mean != label)

            v0 = [v for v in scores0 if v is not None]
            v1 = [v for v in scores1 if v is not None]

            if is_wrong_mean:
                wrong_pairs += 1
                wrong_score_values.extend(v0)
                wrong_score_values.extend(v1)

                u0 = _uncertain_extreme(v0, low=unc_low, high=unc_high)
                u1 = _uncertain_extreme(v1, low=unc_low, high=unc_high)

                if u0 or u1:
                    wrong_uncertain_any += 1
                if u0 and u1:
                    wrong_uncertain_both += 1

                for thr in range_thresholds:
                    r0 = _range_ge(scores0, thr)
                    r1 = _range_ge(scores1, thr)
                    if r0 or r1:
                        wrong_range_any[thr] += 1
                    if r0 and r1:
                        wrong_range_both[thr] += 1
            else:
                correct_score_values.extend(v0)
                correct_score_values.extend(v1)

    # ------------------------------------------------------------------
    # Print summary (same format as variance.py)
    # ------------------------------------------------------------------
    print(f"\n=== Final summary across pairs (merged, n={len(records)}) ===")
    print(f"avg valid score rate:         {np.nanmean(valid_rates):.3f}")
    print(f"avg within-summary std:       {np.nanmean(within_std):.3f}")
    print(f"avg unique scores/summary:    {np.nanmean(within_unique):.3f}")
    print(f"avg tie rate (delta==0):      {np.nanmean(tie_rates):.3f}")
    print(f"avg accuracy (ties=0.5):      {np.nanmean(exp_correct_ties05):.3f}")
    print(f"avg accuracy (strict ties=0): {np.nanmean(exp_correct_strict):.3f}")

    print(f"\n=== Wrong-pair uncertainty (mean-score prediction) ===")
    if wrong_pairs == 0:
        print("wrong_pairs: 0")
    else:
        print(f"wrong_pairs: {wrong_pairs}")
        print(f"uncertain_any among wrong:  {wrong_uncertain_any / wrong_pairs:.3f}")
        print(f"uncertain_both among wrong: {wrong_uncertain_both / wrong_pairs:.3f}")

        bin_size = None if "score100" in parser_name else 0.1

        wh = _histogram(wrong_score_values, bin_size=bin_size)
        ch = _histogram(correct_score_values, bin_size=bin_size)

        print("\nWrong sampled-score histogram:")
        def _hist_sort_key(x: str):
            return float(x) if bin_size is not None else int(x)

        for k in sorted(wh.keys(), key=_hist_sort_key):
            print(f"  {k}: {wh[k]}")

        print("\nCorrect sampled-score histogram:")
        for k in sorted(ch.keys(), key=_hist_sort_key):
            print(f"  {k}: {ch[k]}")

        print(f"\n=== Range-based uncertainty on WRONG pairs (mean-score prediction) ===")
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
