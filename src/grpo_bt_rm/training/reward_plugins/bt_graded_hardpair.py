"""
Graded reward + DWRL-style hard-pair weighting.

Experiment 2: Combines PaTaRM-style bounded graded reward with
DWRL-style misalignment weighting.

Same as bt_graded.py but multiplies reward by a hard-pair weight:
  w_hard = sigmoid(-(delta_bar_pref) / tau_h)

where delta_bar_pref = z_pref_bar - z_non_bar (mean preferred - mean non-preferred).

When the pair is already correct (large positive delta_bar), w_hard is small.
When the pair is wrong or borderline, w_hard is large.

This focuses learning on pairs the model currently gets wrong.

Additional env knobs (on top of bt_graded.py knobs):
  BT_HARD_TAU         (default 1.0)   -> temperature for hard-pair sigmoid
"""
import math
import os
from statistics import mean
from typing import Dict

from swift.rewards import ORM, orms

from grpo_bt_rm.parsing.registry import get_parser


def _get_float(name: str, default: float) -> float:
    v = os.environ.get(name, None)
    return default if v is None else float(v)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _get_params():
    parser_name = os.environ.get("BT_SCORE_PARSER", "score100_last")
    parse_score = get_parser(parser_name)
    score_temp = _get_float("BT_SCORE_TEMP", 20.0)
    grade_thresh = _get_float("BT_GRADE_THRESH", 0.1)
    reward_low = _get_float("BT_REWARD_LOW", 1.2)
    reward_high = _get_float("BT_REWARD_HIGH", 1.4)
    reward_wrong = _get_float("BT_REWARD_WRONG", 0.0)
    reward_parse_fail = _get_float("BT_REWARD_PARSE_FAIL", -0.5)
    hard_tau = _get_float("BT_HARD_TAU", 1.0)
    return parse_score, score_temp, grade_thresh, reward_low, reward_high, reward_wrong, reward_parse_fail, hard_tau


def _group_by_pair(pair_id, side, preferred_side) -> Dict[int, Dict]:
    by_pair: Dict[int, Dict] = {}
    for i in range(len(pair_id)):
        pid = int(pair_id[i])
        s = int(side[i])
        pref = int(preferred_side[i])
        if pid not in by_pair:
            by_pair[pid] = {"pref": pref, 0: [], 1: []}
        by_pair[pid][s].append(i)
    return by_pair


class BTPointwiseGradedHardpairReward(ORM):
    _call_count = 0
    _log_every = int(os.environ.get("BT_LOG_EVERY", "100"))

    def __call__(self, completions, pair_id, side, preferred_side, **kwargs):
        parse_score, score_temp, grade_thresh, reward_low, reward_high, reward_wrong, reward_parse_fail, hard_tau = _get_params()

        n = len(completions)
        rewards = [0.0] * n
        n_correct = 0
        n_wrong = 0
        w_vals = []
        by_pair = _group_by_pair(pair_id, side, preferred_side)

        def grade(delta_raw: float) -> float:
            nonlocal n_correct, n_wrong
            delta_norm = delta_raw / score_temp
            if delta_norm > 0:
                n_correct += 1
                return reward_high if delta_norm > grade_thresh else reward_low
            n_wrong += 1
            return reward_wrong

        for _, d in by_pair.items():
            pref = int(d["pref"])
            idx0 = d[0]
            idx1 = d[1]

            if not idx0 or not idx1:
                for i in idx0 + idx1:
                    rewards[i] = reward_parse_fail
                continue

            z0_map = {i: parse_score(completions[i]) for i in idx0}
            z1_map = {i: parse_score(completions[i]) for i in idx1}

            z0_vals = [z for z in z0_map.values() if z is not None]
            z1_vals = [z for z in z1_map.values() if z is not None]
            if not z0_vals or not z1_vals:
                for i in idx0 + idx1:
                    rewards[i] = reward_parse_fail
                continue

            z0_bar = mean(z0_vals)
            z1_bar = mean(z1_vals)

            # Compute hard-pair weight from mean delta
            if pref == 0:
                delta_bar_pref = (z0_bar - z1_bar) / score_temp
            else:
                delta_bar_pref = (z1_bar - z0_bar) / score_temp

            w_hard = _sigmoid(-delta_bar_pref / hard_tau)
            w_vals.append(w_hard)

            if pref == 0:
                for i in idx0:
                    z = z0_map[i]
                    rewards[i] = reward_parse_fail if z is None else grade(z - z1_bar) * w_hard
                for i in idx1:
                    z = z1_map[i]
                    rewards[i] = reward_parse_fail if z is None else grade(z0_bar - z) * w_hard
            else:
                for i in idx1:
                    z = z1_map[i]
                    rewards[i] = reward_parse_fail if z is None else grade(z - z0_bar) * w_hard
                for i in idx0:
                    z = z0_map[i]
                    rewards[i] = reward_parse_fail if z is None else grade(z1_bar - z) * w_hard

        BTPointwiseGradedHardpairReward._call_count += 1
        total = n_correct + n_wrong
        if total and w_vals and BTPointwiseGradedHardpairReward._call_count % BTPointwiseGradedHardpairReward._log_every == 0:
            import statistics
            w_mean = statistics.mean(w_vals)
            print(f"[BT_GRADED_HARD] n={total} correct={n_correct/total:.1%} w_hard_mean={w_mean:.3f}")

        return rewards


orms["bt_pointwise_graded_hardpair"] = BTPointwiseGradedHardpairReward
