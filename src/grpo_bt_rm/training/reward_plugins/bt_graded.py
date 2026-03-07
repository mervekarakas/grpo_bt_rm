"""
PaTaRM-style graded reward for pointwise BT scoring.

Experiment 1: Bounded graded reward (no hard-pair weighting).

For each pair (pref=0 means side0 preferred):
  z0_bar = mean(z0), z1_bar = mean(z1)

  side0_i (preferred):
    delta = z0_i - z1_bar
    reward = f(delta) if delta > 0 else BT_REWARD_WRONG

  side1_j (non-preferred):
    delta = z0_bar - z1_j
    reward = f(delta) if delta > 0 else BT_REWARD_WRONG

  f(delta) = BT_REWARD_LOW  if 0 < delta/T <= BT_GRADE_THRESH
             BT_REWARD_HIGH if delta/T > BT_GRADE_THRESH

Env knobs:
  BT_SCORE_PARSER     (default score100_last)
  BT_SCORE_TEMP       (default 20.0)  -> normalizes raw score delta
  BT_GRADE_THRESH     (default 0.1)   -> normalized delta threshold for grading
  BT_REWARD_LOW       (default 1.2)   -> reward for small correct margin
  BT_REWARD_HIGH      (default 1.4)   -> reward for large correct margin
  BT_REWARD_WRONG     (default 0.0)   -> reward for wrong direction
  BT_REWARD_PARSE_FAIL (default -0.5) -> reward for parse failure
"""
import os
from statistics import mean
from typing import Dict

from swift.rewards import ORM, orms

from grpo_bt_rm.parsing.registry import get_parser


def _get_float(name: str, default: float) -> float:
    v = os.environ.get(name, None)
    return default if v is None else float(v)


def _get_params():
    parser_name = os.environ.get("BT_SCORE_PARSER", "score100_last")
    parse_score = get_parser(parser_name)
    score_temp = _get_float("BT_SCORE_TEMP", 20.0)
    grade_thresh = _get_float("BT_GRADE_THRESH", 0.1)
    reward_low = _get_float("BT_REWARD_LOW", 1.2)
    reward_high = _get_float("BT_REWARD_HIGH", 1.4)
    reward_wrong = _get_float("BT_REWARD_WRONG", 0.0)
    reward_parse_fail = _get_float("BT_REWARD_PARSE_FAIL", -0.5)
    return parse_score, score_temp, grade_thresh, reward_low, reward_high, reward_wrong, reward_parse_fail


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


class BTPointwiseGradedReward(ORM):
    _call_count = 0
    _log_every = int(os.environ.get("BT_LOG_EVERY", "100"))

    def __call__(self, completions, pair_id, side, preferred_side, **kwargs):
        parse_score, score_temp, grade_thresh, reward_low, reward_high, reward_wrong, reward_parse_fail = _get_params()

        n = len(completions)
        rewards = [0.0] * n
        n_correct = 0
        n_wrong = 0
        n_high = 0
        by_pair = _group_by_pair(pair_id, side, preferred_side)

        def grade(delta_raw: float) -> float:
            nonlocal n_correct, n_wrong, n_high
            delta_norm = delta_raw / score_temp
            if delta_norm > 0:
                n_correct += 1
                if delta_norm > grade_thresh:
                    n_high += 1
                    return reward_high
                return reward_low
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

            if pref == 0:
                for i in idx0:
                    z = z0_map[i]
                    rewards[i] = reward_parse_fail if z is None else grade(z - z1_bar)
                for i in idx1:
                    z = z1_map[i]
                    rewards[i] = reward_parse_fail if z is None else grade(z0_bar - z)
            else:
                for i in idx1:
                    z = z1_map[i]
                    rewards[i] = reward_parse_fail if z is None else grade(z - z0_bar)
                for i in idx0:
                    z = z0_map[i]
                    rewards[i] = reward_parse_fail if z is None else grade(z1_bar - z)

        BTPointwiseGradedReward._call_count += 1
        total = n_correct + n_wrong
        if total and BTPointwiseGradedReward._call_count % BTPointwiseGradedReward._log_every == 0:
            print(f"[BT_GRADED] n={total} correct={n_correct/total:.1%} high_margin={n_high/total:.1%}")

        return rewards


orms["bt_pointwise_graded"] = BTPointwiseGradedReward
