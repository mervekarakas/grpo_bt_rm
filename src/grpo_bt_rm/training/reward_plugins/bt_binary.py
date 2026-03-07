import os
from statistics import mean
from typing import Dict

from swift.rewards import ORM, orms

from grpo_bt_rm.parsing.registry import get_parser


def _get_float(name: str, default: float) -> float:
    v = os.environ.get(name, None)
    return default if v is None else float(v)


def _get_params():
    """
    Env knobs (read at call-time):
      BT_REWARD_POS       (default 1.0)   -> reward for correct direction
      BT_REWARD_NEG       (default -1.0)  -> reward for wrong direction
      BT_SCORE_PARSER     (default score5_last)
    """
    reward_pos = _get_float("BT_REWARD_POS", 1.0)
    reward_neg = _get_float("BT_REWARD_NEG", -1.0)

    parser_name = os.environ.get("BT_SCORE_PARSER", "score5_last")
    parse_score = get_parser(parser_name)

    return reward_pos, reward_neg, parse_score


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


class BTPointwiseBinaryReward(ORM):
    """
    Binary correctness reward for pointwise BT scoring.

    For each pair:
      z0_bar = mean(z0)
      z1_bar = mean(z1)

    If pref == 0 (side0 is preferred):
      side0 i: r = +1 if z0_i > z1_bar else -1   (should score higher than nonpref mean)
      side1 i: r = +1 if z1_i < z0_bar else -1   (should score lower than pref mean)
    Else pref == 1:
      side1 i: r = +1 if z1_i > z0_bar else -1
      side0 i: r = +1 if z0_i < z1_bar else -1

    Ties (z_i == z_bar_other) get the negative reward.

    This converts the continuous BT reward into a binary correctness signal,
    similar to RM-R1's +1/-1 reward. Within a GRPO group, only groups with
    a mix of correct/incorrect samples produce gradient (natural hard-case focusing).
    """

    _call_count = 0
    _log_every = int(os.environ.get("BT_LOG_EVERY", "100"))

    def __call__(self, completions, pair_id, side, preferred_side, **kwargs):
        reward_pos, reward_neg, parse_score = _get_params()

        n = len(completions)
        rewards = [0.0] * n
        n_correct = 0
        n_total = 0
        by_pair = _group_by_pair(pair_id, side, preferred_side)

        for _, d in by_pair.items():
            pref = int(d["pref"])
            idx0 = d[0]
            idx1 = d[1]

            if not idx0 or not idx1:
                for i in idx0 + idx1:
                    rewards[i] = reward_neg
                continue

            z0_map = {i: parse_score(completions[i]) for i in idx0}
            z1_map = {i: parse_score(completions[i]) for i in idx1}

            z0_vals = [z for z in z0_map.values() if z is not None]
            z1_vals = [z for z in z1_map.values() if z is not None]
            if not z0_vals or not z1_vals:
                for i in idx0 + idx1:
                    rewards[i] = reward_neg
                continue

            z0_bar = mean(z0_vals)
            z1_bar = mean(z1_vals)

            if pref == 0:
                # side0 is preferred: should score higher than nonpref mean
                for i in idx0:
                    z = z0_map[i]
                    if z is None:
                        rewards[i] = reward_neg
                    else:
                        correct = z > z1_bar
                        rewards[i] = reward_pos if correct else reward_neg
                        n_total += 1
                        n_correct += int(correct)
                # side1 is non-preferred: should score lower than pref mean
                for i in idx1:
                    z = z1_map[i]
                    if z is None:
                        rewards[i] = reward_neg
                    else:
                        correct = z < z0_bar
                        rewards[i] = reward_pos if correct else reward_neg
                        n_total += 1
                        n_correct += int(correct)
            else:
                # side1 is preferred
                for i in idx1:
                    z = z1_map[i]
                    if z is None:
                        rewards[i] = reward_neg
                    else:
                        correct = z > z0_bar
                        rewards[i] = reward_pos if correct else reward_neg
                        n_total += 1
                        n_correct += int(correct)
                # side0 is non-preferred
                for i in idx0:
                    z = z0_map[i]
                    if z is None:
                        rewards[i] = reward_neg
                    else:
                        correct = z < z1_bar
                        rewards[i] = reward_pos if correct else reward_neg
                        n_total += 1
                        n_correct += int(correct)

        # Periodic logging
        BTPointwiseBinaryReward._call_count += 1
        if n_total and BTPointwiseBinaryReward._call_count % BTPointwiseBinaryReward._log_every == 0:
            acc = n_correct / n_total
            print(f"[BT_BINARY] n={n_total} correctness_rate={acc:.1%}")

        return rewards


orms["bt_pointwise_binary"] = BTPointwiseBinaryReward
