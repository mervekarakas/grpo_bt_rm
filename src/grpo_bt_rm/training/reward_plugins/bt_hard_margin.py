"""
Score-Only Hard-BT reward: marginized BT loss + hard-pair weighting.

Core idea: explicit-score RL with credit assignment fix.
Reward penalizes insufficient margins but stops pushing once margin exceeds target.
Hard-pair weighting focuses learning on pairs the model currently gets wrong.

For each pair (pref=0 means side0 preferred):
  z0_bar = mean(z0), z1_bar = mean(z1)

  Pair-level hard weight:
    delta_bar = (z_pref_bar - z_non_bar) / T
    w_hard = sigmoid(-delta_bar / tau_h)

  Per-sample reward (e.g., preferred side sample i):
    delta_i = (z_i - z_non_bar) / T
    r_i = -w_hard * softplus((gamma - delta_i) / T_m)

  When delta_i > gamma: softplus -> small -> reward near 0 (stop pushing)
  When delta_i < gamma: softplus -> positive -> negative reward (penalty)
  When delta_i < 0 (wrong): softplus -> large -> large negative reward

Env knobs:
  BT_SCORE_PARSER     (default score100_last)
  BT_SCORE_TEMP       (default 20.0)  -> normalizes raw score delta
  BT_MARGIN_GAMMA     (default 0.5)   -> target normalized margin
  BT_MARGIN_TEMP      (default 1.0)   -> softplus temperature
  BT_HARD_TAU         (default 1.0)   -> hard-pair sigmoid temperature
  BT_REWARD_PARSE_FAIL (default -1.0) -> reward for parse failure
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


def _softplus(x: float) -> float:
    if x > 20:
        return x
    if x < -20:
        return 0.0
    return math.log1p(math.exp(x))


def _get_params():
    parser_name = os.environ.get("BT_SCORE_PARSER", "score100_last")
    parse_score = get_parser(parser_name)
    score_temp = _get_float("BT_SCORE_TEMP", 20.0)
    margin_gamma = _get_float("BT_MARGIN_GAMMA", 0.5)
    margin_temp = _get_float("BT_MARGIN_TEMP", 1.0)
    hard_tau = _get_float("BT_HARD_TAU", 1.0)
    reward_parse_fail = _get_float("BT_REWARD_PARSE_FAIL", -1.0)
    return parse_score, score_temp, margin_gamma, margin_temp, hard_tau, reward_parse_fail


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


class BTHardMarginReward(ORM):
    _call_count = 0
    _log_every = int(os.environ.get("BT_LOG_EVERY", "100"))

    def __call__(self, completions, pair_id, side, preferred_side, **kwargs):
        parse_score, score_temp, margin_gamma, margin_temp, hard_tau, reward_parse_fail = _get_params()

        n = len(completions)
        rewards = [0.0] * n
        deltas = []
        w_vals = []
        by_pair = _group_by_pair(pair_id, side, preferred_side)

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

            # Pair-level: hard-pair weight
            if pref == 0:
                delta_bar = (z0_bar - z1_bar) / score_temp
            else:
                delta_bar = (z1_bar - z0_bar) / score_temp
            w_hard = _sigmoid(-delta_bar / hard_tau)
            w_vals.append(w_hard)

            def R(delta_raw: float) -> float:
                delta_norm = delta_raw / score_temp
                deltas.append(delta_norm)
                return -w_hard * _softplus((margin_gamma - delta_norm) / margin_temp)

            if pref == 0:
                for i in idx0:
                    z = z0_map[i]
                    rewards[i] = reward_parse_fail if z is None else R(z - z1_bar)
                for i in idx1:
                    z = z1_map[i]
                    rewards[i] = reward_parse_fail if z is None else R(z0_bar - z)
            else:
                for i in idx1:
                    z = z1_map[i]
                    rewards[i] = reward_parse_fail if z is None else R(z - z0_bar)
                for i in idx0:
                    z = z0_map[i]
                    rewards[i] = reward_parse_fail if z is None else R(z1_bar - z)

        BTHardMarginReward._call_count += 1
        if deltas and w_vals and BTHardMarginReward._call_count % BTHardMarginReward._log_every == 0:
            import statistics
            d_mean = statistics.mean(deltas)
            d_correct = sum(1 for d in deltas if d > 0) / len(deltas)
            w_mean = statistics.mean(w_vals)
            r_vals = [r for r in rewards if r != reward_parse_fail]
            r_mean = statistics.mean(r_vals) if r_vals else 0.0
            print(f"[BT_HARD_MARGIN] n={len(deltas)} correct={d_correct:.1%} "
                  f"delta_mean={d_mean:.3f} w_hard_mean={w_mean:.3f} r_mean={r_mean:.3f}")

        return rewards


orms["bt_hard_margin"] = BTHardMarginReward
