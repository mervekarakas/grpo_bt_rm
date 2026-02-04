import os
from statistics import mean
from typing import Dict

from swift.rewards import ORM, orms
from grpo_bt_rm.parsing.registry import get_parser
from grpo_bt_rm.utils.math import log_sigmoid, scaled_delta


def _get_float(name: str, default: float) -> float:
    v = os.environ.get(name, None)
    return default if v is None else float(v)


def _get_params():
    """
    Env knobs (read at call-time):
      BT_REWARD_SCALE      (default 1.0)
      BT_DELTA_TEMP        (default 1.0)
      BT_DELTA_CLIP        (default 0.0)   -> positive clip (for per-completion reward)
      BT_DELTA_NEG_CLIP    (default 0.0)   -> negative clip (0 => no neg clipping)
      BT_SCORE_PARSER      (default score5_last)

    Hard-example mining knobs (new):
      BT_HARD_LAMBDA       (default 0.5)   -> weight for low-margin correct pairs
      BT_HARD_TAU          (default 0.25)  -> margin threshold in *scaled units* (delta/temp)
    """
    reward_scale = _get_float("BT_REWARD_SCALE", 1.0)
    delta_temp = _get_float("BT_DELTA_TEMP", 1.0)
    delta_clip = _get_float("BT_DELTA_CLIP", 0.0)
    delta_neg_clip = _get_float("BT_DELTA_NEG_CLIP", 0.0)

    hard_lambda = _get_float("BT_HARD_LAMBDA", 0.5)
    hard_tau = _get_float("BT_HARD_TAU", 0.25)
    hard_eps    = _get_float("BT_HARD_EPS", 0.2)

    parser_name = os.environ.get("BT_SCORE_PARSER", "score5_last")
    parse_score = get_parser(parser_name)

    return (reward_scale, delta_temp, delta_clip, delta_neg_clip,
            hard_lambda, hard_tau, hard_eps, parse_score)


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


class BTPointwiseBaselineHardMine(ORM):
    """
    Baseline-directional BT reward *times* a pair-level hard-mining weight.

    Pair mean margin:
      delta_mean_pref = (z0_bar - z1_bar) if pref==0 else (z1_bar - z0_bar)
      delta_scaled = delta_mean_pref / BT_DELTA_TEMP

    Weight:
      w=1           if delta_scaled < 0
      w=lambda      if 0 <= delta_scaled < tau
      w=0           otherwise

    Then per-completion rewards (same as your baseline directional BT reward):
      pref==0:
        side0 i: log σ( scaled_delta(z0_i - z1_bar) ) * scale * w
        side1 i: log σ( scaled_delta(z0_bar - z1_i) ) * scale * w
      pref==1:
        symmetric
    """

    def __call__(self, completions, pair_id, side, preferred_side, **kwargs):
        (reward_scale, delta_temp, delta_clip, delta_neg_clip,
         hard_lambda, hard_tau, hard_eps, parse_score) = _get_params()

        n = len(completions)
        rewards = [0.0] * n
        by_pair = _group_by_pair(pair_id, side, preferred_side)

        def R(delta: float) -> float:
            # this is the per-completion BT loglik shaping (still gives nonzero advantages)
            x = scaled_delta(delta, temp=delta_temp, clip=delta_clip, neg_clip=delta_neg_clip)
            return float(log_sigmoid(x)) * reward_scale

        def hard_weight(delta_mean_pref: float) -> float:
            # compute in "scaled" units so tau is comparable across score scales
            delta_scaled = delta_mean_pref / delta_temp
            if delta_scaled < 0:
                return 1.0
            if delta_scaled < hard_tau:
                return float(hard_lambda)
            return float(hard_eps)

        for _, d in by_pair.items():
            pref = int(d["pref"])
            idx0 = d[0]
            idx1 = d[1]

            if not idx0 or not idx1:
                for i in idx0 + idx1:
                    rewards[i] = -1.0
                continue

            z0_map = {i: parse_score(completions[i]) for i in idx0}
            z1_map = {i: parse_score(completions[i]) for i in idx1}

            z0_vals = [z for z in z0_map.values() if z is not None]
            z1_vals = [z for z in z1_map.values() if z is not None]
            if not z0_vals or not z1_vals:
                for i in idx0 + idx1:
                    rewards[i] = -1.0
                continue

            z0_bar = mean(z0_vals)
            z1_bar = mean(z1_vals)

            # pair-level hardness based on mean margin in the preferred direction
            delta_mean_pref = (z0_bar - z1_bar) if pref == 0 else (z1_bar - z0_bar)
            w = hard_weight(delta_mean_pref)

            # if easy pair, skip updates cleanly (reward=0 -> advantages ~0 for this prompt)
            if w == 0.0:
                for i in idx0 + idx1:
                    rewards[i] = 0.0
                continue

            if pref == 0:
                for i in idx0:
                    z = z0_map[i]
                    rewards[i] = -1.0 if z is None else w * R(z - z1_bar)
                for i in idx1:
                    z = z1_map[i]
                    rewards[i] = -1.0 if z is None else w * R(z0_bar - z)
            else:
                for i in idx1:
                    z = z1_map[i]
                    rewards[i] = -1.0 if z is None else w * R(z - z0_bar)
                for i in idx0:
                    z = z0_map[i]
                    rewards[i] = -1.0 if z is None else w * R(z1_bar - z)

        return rewards


orms["bt_pointwise_baseline_hardmine"] = BTPointwiseBaselineHardMine
