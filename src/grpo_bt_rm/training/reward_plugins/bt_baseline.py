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
      BT_REWARD_SCALE     (default 1.0)
      BT_DELTA_TEMP       (default 1.0)
      BT_DELTA_CLIP       (default 0.0)   -> positive clip
      BT_DELTA_NEG_CLIP   (default 0.0)   -> negative clip (0 => no neg clipping)
      BT_SCORE_PARSER     (default score5_last)

    Behavior:
      - default: positive-only clipping (neg clip disabled)
      - symmetric: set BT_DELTA_NEG_CLIP = BT_DELTA_CLIP
    """
    reward_scale = _get_float("BT_REWARD_SCALE", 1.0)
    delta_temp = _get_float("BT_DELTA_TEMP", 1.0)
    delta_clip = _get_float("BT_DELTA_CLIP", 0.0)
    delta_neg_clip = _get_float("BT_DELTA_NEG_CLIP", 0.0)

    parser_name = os.environ.get("BT_SCORE_PARSER", "score5_last")
    parse_score = get_parser(parser_name)

    return reward_scale, delta_temp, delta_clip, delta_neg_clip, parse_score


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


class BTPointwiseBaselineGeneric(ORM):
    """
    Mean-baseline directional BT reward (credit assignment, avoids tie-attractor).

    For each pair:
      z0_bar = mean(z0)
      z1_bar = mean(z1)

    If pref == 0:
      side0 i: r = log σ( scaled_delta(z0_i - z1_bar) )
      side1 i: r = log σ( scaled_delta(z0_bar - z1_i) )
    Else pref == 1:
      side1 i: r = log σ( scaled_delta(z1_i - z0_bar) )
      side0 i: r = log σ( scaled_delta(z1_bar - z0_i) )

    scaled_delta uses:
      temp=BT_DELTA_TEMP, clip=BT_DELTA_CLIP, neg_clip=BT_DELTA_NEG_CLIP
    """

    def __call__(self, completions, pair_id, side, preferred_side, **kwargs):
        reward_scale, delta_temp, delta_clip, delta_neg_clip, parse_score = _get_params()

        n = len(completions)
        rewards = [0.0] * n
        by_pair = _group_by_pair(pair_id, side, preferred_side)

        def R(delta: float) -> float:
            x = scaled_delta(delta, temp=delta_temp, clip=delta_clip, neg_clip=delta_neg_clip)
            return float(log_sigmoid(x)) * reward_scale

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

            if pref == 0:
                for i in idx0:
                    z = z0_map[i]
                    rewards[i] = -1.0 if z is None else R(z - z1_bar)
                for i in idx1:
                    z = z1_map[i]
                    rewards[i] = -1.0 if z is None else R(z0_bar - z)
            else:
                for i in idx1:
                    z = z1_map[i]
                    rewards[i] = -1.0 if z is None else R(z - z0_bar)
                for i in idx0:
                    z = z0_map[i]
                    rewards[i] = -1.0 if z is None else R(z1_bar - z)

        return rewards


orms["bt_pointwise_baseline"] = BTPointwiseBaselineGeneric
