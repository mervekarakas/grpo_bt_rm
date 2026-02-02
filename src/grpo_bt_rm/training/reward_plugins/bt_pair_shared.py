# src/grpo_bt_rm/training/reward_plugins/bt_pair_shared.py
import os
from typing import Dict

from swift.plugin import ORM, orms

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


class BTPointwisePairShared(ORM):
    """
    Aligned-k shared pair reward:

      For each pair_id:
        sort indices per side
        align (k-th completion side0 with k-th completion side1)
        delta_pref = z_pref - z_non
        r = log_sigmoid( scaled_delta(delta_pref) )
        assign r to BOTH aligned completions

    scaled_delta uses:
      temp=BT_DELTA_TEMP, clip=BT_DELTA_CLIP, neg_clip=BT_DELTA_NEG_CLIP
    """

    def __call__(self, completions, pair_id, side, preferred_side, **kwargs):
        reward_scale, delta_temp, delta_clip, delta_neg_clip, parse_score = _get_params()

        n = len(completions)
        rewards = [0.0] * n
        by_pair = _group_by_pair(pair_id, side, preferred_side)

        for _, d in by_pair.items():
            pref = int(d["pref"])
            idx0 = sorted(d[0])
            idx1 = sorted(d[1])

            m = min(len(idx0), len(idx1))
            if m == 0:
                for i in idx0 + idx1:
                    rewards[i] = -1.0
                continue

            for k in range(m):
                i0, i1 = idx0[k], idx1[k]
                z0 = parse_score(completions[i0])
                z1 = parse_score(completions[i1])

                if z0 is None or z1 is None:
                    rewards[i0] = -1.0
                    rewards[i1] = -1.0
                    continue

                delta_pref = (z0 - z1) if pref == 0 else (z1 - z0)
                x = scaled_delta(float(delta_pref), temp=delta_temp, clip=delta_clip, neg_clip=delta_neg_clip)
                r = float(log_sigmoid(x)) * reward_scale

                rewards[i0] = r
                rewards[i1] = r

            for i in idx0[m:] + idx1[m:]:
                rewards[i] = -1.0

        return rewards


orms["bt_pointwise_pair_shared"] = BTPointwisePairShared
