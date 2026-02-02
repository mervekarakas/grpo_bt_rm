if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"

# Load local overrides if present (secrets + personal paths live here)
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

# Python import path (safe under set -u)
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"

# HF cache defaults
export GRPO_BT_RM_ROOT="${GRPO_BT_RM_ROOT:-$REPO_ROOT}"
export HF_HOME="${HF_HOME:-$GRPO_BT_RM_ROOT/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
unset TRANSFORMERS_CACHE
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

echo "[setup_env] REPO_ROOT=$REPO_ROOT"
echo "[setup_env] PYTHONPATH=$PYTHONPATH"
echo "[setup_env] HF_HOME=$HF_HOME"
echo "[setup_env] WANDB_PROJECT=${WANDB_PROJECT:-<unset>}"
echo "[setup_env] BT_SCORE_PARSER=${BT_SCORE_PARSER:-<unset>}  BT_DELTA_TEMP=${BT_DELTA_TEMP:-<unset>}  BT_DELTA_CLIP=${BT_DELTA_CLIP:-<unset>}  BT_DELTA_NEG_CLIP=${BT_DELTA_NEG_CLIP:-<unset>}  BT_REWARD_SCALE=${BT_REWARD_SCALE:-<unset>} BT_HARD_TAU=${BT_HARD_TAU:-<unset>}  BT_HARD_LAMBDA=${BT_HARD_LAMBDA:-<unset>}"
