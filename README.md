# grpo_bt_rm

Tools + scripts for training and evaluating a **generative reward model** with **GRPO + Bradley–Terry (BT)**-style preference learning on TL;DR comparisons (OpenAI `summarize_from_feedback`).

This repo provides:
- **Prompt + parser registries** (`score5_*`, `score100_*`)
- **Pointwise JSONL dataset builder** (2 rows per preference pair: side=0 then side=1)
- **Reward plugins** for ms-swift GRPO (BT baseline + optional pair-shared + hardmine variants)
- **Evaluation** over checkpoints with accuracy + tie rate + **BT log-loss** + tail diagnostics
- **Variance / uncertainty diagnostics** to sanity-check prompt signal

> ⚠️ Training depends on a **patched ms-swift fork** that adds `PairRepeatSampler` to keep both sides of each BT pair on the same rank.

---

## Repo layout (high level)

- `src/grpo_bt_rm/`
  - `prompts/` prompt functions + registry
  - `parsing/` score parsers + registry
  - `data/` dataset builder + HF dataset helpers
  - `training/reward_plugins/` reward plugins used by ms-swift `--external_plugins`
  - `eval/` `eval_bt.py` and `variance.py` (run with `python -m ...`)
  - `metrics/` stats + reporting helpers
  - `utils/` model loading, generation, math utilities
- `scripts/` shell helpers (env setup, train, eval sweep)
- `tools/` utilities (variance runner, summarize eval logs, W&B uploader)

---

## 1) Setup on a GPU server

### A) Clone + create venv
```bash
git clone git@github.com:mervekarakas/grpo_bt_rm.git
cd grpo_bt_rm

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
```

### B) Install Python dependencies

Right now the repo includes a **full pinned snapshot** at `requirements.lambda.txt` (heavier, but “it works” if your CUDA/driver stack matches).

```bash
pip install -r requirements.lambda.txt
```

> If you want a lighter install later, create a `requirements.core.txt` for just eval/variance/data-building, but the snapshot is easiest.

### C) Install ms-swift fork (editable) pinned to a known commit

```bash
# clone ms-swift anywhere you like (example: next to grpo_bt_rm)
git clone git@github.com:mervekarakas/ms-swift.git ../ms-swift
cd ../ms-swift

# pinned commit that includes PairRepeatSampler + grpo changes you rely on:
git checkout ffb9c73a4178a7956cbceee837c2ef11114d2387

pip install -e .
cd ../grpo_bt_rm
```

Sanity check:
```bash
python - <<'PY'
import swift
print("swift from:", swift.__file__)
from swift.rlhf_trainers.pair_sampler import PairRepeatSampler
print("PairRepeatSampler import OK:", PairRepeatSampler)
PY
```

### D) Make scripts executable
```bash
chmod +x scripts/*.sh tools/*.sh
```

### E) Source environment
This sets `PYTHONPATH`, HF cache directories, and loads `.env` if present.

```bash
source scripts/setup_env.sh
```

---

## 2) Environment configuration

### A) Local overrides: `.env` (not committed)
Create a local `.env` (copy from `.env.example`) for your machine-specific paths + W&B settings.

Example `.env`:
```bash
export WANDB_PROJECT=grpo_bt_rm
export HF_HOME=/data/$USER/hf_home
export HF_DATASETS_CACHE=$HF_HOME/datasets
```

Then:
```bash
source scripts/setup_env.sh
```

### B) Run presets: `configs/runs/*.env` 

These are reusable presets for dataset/output paths + BT reward hyperparams.

Example:
```bash
source scripts/setup_env.sh
source configs/runs/score100_T20_C10_posclip.env
```

---

## 3) Build the pointwise dataset JSONL

Build the full dataset:
```bash
python -m grpo_bt_rm.data.build_dataset \
  --out /data/$USER/datasets/grpo_bt_train_score100_v1.jsonl \
  --split train \
  --prompt score100_v1 \
  --add_meta
```

Smoke test:
```bash
python -m grpo_bt_rm.data.build_dataset \
  --out /tmp/grpo_bt_smoke.jsonl \
  --split train \
  --limit 10 \
  --prompt score100_v1 \
  --add_meta
```

---

## 4) Train GRPO (ms-swift)

### A) Recommended: run with a preset `.env`
```bash
source scripts/setup_env.sh
source configs/runs/score100_T20_C10_posclip.env

# argument is the CUDA_VISIBLE_DEVICES list for this run:
bash scripts/train_grpo.sh 0,1
```

This script expects at least:
- `DATASET` (JSONL path)
- `OUT_DIR` (output directory)
and will pass:
- `PLUGIN` (default points to `src/grpo_bt_rm/training/reward_plugins/bt_baseline.py`)
- `REWARD_NAME` (default `bt_pointwise_baseline`)

### B) Common env vars used by reward plugins
```bash
export BT_SCORE_PARSER=score100_first   # or score5_last
export BT_DELTA_TEMP=20
export BT_DELTA_CLIP=10
export BT_DELTA_NEG_CLIP=0             # 0 => positive-only clip
export BT_REWARD_SCALE=1
```

---

## 5) Evaluate checkpoints (accuracy + BT log-loss)

### A) Single eval (one checkpoint)
```bash
source scripts/setup_env.sh
python -m grpo_bt_rm.eval.eval_bt \
  --run_dir  /data/$USER/experiments/grpo_bt_rm/your_run/v0-... \
  --checkpoint /data/$USER/experiments/grpo_bt_rm/your_run/v0-.../checkpoint-6000 \
  --n_pairs 2000 \
  --seed 0 \
  --dtype bf16 \
  --do_sample --n_samples 2 \
  --temperature 0.7 --top_p 0.9 --top_k 50 \
  --prompt score100_v1 --parser score100_first \
  --bt_temp 20 \
  --unc_low 20 --unc_high 80 \
  --report_uncertainty --report_margins
```

### B) Eval sweep over many checkpoints/seeds (recommended)

```bash
source scripts/setup_env.sh
scripts/run_eval_sweep.sh \
  --run_dir /data/$USER/experiments/grpo_bt_rm/your_run/v0-... \
  --eval_subdir eval_2k_bt20 \
  --ckpts "checkpoint-0 checkpoint-2000 checkpoint-4000 checkpoint-6000 checkpoint-8000" \
  --seeds "0 1 2" \
  --gpus "0 1" \
  --n_pairs 2000 \
  --prompt score100_v1 --parser score100_first \
  --bt_temp 20 \
  --unc_low 20 --unc_high 80
```

This produces:
- `eval_*/eval_checkpoint-*_seed*.log`
- `summary.csv`, `summary.md`
- If `WANDB_PROJECT` is set, it uploads the summary (CSV + MD) to W&B.

---

## 6) Variance / uncertainty diagnostics (prompt signal)

### A) Quick smoke test
```bash
CUDA_VISIBLE_DEVICES=0 bash tools/run_variance_test.sh \
  --prompt score100_v1 \
  --parser score100_first \
  --split validation \
  --n_pairs 50 \
  --n_samples 8 \
  --batch_pairs 8 \
  --max_new_tokens 64 \
  --temperature 0.7 \
  --range_thresholds "10,20,30,40" \
  --unc_low 20 --unc_high 80
```

This reports:
- expected correctness under aligned sampling (ties=0.5 and strict)
- wrong-pair uncertainty rates
- histograms of sampled scores
- range-based uncertainty at thresholds

---

## 7) Troubleshooting

### “Permission denied” when running a script
Make scripts executable:
```bash
chmod +x scripts/*.sh tools/*.sh
```

### `ModuleNotFoundError: grpo_bt_rm`
You must have `PYTHONPATH` set. Do:
```bash
source scripts/setup_env.sh
```

### `ModuleNotFoundError: vllm`
Some ms-swift GRPO codepaths import `vllm`. Install it:
```bash
pip install -U "vllm>=0.5.1"
```

> If you hit NumPy dependency conflicts after installing vllm, the safest workaround is to use a **fresh venv** for training/eval on that box, or pin versions carefully. (We keep the Lambda snapshot in `requirements.txt` for reproducibility on that environment.)

### Hugging Face cache warnings about `TRANSFORMERS_CACHE`
We use `HF_HOME` + `HF_DATASETS_CACHE`. `scripts/setup_env.sh` unsets `TRANSFORMERS_CACHE`.

---

## 8) Lambda setup (optional)

If you use Lambda persistent storage, `setup_lambda.sh` bootstraps:
- persistent venv
- HF cache dirs
- installs `requirements.txt`
- clones your pinned ms-swift fork/commit and installs editable

```bash
bash setup_lambda.sh
```

---

## Notes

- This repo is intended to be used via `PYTHONPATH` + scripts; you do **not** need `pip install -e .` for `grpo_bt_rm`.
- Keep secrets out of git: use `.env` locally, commit only `.env.example`.
