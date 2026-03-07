# grpo_bt_rm

Tools + scripts for training and evaluating a **generative reward model** with **GRPO + Bradley–Terry (BT)**-style preference learning on preference datasets (OpenAI `summarize_from_feedback` and Anthropic `hh-rlhf`).

This repo provides:
- **Prompt + parser registries** (`score5_*`, `score100_*`, `hh_score100_v1`–`v4`, `hh_score5_v1`–`v4`)
- **Pointwise JSONL dataset builder** (2 rows per preference pair: side=0 then side=1)
- **SFT warm-up pipeline**: teacher data generation via API (Claude / Gemini) + SFT training before GRPO
- **Reward plugins** for ms-swift GRPO (BT baseline, graded, hard-margin, binary, hardmine variants)
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
- `scripts/`
  - `train_grpo.sh` GRPO training launcher
  - `train_sft.sh` SFT warm-up training launcher
  - `teacher_generate.py` teacher data generation via API (Anthropic / Gemini)
  - `build_sft_data.py` convert teacher scores to SFT training format
  - `setup_env.sh` env setup, `run_eval_sweep.sh` eval sweep
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

### C.1) Apply ms-swift patches (required for GRPO experiments)

After installing ms-swift, apply these patches to the installed `swift/rlhf_trainers/` directory:

1. **`args_mixin.py`**: Adds `min_reward_std` field to `GRPOArgumentsMixin` (advantage filtering — zeroes out groups with reward std below threshold)
2. **`grpo_trainer.py`**: Implements advantage filtering logic + `frac_reward_low_std` metric logging
3. **`reward_trainer.py`**: Fixes import: `from swift.trainers import SwiftMixin` (instead of relative import)

Find the swift install location and apply:
```bash
SWIFT_DIR=$(python3 -c "import swift; import os; print(os.path.dirname(swift.__file__))")
# Copy the 3 patched files from patches/ directory (or apply manually)
```

The patches are small (15 lines total). See `scripts/setup_lambda_remote.sh` for the automated version.

### D) Make scripts executable
```bash
chmod +x scripts/*.sh tools/*.sh
```

### E) Source environment
Activate your virtual environment first, then source the env script which sets `PYTHONPATH`, HF cache directories, and loads `.env` if present.

```bash
source .venv/bin/activate   # or wherever your venv lives
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

### Summarize from Feedback
```bash
python -m grpo_bt_rm.data.build_dataset \
  --out /data/$USER/datasets/grpo_bt_train_score100_v1.jsonl \
  --split train \
  --prompt score100_v1 \
  --add_meta
```

### Anthropic HH-RLHF
```bash
python -m grpo_bt_rm.data.build_dataset \
  --dataset anthropic_hh \
  --out /data/$USER/datasets/grpo_bt_hh_train_hh_score100_v1.jsonl \
  --split train \
  --prompt hh_score100_v1 \
  --add_meta
```

Smoke test (either dataset):
```bash
python -m grpo_bt_rm.data.build_dataset \
  --dataset anthropic_hh \
  --out /tmp/grpo_bt_smoke.jsonl \
  --split train \
  --limit 10 \
  --prompt hh_score100_v1 \
  --add_meta
```

### Available prompts

| Prompt | Parser | Style | Dataset |
|--------|--------|-------|---------|
| `score100_v1` | `score100_first` | Score-first 0–100 | SFF |
| `hh_score100_v1` | `score100_first` | Score-first 0–100 | HH |
| `hh_score100_v2` | `score100_first` | Concise 0–100 | HH |
| `hh_score100_v3` | `score100_last` | Reason-first 0–100 | HH |
| `hh_score100_v4` | `score100_last` | Pros-cons-first 0–100 | HH |
| `hh_score5_v1`–`v4` | `score5_first`/`last` | Same styles, 1–5 scale | HH |
| `hh_score100_scoreonly` | `score100_last` | Score-only, no reasoning | HH |

**Score-first** prompts output `<s>NN</s>` before the explanation. **Reason-first** prompts write reasoning first, then the score tag. **Score-only** outputs just the score tag with no reasoning (used for score-token-only RL). Reason-first (v3) tends to give better accuracy + variance on the base model.

---

## 4) SFT warm-up (optional but recommended)

When the base model's preference accuracy is low (e.g. ~57% on HH), GRPO struggles to learn because the reward signal is too noisy. An SFT warm-up uses a stronger teacher model (via API) to generate scored examples, then fine-tunes the base model to produce better scores before GRPO.

### A) Generate teacher data

Use `scripts/teacher_generate.py` to score HH pairs with an API model:

```bash
# Anthropic (Claude Sonnet)
PYTHONPATH=src python scripts/teacher_generate.py \
  --backend anthropic --model claude-sonnet-4-20250514 \
  --api_key $ANTHROPIC_API_KEY \
  --n_pairs 1000 --batch_size 1 --scale score100 --split train

# Gemini
PYTHONPATH=src python scripts/teacher_generate.py \
  --backend gemini --model gemini-2.5-flash \
  --api_key $GEMINI_API_KEY \
  --n_pairs 500 --batch_size 50 --scale score100 --split train
```

Supports `--resume` to continue after interruptions, and `--skip_pairs N` for non-overlapping batches.

### B) Build SFT training data

Convert teacher scores into SFT format (user prompt + assistant response with score tag):

```bash
PYTHONPATH=src python scripts/build_sft_data.py \
  --input_dirs /data/$USER/experiments/grpo_bt_rm/teacher_data/score100_* \
  --output /data/$USER/experiments/grpo_bt_rm/sft_data/hh_sft_correct_only.jsonl \
  --prompt hh_score100_v1 \
  --filter_correct \
  --include_both_sides
```

`--filter_correct` keeps only pairs where the teacher scored chosen > rejected (higher quality). `--include_both_sides` creates two examples per pair (one for each response).

### C) Train SFT

```bash
source scripts/setup_env.sh
bash scripts/train_sft.sh <gpu_list> <dataset_jsonl> <output_dir>

# Example:
bash scripts/train_sft.sh 0,1 \
  /data/$USER/experiments/grpo_bt_rm/sft_data/hh_sft_correct_only.jsonl \
  /data/$USER/experiments/grpo_bt_rm/sft_v1
```

### D) Merge LoRA adapter for GRPO

SFT produces a LoRA adapter. Merge it into a full model before GRPO:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, "<sft_checkpoint_path>")
model = model.merge_and_unload()
model.save_pretrained("<merged_output_path>")
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct").save_pretrained("<merged_output_path>")
```

Then use the merged path as `MODEL` for GRPO training (section 5).

---

## 5) Train GRPO (ms-swift)

### A) Recommended: run with a preset `.env`

> If you ran the SFT warm-up (section 4), set `MODEL` to the merged model path and add `EXTRA_ARGS="--model_type qwen2 --template qwen2_5"` before launching.

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

### B) Reward plugins

All plugins are in `src/grpo_bt_rm/training/reward_plugins/` and are loaded via ms-swift's `--external_plugins` flag.

| Plugin | Registered Name | Description |
|--------|----------------|-------------|
| `bt_baseline.py` | `bt_pointwise_baseline` | Continuous log-sigmoid BT reward |
| `bt_graded.py` | `bt_pointwise_graded` | PaTaRM-style bounded graded reward (0/1.2/1.4) |
| `bt_graded_hardpair.py` | `bt_pointwise_graded_hardpair` | Graded + DWRL-style hard-pair weighting |
| `bt_hard_margin.py` | `bt_hard_margin` | Marginized BT + hard-pair weighting (main method) |
| `bt_binary.py` | `bt_pointwise_binary` | Constant +1/-1 binary reward |

**Common env vars** (read at runtime by all BT plugins):
```bash
export BT_SCORE_PARSER=score100_last    # parser for extracting score from completion
export BT_SCORE_TEMP=20                 # temperature for normalizing raw score deltas
```

**Plugin-specific env vars:**

For `bt_baseline`:
```bash
export BT_DELTA_TEMP=20  BT_DELTA_CLIP=10  BT_DELTA_NEG_CLIP=0  BT_REWARD_SCALE=1
```

For `bt_graded` / `bt_graded_hardpair`:
```bash
export BT_GRADE_THRESH=0.1  BT_REWARD_LOW=1.2  BT_REWARD_HIGH=1.4
export BT_REWARD_WRONG=0.0  BT_REWARD_PARSE_FAIL=-0.5
export BT_HARD_TAU=1.0      # hardpair only: sigmoid temperature
```

For `bt_hard_margin`:
```bash
export BT_MARGIN_GAMMA=0.5  BT_MARGIN_TEMP=1.0  BT_HARD_TAU=1.0
export BT_REWARD_PARSE_FAIL=-1.0
```

### C) Experiment configs

Pre-built configs in `configs/runs/`:

| Config | Experiment | Key Idea |
|--------|-----------|----------|
| `hh_exp4_scoreonly_hardmargin.env` | Score-Only Hard-BT (main) | Score-only RL + marginized BT + hard-pair |
| `hh_exp1_graded.env` | Full-token graded (ablation) | Reasoning+score, PaTaRM-style reward |
| `hh_exp3_scoreonly_graded.env` | Score-only graded (ablation) | Score-only, PaTaRM-style reward |
| `hh_exp2_graded_hardpair.env` | Full-token graded+hardpair | Reasoning+score, graded + hard-pair |

Example:
```bash
source scripts/setup_env.sh
source configs/runs/hh_exp4_scoreonly_hardmargin.env
bash scripts/train_grpo.sh 0,1
```

### D) Automated experiment runner

`scripts/launch_lambda_experiments.sh` runs experiments sequentially with auto-eval:
```bash
# Run specific experiments on given GPUs:
bash scripts/launch_lambda_experiments.sh 0,1 exp4 exp1 exp3
```

`scripts/gpu_watcher.sh` watches GPUs and launches experiments when they become idle:
```bash
tmux new -s watcher 'bash scripts/gpu_watcher.sh'
```

---

## 6) Evaluate checkpoints (accuracy + BT log-loss)

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

## 7) Variance / uncertainty diagnostics (prompt signal)

### A) Quick smoke test (Summarize from Feedback)
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

### B) HH-RLHF variance test
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m grpo_bt_rm.eval.variance \
  --dataset anthropic_hh --split test \
  --prompt hh_score100_v3 \
  --n_pairs 200 --n_samples 8 \
  --batch_pairs 4 \
  --range_thresholds "20,30,40"
```

This reports:
- expected correctness under aligned sampling (ties=0.5 and strict)
- wrong-pair uncertainty rates
- histograms of sampled scores
- range-based uncertainty at thresholds

---

## 8) Troubleshooting

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

## 9) Remote GPU setup (Lambda / other servers)

### Quick setup via rsync

From the local machine with all code:
```bash
# Sync code + datasets to remote server
bash scripts/setup_lambda.sh <REMOTE_IP> [SSH_KEY]
```

Then SSH into the remote and run:
```bash
bash /data/grpo_bt_rm/scripts/setup_lambda_remote.sh
```

This installs venv, ms-swift + patches, dependencies, and downloads the model.

### Running experiments
```bash
source /data/venv/bin/activate
cd /data/grpo_bt_rm
wandb login $WANDB_API_KEY

# Option A: Run all experiments sequentially
bash scripts/launch_lambda_experiments.sh 0,1

# Option B: Run specific experiments in parallel
tmux new -s main 'bash scripts/launch_lambda_experiments.sh 0,1 exp4'
tmux new -s abl  'bash scripts/launch_lambda_experiments.sh 2,3 exp1 exp3'
```

---

## Notes

- This repo can be used via `PYTHONPATH` + scripts (`source scripts/setup_env.sh`) **or** installed as an editable package (`pip install -e .`).
- Keep secrets out of git: use `.env` locally, commit only `.env.example`.
