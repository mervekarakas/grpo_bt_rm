"""Analyze BT RM scores on HH test set.

Loads the trained BT RM checkpoint and scores all test pairs.
Outputs per-pair scores, margins, and summary statistics.
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "/data/mkarakas/experiments/grpo_bt_rm/bt_rm_baseline/v5-20260303-111734/checkpoint-500"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BATCH_SIZE = 4
MAX_LENGTH = 2048

print(f"Loading base model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=1, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.config.pad_token_id = tokenizer.pad_token_id

print(f"Loading LoRA adapter: {CHECKPOINT}")
model = PeftModel.from_pretrained(model, CHECKPOINT)
model.eval()

# Load test set
print("Loading HH test set...")
ds = load_dataset("Anthropic/hh-rlhf")["test"]
print(f"Test set: {len(ds)} examples")

# Parse conversations into messages format (same as our convert script)
def to_messages(text):
    parts = [s.strip() for s in re.split(r"\n\nHuman:|\n\nAssistant:|\n\nHum:", text.strip())]
    parts = [p for p in parts if p]
    messages = []
    for i in range(0, len(parts) - 1, 2):
        messages.append({"role": "user", "content": parts[i]})
        messages.append({"role": "assistant", "content": parts[i + 1]})
    if len(parts) % 2 == 1:
        messages.append({"role": "user", "content": parts[-1]})
    return messages


@torch.inference_mode()
def score_batch(texts):
    """Score a batch of tokenized conversations."""
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH
    ).to(model.device)
    outputs = model(**inputs)
    return outputs.logits.squeeze(-1).cpu().float().numpy()


# Score all pairs
print("Scoring all pairs...")
all_results = []
chosen_texts = []
rejected_texts = []

for i, row in enumerate(ds):
    chosen_msgs = to_messages(row["chosen"])
    rejected_msgs = to_messages(row["rejected"])

    chosen_text = tokenizer.apply_chat_template(chosen_msgs, tokenize=False)
    rejected_text = tokenizer.apply_chat_template(rejected_msgs, tokenize=False)

    chosen_texts.append(chosen_text)
    rejected_texts.append(rejected_text)

# Score in batches
chosen_scores = []
rejected_scores = []

for start in range(0, len(chosen_texts), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(chosen_texts))
    batch_chosen = score_batch(chosen_texts[start:end])
    batch_rejected = score_batch(rejected_texts[start:end])
    chosen_scores.extend(batch_chosen.tolist())
    rejected_scores.extend(batch_rejected.tolist())

    if (start // BATCH_SIZE) % 50 == 0:
        n = len(chosen_scores)
        correct = sum(1 for c, r in zip(chosen_scores, rejected_scores) if c > r)
        ties = sum(1 for c, r in zip(chosen_scores, rejected_scores) if c == r)
        acc = (correct + 0.5 * ties) / n
        print(f"  [{n}/{len(chosen_texts)}] running_acc={acc:.4f}")

chosen_scores = np.array(chosen_scores)
rejected_scores = np.array(rejected_scores)
margins = chosen_scores - rejected_scores

correct = margins > 0
ties = margins == 0
wrong = margins < 0

acc = (correct.sum() + 0.5 * ties.sum()) / len(margins)
strict_acc = correct.sum() / len(margins)

print(f"\n{'='*60}")
print(f"RESULTS (checkpoint: {os.path.basename(CHECKPOINT)})")
print(f"{'='*60}")
print(f"N pairs:        {len(margins)}")
print(f"Accuracy:       {acc:.4f} ({acc*100:.1f}%)")
print(f"Strict ACC:     {strict_acc:.4f} ({strict_acc*100:.1f}%)")
print(f"Ties:           {ties.sum()} ({ties.mean()*100:.1f}%)")
print(f"Correct:        {correct.sum()} ({correct.mean()*100:.1f}%)")
print(f"Wrong:          {wrong.sum()} ({wrong.mean()*100:.1f}%)")

print(f"\nScore distribution:")
print(f"  Chosen:   mean={chosen_scores.mean():.3f}  std={chosen_scores.std():.3f}  min={chosen_scores.min():.3f}  max={chosen_scores.max():.3f}")
print(f"  Rejected: mean={rejected_scores.mean():.3f}  std={rejected_scores.std():.3f}  min={rejected_scores.min():.3f}  max={rejected_scores.max():.3f}")
print(f"  Margins:  mean={margins.mean():.3f}  std={margins.std():.3f}  min={margins.min():.3f}  max={margins.max():.3f}")

# Margin buckets
print(f"\nAccuracy by margin magnitude:")
abs_margins = np.abs(margins)
for lo, hi in [(0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, float('inf'))]:
    mask = (abs_margins >= lo) & (abs_margins < hi)
    n = mask.sum()
    if n > 0:
        bucket_acc = correct[mask].mean()
        print(f"  |margin| [{lo:.1f}, {hi:.1f}): n={n:5d}  acc={bucket_acc:.3f}")

# Score percentiles
print(f"\nMargin percentiles:")
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  P{p:2d}: {np.percentile(margins, p):.3f}")

# Save per-pair results
output_path = os.path.join(os.path.dirname(CHECKPOINT), "test_scores.jsonl")
with open(output_path, "w") as f:
    for i in range(len(margins)):
        f.write(json.dumps({
            "idx": i,
            "chosen_score": float(chosen_scores[i]),
            "rejected_score": float(rejected_scores[i]),
            "margin": float(margins[i]),
            "correct": bool(correct[i]),
        }) + "\n")
print(f"\nPer-pair scores saved to: {output_path}")
