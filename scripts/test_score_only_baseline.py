"""Test base accuracy with score-only prompt (no reasoning) on Qwen3-8B.

Compares:
  1. Score-only (no reasoning, thinking OFF) — max_new_tokens=16
  2. Reasoning+score (v3, thinking ON) — max_new_tokens=512 (our current setup)

Both on the same 200 pairs, 1 sample each (n_samples=1), for fast comparison.
"""
import random, sys, os, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from grpo_bt_rm.data.registry import get_dataset
from grpo_bt_rm.prompts.registry import get_prompt
from grpo_bt_rm.parsing.registry import get_parser

N_PAIRS = 200
SEED = 42

model_name = 'Qwen/Qwen3-8B'
print(f'Loading {model_name}...')
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
print('Model loaded.')

adapter = get_dataset('anthropic_hh')
val = adapter.load_split('test')
random.seed(SEED)
all_idxs = list(range(len(val)))
random.shuffle(all_idxs)
idxs = all_idxs[:N_PAIRS]

SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def hh_score100_scoreonly(context: str, response: str) -> str:
    return f"""You are an impartial judge. Score the AI assistant's response on a scale of 0-100.

CONVERSATION:
{context}

RESPONSE:
{response}

Output ONLY your score as: <s>NN</s>
Do not write any explanation or reasoning. Just the score tag."""


@torch.inference_mode()
def generate_one(tok, model, prompt_text, max_new_tokens, enable_thinking):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt_text}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=enable_thinking)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    inputs = tok(text, return_tensors="pt").to("cuda")
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
        top_k=50,
        num_return_sequences=1,
    )
    prompt_len = inputs["input_ids"].shape[1]
    out = gen[0, prompt_len:]
    return tok.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)


conditions = [
    ("score_only (think OFF, max16)",  hh_score100_scoreonly, "score100_first", False, 16),
    ("score_only (think OFF, max32)",  hh_score100_scoreonly, "score100_first", False, 32),
    ("reasoning+score v3 (think ON)",  get_prompt("hh_score100_v3").fn, "score100_last", True, 512),
]

parse100_first = get_parser("score100_first")
parse100_last = get_parser("score100_last")

for cond_name, prompt_fn, parser_name, enable_thinking, max_new_tokens in conditions:
    parse_score = get_parser(parser_name)
    print(f"\n{'='*70}")
    print(f"CONDITION: {cond_name}")
    print(f"{'='*70}")

    correct = 0
    ties = 0
    total = 0
    skipped = 0
    all_scores = []

    for pi, idx in enumerate(idxs):
        row = val[idx]
        context, r0, r1, label = adapter.extract_pair(row)
        # label=0 means r0 is preferred
        r_pref = r0 if label == 0 else r1
        r_nonpref = r1 if label == 0 else r0

        p_pref = prompt_fn(context, r_pref)
        p_nonpref = prompt_fn(context, r_nonpref)

        out_pref = generate_one(tok, model, p_pref, max_new_tokens, enable_thinking)
        out_nonpref = generate_one(tok, model, p_nonpref, max_new_tokens, enable_thinking)

        z_pref = parse_score(out_pref)
        z_nonpref = parse_score(out_nonpref)

        if z_pref is None or z_nonpref is None:
            skipped += 1
            if (pi + 1) % 20 == 0:
                n = total if total > 0 else 1
                print(f"  [{pi+1}/{N_PAIRS}] acc={correct/n:.3f} ties={ties/n:.3f} skipped={skipped}")
            continue

        total += 1
        all_scores.append((z_pref, z_nonpref))

        if z_pref > z_nonpref:
            correct += 1
        elif z_pref == z_nonpref:
            ties += 1
            correct += 0.5

        if (pi + 1) % 20 == 0:
            print(f"  [{pi+1}/{N_PAIRS}] acc={correct/total:.3f} ties={ties/total:.3f} skipped={skipped}")

    if total > 0:
        acc = correct / total
        tie_rate = ties / total
        deltas = [p - n for p, n in all_scores]
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        pref_scores = [p for p, n in all_scores]
        nonpref_scores = [n for p, n in all_scores]
        print(f"\n  FINAL: pairs={total} skipped={skipped}")
        print(f"  ACC={acc:.4f} ties={tie_rate:.4f}")
        print(f"  Mean score (pref): {np.mean(pref_scores):.1f}  (nonpref): {np.mean(nonpref_scores):.1f}")
        print(f"  Delta (pref-nonpref): mean={mean_delta:.1f} std={std_delta:.1f}")
        # show some raw outputs for score_only
        if "score_only" in cond_name:
            print(f"\n  Sample outputs (first 5 pairs):")
            for pi2, idx2 in enumerate(idxs[:5]):
                row2 = val[idx2]
                ctx2, r0_2, r1_2, lab2 = adapter.extract_pair(row2)
                r_pref_2 = r0_2 if lab2 == 0 else r1_2
                out = generate_one(tok, model, prompt_fn(ctx2, r_pref_2), max_new_tokens, enable_thinking)
                print(f"    pair {pi2}: '{out.strip()}'")
    else:
        print(f"\n  FINAL: ALL SKIPPED ({skipped})")
