"""Test score variance under 4 conditions: thinking on/off × score-first/last."""
import random, sys, os, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from grpo_bt_rm.data.registry import get_dataset
from grpo_bt_rm.prompts.registry import get_prompt
from grpo_bt_rm.parsing.registry import get_parser

N_PAIRS = 50
N_SAMPLES = 8
SEED = 42
MAX_NEW_TOKENS = 512

model_name = 'Qwen/Qwen3-8B'
print(f'Loading {model_name}...')
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
print('Model loaded.')

adapter = get_dataset('anthropic_hh')
val = adapter.load_split('test')
random.seed(SEED)
idxs = list(range(len(val)))
random.shuffle(idxs)
idxs = idxs[:N_PAIRS]

SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

conditions = [
    ("think_ON+score_last",  "hh_score100_v3", "score100_last", True),
    ("think_OFF+score_last", "hh_score100_v3", "score100_last", False),
    ("think_ON+score_first", "hh_score100_v1", "score100_first", True),
    ("think_OFF+score_first","hh_score100_v1", "score100_first", False),
]

@torch.inference_mode()
def generate_samples(tok, model, prompt_text, n_samples, max_new_tokens, enable_thinking):
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
        num_return_sequences=n_samples,
    )
    prompt_len = inputs["input_ids"].shape[1]
    outs = []
    for j in range(gen.shape[0]):
        out = gen[j, prompt_len:]
        outs.append(tok.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    return outs


for cond_name, prompt_name, parser_name, enable_thinking in conditions:
    prompt_fn = get_prompt(prompt_name).fn
    parse_fn = get_parser(parser_name)

    all_stds = []
    all_ranges = []
    correct = 0
    ties = 0
    total = 0
    skipped = 0

    print(f"\n{'='*70}")
    print(f"CONDITION: {cond_name}  (prompt={prompt_name}, thinking={enable_thinking})")
    print(f"{'='*70}")

    for i, idx in enumerate(idxs):
        ex = val[idx]
        post, s0, s1, label = adapter.extract_pair(ex)

        p0 = prompt_fn(post, s0)
        p1 = prompt_fn(post, s1)

        outs0 = generate_samples(tok, model, p0, N_SAMPLES, MAX_NEW_TOKENS, enable_thinking)
        outs1 = generate_samples(tok, model, p1, N_SAMPLES, MAX_NEW_TOKENS, enable_thinking)

        scores0 = [parse_fn(o) for o in outs0]
        scores1 = [parse_fn(o) for o in outs1]

        valid0 = [s for s in scores0 if s is not None]
        valid1 = [s for s in scores1 if s is not None]

        if not valid0 or not valid1:
            skipped += 1
            continue

        total += 1
        mean0 = np.mean(valid0)
        mean1 = np.mean(valid1)
        std0 = np.std(valid0)
        std1 = np.std(valid1)

        all_stds.extend([std0, std1])
        all_ranges.extend([max(valid0) - min(valid0), max(valid1) - min(valid1)])

        if abs(mean0 - mean1) < 1e-9:
            ties += 1
            correct += 0.5
        elif (mean0 > mean1 and label == 0) or (mean1 > mean0 and label == 1):
            correct += 1.0
        # else wrong

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{N_PAIRS}] acc={correct/max(total,1):.3f} ties={ties/max(total,1):.3f} "
                  f"mean_std={np.mean(all_stds):.2f} mean_range={np.mean(all_ranges):.1f} skipped={skipped}",
                  flush=True)

    print(f"\n  FINAL: pairs={total} skipped={skipped}")
    print(f"  ACC={correct/max(total,1):.4f} ties={ties/max(total,1):.4f}")
    print(f"  Score STD: mean={np.mean(all_stds):.2f} median={np.median(all_stds):.2f} "
          f"p25={np.percentile(all_stds,25):.2f} p75={np.percentile(all_stds,75):.2f}")
    print(f"  Score RANGE: mean={np.mean(all_ranges):.1f} median={np.median(all_ranges):.1f}")
