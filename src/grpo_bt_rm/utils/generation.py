from typing import List
import torch

SYSTEM_DEFAULT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

@torch.inference_mode()
def generate_batch(tok, model, prompts: List[str], max_new_tokens: int,
                   do_sample: bool, temperature: float, top_p: float, top_k: int,
                   system: str = SYSTEM_DEFAULT) -> List[str]:
    msgs = [[{"role": "system", "content": system}, {"role": "user", "content": p}] for p in prompts]
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    inputs = tok(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p, top_k=top_k))

    gen = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    outs = []
    for j in range(gen.shape[0]):
        out = gen[j, prompt_len:]
        outs.append(tok.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    return outs
