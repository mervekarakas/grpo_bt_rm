from typing import List, Optional, Tuple, Callable
import torch

def format_as_chat(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

@torch.no_grad()
def sample_scores_for_prompt(
    tokenizer,
    model,
    device: str,
    prompt: str,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    use_chat_template: bool,
) -> Tuple[List[Optional[float]], List[str]]:
    """
    Samples n_samples completions from the model for a single prompt.
    Returns:
      - scores: list of parsed scores (None if parse failed)
      - texts: raw decoded completions (for debugging)
    """
    text_in = format_as_chat(tokenizer, prompt, use_chat_template)
    inputs = tokenizer(text_in, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=n_samples,
        pad_token_id=tokenizer.eos_token_id,
    )

    prompt_len = inputs["input_ids"].shape[1]
    scores: List[Optional[float]] = []
    texts: List[str] = []

    for i in range(n_samples):
        gen_ids = outputs[i, prompt_len:]
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        texts.append(decoded)
        scores.append(parse_score(decoded))

    return scores, texts

@torch.no_grad()
def sample_scores_for_prompts_batch(
    tokenizer,
    model,
    device: str,
    prompts: List[str],
    parse_score: Callable[[str], Optional[float]],
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    use_chat_template: bool,
) -> Tuple[List[List[Optional[float]]], List[List[str]]]:
    texts_in = [format_as_chat(tokenizer, p, use_chat_template) for p in prompts]

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    inputs = tokenizer(texts_in, return_tensors="pt", padding=True, truncation=True).to(device)
    input_lens = inputs["attention_mask"].sum(dim=1).tolist()

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=n_samples,
        pad_token_id=tokenizer.eos_token_id,
    )

    scores_per_prompt = [[] for _ in range(len(prompts))]
    texts_per_prompt = [[] for _ in range(len(prompts))]

    for p_idx in range(len(prompts)):
        plen = int(input_lens[p_idx])
        for k in range(n_samples):
            row = p_idx * n_samples + k
            gen_ids = outputs[row, plen:]
            decoded = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            texts_per_prompt[p_idx].append(decoded)
            scores_per_prompt[p_idx].append(parse_score(decoded))

    return scores_per_prompt, texts_per_prompt
