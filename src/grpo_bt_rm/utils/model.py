# src/grpo_bt_rm/utils/model.py
from __future__ import annotations

import json
import logging
import os
import warnings
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

# Optional Omni imports (guarded)
try:
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    HAS_OMNI = True
except Exception:
    HAS_OMNI = False

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger().setLevel(logging.ERROR)


def setup_device() -> Tuple[str, torch.dtype]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return device, dtype


def log_device_info(prefix: str, device: str):
    if device == "cuda":
        print(f"{prefix} on CUDA:")
        print("  CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
        print("  torch.cuda.device_count() =", torch.cuda.device_count())
        print("  current device index:", torch.cuda.current_device())
        print("  device name:", torch.cuda.get_device_name(0))
    else:
        print(f"{prefix} on CPU.")


# -------------------------
# Base model helpers
# -------------------------
def load_qwen_instruct(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Load base instruct model for scoring tools / variance tests."""
    print(f"Loading Instruct model and tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    device, dtype = setup_device()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    log_device_info("Instruct model", device)
    return tok, model, device


def load_qwen_omni(model_name: str = "Qwen/Qwen2.5-Omni-7B"):
    """Load base omni model (optional)."""
    if not HAS_OMNI:
        raise RuntimeError("Omni dependencies not installed (transformers missing Omni classes).")

    print(f"Loading Omni model and processor: {model_name}")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

    device, dtype = setup_device()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    model.disable_talker()
    model.eval()

    log_device_info("Omni model", device)
    return processor, model, device


# -------------------------
# Checkpoint/eval helpers
# -------------------------
def load_base_model_name(run_dir: str) -> str:
    args_path = os.path.join(run_dir, "args.json")
    if not os.path.exists(args_path):
        return "Qwen/Qwen2.5-7B-Instruct"
    with open(args_path, "r", encoding="utf-8") as f:
        args = json.load(f)
    return args.get("model", "Qwen/Qwen2.5-7B-Instruct")


def load_ckpt_model(base_model: str, ckpt_dir: str, dtype: str = "bf16"):
    """
    Load base model + (optional) LoRA adapter from ckpt_dir.
    Used for eval over checkpoints.
    """
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, torch_dtype=torch_dtype
    ).to("cuda")
    model.eval()

    adapter_cfg = os.path.join(ckpt_dir, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        if not HAS_PEFT:
            raise RuntimeError("peft not installed but adapter_config.json found. pip install peft")
        model = PeftModel.from_pretrained(model, ckpt_dir).to("cuda")
        model.eval()

    return tok, model

# TODO: Delete this once old mudules are removed
# Backward compat: older code may still call load_model
def load_model(base_model: str, ckpt_dir: str, dtype: str):
    return load_ckpt_model(base_model, ckpt_dir, dtype)
