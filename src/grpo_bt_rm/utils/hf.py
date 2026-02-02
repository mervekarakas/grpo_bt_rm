from __future__ import annotations

import os
from typing import Optional


def set_hf_cache(hf_home: Optional[str] = None, hf_datasets_cache: Optional[str] = None):
    """
    Convenience helper: set HuggingFace cache env vars.
    """
    if hf_home:
        os.environ["HF_HOME"] = hf_home
    if hf_datasets_cache:
        os.environ["HF_DATASETS_CACHE"] = hf_datasets_cache

    # discourage deprecated var
    os.environ.pop("TRANSFORMERS_CACHE", None)
