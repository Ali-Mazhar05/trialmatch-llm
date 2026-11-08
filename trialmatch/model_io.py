"""Model / adapter loading shared by evaluation, benchmarks and the app."""
from __future__ import annotations

import os

import torch

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_dtype(device: str) -> str:
    # T4/P100 have no bf16; CPU fp16 matmuls are slow/unsupported -> fp32 on CPU
    return "fp16" if device.startswith("cuda") else "fp32"


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(base: str, adapter: str | None = None, device: str | None = None, dtype: str | None = None,
               merge: bool = True, attn_implementation: str = "sdpa"):
    """Load a causal LM, optionally apply a LoRA adapter and merge it.

    Merging folds W + (alpha/r) * B @ A into the base weights so inference code
    (manual KV-cache loop, speculative decoding) sees an ordinary dense model.
    """
    from transformers import AutoModelForCausalLM

    device = pick_device(device)
    dtype = dtype or default_dtype(device)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=DTYPES[dtype], attn_implementation=attn_implementation)
    if adapter and adapter.lower() not in {"none", ""}:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        if merge:
            model = model.merge_and_unload()
    model.to(device).eval()
    return model


def resolve_adapter(adapter: str | None) -> str | None:
    """Accept a local dir or a Hugging Face Hub repo id ("user/repo")."""
    if not adapter or adapter.lower() == "none":
        return None
    if os.path.isdir(adapter):
        return adapter
    from huggingface_hub import snapshot_download

    return snapshot_download(adapter)
