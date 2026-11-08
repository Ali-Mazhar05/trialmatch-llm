"""Shared benchmarking helpers: warm-up, synchronised repeated timing, prompts."""
from __future__ import annotations

import json
import platform
import statistics
from pathlib import Path

import torch


def env_info() -> dict:
    d = {"torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform(),
         "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        d["gpu"] = torch.cuda.get_device_name(0)
        d["gpu_mem_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        d["cpu"] = platform.processor()
    try:
        import transformers
        d["transformers"] = transformers.__version__
    except ImportError:
        pass
    return d


def repeat(fn, warmup: int = 1, repeats: int = 3):
    """Run fn() warmup times (discarded), then `repeats` times. fn returns
    (tokens, GenStats). Returns tokens of the last run and median/min/max stats."""
    for _ in range(warmup):
        fn()
    runs = [fn() for _ in range(repeats)]
    toks = runs[-1][0]
    stats = [r[1].as_dict() for r in runs]
    agg = {}
    for key in ["prefill_s", "decode_s", "total_s", "decode_tok_s", "e2e_tok_s", "peak_mem_mb"]:
        vals = [s[key] for s in stats if s[key] is not None]
        if vals:
            agg[key] = statistics.median(vals)
            agg[key + "_min"], agg[key + "_max"] = min(vals), max(vals)
    for key, val in stats[-1].items():
        agg.setdefault(key, val)
    agg["repeats"] = repeats
    agg["consistent_across_repeats"] = all(r[0] == toks for r in runs)
    return toks, agg


def corpus_ids(tok, texts: list[str], n: int) -> list[int]:
    """Deterministic real-text prompt of exactly n tokens (val notes concatenated)."""
    ids: list[int] = []
    i = 0
    while len(ids) < n:
        ids += tok(texts[i % len(texts)] + "\n", add_special_tokens=False)["input_ids"]
        i += 1
    return ids[:n]


def first_mismatch(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def save(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))
