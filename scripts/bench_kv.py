"""Task 3 benchmark (rubric D2/D3): cached vs uncached greedy decoding.

For every (input_len, output_len) pair, all paths decode the SAME prompt with
EOS disabled (fixed output length). Each path: 1 warm-up + N synchronised
repeats; we report median prefill, decode, tokens/s and peak memory, and check
token-level agreement with the uncached HF reference. On a mismatch we record
the first divergent step and the reference top-2 logit margin there (a tiny
margin means a numerical near-tie, not a logic error).

    python scripts/bench_kv.py --adapter runs/qwen15_lora/final_adapter
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from trialmatch.bench_utils import corpus_ids, env_info, first_mismatch, repeat, save  # noqa: E402
from trialmatch.data import load_processed  # noqa: E402
from trialmatch.decoding import engine_greedy, engine_no_cache_greedy, hf_cache_greedy, no_cache_greedy  # noqa: E402
from trialmatch.engine import QwenEngine  # noqa: E402
from trialmatch.model_io import load_model, load_tokenizer, resolve_adapter  # noqa: E402


@torch.no_grad()
def margin_at(model, prompt, ref, i):
    dev = next(model.parameters()).device
    lg = model(input_ids=torch.tensor([prompt + ref[:i]], device=dev)).logits[0, -1].float()
    top = lg.topk(2).values
    return float(top[0] - top[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--inputs", default="128,512,1024")
    ap.add_argument("--outputs", default="32,128,256")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip_engine_nocache", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "bench" / "kv_cache.json"))
    args = ap.parse_args()

    tok = load_tokenizer(args.model)
    model = load_model(args.model, resolve_adapter(args.adapter), dtype=args.dtype)
    eng = QwenEngine(model)
    texts = list(load_processed(ROOT / "data" / "processed" / "val.jsonl").note.unique())
    cfg = model.config
    hd = cfg.hidden_size // cfg.num_attention_heads
    bytes_per_tok = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * hd * next(model.parameters()).element_size()

    rows = []
    for n_in in map(int, args.inputs.split(",")):
        p = corpus_ids(tok, texts, n_in)
        for n_out in map(int, args.outputs.split(",")):
            paths = {
                "hf_no_cache (reference)": lambda: no_cache_greedy(model, p, n_out),
                "hf_explicit_cache": lambda: hf_cache_greedy(model, p, n_out),
                "engine_kv_cache": lambda: engine_greedy(eng, p, n_out),
            }
            if not args.skip_engine_nocache:
                paths["engine_no_cache"] = lambda: engine_no_cache_greedy(eng, p, n_out)
            ref = None
            for name, fn in paths.items():
                toks, agg = repeat(fn, warmup=1, repeats=args.repeats)
                if ref is None:
                    ref = toks
                mm = first_mismatch(ref, toks)
                row = {"input_len": n_in, "output_len": n_out, "path": name, "tokens_equal_reference": mm is None,
                       "first_mismatch_step": mm, **{k: v for k, v in agg.items() if k != "cache_growth"}}
                if mm is not None:
                    row["ref_top2_margin_at_mismatch"] = margin_at(model, p, ref, mm)
                if name == "engine_kv_cache":
                    row["kv_cache_mb_theory"] = bytes_per_tok * (n_in + n_out - 1) / 2**20
                rows.append(row)
                print(f"in={n_in:5d} out={n_out:4d} {name:26s} equal={mm is None!s:5s} "
                      f"prefill={agg['prefill_s']*1e3:8.1f}ms decode={agg['decode_s']:7.3f}s "
                      f"tok/s={agg['decode_tok_s']:7.1f} mem={agg.get('peak_mem_mb')}")
    save({"env": env_info(), "model": args.model, "adapter": args.adapter,
          "dtype": str(next(model.parameters()).dtype), "decoding": "greedy, EOS disabled, batch 1",
          "kv_bytes_per_token": bytes_per_tok, "rows": rows}, args.out)


if __name__ == "__main__":
    main()
