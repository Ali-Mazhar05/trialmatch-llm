"""Task 5 benchmark (rubric F3): speculative vs ordinary greedy decoding.

Target = fine-tuned Qwen2.5-1.5B (merged LoRA), draft = Qwen2.5-0.5B (base or
its own LoRA). Same tokenizer/vocabulary (151,936 ids), verified at start-up.
Two workloads:
  task : real val screening prompts, EOS enabled (short JSON answers)
  long : the same prompts asked for a free-text rationale, 128 tokens, EOS off
For k in --ks we check token equality with target-only greedy (same engine)
and report acceptance, latency, tokens/s, memory and speed-up.

    python scripts/bench_speculative.py --adapter runs/qwen15_lora/final_adapter
"""
import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from trialmatch.bench_utils import env_info, first_mismatch, repeat, save  # noqa: E402
from trialmatch.data import load_processed, split_note  # noqa: E402
from trialmatch.decoding import engine_greedy, speculative_greedy  # noqa: E402
from trialmatch.engine import QwenEngine  # noqa: E402
from trialmatch.model_io import load_model, load_tokenizer, resolve_adapter  # noqa: E402
from trialmatch.prompting import IM_END, IM_START, encode_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--draft_adapter", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--ks", default="1,2,4,6,8")
    ap.add_argument("--n_prompts", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--long_tokens", type=int, default=128)
    ap.add_argument("--tag", default="base_draft")
    args = ap.parse_args()

    tok = load_tokenizer(args.model)
    dtok = load_tokenizer(args.draft)
    assert tok.get_vocab() == dtok.get_vocab(), "draft/target tokenizers differ"
    target = load_model(args.model, resolve_adapter(args.adapter), dtype=args.dtype)
    draft = load_model(args.draft, resolve_adapter(args.draft_adapter), dtype=args.dtype)
    assert target.config.vocab_size == draft.config.vocab_size
    te, de = QwenEngine(target), QwenEngine(draft)
    eos = {tok.convert_tokens_to_ids(IM_END), tok.eos_token_id}

    val = load_processed(ROOT / "data" / "processed" / "val.jsonl").sample(args.n_prompts, random_state=0)
    task_prompts = [encode_prompt(tok, split_note(r.note), r.criterion_type, r.criterion_text).input_ids
                    for _, r in val.iterrows()]
    long_prompts = [tok(f"{IM_START}user\nSummarise the following patient note for a trial coordinator in plain "
                        f"language.\n{r.note}{IM_END}\n{IM_START}assistant\n", add_special_tokens=False)["input_ids"]
                    for _, r in val.iterrows()]
    workloads = {"task_json_eos": (task_prompts, 64, eos), "long_free_text": (long_prompts, args.long_tokens, None)}

    rows = []
    for wname, (prompts, max_new, stop) in workloads.items():
        for pi, p in enumerate(prompts):
            base_toks, base = repeat(lambda: engine_greedy(te, p, max_new, eos=stop), 1, args.repeats)
            rows.append({"workload": wname, "prompt": pi, "method": "greedy_target", "k": 0, "tokens_equal": True,
                         **base})
            for k in map(int, args.ks.split(",")):
                toks, agg = repeat(lambda: speculative_greedy(te, de, p, max_new, k=k, eos=stop), 1, args.repeats)
                mm = first_mismatch(base_toks, toks)
                agg.pop("accept_hist", None)
                rows.append({"workload": wname, "prompt": pi, "method": "speculative", "k": k,
                             "tokens_equal": mm is None, "first_mismatch": mm,
                             "speedup_vs_greedy": base["total_s"] / agg["total_s"], **agg})
    # summary per workload x k
    summary = []
    for wname in workloads:
        for k in [0] + list(map(int, args.ks.split(","))):
            rs = [r for r in rows if r["workload"] == wname and r["k"] == k]
            s = {"workload": wname, "k": k, "n_prompts": len(rs),
                 "all_tokens_equal": all(r["tokens_equal"] for r in rs),
                 "median_total_s": statistics.median(r["total_s"] for r in rs),
                 "median_e2e_tok_s": statistics.median(r["e2e_tok_s"] for r in rs),
                 "max_peak_mem_mb": max((r["peak_mem_mb"] or 0) for r in rs)}
            if k:
                s["mean_acceptance_rate"] = statistics.mean(r["acceptance_rate"] for r in rs)
                s["mean_tokens_per_target_call"] = statistics.mean(r["tokens_per_target_call"] for r in rs)
                s["median_speedup"] = statistics.median(r["speedup_vs_greedy"] for r in rs)
                s["draft_share_of_time"] = statistics.mean(r["draft_s"] / r["total_s"] for r in rs)
            summary.append(s)
            print(s)
    save({"env": env_info(), "target": args.model, "adapter": args.adapter, "draft": args.draft,
          "draft_adapter": args.draft_adapter, "dtype": str(next(target.parameters()).dtype),
          "summary": summary, "rows": rows},
         ROOT / "results" / "bench" / f"speculative_{args.tag}.json")


if __name__ == "__main__":
    main()
