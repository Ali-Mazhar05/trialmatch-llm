"""Evaluate a (base or fine-tuned) model on val or test (rubric B4, C1-C3).

Decoding: greedy (do_sample=False), max_new_tokens=64, identical prompt for the
base and fine-tuned model. The TEST split requires --final_test so it cannot be
used accidentally during model selection.

    python scripts/evaluate.py --name base      --split val
    python scripts/evaluate.py --name ft_epoch3 --split val --adapter runs/qwen15_lora/checkpoint-114
    python scripts/evaluate.py --name ft_final  --split test --adapter <best> --final_test
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from trialmatch.data import load_processed, split_note  # noqa: E402
from trialmatch.metrics import full_report  # noqa: E402
from trialmatch.model_io import load_model, load_tokenizer, resolve_adapter  # noqa: E402
from trialmatch.predictor import build_explanation, generate_hf  # noqa: E402
from trialmatch.prompting import encode_prompt, parse_output  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final_test", action="store_true", help="required to touch the held-out test split")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--limit", type=int, default=None, help="debug: evaluate first N rows only")
    args = ap.parse_args()
    if args.split == "test" and not args.final_test:
        sys.exit("Refusing to evaluate on TEST without --final_test (test set stays untouched until final evaluation).")

    out = ROOT / "results" / args.split / args.name
    out.mkdir(parents=True, exist_ok=True)
    df = load_processed(ROOT / "data" / "processed" / f"{args.split}.jsonl")
    if args.limit:
        df = df.head(args.limit)

    tok = load_tokenizer(args.model)
    model = load_model(args.model, resolve_adapter(args.adapter), dtype=args.dtype)
    sents = [split_note(n) for n in df.note]
    encs = [encode_prompt(tok, s, t, c) for s, t, c in zip(sents, df.criterion_type, df.criterion_text)]

    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    texts = generate_hf(model, tok, encs, args.max_new_tokens, args.batch_size)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    records = []
    for (_, r), s, txt in zip(df.iterrows(), sents, texts):
        sd = dict(s)
        p = parse_output(txt, r.criterion_type, set(sd))
        records.append({
            "annotation_id": int(r.annotation_id), "patient_id": r.patient_id, "trial_id": r.trial_id,
            "criterion_type": r.criterion_type, "criterion_text": r.criterion_text,
            "gold_label": r.expert_eligibility, "pred_label": p.label,
            "gold_evidence": list(r.expert_sentences), "pred_evidence": p.evidence_ids,
            "evidence_text": {i: sd[i] for i in p.evidence_ids},
            "explanation": build_explanation(p.label, r.criterion_type, p.evidence_ids, sd),
            "parse_ok": p.parse_ok, "raw_output": txt,
        })
    rep = full_report(records)
    rep["run"] = {"model": args.model, "adapter": args.adapter, "split": args.split, "n": len(records),
                  "decoding": {"strategy": "greedy", "max_new_tokens": args.max_new_tokens,
                               "batch_size": args.batch_size, "padding": "left"},
                  "dtype": str(next(model.parameters()).dtype), "seconds": elapsed,
                  "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None}
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "metrics.json").write_text(json.dumps(rep, indent=2))
    c = rep["classification"]
    print(f"[{args.name}/{args.split}] acc={c['overall']['accuracy']:.3f} macroF1={c['overall']['macro_f1']:.3f} "
          f"inc_acc={c['inclusion'].get('accuracy', 0):.3f} exc_acc={c['exclusion'].get('accuracy', 0):.3f} "
          f"evid_microF1={rep['evidence']['overall']['all_rows']['micro']['f1']:.3f} parse_ok={rep['parse_ok_rate']:.3f} "
          f"time={elapsed:.1f}s")


if __name__ == "__main__":
    main()
