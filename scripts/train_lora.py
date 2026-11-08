"""M2: LoRA supervised fine-tuning on the TRAIN split only (rubric B3).

Objective: causal-LM cross-entropy on the answer tokens only (prompt tokens are
masked with -100), so the model learns p(JSON answer | patient, criterion).

    python scripts/train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct --out runs/qwen15_lora
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from trialmatch.data import load_processed, split_note  # noqa: E402
from trialmatch.model_io import DTYPES, load_tokenizer, patch_peft_torchao  # noqa: E402
from trialmatch.prompting import encode_prompt, target_text  # noqa: E402


def build_examples(df, tok, max_prompt_tokens):
    exs, n_trunc = [], 0
    for _, r in df.iterrows():
        enc = encode_prompt(tok, split_note(r.note), r.criterion_type, r.criterion_text, max_prompt_tokens)
        n_trunc += enc.truncated
        tgt = tok(target_text(r.expert_sentences, r.expert_eligibility), add_special_tokens=False)["input_ids"]
        ids = enc.input_ids + tgt
        exs.append({"input_ids": ids, "labels": [-100] * len(enc.input_ids) + tgt})
    return exs, n_trunc


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), self.pad_id, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            lab[i, :n] = torch.tensor(b["labels"])
            att[i, :n] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "qwen15_lora"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_prompt_tokens", type=int, default=1800)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--load_dtype", default="fp32", choices=list(DTYPES), help="base weight dtype")
    ap.add_argument("--amp", default="fp16", choices=["fp16", "bf16", "none"])
    ap.add_argument("--qlora", action="store_true", help="load base in 4-bit NF4 (bitsandbytes)")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    patch_peft_torchao()
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    tok = load_tokenizer(args.model)
    train_df = load_processed(Path(args.data) / "train.jsonl")
    val_df = load_processed(Path(args.data) / "val.jsonl")
    train_ex, tr_trunc = build_examples(train_df, tok, args.max_prompt_tokens)
    val_ex, va_trunc = build_examples(val_df, tok, args.max_prompt_tokens)
    print(f"train={len(train_ex)} val={len(val_ex)} truncated train={tr_trunc} val={va_trunc}")

    kw = {"dtype": DTYPES[args.load_dtype]}
    if args.qlora:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)
        kw["dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="sdpa", **kw)
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.grad_ckpt)
    elif args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(","), task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())

    targs = TrainingArguments(
        output_dir=str(out), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, warmup_steps=args.warmup_ratio,  # float in [0,1) = ratio (transformers v5)
        weight_decay=args.weight_decay, lr_scheduler_type="cosine", logging_steps=5,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=None,
        fp16=args.amp == "fp16", bf16=args.amp == "bf16", report_to="none", seed=args.seed,
        gradient_checkpointing=False, remove_unused_columns=False, dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ex, eval_dataset=val_ex,
                      data_collator=Collator(tok.pad_token_id))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    model.save_pretrained(out / "final_adapter")
    tok.save_pretrained(out / "final_adapter")
    summary = {
        "train_seconds": elapsed,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "trainable_params": n_train, "total_params": n_all,
        "n_train": len(train_ex), "n_val": len(val_ex), "truncated_train": tr_trunc, "truncated_val": va_trunc,
        "log_history": trainer.state.log_history,
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "log_history"}, indent=2))


if __name__ == "__main__":
    main()
