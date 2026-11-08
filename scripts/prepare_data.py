"""M1: download dataset, drop leaky columns, apply the mandated split, write
processed JSONL files, the split manifest, and tokenisation/truncation stats.

    python scripts/prepare_data.py --tokenizer Qwen/Qwen2.5-1.5B-Instruct
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from trialmatch.data import (  # noqa: E402
    assign_split, check_no_leakage, drop_unusable, load_raw, make_split, manifest_dict, sanitize, save_json, split_note,
)
from trialmatch.prompting import encode_prompt, target_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max_prompt_tokens", type=int, default=1800)
    ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
    args = ap.parse_args()

    raw = load_raw()
    print("raw columns:", list(raw.columns))
    df = sanitize(raw)
    print("kept columns:", list(df.columns))

    # sanity: gold evidence IDs must exist in the note
    bad = 0
    for _, r in df.iterrows():
        ids = {i for i, _ in split_note(r.note)}
        if not set(r.expert_sentences) <= ids:
            bad += 1
    print(f"rows whose gold evidence IDs are not in the note: {bad}")

    split = make_split(df.patient_id)  # split on all 53 patients first
    check_no_leakage(df, split)
    df, dropped_rows = drop_unusable(df)
    print("dropped unusable rows:", dropped_rows)
    df = assign_split(df, split)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ["train", "val", "test"]:
        df[df.split == name].to_json(out / f"{name}.jsonl", orient="records", lines=True)
    manifest = manifest_dict(df, split)
    manifest["dropped_rows"] = {"reason": "empty criterion_text", "rows": dropped_rows}

    # tokenisation statistics + truncation audit
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    stats = {}
    for name in ["train", "val", "test"]:
        sub = df[df.split == name]
        lens, tgt_lens, trunc, gold_lost = [], [], 0, 0
        for _, r in sub.iterrows():
            enc = encode_prompt(tok, split_note(r.note), r.criterion_type, r.criterion_text, args.max_prompt_tokens)
            lens.append(len(enc.input_ids))
            tgt_lens.append(len(tok(target_text(r.expert_sentences, r.expert_eligibility), add_special_tokens=False)["input_ids"]))
            if enc.truncated:
                trunc += 1
                gold_lost += bool(set(enc.dropped_sentence_ids) & set(r.expert_sentences))
        lens = np.array(lens)
        stats[name] = {
            "prompt_tokens_mean": float(lens.mean()), "prompt_tokens_p95": float(np.percentile(lens, 95)),
            "prompt_tokens_max": int(lens.max()), "target_tokens_max": int(max(tgt_lens)),
            "n_truncated": trunc, "n_truncated_with_gold_evidence_lost": gold_lost,
        }
    manifest["tokenization"] = {"tokenizer": args.tokenizer, "max_prompt_tokens": args.max_prompt_tokens, "stats": stats}

    save_json(manifest, ROOT / "splits" / "manifest.json")
    for name, s in manifest["splits"].items():
        print(f"{name}: {s['n_patients']} patients, {s['n_annotations']} annotations, labels={s['label_counts']}")
    print("tokenization:", stats)
    print("leakage check passed; manifest -> splits/manifest.json")


if __name__ == "__main__":
    main()
