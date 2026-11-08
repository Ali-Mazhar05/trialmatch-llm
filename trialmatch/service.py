"""Application service layer: model management, validated screening runs,
live decoding benchmarks and report export. The Gradio UI (app/app.py) is a
thin layer over this module, so everything here is unit-testable."""
from __future__ import annotations

import hashlib
import html
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import torch

from .data import LABELS, segment_free_text
from .decoding import engine_greedy, engine_no_cache_greedy, no_cache_greedy, speculative_greedy
from .engine import QwenEngine
from .model_io import load_model, load_tokenizer, pick_device, resolve_adapter
from .predictor import aggregate, build_explanation
from .prompting import IM_END, encode_prompt, parse_output

APP_VERSION = "1.0.0"
DISCLAIMER = ("EDUCATIONAL SCREENING PROTOTYPE - NOT A CLINICAL ENROLLMENT DECISION. Outputs are produced by a "
              "research language model, may be wrong, and must be reviewed by qualified study staff.")

MODES = {
    "KV cache (own engine)": "kv",
    "KV cache + tiled online-softmax attention": "kv_tiled",
    "Speculative decoding (0.5B draft)": "spec",
    "No cache (reference, slow)": "nocache",
}

BENCH_PATHS = [
    "HF forward, no cache (reference)",
    "Own engine, no cache",
    "Own engine, KV cache",
    "Own engine, KV cache + tiled attention",
    "Speculative decoding (0.5B draft)",
]

MAX_NOTE_CHARS, MAX_CRITERIA, MAX_CRITERION_CHARS, MAX_PROMPT_TOKENS = 20000, 40, 1000, 1800


class InputError(ValueError):
    """User-facing validation error."""


@dataclass
class Settings:
    base: str = os.environ.get("TM_BASE", "Qwen/Qwen2.5-1.5B-Instruct")
    adapter: str | None = os.environ.get("TM_ADAPTER", "runs/qwen15_lora/best_adapter")
    draft: str = os.environ.get("TM_DRAFT", "Qwen/Qwen2.5-0.5B-Instruct")
    draft_adapter: str | None = os.environ.get("TM_DRAFT_ADAPTER") or None
    device: str = os.environ.get("TM_DEVICE", "")
    max_new_tokens: int = 64
    spec_k: int = 4


@dataclass
class Models:
    settings: Settings
    tok: object = None
    target: object = None
    engine: QwenEngine | None = None
    draft_engine: QwenEngine | None = None
    adapter_loaded: str | None = None
    status: list[str] = field(default_factory=list)
    eos: set[int] = field(default_factory=set)

    def load(self):
        s = self.settings
        dev = pick_device(s.device or None)
        if dev == "cpu":
            torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.tok = load_tokenizer(s.base)
        try:
            path = resolve_adapter(s.adapter)
            if path is None:
                raise FileNotFoundError("no adapter configured")
            self.target = load_model(s.base, path, device=dev)
            self.adapter_loaded = s.adapter
            self.status.append(f"Loaded {s.base} + fine-tuned LoRA adapter ({s.adapter}), merged, on {dev}.")
        except Exception as e:  # app must still start without the adapter
            self.target = load_model(s.base, None, device=dev)
            self.status.append(f"WARNING: adapter '{s.adapter}' unavailable ({e}); running the BASE model "
                               "(not fine-tuned). Set TM_ADAPTER to the adapter folder or Hub repo.")
        self.engine = QwenEngine(self.target)
        self.eos = {self.tok.convert_tokens_to_ids(IM_END), self.tok.eos_token_id}
        return self

    def ensure_draft(self):
        if self.draft_engine is None:
            s = self.settings
            dtok = load_tokenizer(s.draft)
            if dtok.get_vocab() != self.tok.get_vocab():
                raise RuntimeError("draft tokenizer is not compatible with the target")
            d = load_model(s.draft, resolve_adapter(s.draft_adapter), device=str(self.engine.device))
            self.draft_engine = QwenEngine(d)
            self.status.append(f"Loaded draft {s.draft}" + (f" + {s.draft_adapter}" if s.draft_adapter else ""))
        return self.draft_engine


# ------------------------------------------------------------------ validation
def parse_criteria(inclusion_text: str, exclusion_text: str) -> list[dict]:
    crits = []
    for ctype, text in [("inclusion", inclusion_text or ""), ("exclusion", exclusion_text or "")]:
        for line in text.splitlines():
            line = line.strip().lstrip("-*•").strip()
            if len(line) < 3:
                continue
            if len(line) > MAX_CRITERION_CHARS:
                raise InputError(f"One {ctype} criterion is longer than {MAX_CRITERION_CHARS} characters; "
                                 "split it into separate lines.")
            crits.append({"criterion_type": ctype, "criterion_text": line})
    if not crits:
        raise InputError("Enter at least one inclusion or exclusion criterion (one per line).")
    if len(crits) > MAX_CRITERIA:
        raise InputError(f"Too many criteria ({len(crits)}); the limit is {MAX_CRITERIA} per run.")
    return crits


def validate_note(note: str) -> list[tuple[int, str]]:
    if not note or not note.strip():
        raise InputError("The patient note is empty. Paste a clinical note or load an example.")
    if len(note) > MAX_NOTE_CHARS:
        raise InputError(f"The patient note is too long ({len(note)} characters, limit {MAX_NOTE_CHARS}).")
    sents = segment_free_text(note)
    if len(" ".join(s for _, s in sents).split()) < 5:
        raise InputError("The patient note is too short to screen (fewer than 5 words).")
    return sents


# --------------------------------------------------------------------- running
def screen(models: Models, note: str, inclusion: str, exclusion: str, mode_label: str, trial_id: str = "",
           progress=None) -> dict:
    sents = validate_note(note)
    crits = parse_criteria(inclusion, exclusion)
    mode = MODES.get(mode_label)
    if mode is None:
        raise InputError(f"Unknown decoding mode '{mode_label}'.")
    eng, tok = models.engine, models.tok
    eng.set_attention("tiled" if mode == "kv_tiled" else "sdpa", 64, 64)
    draft = models.ensure_draft() if mode == "spec" else None
    sd = dict(sents)
    results, warnings = [], []
    cache = dcache = None
    cached_prefix: list[int] | None = None
    t_start = time.perf_counter()
    try:
        for i, c in enumerate(crits):
            if progress is not None:
                progress((i, len(crits)), desc=f"Criterion {i + 1}/{len(crits)}")
            enc = encode_prompt(tok, sents, c["criterion_type"], c["criterion_text"], MAX_PROMPT_TOKENS)
            if enc.truncated and not warnings:
                warnings.append(f"Note truncated to fit the context: sentences {enc.dropped_sentence_ids} were "
                                "NOT seen by the model; evidence in them cannot be cited.")
            ids = enc.input_ids
            if mode in {"kv", "kv_tiled", "spec"} and cached_prefix != enc.prefix_ids:
                # prefix caching: patient prefix prefilled ONCE, reused by every criterion
                cache = eng.new_cache(len(ids) + 128)
                eng.forward(torch.tensor([enc.prefix_ids], device=eng.device), cache, 0, last_only=True)
                if draft is not None:
                    dcache = draft.new_cache(len(ids) + 128)
                    draft.forward(torch.tensor([enc.prefix_ids], device=draft.device), dcache, 0, last_only=True)
                cached_prefix = enc.prefix_ids
            P = len(enc.prefix_ids)
            if mode in {"kv", "kv_tiled"}:
                out, st = engine_greedy(eng, ids, models.settings.max_new_tokens, models.eos, cache=cache, prefix_len=P)
            elif mode == "spec":
                out, st = speculative_greedy(eng, draft, ids, models.settings.max_new_tokens, k=models.settings.spec_k,
                                             eos=models.eos, t_cache=cache, d_cache=dcache, prefix_len=P)
            else:
                out, st = engine_no_cache_greedy(eng, ids, models.settings.max_new_tokens, models.eos)
            text = tok.decode(out, skip_special_tokens=False)
            p = parse_output(text, c["criterion_type"], set(enc.kept_sentence_ids))
            results.append({
                "index": i + 1, **c, "label": p.label, "evidence_ids": p.evidence_ids,
                "evidence_text": {str(j): sd[j] for j in p.evidence_ids},
                "explanation": build_explanation(p.label, c["criterion_type"], p.evidence_ids, sd),
                "missing_information": p.label == "not enough information",
                "parse_ok": p.parse_ok, "raw_output": text,
                "timing": {k: v for k, v in st.as_dict().items() if k in
                           ("prefill_s", "decode_s", "n_new", "acceptance_rate", "prefix_reused")},
            })
    finally:
        eng.set_attention("sdpa")
    agg = aggregate(results)
    return {
        "disclaimer": DISCLAIMER, "trial_id": trial_id or "(not specified)",
        "patient_sentences": [{"id": i, "text": s} for i, s in sents],
        "criteria_results": results, "overall": agg, "warnings": warnings,
        "provenance": {
            "app_version": APP_VERSION, "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "base_model": models.settings.base, "adapter": models.adapter_loaded or "NONE (base model)",
            "decoding_mode": mode_label, "decoding": "greedy, max_new_tokens=%d" % models.settings.max_new_tokens,
            "speculative_k": models.settings.spec_k if mode == "spec" else None,
            "draft_model": models.settings.draft if mode == "spec" else None,
            "device": str(eng.device), "dtype": str(eng.dtype),
            "input_sha256": hashlib.sha256((note + "\x00" + inclusion + "\x00" + exclusion).encode()).hexdigest()[:16],
            "total_seconds": round(time.perf_counter() - t_start, 2),
        },
    }


# ------------------------------------------------------------------- benchmark
def live_benchmark(models: Models, note: str, criterion: str, ctype: str, max_new: int, paths: list[str],
                   k: int, progress=None) -> list[dict]:
    sents = validate_note(note)
    if not criterion or not criterion.strip():
        raise InputError("Enter a criterion to benchmark.")
    if ctype not in LABELS:
        raise InputError("Criterion type must be inclusion or exclusion.")
    if not paths:
        raise InputError("Select at least one decoding path.")
    unknown = [p for p in paths if p not in BENCH_PATHS]
    if unknown:
        raise InputError(f"Unknown decoding path(s): {unknown}")
    eng, tok = models.engine, models.tok
    ids = encode_prompt(tok, sents, ctype, criterion.strip(), MAX_PROMPT_TOKENS).input_ids

    def tiled():
        eng.set_attention("tiled")
        try:
            return engine_greedy(eng, ids, max_new)
        finally:
            eng.set_attention("sdpa")

    runners = {
        BENCH_PATHS[0]: lambda: no_cache_greedy(models.target, ids, max_new),
        BENCH_PATHS[1]: lambda: engine_no_cache_greedy(eng, ids, max_new),
        BENCH_PATHS[2]: lambda: engine_greedy(eng, ids, max_new),
        BENCH_PATHS[3]: tiled,
        BENCH_PATHS[4]: lambda: speculative_greedy(eng, models.ensure_draft(), ids, max_new, k=k),
    }
    order = [p for p in BENCH_PATHS if p in paths]   # fixed order: reference first
    rows, ref = [], None
    for j, name in enumerate(order):
        if progress is not None:
            progress((j, len(order)), desc=name)
        runners[name]()            # warm-up run (discarded): first-call allocation / lazy init
        toks, st = runners[name]()
        ref = toks if ref is None else ref
        d = st.as_dict()
        rows.append({"path": name + (f" (k={k})" if name == BENCH_PATHS[4] else ""),
                     "prompt_tokens": len(ids), "new_tokens": d["n_new"],
                     "prefill_ms": round(d["prefill_s"] * 1e3, 1), "decode_s": round(d["decode_s"], 3),
                     "decode_tok_per_s": round(d["decode_tok_s"], 2), "end_to_end_tok_per_s": round(d["e2e_tok_s"], 2),
                     "peak_gpu_mem_mb": round(d["peak_mem_mb"], 1) if d["peak_mem_mb"] else None,
                     "acceptance_rate": round(d["acceptance_rate"], 3) if "acceptance_rate" in d else None,
                     "tokens_identical_to_first_path": toks == ref,
                     "output": tok.decode(toks, skip_special_tokens=True)[:120]})
    return rows


def attention_check(seq_len: int, block: int, heads: int = 12, dim: int = 128) -> list[dict]:
    from .attention import reference_attention, sdpa_attention, tiled_attention

    if not (8 <= seq_len <= 4096) or not (8 <= block <= 1024):
        raise InputError("Sequence length must be 8-4096 and block size 8-1024.")
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, heads, seq_len, dim) for _ in range(3))
    gold = reference_attention(q.double(), k.double(), v.double())
    rows = []
    for name, fn in [("reference (materialised softmax)", lambda: reference_attention(q, k, v)),
                     (f"tiled online softmax (block {block})", lambda: tiled_attention(q, k, v, block_q=block, block_k=block)),
                     ("PyTorch fused SDPA", lambda: sdpa_attention(q, k, v))]:
        fn()
        t = time.perf_counter(); out = fn(); dt = time.perf_counter() - t
        err = (out.double() - gold).abs()
        rows.append({"implementation": name, "time_ms": round(dt * 1e3, 2), "max_abs_err_vs_fp64": float(err.max()),
                     "mean_abs_err_vs_fp64": float(err.mean()),
                     "score_matrix_mb": round(heads * seq_len * seq_len * 4 / 2**20, 1) if name.startswith("reference")
                     else round(heads * block * block * 4 / 2**20, 2) if name.startswith("tiled") else None})
    return rows


# ---------------------------------------------------------------------- export
LABEL_COLORS = {"included": "#1a7f37", "not excluded": "#1a7f37", "not included": "#cf222e", "excluded": "#cf222e",
                "not enough information": "#9a6700", "not applicable": "#57606a", "invalid": "#8250df"}


def report_html(rep: dict) -> str:
    e = html.escape
    cited = {int(i) for r in rep["criteria_results"] for i in r["evidence_ids"]}
    sent_html = "".join(
        f"<li value='{s['id']}' class='{'cited' if s['id'] in cited else ''}'>"
        f"<span class='sid'>[{s['id']}]</span> {e(s['text'])}</li>" for s in rep["patient_sentences"])
    rows = []
    for r in rep["criteria_results"]:
        ev = "".join(f"<div class='ev'><b>[{i}]</b> {e(t)}</div>" for i, t in r["evidence_text"].items()) \
            or "<div class='ev none'>No sentence cited.</div>"
        rows.append(
            f"<tr><td>{r['index']}</td><td>{e(r['criterion_type'])}</td><td>{e(r['criterion_text'])}</td>"
            f"<td><span class='lab' style='background:{LABEL_COLORS.get(r['label'], '#57606a')}'>{e(r['label'])}</span>"
            f"{'<div class=miss>missing information</div>' if r['missing_information'] else ''}</td>"
            f"<td>{ev}</td><td class='expl'>{e(r['explanation'])}</td></tr>")
    prov = "".join(f"<tr><th>{e(k)}</th><td>{e(str(v))}</td></tr>" for k, v in rep["provenance"].items())
    warn = "".join(f"<p class='warn'>{e(w)}</p>" for w in rep["warnings"])
    o = rep["overall"]
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Eligibility screening report</title>
<style>
body{{font:14px/1.45 system-ui,Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1f2328;background:#fff}}
.disc{{background:#fff8c5;border:1px solid #d4a72c;padding:10px 14px;border-radius:6px;font-weight:600}}
.warn{{background:#ffebe9;border:1px solid #cf222e;padding:8px 12px;border-radius:6px}}
.verdict{{font-size:20px;font-weight:700;margin:16px 0 4px}} table{{border-collapse:collapse;width:100%;margin:12px 0}}
td,th{{border:1px solid #d0d7de;padding:6px 8px;vertical-align:top;text-align:left}} th{{background:#f6f8fa}}
.lab{{color:#fff;padding:2px 8px;border-radius:10px;white-space:nowrap;font-size:12px}}
.miss{{color:#9a6700;font-size:12px;margin-top:4px}} .ev{{font-size:13px;margin-bottom:4px}} .none{{color:#57606a}}
.expl{{font-size:12px;color:#424a53}} ol{{padding-left:0;list-style:none}} li{{padding:2px 6px;border-radius:4px}}
li.cited{{background:#ddf4ff}} .sid{{color:#57606a;font-family:monospace}}
</style></head><body>
<p class='disc'>{e(rep['disclaimer'])}</p>
<h1>Clinical-trial eligibility screening report</h1>
<p>Trial: <b>{e(rep['trial_id'])}</b></p>{warn}
<div class='verdict'>Preliminary outcome: {e(o['verdict'])}</div>
<p>Aggregation rule applied: {e(o['rule'])}. ({o['n_criteria']} criteria; {o['n_failing']} failing; {o['n_unknown']} with
missing information.) This outcome is a rule-based aggregation of criterion-level model predictions, not a
ground-truth eligibility label.</p>
<h2>Criterion-level results</h2>
<table><tr><th>#</th><th>Type</th><th>Criterion</th><th>Prediction</th><th>Cited evidence (sentence IDs)</th><th>Explanation</th></tr>
{''.join(rows)}</table>
<h2>Patient note (cited sentences highlighted)</h2><ol>{sent_html}</ol>
<h2>Provenance</h2><table>{prov}</table>
</body></html>"""


def export_files(rep: dict, out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stamp = rep["provenance"]["timestamp_utc"].replace(":", "").replace("-", "")
    h = os.path.join(out_dir, f"screening_report_{stamp}.html")
    j = os.path.join(out_dir, f"screening_report_{stamp}.json")
    with open(h, "w", encoding="utf-8") as f:
        f.write(report_html(rep))
    with open(j, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    return h, j
