"""TrialMatch-LLM browser application (rubric G1-G4).

Start with start.bat / start.sh (see README). Environment variables:
  TM_ADAPTER        local adapter folder or Hugging Face repo id of the fine-tuned LoRA
  TM_DRAFT_ADAPTER  optional LoRA for the 0.5B draft model
  TM_DEVICE         cpu | cuda (default: auto)
  TM_PORT           default 7860
"""
import html
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from trialmatch.service import (BENCH_PATHS, DISCLAIMER, MODES, InputError, Models, Settings,  # noqa: E402
                                attention_check, export_files, live_benchmark, screen)

EXPORT_DIR = Path(tempfile.gettempdir()) / "trialmatch_reports"

# --------------------------------------------------------------------- examples
def load_examples():
    """Example patients/trials from the VALIDATION split (criteria text only;
    expert labels are never shown to or used by the model)."""
    p = ROOT / "data" / "processed" / "val.jsonl"
    if not p.exists():
        return {}
    df = pd.read_json(p, lines=True)
    ex = {}
    for (pid, tid), g in df.groupby(["patient_id", "trial_id"]):
        if len(ex) >= 12:
            break
        ex[f"{pid} x {tid} ({len(g)} criteria)"] = {
            "note": g.note.iloc[0], "trial": f"{tid}: {g.trial_title.iloc[0]}",
            "inc": "\n".join(g[g.criterion_type == "inclusion"].criterion_text),
            "exc": "\n".join(g[g.criterion_type == "exclusion"].criterion_text)}
    return ex


EXAMPLES = load_examples()
UNSEEN = {
    "note": ("A 67-year-old man with a 10-year history of type 2 diabetes presents for follow-up. "
             "His HbA1c is 8.9% despite metformin 1000 mg twice daily. He has stage 3 chronic kidney disease "
             "with an eGFR of 42 mL/min. He had a myocardial infarction two years ago. "
             "He does not smoke. He is not taking insulin."),
    "inc": "Adults aged 40-80 years with type 2 diabetes\nHbA1c between 7.5% and 10%\nCurrently treated with metformin",
    "exc": "Current insulin therapy\neGFR below 30 mL/min\nPregnancy\nHistory of pancreatitis",
}

# ----------------------------------------------------------------------- models
print("Loading models... (first start downloads ~3.5 GB from Hugging Face)")
MODELS = Models(Settings()).load()
for s in MODELS.status:
    print(s)


def status_md():
    return "\n".join(f"- {html.escape(s)}" for s in MODELS.status)


# --------------------------------------------------------------------- handlers
def _err(e):
    if isinstance(e, InputError):
        raise gr.Error(str(e))
    traceback.print_exc()
    raise gr.Error(f"Internal error ({type(e).__name__}): {e}. The app is still running; check inputs and retry.")


def note_markdown(rep, highlight=None):
    cited = set(highlight or [])
    lines = []
    for s in rep["patient_sentences"]:
        t = html.escape(s["text"])
        lines.append(f"<mark>**[{s['id']}]** {t}</mark>" if s["id"] in cited else f"[{s['id']}] {t}")
    return "  \n".join(lines)


def run_screen(note, trial, inc, exc, mode, progress=gr.Progress()):
    try:
        rep = screen(MODELS, note, inc, exc, mode, trial, progress=progress)
    except Exception as e:  # noqa: BLE001
        _err(e)
    df = pd.DataFrame([{
        "#": r["index"], "type": r["criterion_type"], "criterion": r["criterion_text"], "prediction": r["label"],
        "evidence IDs": ", ".join(map(str, r["evidence_ids"])) or "-",
        "evidence text": " | ".join(f"[{k}] {v}" for k, v in r["evidence_text"].items()) or "(none cited)",
        "missing info": "yes" if r["missing_information"] else "",
        "decode s": round(r["timing"]["decode_s"], 2)} for r in rep["criteria_results"]])
    o = rep["overall"]
    verdict = (f"## Preliminary outcome: {o['verdict']}\n{o['rule']}  \n"
               f"{o['n_criteria']} criteria · {o['n_failing']} failing · {o['n_unknown']} missing information · "
               f"{rep['provenance']['total_seconds']} s ({rep['provenance']['decoding_mode']})  \n"
               "_Rule-based aggregation of criterion-level predictions; not a ground-truth label._")
    if rep["warnings"]:
        verdict += "\n\n**Warning:** " + " ".join(rep["warnings"])
    h, j = export_files(rep, str(EXPORT_DIR))
    choices = [f"{r['index']}. [{r['criterion_type']}] {r['criterion_text'][:80]}" for r in rep["criteria_results"]]
    first = detail(choices[0], rep) if choices else ("", "")
    return (rep, df, verdict, [h, j], gr.update(choices=choices, value=choices[0] if choices else None),
            first[0], first[1])


def detail(choice, rep):
    if not rep or not choice:
        return "", ""
    idx = int(choice.split(".")[0]) - 1
    r = rep["criteria_results"][idx]
    md = (f"### {r['criterion_type'].title()} criterion {r['index']}\n> {html.escape(r['criterion_text'])}\n\n"
          f"**Prediction:** `{r['label']}`  \n**Cited sentence IDs:** {r['evidence_ids'] or 'none'}  \n"
          f"**Explanation:** {html.escape(r['explanation'])}  \n"
          f"**Strict JSON output:** {'yes' if r['parse_ok'] else 'no (fallback parser used)'}  \n"
          f"**Raw model output:** `{html.escape(r['raw_output'][:300])}`  \n"
          f"**Timing:** {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r['timing'].items()})}")
    return md, note_markdown(rep, r["evidence_ids"])


def load_example(name):
    if name == "Unseen example (not from dataset)":
        e = UNSEEN
        return e["note"], "Synthetic diabetes trial", e["inc"], e["exc"]
    e = EXAMPLES.get(name)
    if not e:
        raise gr.Error("Example not found.")
    return e["note"], e["trial"], e["inc"], e["exc"]


def run_bench(note, inc, exc, which, max_new, paths, k, progress=gr.Progress()):
    try:
        crit_lines = [l for l in (inc if which == "inclusion" else exc).splitlines() if l.strip()]
        if not crit_lines:
            raise InputError(f"No {which} criterion in the Screening tab to benchmark.")
        rows = live_benchmark(MODELS, note, crit_lines[0], which, int(max_new), paths, int(k), progress=progress)
    except Exception as e:  # noqa: BLE001
        _err(e)
    df = pd.DataFrame(rows)
    plot = df[["path", "end_to_end_tok_per_s"]].copy()
    ok = all(r["tokens_identical_to_first_path"] for r in rows)
    msg = (f"**Benchmarked criterion:** _{html.escape(crit_lines[0][:150])}_  \n"
           f"**Token-level agreement across all selected paths:** {'YES' if ok else 'NO - see table'}")
    return df, plot, msg


def run_attn(seq, block):
    try:
        return pd.DataFrame(attention_check(int(seq), int(block)))
    except Exception as e:  # noqa: BLE001
        _err(e)


def results_tables():
    rows = []
    for split in ["val", "test"]:
        for name in ["base", "ft_best"]:
            p = ROOT / "results" / split / name / "metrics.json"
            if p.exists():
                m = json.loads(p.read_text())
                c, ev = m["classification"], m["evidence"]["overall"]["all_rows"]
                rows.append({"split": split, "model": name, "accuracy": round(c["overall"]["accuracy"], 3),
                             "macro-F1": round(c["overall"]["macro_f1"], 3),
                             "inclusion macro-F1": round(c["inclusion"].get("macro_f1", 0), 3),
                             "exclusion macro-F1": round(c["exclusion"].get("macro_f1", 0), 3),
                             "evidence micro-F1": round(ev["micro"]["f1"], 3),
                             "evidence macro-F1": round(ev["macro"]["f1"], 3)})
    kv = ROOT / "results" / "bench" / "kv_cache.json"
    kv_df = pd.DataFrame(json.loads(kv.read_text())["rows"])[
        ["input_len", "output_len", "path", "tokens_equal_reference", "prefill_s", "decode_s", "decode_tok_s",
         "peak_mem_mb"]] if kv.exists() else pd.DataFrame()
    return pd.DataFrame(rows), kv_df


# --------------------------------------------------------------------------- UI
CSS = """
.disclaimer {background:#fff8c5;border:1px solid #d4a72c;border-radius:8px;padding:10px 14px;font-weight:600;color:#3b2300}
mark {background:#ddf4ff;color:inherit;padding:0 2px}
"""

with gr.Blocks(title="TrialMatch-LLM") as demo:
    gr.HTML(f"<div class='disclaimer'>{html.escape(DISCLAIMER)}</div>")
    gr.Markdown("# TrialMatch-LLM: criterion-level clinical-trial eligibility screening")
    with gr.Accordion("Model status", open=False):
        gr.Markdown(status_md())
    report_state = gr.State(None)

    with gr.Tab("Screening"):
        with gr.Row():
            ex_dd = gr.Dropdown(["Unseen example (not from dataset)"] + list(EXAMPLES), label="Load an example",
                                value=None)
            ex_btn = gr.Button("Load example")
        with gr.Row():
            with gr.Column(scale=1):
                note = gr.Textbox(label="Patient note (free text, or numbered '0. ...' sentences)", lines=12)
                trial = gr.Textbox(label="Trial ID / title (optional)")
            with gr.Column(scale=1):
                inc = gr.Textbox(label="Inclusion criteria (one per line)", lines=6)
                exc = gr.Textbox(label="Exclusion criteria (one per line)", lines=6)
        mode = gr.Radio(list(MODES), value=list(MODES)[0], label="Inference mode")
        run_btn = gr.Button("Run screening", variant="primary")
        verdict = gr.Markdown()
        table = gr.Dataframe(label="Criterion-level results", wrap=True)
        with gr.Row():
            with gr.Column():
                crit_dd = gr.Dropdown(label="Inspect a criterion", choices=[])
                crit_md = gr.Markdown()
            with gr.Column():
                gr.Markdown("**Patient note: sentences cited for the selected criterion are highlighted**")
                note_md = gr.Markdown()
        files = gr.File(label="Download screening report (HTML for printing/PDF, JSON with full provenance)",
                        file_count="multiple")

    with gr.Tab("Inference benchmark"):
        gr.Markdown("Runs the selected decoding paths **live** on the first criterion of the chosen type from the "
                    "Screening tab (same prompt for every path, greedy decoding, 1 warm-up run each). Token "
                    "agreement is checked against the first selected path.")
        with gr.Row():
            which = gr.Radio(["inclusion", "exclusion"], value="inclusion", label="Criterion type to benchmark")
            max_new = gr.Slider(8, 128, value=32, step=8, label="Max new tokens")
            k = gr.Slider(1, 8, value=4, step=1, label="Speculative draft length k")
        paths = gr.CheckboxGroup(BENCH_PATHS, value=BENCH_PATHS[1:3] + [BENCH_PATHS[4]], label="Decoding paths")
        bench_btn = gr.Button("Run benchmark", variant="primary")
        bench_msg = gr.Markdown()
        bench_df = gr.Dataframe(wrap=True)
        bench_plot = gr.BarPlot(x="path", y="end_to_end_tok_per_s", title="End-to-end tokens/second",
                                x_label_angle=-20, height=300)
        gr.Markdown("### Attention kernel check (reference vs tiled online-softmax vs fused SDPA)")
        with gr.Row():
            seq = gr.Slider(16, 2048, value=512, step=16, label="Sequence length")
            block = gr.Slider(16, 512, value=64, step=16, label="Tile size")
            attn_btn = gr.Button("Run attention check")
        attn_df = gr.Dataframe()

    with gr.Tab("Offline results"):
        gr.Markdown("Metrics computed by `scripts/evaluate.py` and `scripts/bench_kv.py` (loaded from `results/`).")
        res_btn = gr.Button("Refresh")
        res_df = gr.Dataframe(label="Validation / held-out test metrics")
        kv_df = gr.Dataframe(label="KV-cache benchmark (GPU run)")

    ex_btn.click(load_example, ex_dd, [note, trial, inc, exc])
    run_btn.click(run_screen, [note, trial, inc, exc, mode],
                  [report_state, table, verdict, files, crit_dd, crit_md, note_md])
    crit_dd.change(detail, [crit_dd, report_state], [crit_md, note_md])
    bench_btn.click(run_bench, [note, inc, exc, which, max_new, paths, k], [bench_df, bench_plot, bench_msg])
    attn_btn.click(run_attn, [seq, block], attn_df)
    res_btn.click(results_tables, None, [res_df, kv_df])
    demo.load(results_tables, None, [res_df, kv_df])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("TM_HOST", "127.0.0.1"), server_port=int(os.environ.get("TM_PORT", 7860)),
        inbrowser=True, css=CSS, allowed_paths=[str(EXPORT_DIR)], show_error=True)
