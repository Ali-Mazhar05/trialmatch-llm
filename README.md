# TrialMatch-LLM

Criterion-level clinical-trial eligibility screening with a fine-tuned open-weight decoder
(Qwen2.5-1.5B-Instruct + LoRA), evidence-sentence citation, a from-scratch KV-cached decoder,
tiled online-softmax attention, greedy speculative decoding, and a browser application.

> **Educational screening prototype — not a clinical enrollment decision tool.**

| | |
|---|---|
| Dataset | [`ncbi/TrialGPT-Criterion-Annotations`](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations) (only) |
| Split | patient-grouped 37/8/8, `np.random.default_rng(42)` → [`splits/manifest.json`](splits/manifest.json) |
| Target model | `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0) + LoRA r=16 |
| Draft model | `Qwen/Qwen2.5-0.5B-Instruct` (same tokenizer, 151,936-token vocabulary) |
| Fine-tuned adapter | [`AliMaz30/trialmatch-qwen2.5-1.5b-lora`](https://huggingface.co/AliMaz30/trialmatch-qwen2.5-1.5b-lora) (downloaded automatically by the app) |
| Draft adapter | [`AliMaz30/trialmatch-qwen2.5-0.5b-draft-lora`](https://huggingface.co/AliMaz30/trialmatch-qwen2.5-0.5b-draft-lora) |
| Held-out test results | accuracy 0.228 → **0.694**, macro-F1 0.192 → **0.386**, evidence micro-F1 0.236 → **0.546** (base → fine-tuned) |
| Demo video | _link_ |
| Report | [`report/main.pdf`](report/) |

## 1. Quick start: run the application (no coding needed)

**Windows:** double-click `start.bat`. **Linux/macOS:** `./start.sh`.

The first run creates `.venv`, installs CPU PyTorch and the requirements, then downloads the base model,
the draft model and the fine-tuned adapter (about 3.5 GB in total). The app then opens at <http://127.0.0.1:7860>.
Requirements: Python 3.10–3.13, about 8 GB of free RAM, and internet access on the first run. A GPU is optional.

The adapter is located automatically in this order:
1. `runs/qwen15_lora/best_adapter/` if it exists locally (for example, unzipped from the Kaggle output);
2. otherwise the Hugging Face repo id in `adapter.txt`;
3. or set it explicitly: `set TM_ADAPTER=<user>/<repo>` (Windows) or `export TM_ADAPTER=...` (Linux/macOS).

If no adapter can be loaded, the app still starts with the base model and shows a warning under *Model status*.
Other variables: `TM_DEVICE=cpu|cuda`, `TM_PORT`, `TM_DRAFT_ADAPTER`.

**App tabs**
* **Screening**: paste a patient note (free text or numbered sentences) and the inclusion/exclusion criteria
  (one per line), or load an example. Pick an inference mode and run. For each criterion the app shows the
  label, cited sentence IDs, the evidence text, an explanation, and a *missing information* flag. It also shows
  a rule-based preliminary outcome. You can click any criterion to highlight its evidence in the note, and
  download the HTML report (printable to PDF) and the JSON report with full provenance.
* **Inference benchmark**: runs no-cache vs KV-cache vs tiled-attention vs speculative decoding *live* on the
  current input. It reports prefill time, decode time, tokens/s, memory, acceptance rate and token agreement.
  It also has a live attention-kernel check (error vs fp64 and timing).
* **Offline results**: shows validation/test metrics and the GPU KV-cache benchmark, read from `results/`.

## 2. Reproducing the experiments

All GPU work runs in one notebook: [`kaggle/trialmatch_pipeline.ipynb`](kaggle/trialmatch_pipeline.ipynb)
(Kaggle, 1× T4, Internet ON). Set `REPO_URL` and the HF repo names in the first cell, then *Run all*.
The individual steps are:

```bash
pip install -r requirements.txt
python scripts/prepare_data.py                      # M1: split, manifest, leakage + truncation audit
python -m pytest -q                                 # unit tests (attention, engine, cache, speculative, metrics)
python scripts/evaluate.py --name base --split val  # before fine-tuning
python scripts/train_lora.py --out runs/qwen15_lora # LoRA SFT on train split only
python scripts/select_best.py --run runs/qwen15_lora          # choose checkpoint on VAL macro-F1
python scripts/evaluate.py --name base    --split test --final_test
python scripts/evaluate.py --name ft_best --split test --final_test --adapter runs/qwen15_lora/best_adapter
python scripts/error_analysis.py --pred results/test/ft_best/predictions.jsonl
python scripts/bench_kv.py --adapter runs/qwen15_lora/best_adapter
python scripts/bench_attention.py
python scripts/bench_speculative.py --adapter runs/qwen15_lora/best_adapter --tag base_draft
python scripts/make_figures.py
```

The test split can only be evaluated with `--final_test`. All model, prompt and threshold choices are made on validation.

## 3. Repository map

| Path | Purpose |
|---|---|
| `trialmatch/data.py` | loading, removal of the GPT-4 / `training` columns, mandated split, leakage checks, sentence IDs |
| `trialmatch/prompting.py` | prompt (patient prefix + criterion suffix), JSON output schema, truncation, robust parser |
| `trialmatch/metrics.py` | accuracy, macro-F1, per-class P/R/F1, confusion matrices, evidence micro/macro P/R/F1 |
| `trialmatch/engine.py` | **from-scratch Qwen2 forward pass + own KV cache** (RoPE, GQA, RMSNorm, SwiGLU) |
| `trialmatch/attention.py` | reference SDPA, **tiled online-softmax attention**, bottom-right causal masks |
| `trialmatch/decoding.py` | no-cache reference, explicit HF cache loop, engine KV loop, **speculative decoding** |
| `trialmatch/predictor.py` | evidence-grounded explanations, rule-based aggregation, batched evaluation generation |
| `trialmatch/service.py`, `app/app.py` | application service layer + Gradio UI |
| `scripts/` | data prep, training, evaluation, selection, error analysis, benchmarks, figures |
| `tests/` | 41 unit tests (engine logits vs HF, token equality across all decoding paths, speculative == greedy, ...) |
| `results/` | metrics, predictions, benchmark logs (JSON) |
| `report/` | IEEE LaTeX paper |

## 4. Rubric evidence map

| Code | Where the evidence is |
|---|---|
| A1–A3 | report §II Related Work, §III-C model choice table, §VI design decisions & failures |
| B1 | `splits/manifest.json`, `trialmatch/data.py::make_split/check_no_leakage`, `tests/test_data_metrics.py` |
| B2 | `trialmatch/data.py::sanitize` (GPT-4 columns dropped), `prompting.py`, truncation audit in manifest |
| B3 | `scripts/train_lora.py`, `runs/*/run_config.json`, `results/train_summary.json`, HF adapter link |
| B4 | `results/{val,test}/{base,ft_best}/metrics.json` (identical rows + greedy decoding) |
| C1–C3 | `trialmatch/metrics.py`, `results/test/ft_best/metrics.json` |
| C4 | `results/test/ft_best/error_analysis.md`, explanations in `predictions.jsonl` |
| D1–D3 | `trialmatch/engine.py`, `trialmatch/decoding.py`, `tests/test_engine.py`, `results/bench/kv_cache.json` |
| E1–E3 | `trialmatch/attention.py`, `tests/test_attention.py`, `results/bench/attention.json` |
| F1–F3 | `trialmatch/decoding.py::speculative_greedy`, `results/bench/speculative_*.json` |
| G1–G4 | `start.bat`/`start.sh`, `app/app.py`, `trialmatch/service.py` (validation, export, provenance) |
| H1–H2 | `report/`, this README, demo video |

## 5. Troubleshooting
* *Slow on CPU*: a 1.5B model on a laptop CPU takes about 5–10 s per criterion. "No cache" mode is deliberately slow.
* *Adapter warning*: check `TM_ADAPTER` / `adapter.txt`.
* *Port busy*: `set TM_PORT=7861`.

## 6. AI-assistance acknowledgement
An AI coding assistant (Claude) helped scaffold code and documentation. All components were reviewed,
tested (`pytest`) and run by the author. See the report's AI-use appendix.
