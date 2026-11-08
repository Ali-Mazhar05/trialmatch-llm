"""Turn logged results into report figures (report/figures/*.pdf|png) and a
LaTeX-ready summary table file (report/generated_tables.tex). Missing inputs
are skipped so it can be run at any milestone.

    python scripts/make_figures.py --run runs/qwen15_lora
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
plt.rcParams.update({"font.size": 8, "figure.dpi": 200, "savefig.bbox": "tight"})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf"); fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("figure", name)


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def training_curves(run):
    s = load(Path(run) / "train_summary.json")
    if not s:
        return
    h = s["log_history"]
    tr = [(e["epoch"], e["loss"]) for e in h if "loss" in e]
    ev = [(e["epoch"], e["eval_loss"]) for e in h if "eval_loss" in e]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(*zip(*tr), label="train loss", lw=1)
    if ev:
        ax.plot(*zip(*ev), "o-", label="val loss", lw=1)
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy (answer tokens)"); ax.legend(); ax.grid(alpha=.3)
    save(fig, "training_curves")


def confusion(name, split="test"):
    m = load(ROOT / "results" / split / name / "metrics.json")
    if not m:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    for ax, t in zip(axes, ["inclusion", "exclusion"]):
        cm = m["classification"][t]["confusion_matrix"]
        mat = np.array(cm["matrix"])
        ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(len(cm["cols_pred"])), [c.replace("not enough information", "NEI") for c in cm["cols_pred"]],
                      rotation=35, ha="right")
        ax.set_yticks(range(len(cm["rows_gold"])), [c.replace("not enough information", "NEI") for c in cm["rows_gold"]])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, mat[i, j], ha="center", va="center", color="white" if mat[i, j] > mat.max() / 2 else "black")
        ax.set_title(f"{t} ({name})"); ax.set_xlabel("predicted"); ax.set_ylabel("gold")
    save(fig, f"confusion_{split}_{name}")


def kv_plots():
    d = load(ROOT / "results" / "bench" / "kv_cache.json")
    if not d:
        return
    rows = d["rows"]
    paths = sorted({r["path"] for r in rows})
    outs = sorted({r["output_len"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.4))
    big = max(outs)
    for pth in paths:
        rs = sorted([r for r in rows if r["path"] == pth and r["output_len"] == big], key=lambda r: r["input_len"])
        axes[0].plot([r["input_len"] for r in rs], [r["decode_tok_s"] for r in rs], "o-", label=pth, lw=1)
        axes[1].plot([r["input_len"] for r in rs], [r["peak_mem_mb"] or 0 for r in rs], "o-", label=pth, lw=1)
    axes[0].set_yscale("log"); axes[0].set_xlabel("input tokens"); axes[0].set_ylabel(f"decode tok/s (out={big})")
    axes[1].set_xlabel("input tokens"); axes[1].set_ylabel("peak GPU memory (MB)")
    axes[0].legend(fontsize=6); [a.grid(alpha=.3) for a in axes]
    save(fig, "kv_cache_bench")


def attention_plots():
    d = load(ROOT / "results" / "bench" / "attention.json")
    if not d:
        return
    perf = d["performance"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.4))
    for impl in sorted({r["impl"] for r in perf}):
        rs = [r for r in perf if r["impl"] == impl and r["median_ms"]]
        axes[0].plot([r["seq_len"] for r in rs], [r["median_ms"] for r in rs], "o-", label=impl, lw=1)
        mem = [r for r in rs if isinstance(r["peak_extra_mem_mb"], (int, float))]
        if mem:
            axes[1].plot([r["seq_len"] for r in mem], [r["peak_extra_mem_mb"] for r in mem], "o-", label=impl, lw=1)
    for a, yl in zip(axes, ["median time (ms)", "peak extra memory (MB)"]):
        a.set_xscale("log", base=2); a.set_yscale("log"); a.set_xlabel("sequence length"); a.set_ylabel(yl)
        a.grid(alpha=.3); a.legend(fontsize=6)
    save(fig, "attention_bench")


def spec_plots():
    files = sorted((ROOT / "results" / "bench").glob("speculative_*.json"))
    files = [f for f in files if "smoke" not in f.name]
    if not files:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.4))
    for f in files:
        d = json.loads(f.read_text())
        for w in sorted({s["workload"] for s in d["summary"]}):
            ss = [s for s in d["summary"] if s["workload"] == w and s["k"] > 0]
            lab = f"{f.stem.replace('speculative_', '')}/{w}"
            axes[0].plot([s["k"] for s in ss], [s["median_speedup"] for s in ss], "o-", label=lab, lw=1)
            axes[1].plot([s["k"] for s in ss], [s["mean_acceptance_rate"] for s in ss], "o-", label=lab, lw=1)
    axes[0].axhline(1, color="k", lw=.6, ls="--")
    axes[0].set_ylabel("speed-up vs greedy"); axes[1].set_ylabel("draft acceptance rate")
    for a in axes:
        a.set_xlabel("draft length k"); a.grid(alpha=.3); a.legend(fontsize=6)
    save(fig, "speculative_bench")


def tables(names):
    out = []
    for split in ["val", "test"]:
        for n in names:
            m = load(ROOT / "results" / split / n / "metrics.json")
            if not m:
                continue
            c, e = m["classification"], m["evidence"]["overall"]
            out.append(f"{split} & {n.replace('_', '-')} & {c['overall']['accuracy']:.3f} & {c['overall']['macro_f1']:.3f} & "
                       f"{c['inclusion'].get('macro_f1', 0):.3f} & {c['exclusion'].get('macro_f1', 0):.3f} & "
                       f"{e['all_rows']['micro']['f1']:.3f} & {e['all_rows']['macro']['f1']:.3f} & "
                       f"{e['gold_nonempty_rows']['micro']['f1']:.3f} & {m['parse_ok_rate']:.2f} \\\\")
    (ROOT / "report").mkdir(exist_ok=True)
    (ROOT / "report" / "generated_tables.tex").write_text(
        "% split & model & acc & macroF1 & inc-mF1 & exc-mF1 & ev-microF1 & ev-macroF1 & ev-microF1(gold>0) & parse\n"
        + "\n".join(out) + "\n")
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs" / "qwen15_lora"))
    ap.add_argument("--names", default="base,ft_best")
    args = ap.parse_args()
    training_curves(args.run)
    for n in args.names.split(","):
        confusion(n, "test"); confusion(n, "val")
    kv_plots(); attention_plots(); spec_plots(); tables(args.names.split(","))


if __name__ == "__main__":
    main()
