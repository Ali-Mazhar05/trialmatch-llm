"""Model selection on VALIDATION only: evaluate every saved checkpoint of a run
on val and pick the best by overall macro-F1 (tie-break: evidence micro-F1).

    python scripts/select_best.py --run runs/qwen15_lora
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs" / "qwen15_lora"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--batch_size", default="8")
    args = ap.parse_args()
    run = Path(args.run)
    ckpts = sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    rows = []
    for c in ckpts:
        name = f"{run.name}_{c.name}"
        mpath = ROOT / "results" / "val" / name / "metrics.json"
        if not mpath.exists():
            subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--model", args.model,
                            "--adapter", str(c), "--name", name, "--split", "val",
                            "--batch_size", args.batch_size], check=True)
        m = json.loads(mpath.read_text())
        rows.append({"checkpoint": str(c), "name": name,
                     "val_macro_f1": m["classification"]["overall"]["macro_f1"],
                     "val_accuracy": m["classification"]["overall"]["accuracy"],
                     "val_evidence_micro_f1": m["evidence"]["overall"]["all_rows"]["micro"]["f1"]})
    best = max(rows, key=lambda r: (r["val_macro_f1"], r["val_evidence_micro_f1"]))
    out = {"criterion": "max val overall macro-F1, tie-break val evidence micro-F1", "candidates": rows, "best": best}
    (run / "selection.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
