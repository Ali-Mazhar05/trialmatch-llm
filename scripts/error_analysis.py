"""Rubric C4: categorise errors of a prediction file and dump examples.

    python scripts/error_analysis.py --pred results/test/ft_final/predictions.jsonl

Categories (a row can fall in several):
  E1 invalid / unparseable output
  E2 unsupported certainty: gold 'not enough information', model gives a definite label
  E3 missed available information: gold definite, model says 'not enough information'
  E4 polarity error: included<->not included, excluded<->not excluded
  E5 applicability error: 'not applicable' confused with another label
  E6 definite label with NO cited evidence (unsupported decision)
  E7 evidence cited although gold evidence is empty (spurious citation)
  E8 correct label but wrong / incomplete evidence (evidence F1 < 1, gold non-empty)
  E9 label/evidence contradiction: model cites exactly the gold sentences but the label is wrong
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

DEFINITE = {"included", "not included", "excluded", "not excluded"}
POLAR = {frozenset({"included", "not included"}), frozenset({"excluded", "not excluded"})}


def categorise(r):
    g, p = r["gold_label"], r["pred_label"]
    ge, pe = set(r["gold_evidence"]), set(r["pred_evidence"])
    cats = []
    if p == "invalid":
        cats.append("E1_invalid_output")
    if g == "not enough information" and p in DEFINITE:
        cats.append("E2_unsupported_certainty")
    if g in DEFINITE and p == "not enough information":
        cats.append("E3_missed_available_information")
    if frozenset({g, p}) in POLAR:
        cats.append("E4_polarity_error")
    if (g == "not applicable") != (p == "not applicable"):
        cats.append("E5_applicability_error")
    if p in DEFINITE and not pe:
        cats.append("E6_definite_label_without_evidence")
    if not ge and pe:
        cats.append("E7_spurious_citation")
    if g == p and ge and pe != ge:
        cats.append("E8_correct_label_wrong_evidence")
    if g != p and ge and pe == ge:
        cats.append("E9_right_evidence_wrong_label")
    return cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--n_examples", type=int, default=3)
    args = ap.parse_args()
    rows = [json.loads(l) for l in Path(args.pred).open(encoding="utf-8")]
    by = defaultdict(list)
    for r in rows:
        for c in categorise(r):
            by[c].append(r)
    out = Path(args.pred).with_name("error_analysis.md")
    lines = [f"# Error analysis: {args.pred}", "", f"Rows: {len(rows)}; label errors: "
             f"{sum(r['gold_label'] != r['pred_label'] for r in rows)}", "", "| Category | Count |", "|---|---|"]
    for c in sorted(by):
        lines.append(f"| {c} | {len(by[c])} |")
    for c in sorted(by):
        lines += ["", f"## {c} ({len(by[c])})"]
        for r in by[c][: args.n_examples]:
            lines += [f"- **{r['patient_id']} / {r['trial_id']} ({r['criterion_type']})** — criterion: "
                      f"_{r['criterion_text'][:300]}_",
                      f"  - gold: `{r['gold_label']}` evidence {r['gold_evidence']} | "
                      f"pred: `{r['pred_label']}` evidence {r['pred_evidence']}",
                      f"  - cited text: {json.dumps(r['evidence_text'], ensure_ascii=False)[:400]}",
                      f"  - raw output: `{r['raw_output'][:200]}`"]
    out.write_text("\n".join(lines), encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps({c: [r["annotation_id"] for r in v] for c, v in by.items()},
                                                     indent=2))
    print("\n".join(lines[:20]))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
