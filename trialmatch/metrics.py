"""Criterion-label and evidence-sentence metrics (rubric C1-C3).

Label metrics: accuracy, macro-F1, per-class P/R/F1 and confusion matrices,
overall and separately for inclusion / exclusion. Unparseable outputs get the
label "invalid" and are always counted as errors (they appear as an extra
predicted column in the confusion matrix).

Evidence metrics (sentence-ID sets):
  * micro: TP/FP/FN summed over all rows, then P/R/F1.
  * macro: P/R/F1 per row, then averaged. Empty-set conventions (stated in the
    report):
        gold = {} and pred = {}  -> P = R = F1 = 1  (correctly cites nothing)
        gold = {} and pred != {} -> P = R = F1 = 0  (cites irrelevant text)
        gold != {} and pred = {} -> P = R = F1 = 0  (misses all evidence)
  * every evidence metric is also reported on the gold-non-empty subset only,
    where none of these conventions are needed, plus counts of the 4 patterns.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .data import LABELS
from .prompting import INVALID


def label_metrics(gold: list[str], pred: list[str], labels: list[str]) -> dict:
    if not gold:
        return {"n": 0}
    present = [l for l in labels if l in gold or l in pred]
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=present, zero_division=0)
    cm_cols = labels + ([INVALID] if INVALID in pred else [])
    return {
        "n": len(gold),
        "accuracy": float(accuracy_score(gold, pred)),
        # macro-F1 over the classes that occur in the gold labels of this subset
        "macro_f1": float(f1_score(gold, pred, labels=[l for l in labels if l in gold], average="macro", zero_division=0)),
        "invalid_rate": float(np.mean([x == INVALID for x in pred])),
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, l in enumerate(present)},
        "confusion_matrix": {"rows_gold": labels, "cols_pred": cm_cols,
                             "matrix": confusion_matrix(gold, pred, labels=cm_cols)[: len(labels)].tolist()},
    }


def classification_report(records: list[dict]) -> dict:
    """records: dicts with criterion_type, gold_label, pred_label."""
    out = {}
    for ctype in ["inclusion", "exclusion"]:
        sub = [r for r in records if r["criterion_type"] == ctype]
        out[ctype] = label_metrics([r["gold_label"] for r in sub], [r["pred_label"] for r in sub], LABELS[ctype])
    # overall: labels are namespaced by type so "not enough information" for
    # inclusion and exclusion stay distinct classes in the 8-way matrix
    g = [f"{r['criterion_type'][:3]}:{r['gold_label']}" for r in records]
    p = [f"{r['criterion_type'][:3]}:{r['pred_label']}" for r in records]
    all_labels = [f"{t[:3]}:{l}" for t in ["inclusion", "exclusion"] for l in LABELS[t]]
    overall = label_metrics(g, [x if not x.endswith(INVALID) else INVALID for x in p], all_labels)
    out["overall"] = overall
    return out


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evidence_metrics(gold_sets: list[set], pred_sets: list[set]) -> dict:
    def compute(pairs):
        if not pairs:
            return {"n": 0}
        TP = FP = FN = 0
        per = []
        for g, p in pairs:
            tp, fp, fn = len(g & p), len(p - g), len(g - p)
            TP, FP, FN = TP + tp, FP + fp, FN + fn
            if not g and not p:
                per.append((1.0, 1.0, 1.0))
            elif not g or not p:
                per.append((0.0, 0.0, 0.0))
            else:
                per.append(_prf(tp, fp, fn))
        mp, mr, mf = _prf(TP, FP, FN)
        per = np.array(per)
        return {
            "n": len(pairs),
            "micro": {"precision": mp, "recall": mr, "f1": mf, "tp": TP, "fp": FP, "fn": FN},
            "macro": {"precision": float(per[:, 0].mean()), "recall": float(per[:, 1].mean()), "f1": float(per[:, 2].mean())},
            "exact_set_match": float(np.mean([g == p for g, p in pairs])),
        }

    pairs = [(set(g), set(p)) for g, p in zip(gold_sets, pred_sets)]
    return {
        "all_rows": compute(pairs),
        "gold_nonempty_rows": compute([(g, p) for g, p in pairs if g]),
        "empty_patterns": {
            "gold_empty_pred_empty": sum(1 for g, p in pairs if not g and not p),
            "gold_empty_pred_nonempty": sum(1 for g, p in pairs if not g and p),
            "gold_nonempty_pred_empty": sum(1 for g, p in pairs if g and not p),
            "gold_nonempty_pred_nonempty": sum(1 for g, p in pairs if g and p),
        },
    }


def full_report(records: list[dict]) -> dict:
    """records also need gold_evidence / pred_evidence (lists of ints)."""
    rep = {"classification": classification_report(records), "evidence": {}}
    for name, sub in [("overall", records),
                      ("inclusion", [r for r in records if r["criterion_type"] == "inclusion"]),
                      ("exclusion", [r for r in records if r["criterion_type"] == "exclusion"])]:
        rep["evidence"][name] = evidence_metrics([r["gold_evidence"] for r in sub], [r["pred_evidence"] for r in sub])
    rep["parse_ok_rate"] = float(np.mean([r.get("parse_ok", True) for r in records])) if records else 0.0
    return rep
