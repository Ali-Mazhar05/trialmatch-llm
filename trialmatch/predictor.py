"""Criterion-level prediction, traceable explanations and rule-based aggregation."""
from __future__ import annotations

import torch

from .prompting import INVALID, Encoded

LABEL_MEANING = {
    "included": "the patient meets this inclusion criterion",
    "not included": "the patient does not meet this inclusion criterion",
    "excluded": "the patient meets this exclusion criterion (would be excluded)",
    "not excluded": "the patient does not meet this exclusion criterion",
    "not enough information": "the note does not contain enough information to decide",
    "not applicable": "the criterion does not apply to this patient",
    INVALID: "the model output could not be parsed into an allowed label",
}


def build_explanation(label: str, criterion_type: str, evidence_ids: list[int], sentences: dict[int, str]) -> str:
    """Deterministic, evidence-grounded explanation.

    It only restates the predicted label and quotes the cited patient sentences,
    so it cannot introduce facts that are not in the record. Absence of evidence
    is stated explicitly and never presented as proof of (in)eligibility.
    """
    parts = [f"Prediction for this {criterion_type} criterion: '{label}' -- {LABEL_MEANING.get(label, label)}."]
    cited = [i for i in evidence_ids if i in sentences]
    if cited:
        quotes = " ".join(f"[{i}] \"{sentences[i]}\"" for i in cited)
        parts.append(f"Supporting patient sentences: {quotes}")
    else:
        parts.append("No patient sentence was cited as evidence.")
        if label in {"not included", "excluded", "included", "not excluded"}:
            parts.append("Caution: a definite label without cited evidence is unsupported and should be reviewed.")
    if label == "not enough information":
        parts.append("Missing information is not treated as proof of eligibility or ineligibility.")
    return " ".join(parts)


def aggregate(results: list[dict]) -> dict:
    """Explicit rule-based patient-trial screening outcome (NOT a ground-truth label).

    R1 any inclusion 'not included' or exclusion 'excluded'   -> LIKELY INELIGIBLE
    R2 else any 'not enough information' or 'invalid'          -> NEEDS MANUAL REVIEW
    R3 else (all inclusion included/n.a., exclusion not excluded/n.a.) -> LIKELY ELIGIBLE
    """
    failing = [r for r in results if (r["criterion_type"] == "inclusion" and r["label"] == "not included")
               or (r["criterion_type"] == "exclusion" and r["label"] == "excluded")]
    unknown = [r for r in results if r["label"] in {"not enough information", INVALID}]
    if failing:
        verdict, rule = "LIKELY INELIGIBLE", "R1: at least one inclusion criterion not met or exclusion criterion met"
    elif unknown:
        verdict, rule = "NEEDS MANUAL REVIEW", "R2: no criterion failed, but some criteria lack information"
    else:
        verdict, rule = "LIKELY ELIGIBLE", "R3: all criteria satisfied or not applicable"
    return {"verdict": verdict, "rule": rule, "n_failing": len(failing), "n_unknown": len(unknown),
            "n_criteria": len(results)}


@torch.no_grad()
def generate_hf(model, tok, encs: list[Encoded], max_new_tokens: int = 64, batch_size: int = 8) -> list[str]:
    """Greedy batched generation with HF generate() -- used for metric evaluation.

    Left padding keeps every prompt's last token aligned at the right edge.
    """
    device = next(model.parameters()).device
    outs: list[str] = []
    for s in range(0, len(encs), batch_size):
        chunk = [e.input_ids for e in encs[s:s + batch_size]]
        L = max(len(c) for c in chunk)
        ids = torch.full((len(chunk), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for i, c in enumerate(chunk):
            ids[i, L - len(c):] = torch.tensor(c)
            att[i, L - len(c):] = 1
        gen = model.generate(input_ids=ids.to(device), attention_mask=att.to(device), max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
        for row in gen[:, L:]:
            outs.append(tok.decode(row, skip_special_tokens=False).replace(tok.pad_token, ""))
    return outs
