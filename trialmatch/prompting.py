"""Prompt format, output schema, tokenisation and robust output parsing.

The prompt is split into a *patient prefix* (system message + numbered patient
sentences) and a *criterion suffix*. Both are tokenised separately and
concatenated; training, evaluation and the application all use this exact
function, so prefix KV caches can be reused across criteria of one patient
without any tokenisation mismatch.

Output schema (the training target):
    {"evidence_ids": [2, 5], "label": "not included"}
Evidence comes first so the label is generated *conditioned on* the cited
sentences. The free-text explanation is built afterwards from the cited
sentences (see predictor.py) -- GPT-4 explanations are never used as targets.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .data import LABELS

SYSTEM = (
    "You are a clinical-trial eligibility screening assistant. You are given a patient note "
    "split into numbered sentences and ONE eligibility criterion from a clinical trial. "
    "Decide how the criterion applies to the patient using ONLY the information in the note. "
    "Cite the IDs of the patient sentences that support your decision (an empty list if no "
    "sentence is relevant). Missing information is not evidence: if the note does not say "
    "enough, answer 'not enough information'. Respond with a single JSON object."
)

IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def format_patient(sentences: list[tuple[int, str]]) -> str:
    return "\n".join(f"[{i}] {s}" for i, s in sentences)


def prefix_text(sentences: list[tuple[int, str]]) -> str:
    return (
        f"{IM_START}system\n{SYSTEM}{IM_END}\n"
        f"{IM_START}user\nPatient note:\n{format_patient(sentences)}\n\n"
    )


def suffix_text(criterion_type: str, criterion_text: str) -> str:
    labels = LABELS[criterion_type]
    return (
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {criterion_text.strip()}\n"
        f"Allowed labels: {', '.join(labels)}\n"
        'Return JSON: {"evidence_ids": [<sentence ids>], "label": "<one allowed label>"}'
        f"{IM_END}\n{IM_START}assistant\n"
    )


def target_text(evidence_ids: list[int], label: str) -> str:
    return json.dumps({"evidence_ids": sorted(int(i) for i in evidence_ids), "label": label}) + IM_END


@dataclass
class Encoded:
    prefix_ids: list[int]
    suffix_ids: list[int]
    kept_sentence_ids: list[int]
    dropped_sentence_ids: list[int] = field(default_factory=list)

    @property
    def input_ids(self) -> list[int]:
        return self.prefix_ids + self.suffix_ids

    @property
    def truncated(self) -> bool:
        return bool(self.dropped_sentence_ids)


def encode_prompt(tok, sentences, criterion_type, criterion_text, max_prompt_tokens: int = 1800) -> Encoded:
    """Tokenise prompt; if too long, drop patient sentences from the END only.

    Truncation never touches the criterion. Dropped sentence IDs are returned so
    callers can log whether gold evidence was removed (documented in the report).
    """
    suffix_ids = tok(suffix_text(criterion_type, criterion_text), add_special_tokens=False)["input_ids"]
    kept = list(sentences)
    dropped: list[int] = []
    while True:
        prefix_ids = tok(prefix_text(kept), add_special_tokens=False)["input_ids"]
        if len(prefix_ids) + len(suffix_ids) <= max_prompt_tokens or len(kept) <= 1:
            break
        dropped.insert(0, kept.pop()[0])
    return Encoded(prefix_ids, suffix_ids, [i for i, _ in kept], dropped)


# --------------------------------------------------------------------- parsing
_JSON_RE = re.compile(r"\{.*?\}", re.S)
INVALID = "invalid"


@dataclass
class Parsed:
    label: str
    evidence_ids: list[int]
    parse_ok: bool       # strict JSON with a valid label
    raw: str


def parse_output(text: str, criterion_type: str, valid_ids: set[int] | None = None) -> Parsed:
    """Parse model output; tolerate prose but never invent a label.

    1. strict JSON object with an allowed label  -> parse_ok=True
    2. otherwise search for an allowed label string (longest first)
    3. otherwise label = "invalid" (counted as an error in all metrics)
    Evidence IDs outside the patient's sentence IDs are discarded.
    """
    text = text.replace(IM_END, "").strip()
    allowed = LABELS[criterion_type]
    label, ids, ok = None, [], False
    for m in _JSON_RE.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        lab = str(obj.get("label", "")).strip().lower()
        raw_ids = obj.get("evidence_ids", obj.get("evidence", []))
        try:
            ids = sorted({int(i) for i in raw_ids})
        except (TypeError, ValueError):
            ids = []
        if lab in allowed:
            label, ok = lab, True
            break
    if label is None:
        low = text.lower()
        for lab in sorted(allowed, key=len, reverse=True):
            # "included"/"excluded" must not match inside "not included"/"not excluded"
            pattern = rf"\b{re.escape(lab)}\b" if lab.startswith("not") else rf"(?<!not )\b{re.escape(lab)}\b"
            if re.search(pattern, low):
                label = lab
                break
        if not ids:
            m = re.search(r"evidence[^\[]*\[([\d,\s]*)\]", low)
            if m:
                ids = sorted({int(x) for x in re.findall(r"\d+", m.group(1))})
    if valid_ids is not None:
        ids = [i for i in ids if i in valid_ids]
    return Parsed(label or INVALID, ids, ok, text)
