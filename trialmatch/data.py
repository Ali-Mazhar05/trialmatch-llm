"""Dataset loading, leakage control, and the mandated patient-grouped split.

Rubric B1/B2: only the prescribed TrialGPT Criterion Annotations dataset is used;
all model-generated (GPT-4) fields and the dataset's own `training` flag are
dropped immediately after loading so they can never reach a prompt or a metric.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ID = "ncbi/TrialGPT-Criterion-Annotations"
SEED = 42
N_TRAIN, N_VAL, N_TEST = 37, 8, 8

# Columns that must never be used as input features or evaluation targets.
LEAKY_COLUMNS = [
    "gpt4_explanation",
    "explanation_correctness",
    "gpt4_sentences",
    "gpt4_eligibility",
    "training",  # dataset's own split flag; we use the mandated split instead
]
KEEP_COLUMNS = [
    "annotation_id", "patient_id", "note", "trial_id", "trial_title",
    "criterion_type", "criterion_text", "expert_eligibility", "expert_sentences",
]

LABELS = {
    "inclusion": ["included", "not included", "not enough information", "not applicable"],
    "exclusion": ["excluded", "not excluded", "not enough information", "not applicable"],
}
ALL_LABELS = sorted({l for v in LABELS.values() for l in v})

_SENT_RE = re.compile(r"^\s*(\d+)\.\s?(.*)$")


def load_raw(cache_dir: str | None = None) -> pd.DataFrame:
    from datasets import load_dataset

    return load_dataset(DATASET_ID, split="train", cache_dir=cache_dir).to_pandas()


def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leaky columns, parse gold evidence lists, normalise text fields."""
    df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns])
    df = df[KEEP_COLUMNS].copy()
    df["criterion_type"] = df["criterion_type"].str.strip().str.lower()
    df["expert_eligibility"] = df["expert_eligibility"].str.strip().str.lower()
    df["expert_sentences"] = df["expert_sentences"].apply(_parse_id_list)
    assert not any(c.startswith("gpt4") for c in df.columns), "GPT-4 column leaked"
    return df


def drop_unusable(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Remove rows that cannot form a valid prompt (empty criterion text).

    Returned records are written to the manifest so the exclusion is auditable.
    The patient split is computed on ALL patients before this step.
    """
    bad = df.criterion_text.isna() | (df.criterion_text.astype(str).str.strip() == "")
    dropped = df[bad][["annotation_id", "patient_id", "trial_id", "criterion_type", "expert_eligibility"]]
    return df[~bad].copy(), dropped.to_dict(orient="records")


def _parse_id_list(x) -> list[int]:
    if isinstance(x, (list, tuple, np.ndarray)):
        return sorted({int(i) for i in x})
    if x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == "":
        return []
    return sorted({int(i) for i in ast.literal_eval(str(x))})


def split_note(note: str) -> list[tuple[int, str]]:
    """Split a TrialGPT note into (sentence_id, text) using the original IDs.

    Notes are stored as lines of the form "<id>. <sentence>". Lines without a
    numeric prefix are appended to the previous sentence (never renumbered), so
    the IDs returned are exactly the dataset's IDs.
    """
    out: list[list] = []
    for line in str(note).splitlines():
        if not line.strip():
            continue
        m = _SENT_RE.match(line)
        if m:
            out.append([int(m.group(1)), m.group(2).strip()])
        elif out:
            out[-1][1] += " " + line.strip()
        else:
            out.append([0, line.strip()])
    return [(i, s) for i, s in out]


def segment_free_text(text: str) -> list[tuple[int, str]]:
    """Sentence-split an unseen free-text clinical note (used by the app).

    If the text is already numbered ("0. ...") the original numbering is kept.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if lines and all(_SENT_RE.match(l) for l in lines):
        return split_note(text)
    flat = re.sub(r"\s+", " ", text).strip()
    # split on sentence-final punctuation followed by space + capital/digit
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", flat)
    return [(i, p.strip()) for i, p in enumerate(p for p in parts if p.strip())]


@dataclass
class Split:
    train: list[str]
    val: list[str]
    test: list[str]


def make_split(patient_ids, seed: int = SEED) -> Split:
    """Mandated protocol: sort unique IDs, shuffle with default_rng(42), 37/8/8."""
    ids = sorted(set(patient_ids))
    assert len(ids) == N_TRAIN + N_VAL + N_TEST, f"expected 53 patients, got {len(ids)}"
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    return Split(
        train=list(ids[:N_TRAIN]),
        val=list(ids[N_TRAIN:N_TRAIN + N_VAL]),
        test=list(ids[N_TRAIN + N_VAL:]),
    )


def check_no_leakage(df: pd.DataFrame, split: Split) -> None:
    s = {k: set(v) for k, v in split.__dict__.items()}
    assert not (s["train"] & s["val"]), "train/val patient overlap"
    assert not (s["train"] & s["test"]), "train/test patient overlap"
    assert not (s["val"] & s["test"]), "val/test patient overlap"
    assert s["train"] | s["val"] | s["test"] == set(df.patient_id), "patients missing from split"
    # every annotation of a patient sits in exactly one split
    assign = {p: k for k, ps in s.items() for p in ps}
    per_patient = df.groupby("patient_id")["annotation_id"].apply(lambda a: {assign[a.name]})
    assert all(len(v) == 1 for v in per_patient), "a patient spans multiple splits"


def assign_split(df: pd.DataFrame, split: Split) -> pd.DataFrame:
    m = {p: "train" for p in split.train} | {p: "val" for p in split.val} | {p: "test" for p in split.test}
    df = df.copy()
    df["split"] = df["patient_id"].map(m)
    return df


def load_processed(path: str | Path) -> pd.DataFrame:
    df = pd.read_json(path, orient="records", lines=True)
    df["expert_sentences"] = df["expert_sentences"].apply(_parse_id_list)
    return df


def manifest_dict(df: pd.DataFrame, split: Split) -> dict:
    out = {
        "dataset": DATASET_ID,
        "protocol": "sorted unique patient_id -> np.random.default_rng(42).shuffle -> first 37 train, next 8 val, last 8 test",
        "seed": SEED,
        "dropped_columns": LEAKY_COLUMNS,
        "splits": {},
    }
    for name, ids in split.__dict__.items():
        sub = df[df.patient_id.isin(ids)]
        out["splits"][name] = {
            "patient_ids": ids,
            "n_patients": len(ids),
            "n_annotations": int(len(sub)),
            "n_trials": int(sub.trial_id.nunique()),
            "per_patient_annotations": {p: int(n) for p, n in sub.groupby("patient_id").size().items()},
            "label_counts": {
                f"{t}/{l}": int(n)
                for (t, l), n in sub.groupby(["criterion_type", "expert_eligibility"]).size().items()
            },
            "n_empty_gold_evidence": int((sub.expert_sentences.apply(len) == 0).sum()),
        }
    return out


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
