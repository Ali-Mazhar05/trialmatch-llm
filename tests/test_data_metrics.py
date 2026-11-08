import json
from pathlib import Path

import numpy as np

from trialmatch.data import make_split, split_note
from trialmatch.metrics import evidence_metrics
from trialmatch.prompting import parse_output

ROOT = Path(__file__).resolve().parents[1]


def test_split_protocol_and_manifest():
    ids = [f"p{i:02d}" for i in range(53)]
    s = make_split(ids)
    exp = sorted(ids); np.random.default_rng(42).shuffle(exp)
    assert s.train == exp[:37] and s.val == exp[37:45] and s.test == exp[45:]
    man = ROOT / "splits" / "manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        sets = [set(m["splits"][k]["patient_ids"]) for k in ["train", "val", "test"]]
        assert [len(x) for x in sets] == [37, 8, 8]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        assert "gpt4_eligibility" in m["dropped_columns"]


def test_processed_files_have_no_gpt4_columns():
    for name in ["train", "val", "test"]:
        p = ROOT / "data" / "processed" / f"{name}.jsonl"
        if p.exists():
            row = json.loads(p.open(encoding="utf-8").readline())
            assert not any(c.startswith("gpt4") or c == "explanation_correctness" for c in row)


def test_split_note_keeps_ids():
    assert split_note("0. A b.\n1. C d.\n3. E.") == [(0, "A b."), (1, "C d."), (3, "E.")]


def test_parse_output():
    p = parse_output('{"evidence_ids": [2, 9, 1], "label": "not included"}<|im_end|>', "inclusion", {1, 2})
    assert p.parse_ok and p.label == "not included" and p.evidence_ids == [1, 2]
    p = parse_output("The patient is not included because...", "inclusion")
    assert not p.parse_ok and p.label == "not included"
    p = parse_output("The patient is included.", "inclusion")
    assert p.label == "included"
    p = parse_output('{"label": "included"}', "exclusion")      # wrong label set
    assert p.label == "invalid"


def test_evidence_empty_conventions():
    m = evidence_metrics([set(), set(), {1}, {1, 2}], [set(), {3}, set(), {1}])
    assert m["empty_patterns"] == {"gold_empty_pred_empty": 1, "gold_empty_pred_nonempty": 1,
                                   "gold_nonempty_pred_empty": 1, "gold_nonempty_pred_nonempty": 1}
    a = m["all_rows"]
    assert a["micro"]["tp"] == 1 and a["micro"]["fp"] == 1 and a["micro"]["fn"] == 2
    assert abs(a["macro"]["f1"] - (1 + 0 + 0 + 2 / 3) / 4) < 1e-9
    assert m["gold_nonempty_rows"]["n"] == 2
