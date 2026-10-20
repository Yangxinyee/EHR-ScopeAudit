"""ICML plan §4: physician-set re-scoring (scripts/scopebench/rescore_physician_rt.py) — strict-majority sets and the
scorer's completeness / citation-compliance definitions with the physicians' sets in place of R_t and L_t."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/scopebench"))

from rescore_physician_rt import physician_sets, rescore  # noqa: E402


def test_strict_majority_and_rescore():
    key = {"p1": {"task_id": "t1", "tier": "B", "required_ids": ["a"]}}
    card = lambda al, nd: {"packet_type": "direct_set", "allowed_ids": al, "needed_ids": nd}  # noqa: E731
    ratings = {"R1": {"p1": card(["a", "b", "c"], ["a", "b"])}, "R2": {"p1": card(["a", "b"], ["a"])}, "R3": {"p1": card(["a", "c"], ["a", "c"])}}
    s = physician_sets(key, ratings)["t1"]
    assert s["majority"] == ({"a", "b", "c"}, {"a"})            # allowed: each of b, c chosen by 2 of 3; needed: only a by > half
    assert physician_sets(key, {"R1": ratings["R1"], "R2": ratings["R2"]})["t1"]["majority"] == ({"a", "b"}, {"a"})   # two raters: both
    row = {"task_type": "lab_trend_k3", "parsed_answer": {"status": "ok", "results": [{"citation": "a"}, {"citation": "b"}, {"citation": "d"}]}}
    comp, legal = rescore(row, {"a", "b", "c"}, {"a", "c"})
    assert comp == 0.5 and legal is False                         # d is outside the physicians' allowed set
    comp, legal = rescore(row, {"a", "b", "d"}, {"a"})
    assert comp == 1.0 and legal is True
