"""Scoring of the free-text task types is rule-based and reads structured fields only (parser v1 extracts one JSON
object; no language model, no interpretation of prose): reconstruction by the ordered cited line ids, summarization
by the cited impression lines and their verbatim text after case/whitespace normalization, claims by the label."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench import tasktypes_tierb  # noqa: E402,F401
from scopebench.parser import extract_json  # noqa: E402
from scopebench.tasktypes import get  # noqa: E402


class T:   # the scorers read only task.gold
    def __init__(self, gold):
        self.gold = gold


def test_parser_v1_extracts_one_object():
    """Frozen behaviour: strip one fence, else take the first '{' to the last '}'. (The module docstring says prose
    after a fenced object is unparseable; the frozen code recovers the object through the brace fallback.)"""
    assert extract_json('```json\n{"status": "ok"}\n```') == {"status": "ok"}
    assert extract_json('Answer: {"status": "ok"} done') == {"status": "ok"}
    assert extract_json('```json\n{"status": "ok"}\n```\nThe impression is normal.') == {"status": "ok"}
    assert extract_json('{"a": 1} and {"b": 2}') is None                              # two objects: no credit
    assert extract_json("no json here") is None and extract_json(None) is None and extract_json("[1, 2]") is None


def test_reconstruction_is_scored_by_the_ordered_cited_lines():
    tt, gold = get("reconstruct_v1"), {"status": "ok", "line_ids": ["l1", "l2"], "text": "1. Normal study."}
    ok = {"status": "ok", "lines": [{"citation": "l1", "text": "anything"}, {"citation": "l2", "text": "else"}]}
    assert tt.answer_correct(ok, T(gold))                                             # the prose of the lines is not judged
    assert not tt.answer_correct({"status": "ok", "lines": ok["lines"][::-1]}, T(gold))
    assert not tt.answer_correct({"status": "insufficient", "lines": []}, T(gold))
    assert tt.answer_correct({"status": "insufficient", "lines": []}, T({"status": "insufficient"}))


def test_summarization_matches_cited_lines_and_verbatim_text_after_normalization():
    tt = get("summarize_exams_v1")
    gold = {"status": "ok", "exams": [{"citation": "x1", "exam_name": "CT Head", "first_impression_line": "No acute hemorrhage."}]}
    got = {"status": "ok", "exams": [{"citation": "x1", "exam_name": "ct  head", "first_impression_line": "NO ACUTE HEMORRHAGE."}]}
    assert tt.answer_correct(got, T(gold))
    para = {"status": "ok", "exams": [{"citation": "x1", "exam_name": "CT Head", "first_impression_line": "No bleed."}]}
    assert not tt.answer_correct(para, T(gold))                                       # a paraphrase earns no credit


def test_claim_is_scored_by_the_label():
    tt = get("claim_v1")
    assert tt.answer_correct({"label": "supported", "citations": ["a"]}, T({"label": "supported"}))
    assert not tt.answer_correct({"label": "insufficient", "citations": []}, T({"label": "supported"}))
