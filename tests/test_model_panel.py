"""Invariants for the extended checkpoint panel and its reporting code."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import generate_submission_tables as tables
from scripts import run_scopepair_all6_cot as cot


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments" / "SCOPEPAIR_MODEL_PANEL_FREEZE.json"


def test_panel_labels_are_unique_and_ordered_by_scale():
    labels = [label for _, label in tables.PANEL]
    assert len(labels) == len(set(labels))
    assert len({directory for directory, _ in tables.PANEL}) == len(tables.PANEL)
    assert tables.PANEL_ORDER == labels
    assert set(tables.MACRO_ALIAS) == set(labels)
    # Macro aliases must be LaTeX-safe control-sequence names.
    assert all(alias.isalpha() for alias in tables.MACRO_ALIAS.values())


def test_freeze_record_matches_the_reporting_panel():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    frozen_paths = {
        Path(entry["path"]).name
        for entry in freeze["previously_frozen_checkpoints"] + freeze["new_checkpoints"]
    }
    assert frozen_paths == set(tables.MODEL_LABELS)
    assert freeze["formal_execution_started_at_freeze"] is False
    assert freeze["runtime_policy"]["cpu_inference_allowed"] is False


def test_unlabelled_checkpoint_is_rejected_rather_than_silently_dropped():
    with pytest.raises(ValueError):
        tables.known_models([{"model": "some__unregistered-checkpoint"}])


def test_present_models_preserves_panel_order_and_ignores_absent_checkpoints():
    rows = {
        ("Qwen3-30B", "scope_first"): {},
        ("Qwen3-4B", "scope_first"): {},
    }
    assert tables.present_models(rows) == ["Qwen3-4B", "Qwen3-30B"]


def test_percentage_formatting_is_stable():
    assert tables.pct("0.6796875") == "67.97"
    assert tables.pct(1.0) == "100.00"
    assert tables.signed_pct(-0.15625) == "-15.62"
    assert tables.signed_pct(0.15625) == "+15.62"


def test_p_value_formatting_switches_to_scientific_below_one_in_a_thousand():
    assert tables.tex_p(0.002) == "0.002"
    assert tables.tex_p(2.38e-07) == "2.38\\times10^{-7}"


def test_permissive_parser_prefers_the_last_fenced_object():
    text = (
        'Reasoning: record {A} then {B}.\n'
        '```json\n{"status":"insufficient"}\n```\n'
        'On reflection:\n```json\n{"status":"ok","delta":2.0}\n```'
    )
    parsed = cot.permissive_extract_json(text)
    assert parsed == {"status": "ok", "delta": 2.0}


def test_permissive_parser_survives_braces_inside_reasoning_text():
    text = 'The set {a, b} is out of scope.\n{"status":"ok","results":[]}'
    assert cot.permissive_extract_json(text) == {"status": "ok", "results": []}


def test_permissive_parser_ignores_braces_inside_json_strings():
    text = '{"status":"ok","note":"a } brace in a string","delta":1}'
    assert cot.permissive_extract_json(text)["note"] == "a } brace in a string"


def test_permissive_parser_returns_none_without_an_object():
    assert cot.permissive_extract_json("no object here") is None


def test_cot_condition_only_relaxes_the_baseline():
    """The reasoning condition must not touch any guarded system."""
    assert set(cot.FAMILIES) == {"lab", "med"}
    for family in cot.FAMILIES.values():
        assert family["system"].endswith("_cot")
        # The reasoning suffix is appended to, never replaces, the frozen prompt.
        assert family["base_prompt"] in family["base_prompt"] + cot.REASONING_SUFFIX
    assert "reason step by step" in cot.REASONING_SUFFIX
