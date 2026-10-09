"""End-to-end publication smoke checks independent of restricted data."""
import json
from pathlib import Path

from examples.scope_demo import demonstrate

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_correct_answer_and_legal_citations_can_hide_illegal_access():
    rows = demonstrate()
    assert all(row["answer_correct"] and row["citation_legal"] for row in rows.values())
    assert rows["prompt_only"]["accessed_objects"] == 6
    assert not rows["prompt_only"]["access_legal"]
    for name in ("scope_aware", "full"):
        assert rows[name]["accessed_objects"] == 3
        assert rows[name]["access_legal"]
        assert rows[name]["completeness"] == 1.0


def test_introductory_estimates_retain_the_all_task_denominator():
    panel = json.loads((ROOT / "results/scopebench/tierB_panel.json").read_text())
    values = json.loads((ROOT / "results/paper/macros.json").read_text())
    assert values["B.exposed"] == f"{100 - panel['checkpoint_average']['prompt_only']['access_legal']:.1f}"
    rates = [row["legal_citations_after_illegal_exposure_pct"] for row in panel["access_citation_divergence"]["prompt_only"].values()]
    assert values["B.div.prompt_only"] == f"{sum(rates) / len(rates):.1f}"
    assert len(rates) == 6
