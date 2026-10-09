"""DEV-26 / DEV-27 unit checks (no FHIR data needed): the covered-template mapping and the late-filter access rule."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

VALIDATION = ROOT / "experiments/scopebench_tierC/skill_validation.json"


@pytest.mark.skipif(not VALIDATION.exists(), reason="skill validation record not present")
def test_mapped_skills_cover_exactly_the_validated_templates():
    from scopebench.tierc.skills import covered_templates, mapped_skills
    d = json.loads(VALIDATION.read_text())
    covered = covered_templates(VALIDATION)
    assert len(covered) == d["covered_templates"] == 45
    skills = mapped_skills(VALIDATION, {})
    assert set(skills) == covered and all(callable(f) for f in skills.values())
    assert all(not d["templates"][t]["covered"] for t in d["templates"] if t not in covered)


def test_late_filter_access_rule():
    from scopebench.tierc.agent import ID_LIST_CAP, late_filter_access
    row = {"access_ids": ["a", "b"], "n_access_ids": 2}
    access, capped = late_filter_access(row, ["a", "z"])
    assert access == {"a", "b"} and not capped          # uncapped list: A_t as recorded, kept citations add nothing
    row = {"access_ids": [], "n_access_ids": ID_LIST_CAP + 1}
    access, capped = late_filter_access(row, ["x", "y"])
    assert access == {"x", "y"} and capped              # capped list: the kept citations count as accessed
    assert late_filter_access({"access_ids": ["q"]}, ["q"]) == ({"q"}, False)


def test_python_view_isolates_the_shared_store():
    """DEV-28: code run by execute_python_code edits its own copies; the store's resources stay pristine, and the episode's
    edits persist across its own calls."""
    from scopebench.schema import Contract
    from scopebench.tierc.store import TaskStore
    r = {"resourceType": "Observation", "id": "obs-1", "effectiveDateTime": "2130-01-02T03:04:05-05:00", "valueQuantity": {"value": 1.5}}
    ts = TaskStore(store=object(), patient_fhir_id="p", patient_hash="h", contract=Contract(index_time="2130-01-03T00:00:00Z", scope="pre", basis="occurred"))
    ts.retrieved = {"Observation": [r]}
    view = ts.python_view()
    assert view["Observation"][0] == r and view["Observation"][0] is not r
    view["Observation"][0]["effectiveDateTime"] = object()           # what a model's datetime conversion does
    view["Observation"][0]["valueQuantity"]["value"] = 99
    assert r["effectiveDateTime"] == "2130-01-02T03:04:05-05:00" and r["valueQuantity"]["value"] == 1.5
    again = ts.python_view()
    assert again["Observation"][0] is view["Observation"][0] and again["Observation"][0]["valueQuantity"]["value"] == 99
    ts.retrieved["Observation"].append({"resourceType": "Observation", "id": "obs-2"})
    assert [x["id"] for x in ts.python_view()["Observation"]] == ["obs-1", "obs-2"]
