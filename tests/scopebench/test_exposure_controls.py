"""DEV-35 exposure injection and DEV-36 self-revision (experiments/SCOPEBENCH_EXPOSURE_PROTOCOL.md,
experiments/SCOPEBENCH_CONTROLS_PROTOCOL.md): the injected records are appended after the scoped selection, logged as
accessed, and nothing else changes; `self_revise` sends exactly one fixed, gold-free review turn and reads no reference
input."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_plans import RecordingBackend, lab_task, scrambled  # noqa: E402

from scopebench.configurations import CONFIGS  # noqa: E402
from scopebench.guard import SELF_REVISE_MESSAGE  # noqa: E402
from scopebench.predicate import legal  # noqa: E402
from scopebench.retrieval import TemporalNearestRetriever  # noqa: E402
from scopebench.runner import RunSpec, run  # noqa: E402
from scopebench.schema import read_jsonl  # noqa: E402
from scopebench.skills import OPERATIONS  # noqa: E402


def _run(config: str, out: Path, task=None, **spec_kw):
    t, case = lab_task()
    task = task or t
    backend = RecordingBackend()
    spec = RunSpec(configuration=CONFIGS[config], retriever=TemporalNearestRetriever(), k=3, skills=dict(OPERATIONS), **spec_kw)
    run(spec, [task], {case.case_id: case}, backend, out)
    row = next(iter(read_jsonl(out / "predictions.jsonl")))
    access = next(iter(read_jsonl(out / "access_log.jsonl")))
    return backend.requests, row, access, case


def _shown(request_json: str) -> list[str]:
    messages = json.loads(request_json)
    return [e["id"] for e in json.loads(messages[1]["content"])["supplied_evidence"]]


def test_injection_appends_after_the_scoped_selection_and_is_logged(tmp_path):
    base_req, base_row, _, case = _run("scope_aware", tmp_path / "base")
    inject = {"t-pre": {"future": ["f", "e"], "inscope": ["a"]}}
    fut_req, fut_row, fut_access, _ = _run("scope_aware", tmp_path / "future", inject="future", inject_ids=inject)
    base_ids = _shown(base_req[0][1])
    assert _shown(fut_req[0][1]) == base_ids + ["f", "e"]                       # appended, in the fixed order, nothing else moves
    assert fut_row["selected_evidence_ids"] == base_ids + ["f", "e"]
    assert fut_row["injected_ids"] == ["f", "e"] and fut_row["exposure_arm"] == "future"
    assert {"f", "e"} <= set(fut_row["access_ids"])                              # an injected record counts as accessed
    assert fut_row["metrics"]["n_illegal_access"] == 2 and base_row["metrics"]["n_illegal_access"] == 0
    stages = [e["stage"] for e in fut_access["entries"]]
    assert stages.index("injected") < stages.index("prompt")
    pool = case.by_id()
    task, _ = lab_task()
    assert not any(legal(pool[i], task.contract) for i in ("f", "e"))


def test_inscope_arm_and_absent_task_leave_the_baseline_alone(tmp_path):
    base_req, base_row, _, _ = _run("scope_aware", tmp_path / "base")
    # a record the scoped top-3 did not show is appended; an arm with no entry for the task changes nothing
    in_req, in_row, _, _ = _run("scope_aware", tmp_path / "in", inject="inscope", inject_ids={"t-pre": {"inscope": ["a"]}})
    assert _shown(in_req[0][1]) == _shown(base_req[0][1]) + ["a"]
    none_req, none_row, _, _ = _run("scope_aware", tmp_path / "none", inject="future", inject_ids={"other": {"future": ["f"]}})
    assert none_req == base_req
    assert {k: v for k, v in none_row.items() if k not in ("exposure_arm", "injected_ids")} == base_row


def test_self_revise_sends_one_fixed_gold_free_review_turn(tmp_path):
    req, row, _, _ = _run("self_revise", tmp_path / "rev")
    assert len(req) == 2
    second = json.loads(req[1][1])
    assert len(second) == 4 and second[2]["role"] == "assistant" and second[3] == {"role": "user", "content": SELF_REVISE_MESSAGE}
    assert row["self_revised"] is True and row["metrics"]["calls"] == 2
    assert row["guard_repaired"] is False and row["final_guard"] is None
    # nothing the model sees depends on R_t, the gold or the generator's skill parameters
    t, _ = lab_task()
    req2, _, _, _ = _run("self_revise", tmp_path / "rev_scrambled", task=scrambled(t))
    assert req2 == req


def test_existing_configurations_are_untouched_by_the_new_fields(tmp_path):
    for cfg in ("scope_aware", "audit", "full", "plan_full", "retry_generic"):
        assert CONFIGS[cfg].review is False
    req, row, _, _ = _run("scope_aware", tmp_path / "sa")
    assert len(req) == 1 and "self_revised" not in row and "exposure_arm" not in row
