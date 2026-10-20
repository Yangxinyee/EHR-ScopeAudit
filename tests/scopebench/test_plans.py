"""DEV-34 evidence plans (experiments/SCOPEBENCH_PLAN_PROTOCOL.md): plan reading and execution, the reference-plan
identity, and the property the configuration exists for — at inference time the plan-audited pipelines read no
reference input. The invariance tests run a pipeline twice with a scripted backend whose replies do not depend on the
gold: once on the real tasks and once on copies whose R_t, gold and generator metadata are scrambled. Every prompt,
audit report, repair and final answer must be identical for the plan configurations, and must differ for the
reference `full` configuration (so the test can see a leak when there is one)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "scripts"))

from scopebench import plans  # noqa: E402
from scopebench.backends.base import Generation  # noqa: E402
from scopebench.configurations import CONFIGS  # noqa: E402
from scopebench.guard import check  # noqa: E402
from scopebench.retrieval import TemporalNearestRetriever  # noqa: E402
from scopebench.runner import RunSpec, legal_all, run  # noqa: E402
from scopebench.schema import Case, Contract, EvidenceObject, Task, read_jsonl  # noqa: E402
from scopebench.skills import OPERATIONS  # noqa: E402
from scopebench.tasktypes import LAB_TREND_K3  # noqa: E402
from scopebench import tasktypes_tierb  # noqa: E402,F401

INDEX = "2187-02-17T11:52:30Z"


def ev(i: str, avail: str, value: float) -> EvidenceObject:
    return EvidenceObject(id=i, source="labevents", patient="p", occurred=avail, available=avail, fields={"display": "PT", "value": value, "unit": "sec"})


POOL = [ev("a", "2187-02-13T09:08:00Z", 14.1), ev("b", "2187-02-14T09:08:00Z", 13.9), ev("c", "2187-02-15T09:29:00Z", 13.5),
        ev("d", "2187-02-16T09:42:00Z", 13.0), ev("e", "2187-02-17T12:30:00Z", 12.8), ev("f", "2187-02-18T14:03:00Z", 12.5)]


def lab_task(task_id: str = "t-pre") -> tuple[Task, Case]:
    contract = Contract(INDEX, "pre", "available", permitted_sources=("labevents",))
    case = Case("c", "p", {"index_time": INDEX}, POOL)
    ref = OPERATIONS["lab_trend_k3"](legal_all(Task(task_id, "c", "p", "B", "lab_trend_k3", "q", contract, {}, ()), case), contract, {"k": 3})
    gold = {"status": "ok", "results": ref["provenance"], "delta": ref["delta_last_minus_first"], "direction": ref["direction"]}
    task = Task(task_id, "c", "p", "B", "lab_trend_k3", "Return the three PT results nearest to but strictly before the index.", contract,
                gold=gold, required_ids=tuple(ref["ordered_result_ids"]), metadata={"skill_params": {"k": 3}})
    return task, case


def scrambled(task: Task) -> Task:
    """The same request with every reference input replaced: R_t, gold, and the generator's skill parameters."""
    t = copy.deepcopy(task)
    t.required_ids = ("a", "f")
    t.gold = {"status": "insufficient", "results": [], "delta": None, "direction": None, "answer": [["wrong"]], "fhir_ids": {}}
    t.metadata = {**t.metadata, "skill_params": {"k": 1, "values": {"patient_id": "0"}, "ops": {}, "time": {}},
                  "template": "What is the gender of patient {patient_id}?", "main_table": "patients", "criteria": t.metadata.get("criteria")}
    return t


# ------------------------------------------------------------------------------------------- reading plans
def test_tier_b_plan_reading():
    assert plans.read({"status": "ok", "evidence_plan": {"operation": "lab_trend_k3"}}).plan == plans.Plan("lab_trend_k3", {})
    assert plans.read({"evidence_plan": None}).plan == plans.Plan(None)
    assert plans.read({"evidence_plan": {"operation": None}}).plan == plans.Plan(None)
    missing = plans.read({"status": "ok"})
    assert not missing.present and missing.plan is None and "evidence_plan" in missing.errors[0]
    assert plans.read({"evidence_plan": {"operation": "nearest"}}).errors[0].startswith("invalid evidence_plan: unknown operation")
    # v0.3: a plan is an operation name; the earlier outcome_lookup + target form and bare strings are read too
    assert plans.read({"evidence_plan": {"operation": "outcome_lookup", "target": "icu"}}).errors[0].startswith("invalid evidence_plan: unknown operation")
    assert plans.read({"evidence_plan": {"operation": "outcome_lookup", "target": "first_careunit"}}).plan == plans.Plan("first_careunit", {})
    assert plans.read({"evidence_plan": "discharge_disposition"}).plan == plans.Plan("discharge_disposition", {})
    assert plans.read({"evidence_plan": "null"}).plan == plans.Plan(None)
    assert plans.read({"evidence_plan": ["lab_trend_k3"]}).errors
    # v0.2: the cohort criteria are the request's; the plan names the operation (criteria written into it are ignored)
    assert plans.read({"evidence_plan": {"operation": "cohort_criteria"}}).plan == plans.Plan("cohort_criteria", {})
    assert plans.read({"evidence_plan": {"operation": "cohort_criteria", "criteria": [{"kind": "x"}]}}).plan == plans.Plan("cohort_criteria", {})
    assert plans.read(None).errors and not plans.read(None).present


def test_reference_plan_reproduces_the_reference_skill():
    task, case = lab_task()
    evidence = legal_all(task, case)
    res = plans.execute(plans.reference_plan(task), evidence, task, OPERATIONS)
    assert res.ok and res.summary == OPERATIONS["lab_trend_k3"](evidence, task.contract, {"k": 3})
    assert res.required == list(task.required_ids) == ["b", "c", "d"]
    assert plans.execute(plans.Plan(None), evidence, task, OPERATIONS) is None
    assert plans.plan_accuracy(plans.Plan("lab_trend_k3", {}), plans.reference_plan(task))["exact"]
    assert plans.required_overlap(["b", "c"], ["b", "c", "d"]) == {"n_pred": 2, "n_ref": 3, "precision": 1.0, "recall": 2 / 3, "exact": False}


def test_guard_required_override_keeps_the_default():
    task, case = lab_task()
    pool = {e.id: e for e in POOL}
    answer = {"status": "ok", "results": [{"citation": "c"}, {"citation": "d"}, {"citation": "b"}], "delta": 0, "direction": "stable"}
    default = check(answer, task, LAB_TREND_K3, pool, POOL[:4], {"b", "c", "d"}, check_completeness=True)
    explicit = check(answer, task, LAB_TREND_K3, pool, POOL[:4], {"b", "c", "d"}, check_completeness=True, required=task.required_ids)
    assert default.to_dict() == explicit.to_dict() and default.missing_required == []
    assert check(answer, task, LAB_TREND_K3, pool, POOL[:4], set(), check_completeness=True, required=("a",)).missing_required == ["a"]
    assert check(answer, task, LAB_TREND_K3, pool, POOL[:4], set(), check_completeness=True, required=()).missing_required == []


# ------------------------------------------------------------------------------------------- Tier-B invariance
class RecordingBackend:
    """Replies by turn, never by gold: turn 1 cites two of the supplied results (incomplete) and declares the plan;
    the repair turn cites the three nearest supplied results with a wrong delta (so plan_full still disagrees and
    replaces). Records every request it receives."""
    name = "recording"

    def __init__(self):
        self.requests: list = []

    def _reply(self, messages) -> str:
        payload = json.loads(messages[1]["content"])
        ids = [e["id"] for e in payload.get("supplied_evidence", [])]
        plan = {"operation": "lab_trend_k3"}
        if len(messages) == 2:
            return json.dumps({"status": "ok", "results": [{"citation": i, "value": 0} for i in ids[:2]], "delta": 0.0, "direction": "stable", "evidence_plan": plan})
        return json.dumps({"status": "ok", "results": [{"citation": i, "value": 0} for i in ids[:3]], "delta": 9.0, "direction": "increasing", "evidence_plan": plan})

    def generate_many(self, requests):
        self.requests += [(r.key, json.dumps(r.messages, sort_keys=True)) for r in requests]
        return {r.key: Generation(text=self._reply(r.messages), tokens_in=10, tokens_out=5, latency_seconds=0.0) for r in requests}

    def metadata(self):
        return {"backend": "recording"}


INFERENCE_FIELDS = ("selected_evidence_ids", "repair_evidence_ids", "response", "repair_response", "parsed_initial", "parsed_answer", "initial_guard",
                    "final_guard", "initial_guard_errors", "final_guard_errors", "guard_repaired", "guard_deterministic_fallback", "skill_summary", "access_ids",
                    "evidence_plan_initial", "evidence_plan_final", "plan_result_initial", "plan_result_final")


def _run_b(config: str, task: Task, case: Case, out: Path):
    backend = RecordingBackend()
    spec = RunSpec(configuration=CONFIGS[config], retriever=TemporalNearestRetriever(), k=5, skills=dict(OPERATIONS))
    run(spec, [task], {case.case_id: case}, backend, out)
    row = next(iter(read_jsonl(out / "predictions.jsonl")))
    return backend.requests, {k: row.get(k) for k in INFERENCE_FIELDS}, row


@pytest.mark.parametrize("config", ["plan_audit", "plan_full"])
def test_tier_b_plan_pipeline_reads_no_reference_input(config, tmp_path):
    task, case = lab_task()
    req_a, inf_a, row_a = _run_b(config, task, case, tmp_path / "real")
    req_b, inf_b, _ = _run_b(config, scrambled(task), case, tmp_path / "scrambled")
    assert req_a == req_b and inf_a == inf_b
    # the pipeline did what it is for: an incomplete first answer is repaired from R̂_t; plan_full replaces the wrong delta
    assert row_a["guard_repaired"] and row_a["initial_guard"]["missing_required"]
    assert row_a["plan_result_initial"]["required"] == list(task.required_ids)
    if config == "plan_full":
        assert row_a["guard_deterministic_fallback"] and row_a["metrics"]["answer_correct"] and row_a["metrics"]["completeness"] == 1.0
        assert "evidence_plan_result" in json.loads(json.loads(req_a[-1][1])[-1]["content"])
    else:
        assert not row_a["guard_deterministic_fallback"]


def test_plan_replacement_that_fails_its_own_audit_is_rejected(tmp_path, monkeypatch):
    """Tier-A addendum 2: when the declared operation's result is not a valid answer of the task type (on Tier-A, a
    lookup declared on an outcome-count task), plan_full keeps the submission, withdraws the replacement access and
    records why, instead of aborting the run. Simulated by a task-type answer builder that yields an invalid answer."""
    task, case = lab_task()
    _, _, ok_row = _run_b("plan_full", task, case, tmp_path / "ok")
    assert ok_row["guard_deterministic_fallback"] and "plan_replacement_rejected" not in ok_row
    tt = __import__("scopebench.tasktypes", fromlist=["get"]).get(task.task_type)
    monkeypatch.setattr(tt, "skill_answer", lambda summary, task, evidence: {"status": "ok", "results": []}, raising=True)
    _, _, row = _run_b("plan_full", task, case, tmp_path / "bad")
    assert not row["guard_deterministic_fallback"] and row["plan_replacement_rejected"]
    assert row["parsed_answer"]["delta"] == 9.0 and row["final_guard_errors"]          # the submission stands, still failing
    access = next(iter(read_jsonl(tmp_path / "bad" / "access_log.jsonl")))["entries"]
    assert not any(e["stage"] == "replacement" for e in access)


def test_plan_of_another_result_kind_is_a_disagreement_not_a_crash(tmp_path, monkeypatch):
    """Tier-A addendum 2: a declared operation whose result the task type's agreement check cannot read (a cohort decision
    on a lab-trend task) is reported as a disagreement; the run completes."""
    task, case = lab_task()
    tt = __import__("scopebench.tasktypes", fromlist=["get"]).get(task.task_type)

    def boom(parsed, summary, task):
        raise KeyError("delta_last_minus_first")
    monkeypatch.setattr(tt, "skill_disagreements", boom, raising=True)
    _, _, row = _run_b("plan_full", task, case, tmp_path / "kind")
    assert "the declared operation does not compute an answer of this task type" in row["initial_guard"]["skill_disagreements"]


def test_tier_b_reference_pipelines_do_read_reference_inputs(tmp_path):
    task, case = lab_task()
    req_a, inf_a, _ = _run_b("audit", task, case, tmp_path / "real")
    req_b, inf_b, _ = _run_b("audit", scrambled(task), case, tmp_path / "scrambled")
    assert req_a != req_b and inf_a != inf_b          # R_t drives the completeness check and the repair refill
    with pytest.raises(AssertionError, match="verifier-generated skill answer failed"):
        _run_b("full", scrambled(task), case, tmp_path / "full")   # its replacement presumes R_t = the skill's provenance


# ------------------------------------------------------------------------------------------- Tier-C
from scopebench.config import PUBLIC_DATA  # noqa: E402

FHIR_DIR = PUBLIC_DATA / "mimic-iv-clinical-database-demo-on-fhir-2.1.0" / "fhir"
MANIFEST = ROOT / "experiments/scopebench_tierC/fhir_agentbench"
PMAP = PUBLIC_DATA / ".clinehr_cache/scopebench_tierc_patients/fhir_agentbench_patient_map.json"
tierc_data = pytest.mark.skipif(not (FHIR_DIR.exists() and MANIFEST.exists() and PMAP.exists()), reason="FHIR demo / Tier-C manifest not present")


def test_tier_c_plan_reading():
    from scopebench.tierc.plans import CATALOG_PATH, load_catalog, read

    if not CATALOG_PATH.exists():
        pytest.skip("Tier-C plan catalog not present (built from the FHIR-AgentBench dev split by build_tierc_plan_catalog.py)")

    cat = load_catalog()
    text = 'The final answer is: [[1]]\nCitations: ["x"]\nPlan: {"template": "C35", "values": {"patient_id": 10005866}}'
    rest, reading = read(text, cat)
    assert rest == 'The final answer is: [[1]]\nCitations: ["x"]' and reading.plan == plans.Plan("C35", {"values": {}})   # the task fixes the patient
    assert read("The final answer is: [[1]]\nPlan: null", cat)[1].plan == plans.Plan(None)
    assert not read("The final answer is: [[1]]", cat)[1].present
    assert "needs order" in read('Plan: {"template": "C05", "values": {"patient_id": "1"}}', cat)[1].errors[0]
    assert "unknown template" in read('Plan: {"template": "C99", "values": {}}', cat)[1].errors[0]
    assert read('Plan: {"template": "C05", "values": {"patient_id": "1"}, "order": "first"}', cat)[1].plan.args == {"values": {}, "order": "first"}
    # v0.2 lenient reading: zero-padded ids, the benchmark's slot names, everyday words, an `operators` object
    assert read('Plan: {"template": "C023", "operators": {"time_filter_exact1": "the first time"}}', cat)[1].plan == plans.Plan("C23", {"values": {}, "order": "first"})
    assert read('Plan: {"template": "C21", "values": {"lab_name": "pt"}, "agg_function": "maximum"}', cat)[1].plan.args == {"values": {"lab_name": "pt"}, "aggregate": "max"}
    assert "needs order" in read('Plan: {"template": "C05", "operator": "order"}', cat)[1].errors[0]          # a slot name is not a choice
    assert "needs extreme" in read('Plan: {"template": "C24", "values": {"lab_name": "x"}, "extreme": "max or min", "order": "last"}', cat)[1].errors[0]
    assert read("**Plan:** not json", cat)[1].errors == ["the Plan line must be one JSON object or null"]


class ScriptedC:
    """Turn 1: one Observation call; turn 2: a wrong answer, no citations, and the scripted plan line (fixed in
    advance, independent of the task's gold); repair: the same. Records every request."""
    name = "scripted-c"

    def __init__(self, plan_lines: dict[str, str]):
        self.plan_lines = plan_lines
        self.turns: dict[str, int] = {}
        self.requests: list = []

    def generate_many(self, requests):
        out = {}
        for r in requests:
            self.requests.append((r.key, json.dumps(r.messages, sort_keys=True)))
            n = self.turns[r.key] = self.turns.get(r.key, 0) + 1
            if n == 1:
                pid = [m for m in r.messages if m["role"] == "user"][0]["content"].split("Patient FHIR ID: ")[1].split("\n")[0]
                text = json.dumps({"name": "get_resources_by_patient_fhir_id", "arguments": {"resource_type": "Observation", "patient_fhir_id": pid}}).join(["<tool_call>\n", "\n</tool_call>"])
            else:
                text = f'The final answer is: [["no"]]\nCitations: []\nPlan: {self.plan_lines[r.key]}'
            out[r.key] = Generation(text=text, tokens_in=10, tokens_out=5, latency_seconds=0.0)
        return out

    def metadata(self):
        return {"backend": "scripted-c"}


@pytest.fixture(scope="module")
def store():
    from scopebench.tierc.store import load_store
    return load_store(FHIR_DIR)


TIER_C_FIELDS = INFERENCE_FIELDS + ("trace", "replacement_reason", "n_access_ids", "illegal_access_ids")


@tierc_data
@pytest.mark.parametrize("config", ["plan_audit", "plan_full"])
def test_tier_c_plan_pipeline_reads_no_reference_input(config, store, tmp_path):
    from scopebench.tierc.agent import TierCSpec, run_tierc
    from scopebench.tierc.plans import Executor, reference_plan

    pmap = json.load(open(PMAP))
    ex = Executor(pmap)
    tasks = [Task.from_dict(r) for r in read_jsonl(MANIFEST / "tasks.jsonl") if r["split"] == "valid"]
    picked = [t for t in tasks if reference_plan(t, ex.catalog).operation in ("C21", "C36", "C35")][:3]
    assert len(picked) == 3
    lines = {t.task_id: json.dumps({"template": (p := reference_plan(t, ex.catalog)).operation, **p.args}) for t in picked}

    def go(ts, out):
        backend = ScriptedC(lines)
        run_tierc(TierCSpec(configuration=config, adapter="text", plan_executor=Executor(pmap)), ts, store, pmap, backend, None, out, manifest_path=MANIFEST)
        rows = {r["task_id"]: r for r in read_jsonl(out / "predictions.jsonl")}
        return backend.requests, {k: {f: r.get(f) for f in TIER_C_FIELDS} for k, r in rows.items()}, rows

    req_a, inf_a, rows = go(picked, tmp_path / "real")
    req_b, inf_b, _ = go([scrambled(t) for t in picked], tmp_path / "scrambled")
    assert req_a == req_b and inf_a == inf_b
    for r in rows.values():
        assert r["evidence_plan_initial"]["plan"] is not None and r["plan_result_initial"]["error"] is None
        assert r["guard_repaired"]
        assert r["guard_deterministic_fallback"] == (config == "plan_full")
        if config == "plan_full":
            assert r["metrics"]["answer_correct"]   # the declared (here: reference) plan's answer


@tierc_data
def test_tier_c_reference_full_does_read_reference_inputs(store, tmp_path):
    from scopebench.tierc.agent import TierCSpec, run_tierc
    from scopebench.tierc.plans import Executor, reference_plan
    from scopebench.tierc.skills import mapped_skills
    SKILL_VALIDATION = ROOT / "experiments/scopebench_tierC/skill_validation_protocol_dev31.json"

    pmap = json.load(open(PMAP))
    cat = Executor(pmap).catalog
    picked = [t for t in (Task.from_dict(r) for r in read_jsonl(MANIFEST / "tasks.jsonl") if r["split"] == "valid") if reference_plan(t, cat).operation == "C21"][:2]
    lines = {t.task_id: "null" for t in picked}

    def go(ts, out):
        backend = ScriptedC(lines)
        run_tierc(TierCSpec(configuration="full", adapter="text", skills=mapped_skills(SKILL_VALIDATION, pmap)), ts, store, pmap, backend, None, out, manifest_path=MANIFEST)
        return backend.requests, {r["task_id"]: {f: r.get(f) for f in TIER_C_FIELDS} for r in read_jsonl(out / "predictions.jsonl")}

    assert go(picked, tmp_path / "real") != go([scrambled(t) for t in picked], tmp_path / "scrambled")


def test_outcome_operation_fields_follow_the_source_schema():
    """DEV-34 addendum 2: the disposition lives in `discharge_location` (MIMIC, Tier-B) and in `admission_type` (CCDA,
    Tier-A); the Tier-B parameters are exactly the frozen ones."""
    import dataclasses

    t, _ = lab_task()
    tb = dataclasses.replace(t, tier="B")
    ta = dataclasses.replace(t, tier="A")
    op = plans.TIER_B_OPERATIONS["discharge_disposition"]
    assert op.params(tb) == {"target": "discharge_disposition", "field": "discharge_location", "basis": "occurred", "drug": None}
    assert op.params(ta) == {"target": "discharge_disposition", "field": "admission_type", "basis": "occurred", "drug": None}
    assert plans.TIER_B_OPERATIONS["first_careunit"].params(ta) == plans.TIER_B_OPERATIONS["first_careunit"].params(tb)
