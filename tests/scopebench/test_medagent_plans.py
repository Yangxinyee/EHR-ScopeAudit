"""DEV-34 evidence plans for MedAgentBench (scopebench/tierc/medagent_plans.py): the executor reproduces the reference
skill on every covered category, plan lines are read leniently, the plan configurations read no reference input
(R_t, gold, the category's parameters), and the existing configurations are unchanged. Offline: a fake FHIR server
answers the logged HTTP layer and the released grader; the last test checks the reference-plan identity against the
official server (skipped when it is down)."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "scripts"))

from scopebench.backends.base import Generation  # noqa: E402
from scopebench.schema import parse_time, read_jsonl  # noqa: E402
from scopebench.tierc import medagent  # noqa: E402
from scopebench.tierc.medagent import FHIR_BASE, MedAgentSpec, MedAgentStore, make_task, run_medagent  # noqa: E402

TASK_DATA = ROOT / "external_benchmarks/MedAgentBench/data/medagentbench/test_data_v2.json"
FUNCTIONS = ROOT / "external_benchmarks/MedAgentBench/data/medagentbench/funcs_v1.json"
MANIFEST = ROOT / "experiments/scopebench_tierC/medagentbench"
NOW = "It's 2023-11-13T10:15:00+00:00 now."


# ------------------------------------------------------------------------------------------- a fake FHIR server
def obs(rid: str, mrn: str, code: str, when: str, value: float | None) -> dict:
    r = {"resourceType": "Observation", "id": rid, "status": "final", "subject": {"reference": f"Patient/{mrn}"}, "code": {"coding": [{"code": code}], "text": code},
         "effectiveDateTime": when, "issued": when}
    if value is not None:
        r["valueQuantity"] = {"value": value, "unit": "x"}
    return r


PATIENTS = {"S1": {"resourceType": "Patient", "id": "S1", "birthDate": "1950-12-01", "name": [{"given": ["Ann"], "family": "Lee"}]},
            "S2": {"resourceType": "Patient", "id": "S2", "birthDate": "1960-01-02"}}
OBSERVATIONS = [   # the served chart (consistent with the released graders, which read the whole chart)
    obs("m1", "S2", "MG", "2023-11-13T08:00:00+00:00", 1.7), obs("m2", "S2", "MG", "2023-11-12T05:00:00+00:00", 2.0),
    obs("g1", "S3", "GLU", "2023-11-12T12:00:00+00:00", 100), obs("g2", "S3", "GLU", "2023-11-13T06:00:00+00:00", 120), obs("g3", "S3", "GLU", "2023-11-11T00:00:00+00:00", 200),
    obs("h1", "S4", "GLU", "2023-11-01T00:00:00+00:00", 95), obs("h2", "S4", "GLU", "2023-11-10T00:00:00+00:00", 105),
    obs("a1", "S5", "A1C", "2021-01-01T00:00:00+00:00", 6.1), obs("a2", "S5", "A1C", "2022-06-01T00:00:00+00:00", 6.5),
]
# edge cases for the executor identity (never served): after the index, no value, another code, boundary times
EXTRA = [obs("m3", "S2", "MG", "2023-11-13T12:00:00+00:00", 9.9), obs("m4", "S2", "MG", "2023-11-12T10:15:00+00:00", 1.1), obs("m5", "S2", "MG", "2023-11-13T10:15:00+00:00", 1.3),
         obs("g4", "S3", "GLU", "2023-11-13T07:00:00+00:00", None), obs("h3", "S4", "GLU", "2023-11-14T00:00:00+00:00", 300),
         obs("k1", "S7", "K", "2023-11-12T09:00:00+00:00", 3.1), obs("x1", "S3", "mg", "2023-11-13T09:00:00+00:00", 5.0),
         {**obs("x2", "S3", "GLU", "2023-11-13T09:30:00+00:00", 130), "code": {"coding": [{"code": "GLU"}]}},
         {**obs("x3", "S2", "MG", "2023-11-13T09:30:00+00:00", 1.2), "code": {"text": "MG"}}]


def fake_get(url: str, timeout: int = 120) -> dict:
    """The compatibility form of `get_json_text` over the synthetic chart (search by patient/code or identifier, read by id)."""
    parts = urlsplit(url)
    path = parts.path.split("/fhir/", 1)[-1].strip("/")
    q = dict(parse_qsl(parts.query))
    if path == "metadata":
        return {"status_code": 200, "data": "{}"}
    if path == "Patient":
        found = [PATIENTS[q["identifier"]]] if q.get("identifier") in PATIENTS else []
    elif path == "Observation":
        found = [o for o in OBSERVATIONS if o["subject"]["reference"] == f"Patient/{q.get('patient')}" and (not q.get("code") or o["code"]["text"] == q["code"])]
    elif "/" in path:
        rtype, rid = path.split("/", 1)
        hit = PATIENTS.get(rid) if rtype == "Patient" else next((o for o in OBSERVATIONS if o["id"] == rid and rtype == "Observation"), None)
        return {"status_code": 200, "data": json.dumps(hit)} if hit else {"error": "404 Not Found"}
    else:
        found = []
    bundle = {"resourceType": "Bundle", "id": "fixed", "type": "searchset", "total": len(found), "entry": [{"resource": copy.deepcopy(r)} for r in found]}
    return {"status_code": 200, "data": json.dumps(bundle)}


CODES = {"task4": "MG", "task5": "MG", "task6": "GLU", "task7": "GLU", "task9": "K", "task10": "A1C"}
CASES = [
    {"id": "task1_1", "instruction": "What's the MRN of the patient with name Ann Lee and DOB of 1950-12-01?", "context": "", "sol": ["S1"]},
    {"id": "task2_1", "instruction": "What's the age of the patient with MRN of S1?", "context": NOW + " The answer should be rounded down to an integer.", "sol": [72], "eval_MRN": "S1"},
    {"id": "task4_1", "instruction": "What's the most recent magnesium level of the patient S2 within last 24 hours?", "context": NOW + ' The code for magnesium is "MG".', "sol": [1.7], "eval_MRN": "S2"},
    {"id": "task4_2", "instruction": "What's the most recent magnesium level of the patient S6 within last 24 hours?", "context": NOW + ' The code for magnesium is "MG".', "sol": [-1], "eval_MRN": "S6"},
    {"id": "task6_1", "instruction": "What is the average CBG of the patient S3 over the last 24 hours?", "context": NOW + ' The code for CBG is "GLU".', "sol": [110.0], "eval_MRN": "S3"},
    {"id": "task7_1", "instruction": "What is the most recent CBG of the patient S4?", "context": NOW + ' The code for CBG is "GLU"', "sol": [105], "eval_MRN": "S4"},
    {"id": "task10_1", "instruction": "What's the last HbA1C value in the chart for patient S5 and when was it recorded?", "context": NOW + ' The code for HbA1C is "A1C".', "sol": [], "eval_MRN": "S5"},
]


def get_line(case: dict) -> str:
    cat = case["id"].split("_", 1)[0]
    if cat in CODES:
        return f"GET {FHIR_BASE}Observation?patient={case['eval_MRN']}&code={CODES[cat]}&_count=5000"
    return f"GET {FHIR_BASE}Patient?identifier={case.get('eval_MRN') or 'S1'}"


# fixed replies (not a function of any task's gold): a wrong first answer citing nothing useful, a still-wrong repair
FINALS = {c["id"]: ('FINISH([7])\nCitations: ["Observation/not-a-real-id"]', 'FINISH([8])\nCitations: []') for c in CASES}


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return "\n".join(f"<{m['role']}>{m['content']}" for m in messages) + "\n<assistant>"


class ScriptedBackend:
    """Turn 1 GETs the task's chart, turn 2 and the repair turn answer from `finals`; records every prompt."""
    name = "scripted"

    def __init__(self, finals: dict[str, tuple[str, str]]):
        self.finals = finals
        self.turns: dict[str, int] = {}
        self.prompts: list[tuple[str, str]] = []

    def generate_many_raw(self, requests):
        out = {}
        for r in requests:
            n = self.turns[r.key] = self.turns.get(r.key, 0) + 1
            self.prompts.append((r.key, r.prompt))
            text = get_line(CASE_BY_ID[r.key]) if n == 1 else self.finals[r.key][min(n - 2, 1)]
            out[r.key] = Generation(text=text, tokens_in=400, tokens_out=20, latency_seconds=0.01)
        return out

    def metadata(self):
        return {"backend": "scripted"}


CASE_BY_ID = {c["id"]: c for c in CASES}


@pytest.fixture
def fake_fhir(monkeypatch):
    from run_medagentbench_corrected import OFFICIAL_TASK_MODULE, extract_official_prompt, load_official_grader
    if not (OFFICIAL_TASK_MODULE.exists() and FUNCTIONS.exists()):
        pytest.skip("Clone the pinned MedAgentBench source for this integration check")
    monkeypatch.setattr(medagent, "get_json_text", fake_get)
    refsol, _ = load_official_grader()
    refsol.send_get_request = fake_get
    tasks = {c["id"]: make_task(c, FHIR_BASE) for c in CASES}
    return refsol, extract_official_prompt(OFFICIAL_TASK_MODULE), json.load(open(FUNCTIONS)), tasks


def run_config(env, config: str, out: Path, finals=FINALS, tasks=None, cases=CASES):
    refsol, prompt, functions, real_tasks = env
    backend = ScriptedBackend(finals)
    budget = 600 if medagent.CONFIGS[config].get("budget") else None
    meta = run_medagent(MedAgentSpec(configuration=config, token_budget=budget), copy.deepcopy(cases), tasks or real_tasks, prompt, functions, backend,
                        FakeTokenizer(), refsol, out)
    rows = {r["task_id"]: r for r in read_jsonl(out / "predictions.jsonl")}
    return backend.prompts, rows, meta


# ------------------------------------------------------------------------------------------- (d) existing configurations
EXISTING = ["prompt_only", "wrong_basis", "scope_aware", "audit", "full", "token_matched", "high_compute", "security_label_consent", "direct_api"]
# prompts, rows, access logs and stages of every existing configuration with this harness. The parent commit (5afbe11)
# gave f266ef8d…; DEV-37 (the released grader's answer shapes for task4 / task10) changes only the `full` rows whose
# answer the skill replaces (task4_1, task10_1: parsed_answer, skill_summary and the grade), checked by a row diff.
EXISTING_DIGEST = "0710977f83b68194f0c59ae1fe936e22fb5e57f5ce302603d36dd87c7cbe5cd4"
ADDENDUM_SHA256 = "4c01bb8873bf18883bf78a82d0690444ac7dfdfdd722d784890c294cbd9a2bee"


def existing_digest(env, tmp_path: Path) -> str:
    h = hashlib.sha256()
    for config in EXISTING:
        prompts, rows, _ = run_config(env, config, tmp_path / config)
        h.update(json.dumps({"config": config, "prompts": prompts, "rows": rows}, sort_keys=True).encode())
        for name in ("access_log.jsonl", "stages.jsonl"):
            h.update((tmp_path / config / name).read_bytes())
    return h.hexdigest()


def test_existing_configurations_are_unchanged(fake_fhir, tmp_path):
    assert hashlib.sha256(medagent.SCOPEBENCH_ADDENDUM.encode()).hexdigest() == ADDENDUM_SHA256
    assert existing_digest(fake_fhir, tmp_path) == EXISTING_DIGEST


# ------------------------------------------------------------------------------------------- (a) executor = reference skill
def skill_answer_v0(ep) -> dict | None:
    """`medagent.skill_answer` as of the parent commit (5afbe11), verbatim, before it was moved onto the shared executor."""
    from scopebench.tierc.medagent import CATEGORY_RULES
    cat = ep.task.metadata["category"]; rule = CATEGORY_RULES[cat]
    index = parse_time(ep.task.contract.index_time).replace(tzinfo=None)
    if rule["rt"] in ("latest", "latest_in_window", "all_in_window"):
        obs = [r for r in ep.store.retrieved.values() if r.get("resourceType") == "Observation" and ep.store.is_legal(r)]
        code = rule["code"]
        def has_code(r):
            c = r.get("code") or {}
            return (c.get("text") == code) or any(x.get("code") == code for x in c.get("coding", []))
        obs = [r for r in obs if has_code(r) and r.get("effectiveDateTime")]
        if not obs:
            return {"status": "insufficient", "answer": [-1], "provenance": []}
        timed = sorted(((datetime.fromisoformat(r["effectiveDateTime"]).replace(tzinfo=None), str(r["id"]), (r.get("valueQuantity") or {}).get("value")) for r in obs))
        if rule["rt"] == "all_in_window":
            lo = index - timedelta(hours=rule["window_hours"])
            sel = [t for t in timed if lo < t[0] <= index and t[2] is not None]
            return {"status": "ok", "answer": [sum(t[2] for t in sel) / len(sel) if sel else -1], "provenance": [t[1] for t in sel]}
        if rule["rt"] == "latest_in_window":
            lo = index - timedelta(hours=rule["window_hours"])
            sel = [t for t in timed if lo < t[0] <= index]
        else:
            sel = [t for t in timed if t[0] <= index]
        if not sel:
            return {"status": "ok", "answer": [-1], "provenance": []}
        best = sel[-1]
        if cat == "task4":
            return {"status": "ok", "answer": [best[2], best[0].strftime("%Y-%m-%dT%H:%M:%S+00:00")], "provenance": [best[1]]}
        return {"status": "ok", "answer": [best[2]], "provenance": [best[1]]}
    if cat == "task2":
        pats = [r for r in ep.store.retrieved.values() if r.get("resourceType") == "Patient" and r.get("birthDate")]
        if not pats:
            return None
        p = pats[0]
        dob = datetime.fromisoformat(p["birthDate"])
        age = index.year - dob.year - ((index.month, index.day) < (dob.month, dob.day))
        return {"status": "ok", "answer": [age], "provenance": [str(p["id"])]}
    return None


def skill_answer_dev37(ep) -> dict | None:
    """skill_answer_v0 with the DEV-37 correction only: the released grader's answer shapes (task4 [value]; task10
    [value, the recorded effectiveDateTime string]); everything else as v0."""
    ref = skill_answer_v0(ep)
    cat = ep.task.metadata["category"]
    if isinstance(ref, dict) and ref["status"] == "ok" and ref["provenance"]:
        if cat == "task4":
            ref = {**ref, "answer": ref["answer"][:1]}
        elif cat == "task10":
            ref = {**ref, "answer": [ref["answer"][0], ep.store.retrieved[ref["provenance"][0]]["effectiveDateTime"]]}
    return ref


IDENTITY_CASES = CASES + [
    {"id": "task9_1", "instruction": "Check patient S7's most recent potassium level.", "context": NOW + ' The code for potassium is "K".', "eval_MRN": "S7"},
    {"id": "task5_1", "instruction": "Check patient S2's last serum magnesium level within last 24 hours.", "context": NOW + ' The code for magnesium is "MG".', "eval_MRN": "S2"},
    {"id": "task3_1", "instruction": "Record a blood pressure for S2.", "context": NOW, "eval_MRN": "S2"},
    {"id": "task8_1", "instruction": "Order a referral for S2.", "context": NOW, "eval_MRN": "S2"}]
RESOURCES = list(PATIENTS.values()) + [{"resourceType": "Patient", "id": "S9"}] + OBSERVATIONS + EXTRA


def store_variants():
    """Retrieved sets: everything, nothing, the served chart, the edge cases, each resource alone and all but each."""
    yield "all", RESOURCES
    yield "empty", []
    yield "served", list(PATIENTS.values()) + OBSERVATIONS
    yield "extra", EXTRA
    for i, r in enumerate(RESOURCES):
        yield f"only_{r['id']}", [r]
        yield f"without_{r['id']}", RESOURCES[:i] + RESOURCES[i + 1:]


def _outcome(fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — both implementations must fail alike
        return f"{type(exc).__name__}: {exc}"


@pytest.mark.parametrize("enforcement", ["none", "contract"])
def test_executor_of_the_reference_plan_is_the_reference_skill(fake_fhir, enforcement):
    from scopebench.tierc.medagent import SKILL_COVERED, Episode, skill_answer
    from scopebench.tierc.medagent_plans import execute, reference_plan
    n = 0
    for case in IDENTITY_CASES:
        task = make_task(case, FHIR_BASE)
        plan = reference_plan(task)
        assert (plan.operation is not None) == (task.metadata["category"] in SKILL_COVERED)
        for name, resources in store_variants():
            store = MedAgentStore(base_url=FHIR_BASE, contract=task.contract, enforcement=enforcement, patient_hash=task.patient,
                                  retrieved={r["id"]: copy.deepcopy(r) for r in resources})
            ep = Episode(task=task, case=case, store=store, messages=[])
            ref = _outcome(lambda: skill_answer_dev37(ep))
            assert _outcome(lambda: skill_answer(ep)) == ref, (case["id"], name)
            if plan.operation is None:
                continue
            res = execute(plan, store, task.contract.index_time)
            if isinstance(ref, str):
                assert res.error == ref and res.required == [], (case["id"], name)
            elif ref is None:
                assert res.summary["status"] == "insufficient" and res.required == [] and not res.ok, (case["id"], name)
            else:
                assert json.dumps(res.summary) == json.dumps(ref) and res.required == (ref["provenance"] if ref["status"] == "ok" else []), (case["id"], name)
            n += 1
    assert n == 7 * (4 + 2 * len(RESOURCES))   # seven covered tasks (task2, task4 x2, task6, task7, task9, task10)


def test_reference_plans():
    from scopebench.plans import Plan
    from scopebench.schema import Task
    from scopebench.tierc.medagent_plans import reference_plan
    def plan(cat):
        return reference_plan(Task(cat, cat, "p", "C", "medagentbench_v1", "q", None, {}, (), metadata={"category": cat}))  # type: ignore[arg-type]
    assert plan("task2") == Plan("patient_age", {})
    assert plan("task4") == Plan("latest_lab_in_window", {"code": "MG", "window_hours": 24.0})
    assert plan("task6") == Plan("mean_lab_in_window", {"code": "GLU", "window_hours": 24.0})
    assert [plan(c) for c in ("task7", "task9")] == [Plan("latest_lab", {"code": c}) for c in ("GLU", "K")]
    assert plan("task10") == Plan("latest_lab_with_time", {"code": "A1C"})
    assert all(plan(c) == Plan(None) for c in ("task1", "task3", "task5", "task8"))


# ------------------------------------------------------------------------------------------- (b) reading plans
def test_plan_reading_is_lenient_and_reports_errors():
    from scopebench.plans import Plan
    from scopebench.tierc.medagent import parse_final
    from scopebench.tierc.medagent_plans import answers_agree, read
    head = 'FINISH([1.5])\nCitations: ["Observation/m1"]'
    rest, r = read(head + '\nPlan: {"operation": "Latest_Lab", "code": "glu ", "extra": 1}')
    assert rest == head and r.present and r.errors == [] and r.plan == Plan("latest_lab", {"code": "GLU"})
    assert parse_final(rest)["answer"] == [1.5] and parse_final(rest)["citations"] == ["m1"]
    ok = {'Plan: null': Plan(None), 'plan: None': Plan(None), 'Plan: "none"': Plan(None), 'Plan: {"operation": null}': Plan(None),
          'Plan: patient_age': Plan("patient_age", {}), 'Plan: "PATIENT_AGE"': Plan("patient_age", {}), '**Plan**: `{"operation": "patient_age"}`': Plan("patient_age", {}),
          'Plan: {"operation": "mean_lab_in_window", "code": "GLU", "window_hours": "24"}': Plan("mean_lab_in_window", {"code": "GLU", "window_hours": 24.0}),
          'Plan: {"operation": "latest_lab_in_window", "code": "MG", "window_hours": 12}': Plan("latest_lab_in_window", {"code": "MG", "window_hours": 12.0})}
    for line, want in ok.items():
        rest, r = read(head + "\n" + line)
        assert rest == head and r.errors == [] and r.plan == want, line
    # the Plan line before the Citations line, and the last Plan line counts
    rest, r = read('FINISH([1])\nPlan: patient_age\nCitations: []')
    assert parse_final(rest)["citations_line_present"] and r.plan == Plan("patient_age", {})
    assert read(head + "\nPlan: null\nPlan: patient_age")[1].plan == Plan("patient_age", {})
    bad = {'Plan: {"operation": "median_lab"}': "unknown operation", 'Plan: {"operation": "latest_lab"}': "needs \"code\"",
           'Plan: {"operation": "latest_lab", "code": 7}': "needs \"code\"",
           'Plan: {"operation": "mean_lab_in_window", "code": "GLU"}': "window_hours", 'Plan: {"operation": "mean_lab_in_window", "code": "GLU", "window_hours": "a day"}': "window_hours",
           'Plan: {"operation": "mean_lab_in_window", "code": "GLU", "window_hours": 0}': "window_hours",
           'Plan: {"operation": "mean_lab_in_window", "code": "GLU", "window_hours": true}': "window_hours",
           'Plan: [1, 2]': "must be a JSON object", 'Plan: {operation: latest_lab}': "must be one JSON object", 'Plan: latest lab': "must be one JSON object"}
    for line, msg in bad.items():
        rest, r = read(head + "\n" + line)
        assert rest == head and r.present and r.plan is None and len(r.errors) == 1 and msg in r.errors[0], (line, r.errors)
    for text in (head, "", None):
        _, r = read(text)
        assert not r.present and r.plan is None and "Plan: null" in r.errors[0]
    assert answers_agree([1.7, "2023-11-13T08:00:00Z"], [1.7000000001, "2023-11-13T08:00:00+00:00"]) and answers_agree([-1], [-1.0])
    assert not answers_agree([1.7], [1.7, "2023-11-13T08:00:00+00:00"]) and not answers_agree(["1.7"], [1.7]) and not answers_agree([1.71], [1.7])
    assert not answers_agree(None, [1]) and not answers_agree([True], [1]) and not answers_agree([1, "2023-11-13T09:00:00Z"], [1, "2023-11-13T08:00:00+00:00"])


# ------------------------------------------------------------------------------------------- (c) no reference input
def plan_finals(tasks) -> dict[str, tuple[str, str]]:
    """FINALS with the reference plan declared (fixed text, computed once from the real tasks)."""
    from scopebench.tierc.medagent_plans import reference_plan
    out = {}
    for tid, (a, b) in FINALS.items():
        line = "Plan: " + json.dumps(reference_plan(tasks[tid]).to_dict())
        out[tid] = (a + "\n" + line, b + "\n" + line)
    return out


def scrambled(task):
    """The same request with every reference input replaced: R_t, the gold, and the category (hence its parameters)."""
    t = copy.deepcopy(task)
    t.required_ids = ("h1", "g3", "not-a-real-id")
    t.gold = {"sol": [123], "rule": "latest", "value": 123, "time": "2020-01-01T00:00:00Z"}
    t.metadata = {**t.metadata, "category": "task2" if t.metadata["category"] != "task2" else "task7"}
    return t


INFERENCE_FIELDS = ("response", "repair_response", "parsed_initial", "parsed_answer", "access_ids", "illegal_access_ids", "repair_evidence_ids", "guard_repaired",
                    "guard_deterministic_fallback", "initial_guard_errors", "final_guard_errors", "skill_summary", "posts",
                    "evidence_plan_initial", "evidence_plan_final", "plan_result_initial", "plan_result_final")


def _real_and_scrambled(env, config, tmp_path, finals):
    tasks = env[3]
    cases_s = [{**c, "sol": [123]} for c in CASES]
    real = run_config(env, config, tmp_path / "real", finals=finals)
    scr = run_config(env, config, tmp_path / "scrambled", finals=finals, tasks={k: scrambled(t) for k, t in tasks.items()}, cases=cases_s)
    return real, scr


@pytest.mark.parametrize("config", ["plan_audit", "plan_full"])
def test_plan_configurations_read_no_reference_input(fake_fhir, config, tmp_path):
    from scopebench.tierc.medagent_plans import PLAN_ADDENDUM
    tasks = fake_fhir[3]
    (prompts_a, rows_a, meta), (prompts_b, rows_b, _) = _real_and_scrambled(fake_fhir, config, tmp_path, plan_finals(tasks))
    assert prompts_a == prompts_b
    assert {k: {f: r.get(f) for f in INFERENCE_FIELDS} for k, r in rows_a.items()} == {k: {f: r.get(f) for f in INFERENCE_FIELDS} for k, r in rows_b.items()}
    # the pipeline did what it is for
    assert meta["plan_addendum_sha256"] == hashlib.sha256(PLAN_ADDENDUM.encode()).hexdigest()
    assert all(p.endswith(PLAN_ADDENDUM + "\n<assistant>") for k, p in prompts_a if p.count("<user>") == 1)
    for tid, r in rows_a.items():
        task = tasks[tid]
        assert r["benchmark_native"]["status"] == "completed" and r["guard_repaired"], tid    # the Plan line does not make FINISH an invalid action
        assert any(e.startswith("resolution") for e in r["initial_guard_errors"]), tid
        res = r["plan_result_initial"]
        if task.metadata["category"] == "task1":
            assert res is None and r["evidence_plan_initial"]["declared_none"] and not r["guard_deterministic_fallback"]
            continue
        assert res["required"] == list(task.required_ids), tid                 # R̂_t of the reference plan is R_t here
        repair = next(p for k, p in prompts_a if k == tid and "scopeguard_errors" in p)
        assert ("evidence_plan_result" in repair) == (config == "plan_full" and res["summary"]["status"] == "ok"), tid
        replaced = config == "plan_full" and res["summary"]["status"] == "ok"
        assert r["guard_deterministic_fallback"] == replaced, tid
        if replaced:
            assert r["parsed_answer"]["answer"] == res["summary"]["answer"] and r["final_guard_errors"] == [] and r["metrics"]["completeness"] == 1.0, tid
            assert any(e.startswith("agreement") for e in r["initial_guard_errors"]), tid
    if config == "plan_full":   # the released grader accepts the replaced answers (DEV-37 shapes)
        assert all(rows_a[t]["metrics"]["answer_correct"] for t in ("task2_1", "task4_1", "task6_1", "task7_1"))


def test_reference_audit_does_read_reference_inputs(fake_fhir, tmp_path):
    """Positive control: the same comparison sees the reference configurations read R_t and the category."""
    for config in ("audit", "full"):
        (prompts_a, rows_a, _), (prompts_b, rows_b, _) = _real_and_scrambled(fake_fhir, config, tmp_path / config, FINALS)
        assert prompts_a != prompts_b and any(rows_a[k]["final_guard_errors"] != rows_b[k]["final_guard_errors"] for k in rows_a)


# ------------------------------------------------------------------------------------------- identity on the official server
def test_reference_plan_identity_on_the_official_server():
    """Every covered MedAgentBench task: with all of the patient's observations of the category code (and the Patient)
    retrieved, executing the reference plan gives the reference skill's result."""
    from scopebench.schema import Task
    from scopebench.tierc.medagent import CATEGORY_RULES, SKILL_COVERED, Episode, get_json_text, skill_answer
    from scopebench.tierc.medagent_plans import execute, reference_plan
    if not (MANIFEST / "tasks.jsonl").exists() or "error" in get_json_text(f"{FHIR_BASE}metadata", timeout=5):
        pytest.skip("MedAgentBench manifest or FHIR server not available")
    cases = {c["id"]: c for c in json.load(open(TASK_DATA))}
    n = 0
    for row in read_jsonl(MANIFEST / "tasks.jsonl"):
        task = Task.from_dict(row)
        cat = task.metadata["category"]
        if cat not in SKILL_COVERED:
            continue
        store = MedAgentStore(base_url=FHIR_BASE, contract=task.contract, enforcement="none", patient_hash=task.patient)
        mrn = task.metadata["eval_mrn"]
        store.get(1, f"{FHIR_BASE}Patient?identifier={mrn}" if cat == "task2" else f"{FHIR_BASE}Observation?patient={mrn}&code={CATEGORY_RULES[cat]['code']}&_count=5000")
        assert not store.log[-1].error, task.task_id
        ep = Episode(task=task, case=cases[task.task_id], store=store, messages=[])
        ref = skill_answer_dev37(ep)
        res = execute(reference_plan(task), store, task.contract.index_time)
        assert skill_answer(ep) == ref and ref is not None and json.dumps(res.summary) == json.dumps(ref), task.task_id
        assert res.required == (ref["provenance"] if ref["status"] == "ok" else []), task.task_id
        n += 1
    assert n == 180
