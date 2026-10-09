"""Tier-C loop mechanics with a scripted backend over the real FHIR demo store (integration; skipped without the data):
tool calls are executed and logged, scope enforcement filters results, the citations line is read, the guard's
repair refill supplies missing legal R_t resources, and the output rows carry the shared metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scopebench.config import PUBLIC_DATA  # noqa: E402
from scopebench.schema import Task, read_jsonl  # noqa: E402

FHIR_DIR = PUBLIC_DATA / "mimic-iv-clinical-database-demo-on-fhir-2.1.0" / "fhir"
MANIFEST = ROOT / "experiments/scopebench_tierC/fhir_agentbench"
PMAP = PUBLIC_DATA / ".clinehr_cache/scopebench_tierc_patients/fhir_agentbench_patient_map.json"

pytestmark = pytest.mark.skipif(not (FHIR_DIR.exists() and MANIFEST.exists() and PMAP.exists()), reason="FHIR demo / Tier-C manifest not present")


class ScriptedBackend:
    """Turn 1: retrieve Observations for the patient; turn 2: final answer citing R_t (or a wrong id when told to)."""
    name = "scripted"

    def __init__(self, cite_required: dict[str, list[str]], wrong_first: set[str] = frozenset()):
        self.cite_required = cite_required
        self.wrong_first = set(wrong_first)
        self.turns: dict[str, int] = {}

    def _answer(self, key: str, messages) -> str:
        n = self.turns.get(key, 0) + 1
        self.turns[key] = n
        if n == 1:
            pid = [m for m in messages if m["role"] == "user"][0]["content"].split("Patient FHIR ID: ")[1].split("\n")[0]
            return json.dumps({"name": "get_resources_by_patient_fhir_id", "arguments": {"resource_type": "Observation", "patient_fhir_id": pid}}).join(["<tool_call>\n", "\n</tool_call>"])
        cites = ["not-a-real-id"] if (key in self.wrong_first and n == 2) else self.cite_required[key]
        return f'The final answer is: [["2133-12-31 02:00:00"]]\nCitations: {json.dumps(cites)}'

    def generate_many(self, requests):
        from scopebench.backends.base import Generation
        return {r.key: Generation(text=self._answer(r.key, r.messages), tokens_in=100, tokens_out=20, latency_seconds=0.01) for r in requests}

    def metadata(self):
        return {"backend": "scripted"}


@pytest.fixture(scope="module")
def store():
    from scopebench.tierc.store import load_store
    return load_store(FHIR_DIR)


def _tasks(n: int = 3) -> list[Task]:
    tasks = [Task.from_dict(r) for r in read_jsonl(MANIFEST / "tasks.jsonl")]
    picked = [t for t in tasks if t.contract.scope == "pre" and t.metadata["main_table"] == "chartevents" and t.metadata["rt_in_lt"]][:n]
    assert len(picked) == n
    return picked


def test_loop_scope_aware_and_audit_repair(store, tmp_path):
    from scopebench.tierc.agent import TierCSpec, run_tierc
    tasks = _tasks(3)
    pmap = json.load(open(PMAP))
    cites = {t.task_id: list(t.required_ids) for t in tasks}
    # scope_aware: enforcement filters the Observation call; the scripted answer cites R_t -> complete, legal, auditable
    backend = ScriptedBackend(cites)
    meta = run_tierc(TierCSpec(configuration="scope_aware", adapter="text"), tasks, store, pmap, backend, None, tmp_path / "scope_aware", manifest_path=MANIFEST)
    rows = list(read_jsonl(tmp_path / "scope_aware/predictions.jsonl"))
    assert meta["n_tasks_this_run"] == 3 and len(rows) == 3
    for r in rows:
        assert r["metrics"]["access_legal"] and r["metrics"]["citation_legal"] and r["metrics"]["completeness"] == 1.0 and r["metrics"]["auditable"]
        assert r["benchmark_native"]["rounds"] == 2 and r["metrics"]["calls"] == 2
        assert r["n_illegal_access_ids"] == 0 and r["n_access_ids"] > 0
    log = list(read_jsonl(tmp_path / "scope_aware/access_log.jsonl"))
    assert all(e["entries"][0]["n_filtered_out"] > 0 for e in log)   # out-of-scope observations never reached the model
    # prompt_only: nothing filtered, illegal access recorded
    backend = ScriptedBackend(cites)
    run_tierc(TierCSpec(configuration="prompt_only", adapter="text"), tasks, store, pmap, backend, None, tmp_path / "prompt_only", manifest_path=MANIFEST)
    rows = list(read_jsonl(tmp_path / "prompt_only/predictions.jsonl"))
    assert all(not r["metrics"]["access_legal"] and r["n_illegal_access_ids"] > 0 for r in rows)
    # audit: the first task cites an unresolvable id -> guard error -> repair turn -> correct citations; the others pass untouched
    first = tasks[0].task_id
    backend = ScriptedBackend(cites, wrong_first={first})
    run_tierc(TierCSpec(configuration="audit", adapter="text"), tasks, store, pmap, backend, None, tmp_path / "audit", manifest_path=MANIFEST)
    rows = {r["task_id"]: r for r in read_jsonl(tmp_path / "audit/predictions.jsonl")}
    assert rows[first]["guard_repaired"] and rows[first]["initial_guard_errors"] and rows[first]["final_guard_errors"] == []
    assert rows[first]["metrics"]["completeness"] == 1.0 and not rows[first]["initial_metrics"]["citation_legal"]
    assert all(not rows[t.task_id]["guard_repaired"] for t in tasks[1:])
    # resumable: a second call does nothing
    meta2 = run_tierc(TierCSpec(configuration="audit", adapter="text"), tasks, store, pmap, backend, None, tmp_path / "audit", manifest_path=MANIFEST)
    assert meta2["n_tasks_this_run"] == 0 and meta2["n_tasks_previously_done"] == 3


def test_full_verifies_against_the_template_skill_and_replaces(store, tmp_path):
    """DEV-26: under `full` a covered template's skill verifies the submission; a submission that passes the guard but
    disagrees with the skill is replaced deterministically, and the replacement's provenance enters A_t as legal access."""
    from types import SimpleNamespace

    from fhir_benchmark_core import canonical_answer_correct
    from scopebench.tierc.agent import TierCSpec, run_tierc
    from scopebench.tierc.skills import mapped_skills

    pmap = json.load(open(PMAP))
    validation = ROOT / "experiments/scopebench_tierC/skill_validation.json"
    skills = mapped_skills(validation, pmap)
    assert skills, "no covered template"
    per_task = {r["task_id"]: r for r in read_jsonl(validation.with_suffix(".per_task.jsonl"))}
    fake = '[["2133-12-31 02:00:00"]]'
    tasks = [Task.from_dict(r) for r in read_jsonl(MANIFEST / "tasks.jsonl")]
    picked = None
    for t in tasks:
        v = per_task.get(t.task_id)
        if not v or v.get("status") != "ok" or not v.get("agree") or t.metadata["template"] not in skills or not t.metadata.get("rt_in_lt"):
            continue
        res = skills[t.metadata["template"]](SimpleNamespace(store=store), t)
        gold = t.gold.get("answer")
        if res.get("status") == "ok" and set(t.required_ids) <= set(res["provenance"]) and not canonical_answer_correct(fake, gold if isinstance(gold, str) else json.dumps(gold)):
            picked = t; break
    assert picked is not None
    cites = {picked.task_id: list(picked.required_ids)}
    # audit: the fake answer cites R_t, so the guard passes and the wrong value stands
    run_tierc(TierCSpec(configuration="audit", adapter="text"), [picked], store, pmap, ScriptedBackend(cites), None, tmp_path / "audit", manifest_path=MANIFEST)
    a = next(read_jsonl(tmp_path / "audit/predictions.jsonl"))
    assert not a["guard_deterministic_fallback"] and not a["metrics"]["answer_correct"] and a["skill_summary"] is None
    # full with the mapped skill: disagreement -> replacement -> correct, complete, legal, auditable
    spec = TierCSpec(configuration="full", adapter="text", skills=skills)
    meta = run_tierc(spec, [picked], store, pmap, ScriptedBackend(cites), None, tmp_path / "full", manifest_path=MANIFEST)
    assert meta["skills_mapped_templates"] == sorted(skills)
    f = next(read_jsonl(tmp_path / "full/predictions.jsonl"))
    assert f["guard_deterministic_fallback"] and f["parsed_answer"]["replaced"]
    assert f["skill_summary"]["status"] == "ok" and f["skill_summary"]["agrees_with_submission"] is False and "_provenance_pairs" not in f["skill_summary"]
    assert any("template skill" in r for r in f["replacement_reason"]) and f["final_guard_errors"] == []
    m = f["metrics"]
    assert m["answer_correct"] and m["completeness"] == 1.0 and m["access_legal"] and m["citation_legal"] and m["auditable"]
    assert set(f["parsed_answer"]["citations"]) == set(f["skill_summary"]["provenance"]) and f["n_illegal_access_ids"] == 0
    log = next(e for e in read_jsonl(tmp_path / "full/access_log.jsonl") if e["task_id"] == picked.task_id)
    assert any(e["tool"] == "skill_provenance" for e in log["entries"])
    # the same submission that agrees with the skill is left alone
    agreeing = ScriptedBackend(cites)
    agreeing._answer = lambda key, messages, _orig=agreeing._answer: (_orig(key, messages) if agreeing.turns.get(key, 0) == 0
                                                                    else f"The final answer is: {json.dumps(f['skill_summary']['answer'])}\nCitations: {json.dumps(cites[key])}")
    run_tierc(spec, [picked], store, pmap, agreeing, None, tmp_path / "full_agree", manifest_path=MANIFEST)
    g = next(read_jsonl(tmp_path / "full_agree/predictions.jsonl"))
    assert not g["guard_deterministic_fallback"] and g["skill_summary"]["agrees_with_submission"] is True and g["metrics"]["answer_correct"]


def test_negative_skill_results_cite_the_examined_resources(store):
    """A skill result with nothing to cite (no open visit, nothing matched) cites the resources it examined, all legal."""
    from types import SimpleNamespace

    from scopebench.predicate import legal
    from scopebench.tierc.contracts import resource_to_evidence
    from scopebench.tierc.skills import ABSENCE_PROVENANCE_CAP, _solve, legal_set_with_references, mapped_skills, patient_pairs

    pmap = json.load(open(PMAP))
    skills = mapped_skills(ROOT / "experiments/scopebench_tierC/skill_validation.json", pmap)
    tasks = [Task.from_dict(r) for r in read_jsonl(MANIFEST / "tasks.jsonl")]
    checked = 0
    for t in tasks:
        if t.metadata["template"] not in skills:
            continue
        pid = pmap[t.patient]
        legal_rs, extra = legal_set_with_references(store, patient_pairs(store, pid, t.patient), t.contract, pid)
        raw = _solve(t.metadata, legal_rs + extra)
        if raw.get("status") != "ok" or raw.get("provenance"):
            continue
        res = skills[t.metadata["template"]](SimpleNamespace(store=store), t)
        if not res["provenance"]:
            continue   # nothing of that table in the legal set: genuinely nothing to cite
        assert res["provenance_kind"] == "examined" and len(res["provenance"]) <= ABSENCE_PROVENANCE_CAP
        assert res["answer"] == raw["answer"]
        legal_ids = {str(r["id"]) for r in legal_rs + extra}
        assert set(res["provenance"]) <= legal_ids
        checked += 1
        if checked >= 25:
            break
    assert checked > 0
