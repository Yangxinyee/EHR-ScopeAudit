"""MedAgentBench loop mechanics with a scripted backend against the official FHIR server (integration; skipped when the
server or the manifest is absent): GET results are logged as A_t and filtered under scope_aware, the FINISH answer and
its Citations line are parsed, the released grader scores the answer, the guard's repair refill supplies the missing
legal required resource under audit, and the output rows carry the shared metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scopebench.backends.base import Generation  # noqa: E402
from scopebench.schema import Task, read_jsonl  # noqa: E402
from scopebench.tierc.medagent import FHIR_BASE, MedAgentSpec, get_json_text, run_medagent  # noqa: E402

MANIFEST = ROOT / "experiments/scopebench_tierC/medagentbench"
TASK_DATA = ROOT / "external_benchmarks/MedAgentBench/data/medagentbench/test_data_v2.json"
FUNCTIONS = ROOT / "external_benchmarks/MedAgentBench/data/medagentbench/funcs_v1.json"

pytestmark = pytest.mark.skipif(not (MANIFEST / "tasks.jsonl").exists() or "error" in get_json_text(f"{FHIR_BASE}metadata", timeout=5),
                                reason="MedAgentBench manifest or FHIR server not available")


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return "\n".join(f"<{m['role']}>{m['content']}" for m in messages) + "\n<assistant>"


class ScriptedBackend:
    """Turn 1: GET the GLU observations of the task's patient; turn 2: FINISH with the gold value, citing the required id
    (or a wrong id first when `wrong_first`, so the audit repair has something to fix)."""
    name = "scripted"

    def __init__(self, tasks: dict[str, Task], wrong_first: bool = False):
        self.tasks = tasks
        self.wrong_first = wrong_first
        self.turns: dict[str, int] = {}

    def generate_many_raw(self, requests):
        out = {}
        for r in requests:
            n = self.turns.get(r.key, 0) + 1
            self.turns[r.key] = n
            t = self.tasks[r.key]
            if n == 1:
                text = f"GET {FHIR_BASE}Observation?patient={t.metadata['eval_mrn']}&code=GLU&_count=5000"
            else:
                value = t.gold.get("value"); value = value if value is not None else -1
                cites = ["not-a-real-id"] if (self.wrong_first and n == 2) else list(t.required_ids)
                text = f"FINISH([{json.dumps(value)}])\nCitations: {json.dumps(cites)}"
            out[r.key] = Generation(text=text, tokens_in=500, tokens_out=30, latency_seconds=0.01)
        return out

    def metadata(self):
        return {"backend": "scripted"}


@pytest.fixture(scope="module")
def env():
    from run_medagentbench_corrected import OFFICIAL_TASK_MODULE, extract_official_prompt, load_official_grader
    refsol, _ = load_official_grader()
    refsol.send_get_request = get_json_text
    tasks = {r["task_id"]: Task.from_dict(r) for r in read_jsonl(MANIFEST / "tasks.jsonl")}
    cases = {c["id"]: c for c in json.load(open(TASK_DATA))}
    picked = [t for t in tasks.values() if t.metadata["category"] == "task7" and t.required_ids][:3]
    assert len(picked) == 3
    return refsol, extract_official_prompt(OFFICIAL_TASK_MODULE), json.load(open(FUNCTIONS)), tasks, [cases[t.task_id] for t in picked]


def _run(env, config, tmp_path, wrong_first=False):
    refsol, prompt, functions, tasks, cases = env
    spec = MedAgentSpec(configuration=config)
    meta = run_medagent(spec, cases, tasks, prompt, functions, ScriptedBackend(tasks, wrong_first), FakeTokenizer(), refsol, tmp_path / config)
    rows = list(read_jsonl(tmp_path / config / "predictions.jsonl"))
    return meta, rows


def test_prompt_only_logs_access_and_grades(env, tmp_path):
    meta, rows = _run(env, "prompt_only", tmp_path)
    assert meta["n_tasks_this_run"] == 3 and len(rows) == 3
    for r in rows:
        assert r["n_access_ids"] >= 1 and not r["generation_errors"]
        assert r["metrics"]["answer_correct"] and r["benchmark_native"]["correct"], r["task_id"]
        assert r["metrics"]["completeness"] == 1.0 and r["metrics"]["citation_legal"] and r["metrics"]["auditable"]
        assert r["benchmark_native"]["rounds"] == 2
        assert r["benchmark_native"]["status"] == "completed"           # DEV-23: FINISH followed by its Citations line is a completed episode
    stages = {s["task_id"]: s for s in read_jsonl(tmp_path / "prompt_only/stages.jsonl")}
    assert all(stages[r["task_id"]]["status"] == "completed" for r in rows)


def test_scope_aware_filters_out_of_scope_results(env, tmp_path):
    meta, rows = _run(env, "scope_aware", tmp_path)
    for r in rows:
        assert r["metrics"]["access_legal"], r["task_id"]          # nothing outside the contract reached the agent
        assert set(r["repair_evidence_ids"]) == set()
    log = list(read_jsonl(tmp_path / "scope_aware/access_log.jsonl"))
    assert all(e["entries"][0]["tool"] == "GET" for e in log)


def test_audit_repairs_a_wrong_citation(env, tmp_path):
    meta, rows = _run(env, "audit", tmp_path, wrong_first=True)
    for r in rows:
        assert r["guard_repaired"] and any("resolution" in e or "completeness" in e for e in r["initial_guard_errors"]), r["task_id"]
        assert r["metrics"]["completeness"] == 1.0 and r["metrics"]["answer_correct"], r["task_id"]
        assert r["final_guard_errors"] == []


def test_late_filter_drops_illegal_citations_and_keeps_the_answer(env, tmp_path):
    from scopebench.tierc.medagent import late_filter_row
    _, rows = _run(env, "prompt_only", tmp_path)
    tasks = env[3]
    for r in rows:
        task = tasks[r["task_id"]]
        illegal = list(r.get("illegal_access_ids") or [])[:1]
        bad = dict(r)
        bad["parsed_answer"] = {**r["parsed_answer"], "citations": list(task.required_ids) + ["not-a-real-id", task.metadata["eval_mrn"]] + illegal}
        out = late_filter_row(bad, task)
        assert out["configuration"] == "late_filter" and out["parsed_answer"]["citations"] == list(task.required_ids), r["task_id"]
        assert out["late_filter_removed"] == 2 + len(illegal)
        assert out["metrics"]["answer_correct"] == r["metrics"]["answer_correct"] and out["benchmark_native"] == r["benchmark_native"]
        assert out["metrics"]["completeness"] == 1.0 and out["metrics"]["citation_legal"] and out["metrics"]["auditable"]


SGLANG_OVERFLOW = ('HTTP 400: {"object":"error","message":"The input (33012 tokens) is longer than the model\'s context length '
                   '(32768 tokens).","type":"BadRequestError","code":400}')


class OverflowBackend(ScriptedBackend):
    """Two GETs, then — for the first task only — the third request is rejected at the context limit (DEV-19); the
    guarded request (the environment's last response withheld) is answered with FINISH."""

    def __init__(self, tasks, overflow_key):
        super().__init__(tasks)
        self.overflow_key = overflow_key
        self.seen: dict[str, list] = {}

    def generate_many_raw(self, requests):
        out = {}
        for r in requests:
            self.seen.setdefault(r.key, []).append(r)
            n = len(self.seen[r.key])
            t = self.tasks[r.key]
            if "reached the model's context limit" in r.prompt or n >= 3 and r.key != self.overflow_key:
                value = t.gold.get("value"); value = value if value is not None else -1
                out[r.key] = Generation(text=f"FINISH([{json.dumps(value)}])\nCitations: {json.dumps(list(t.required_ids))}", tokens_in=30000, tokens_out=30, latency_seconds=0.01)
            elif n >= 3:
                out[r.key] = Generation(text="", tokens_in=0, tokens_out=0, latency_seconds=0.0, error=SGLANG_OVERFLOW)
            else:
                out[r.key] = Generation(text=f"GET {FHIR_BASE}Observation?patient={t.metadata['eval_mrn']}&code=GLU&_count=5000",
                                        tokens_in=31000 if n == 2 else 2000, tokens_out=30, latency_seconds=0.01)
        return out


def test_context_guard_asks_for_finish_and_keeps_the_agent_turns(env, tmp_path):
    from scopebench.context_guard import final_max_tokens
    refsol, prompt, functions, tasks, cases = env
    first = cases[0]["id"]
    backend = OverflowBackend(tasks, first)
    run_medagent(MedAgentSpec(configuration="prompt_only"), cases, tasks, prompt, functions, backend, FakeTokenizer(), refsol, tmp_path / "po")
    rows = {r["task_id"]: r for r in read_jsonl(tmp_path / "po" / "predictions.jsonl")}
    a = rows[first]
    assert not a["generation_errors"] and a["context_guard"] and a["metrics"]["answer_correct"] and a["benchmark_native"]["correct"]
    assert backend.seen[first][-1].max_tokens == final_max_tokens(MedAgentSpec(configuration="prompt_only").max_new_tokens, 32768, 31000, 30)
    assert "reached the model's context limit" in backend.seen[first][-1].prompt and len(backend.seen[first]) == 4
    for c in cases[1:]:
        r = rows[c["id"]]
        assert not r["context_guard"] and not r["generation_errors"] and len(backend.seen[c["id"]]) == 3


class DiesBeforeRound2(ScriptedBackend):
    """Answers round 1; the next request kills the harness (DEV-22 checkpoint test)."""

    def generate_many_raw(self, requests):
        if any(self.turns.get(r.key) for r in requests):
            raise RuntimeError("simulated harness death between rounds")
        return super().generate_many_raw(requests)


def test_checkpoint_resumes_without_reissuing_earlier_rounds(env, tmp_path):
    from scopebench.tierc.medagent import CHECKPOINT_NAME
    refsol, prompt, functions, tasks, cases = env
    spec = MedAgentSpec(configuration="prompt_only")
    out = tmp_path / "ck"
    with pytest.raises(RuntimeError):
        run_medagent(spec, cases, tasks, prompt, functions, DiesBeforeRound2(tasks), FakeTokenizer(), refsol, out)
    ck = json.loads((out / CHECKPOINT_NAME).read_text())
    assert ck["configuration"] == "prompt_only" and ck["next_round"] == 1 and {d["task_id"] for d in ck["episodes"]} == {c["id"] for c in cases}
    assert all(d["rounds"] == 1 and d["active"] and d["messages"][-1]["role"] == "user" and d["store"]["log"][0]["tool"] == "GET" for d in ck["episodes"])
    assert not (out / "predictions.jsonl").exists()
    # the restarted harness: its backend is only ever asked for round 2 (the scripted turn counter starts at 1 → FINISH)
    second = ScriptedBackend(tasks)
    second.turns = {c["id"]: 1 for c in cases}
    meta = run_medagent(spec, cases, tasks, prompt, functions, second, FakeTokenizer(), refsol, out)
    assert meta["checkpoint"] == {"resumed": True, "resumed_from_round": 1, "ignored": None}
    assert second.turns == {c["id"]: 2 for c in cases} and not (out / CHECKPOINT_NAME).exists()
    rows = {r["task_id"]: r for r in read_jsonl(out / "predictions.jsonl")}
    _, ref = _run(env, "prompt_only", tmp_path)
    for r in ref:
        a = rows[r["task_id"]]
        assert a["benchmark_native"]["rounds"] == 2 and a["benchmark_native"]["correct"] == r["benchmark_native"]["correct"]
        assert a["metrics"] == r["metrics"] and a["n_access_ids"] == r["n_access_ids"] and not a["generation_errors"]
    log = {e["task_id"]: e for e in read_jsonl(out / "access_log.jsonl")}
    assert all(len(log[t]["entries"]) == 1 and log[t]["entries"][0]["round"] == 1 for t in rows)      # the GET was not re-executed
    # a checkpoint of another run is ignored, not applied
    (out / CHECKPOINT_NAME).write_text(json.dumps({"configuration": "audit", "next_round": 1, "episodes": ck["episodes"]}))
    meta = run_medagent(spec, cases, tasks, prompt, functions, ScriptedBackend(tasks), FakeTokenizer(), refsol, out)
    assert meta["n_tasks_this_run"] == 0 and meta["checkpoint"]["resumed"] is False and meta["checkpoint"]["ignored"]
