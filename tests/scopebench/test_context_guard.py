"""DEV-18 / DEV-19: the masking replay tolerates malformed citations, the tool loops ask for the final answer when the
server rejects a turn at the context limit (and only then), and the compute-control loop honours --retry-errors."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scopebench.backends.base import Generation  # noqa: E402
from scopebench.context_guard import MIN_FINAL_TOKENS, NOTICE_TOKENS, final_max_tokens, parse_overflow  # noqa: E402

DEV = Path(os.environ.get("SCOPEBENCH_TIERB_DIR", ROOT / "experiments/scopebench_tierB_dev"))
SGLANG_INPUT = ('HTTP 400: {"object":"error","message":"The input (17012 tokens) is longer than the model\'s context length '
                '(16384 tokens).","type":"BadRequestError","param":null,"code":400}')
SGLANG_TOTAL = ('HTTP 400: {"object":"error","message":"Requested token count exceeds the model\'s maximum context length of 16384 '
                'tokens. You requested a total of 16900 tokens: 15876 tokens from the input messages and 1024 tokens for the '
                'completion. Please reduce the number of tokens in the input messages or the completion to fit within the limit."}')


# ------------------------------------------------------------------------------------------------------------- unit
def test_parse_overflow_reads_both_sglang_messages():
    assert parse_overflow(SGLANG_INPUT) == (17012, 16384)
    assert parse_overflow(SGLANG_TOTAL) == (15876, 16384)
    assert parse_overflow("HTTPStatusError: Client error '400 Bad Request' for url 'http://x/v1/chat/completions'") is None
    assert parse_overflow("RuntimeError: HTTP 503: busy") is None and parse_overflow(None) is None


def test_final_max_tokens_fits_after_the_last_request_that_fitted():
    assert final_max_tokens(1024, 16384, 3000, 40) == 1024
    assert final_max_tokens(1024, 16384, 16000, 40) == 16384 - 16040 - NOTICE_TOKENS
    assert final_max_tokens(1024, 16384, 16300, 80) == MIN_FINAL_TOKENS


# ------------------------------------------------------------------------------------------- compute-control loop
needs_dev = pytest.mark.skipif(not (DEV / "tasks.jsonl").exists(), reason="development manifest not present")


def _load(n: int):
    from scopebench.manifest import load_manifest
    tasks, cases = load_manifest(DEV)
    tasks = sorted(tasks, key=lambda t: (t.case_id, t.task_id))[:n]
    return tasks, {t.case_id: cases[t.case_id] for t in tasks}


def _answer(ids):
    return json.dumps({"status": "ok", "results": [{"citation": i, "value": 1.0, "unit": "u", "issued_time": "2020-01-01T00:00:00Z"} for i in ids],
                       "delta": 0.0, "direction": "stable", "decision": "exclude", "citations": ids, "label": "insufficient"})


class TwoSearchAgent:
    """Search, search again, answer. For keys in `overflow` the server rejects any request holding two tool results
    (context limit); the second search's prompt is reported as 16,000 tokens so the guarded request must be capped."""
    name = "two-search"

    def __init__(self, overflow: set[str]):
        self.overflow = overflow
        self.requests: dict[str, list] = {}

    def generate_many(self, requests):
        out = {}
        for r in requests:
            self.requests.setdefault(r.key, []).append(r)
            n_results = sum(1 for m in r.messages if m["role"] == "user" and m["content"].startswith('{"tool_result"'))
            last = r.messages[-1]["content"]
            if '"context_limit_reached": true' in last:
                prior = [json.loads(m["content"]) for m in r.messages if m["role"] == "user" and m["content"].startswith('{"tool_result"')]
                out[r.key] = Generation(text=_answer([e["id"] for e in prior[-1]["tool_result"]][:2]), tokens_in=16100, tokens_out=40, latency_seconds=0.01)
            elif n_results >= 2 and r.key in self.overflow:
                out[r.key] = Generation(text="", tokens_in=0, tokens_out=0, latency_seconds=0.0, error=SGLANG_INPUT)
            elif n_results == 0:
                out[r.key] = Generation(text=json.dumps({"tool": "search", "query": "sodium", "k": 3}), tokens_in=300, tokens_out=30, latency_seconds=0.01)
            elif n_results == 1:
                out[r.key] = Generation(text=json.dumps({"tool": "search", "query": "creatinine", "k": 3}), tokens_in=16000, tokens_out=40, latency_seconds=0.01)
            else:
                res = json.loads(last)
                out[r.key] = Generation(text=_answer([e["id"] for e in res["tool_result"]][:2]), tokens_in=17000, tokens_out=40, latency_seconds=0.01)
        return out

    def metadata(self):
        return {"backend": "scripted"}


@needs_dev
def test_tools_loop_guard_only_where_the_server_rejects(tmp_path):
    from scopebench.configurations import CONFIGS
    from scopebench.retrieval import TemporalNearestRetriever
    from scopebench.runner import RunSpec
    from scopebench.schema import read_jsonl
    from scopebench.tools_loop import run_tools
    tasks, cases = _load(4)
    over = {tasks[0].task_id, tasks[2].task_id}
    agent = TwoSearchAgent(over)
    spec = RunSpec(configuration=CONFIGS["high_compute"], retriever=TemporalNearestRetriever(), k=30, pool_k=50)
    meta = run_tools(spec, tasks, cases, agent, tmp_path / "hc", token_budget=10 ** 6, max_turns=8, manifest_path=DEV)
    rows = {r["task_id"]: r for r in read_jsonl(tmp_path / "hc/predictions.jsonl")}
    assert meta["context_guard_episodes"] == 2 and len(rows) == 4
    for t in tasks:
        r, reqs = rows[t.task_id], agent.requests[t.task_id]
        assert not r["generation_errors"], t.task_id
        if t.task_id in over:
            rejected, guarded = reqs[2], reqs[3]
            assert len(reqs) == 4                                              # 2 searches, the rejected turn, the guarded final turn
            assert guarded.messages[:-1] == rejected.messages[:-1]             # same conversation; only the unfittable tool result withheld
            assert '"context_limit_reached": true' in guarded.messages[-1]["content"]
            assert guarded.max_tokens == 16384 - (16000 + 40) - NOTICE_TOKENS  # capped to fit after the last request that fitted
        else:
            assert len(reqs) == 3 and all('"context_limit_reached"' not in m["content"] for q in reqs for m in q.messages)


class FlakyThenGood:
    """Every request fails with a non-context infrastructure error on the first pass; answers normally afterwards."""
    name = "flaky"

    def __init__(self, fail: bool):
        self.fail = fail

    def generate_many(self, requests):
        out = {}
        for r in requests:
            if self.fail:
                out[r.key] = Generation(text="", tokens_in=0, tokens_out=0, latency_seconds=0.0, error="RuntimeError: HTTP 503: busy")
            elif not any(m["content"].startswith('{"tool_result"') for m in r.messages):
                out[r.key] = Generation(text=json.dumps({"tool": "search", "query": "sodium", "k": 3}), tokens_in=300, tokens_out=30, latency_seconds=0.01)
            else:
                res = json.loads(r.messages[-1]["content"])
                out[r.key] = Generation(text=_answer([e["id"] for e in res["tool_result"]][:2]), tokens_in=600, tokens_out=40, latency_seconds=0.01)
        return out

    def metadata(self):
        return {"backend": "scripted"}


@needs_dev
def test_tools_loop_retry_errors_reruns_failed_rows(tmp_path):
    from scopebench.configurations import CONFIGS
    from scopebench.retrieval import TemporalNearestRetriever
    from scopebench.runner import RunSpec
    from scopebench.schema import read_jsonl
    from scopebench.tools_loop import run_tools
    tasks, cases = _load(3)
    spec = RunSpec(configuration=CONFIGS["token_matched"], retriever=TemporalNearestRetriever(), k=30, pool_k=50)
    run_tools(spec, tasks, cases, FlakyThenGood(True), tmp_path / "tm", token_budget=5000, max_turns=8, manifest_path=DEV)
    assert all(r["generation_errors"] for r in read_jsonl(tmp_path / "tm/predictions.jsonl"))
    meta = run_tools(spec, tasks, cases, FlakyThenGood(False), tmp_path / "tm", token_budget=5000, max_turns=8, manifest_path=DEV)
    assert meta["n_tasks_previously_done"] == 3 and meta["retried_error_rows"] == 0  # without the flag nothing is re-run
    meta = run_tools(spec, tasks, cases, FlakyThenGood(False), tmp_path / "tm", token_budget=5000, max_turns=8, manifest_path=DEV, retry_errors=True)
    rows = list(read_jsonl(tmp_path / "tm/predictions.jsonl"))
    assert meta["retried_error_rows"] == 3 and meta["n_tasks_this_run"] == 3 and len(rows) == 3
    assert not any(r["generation_errors"] for r in rows)


# ---------------------------------------------------------------------------------------------------- masking replay
@needs_dev
def test_masking_ignores_non_string_citations():
    from scopebench.configurations import CONFIGS
    from scopebench.retrieval import AccessLog, TemporalNearestRetriever
    from scopebench.runner import RunSpec, _finish_selection
    tasks, cases = _load(1)
    task, case = tasks[0], cases[tasks[0].case_id]
    chosen = list(case.evidence[:6])
    accessed = [e.id for e in chosen[:4]]
    src = {task.task_id: {"parsed_answer": {"citations": [[accessed[0]], [accessed[1]]]}, "final_guard": None, "access_ids": accessed}}
    for mask, expect_removed in (("cited_admissible", set()), ("uncited_accessed", set(accessed))):
        spec = RunSpec(configuration=CONFIGS["prompt_only"], retriever=TemporalNearestRetriever(), k=30, pool_k=50, mask=mask, mask_source=src)
        kept = _finish_selection(spec, task, case, AccessLog(), list(chosen))
        assert {e.id for e in chosen} - {e.id for e in kept} == expect_removed, mask


# ------------------------------------------------------------------------------------------ Tier-C FHIR-AgentBench
def _fhir_paths():
    from scopebench.config import PUBLIC_DATA
    return (PUBLIC_DATA / "mimic-iv-clinical-database-demo-on-fhir-2.1.0" / "fhir", ROOT / "experiments/scopebench_tierC/fhir_agentbench",
            PUBLIC_DATA / ".clinehr_cache/scopebench_tierc_patients/fhir_agentbench_patient_map.json")


@pytest.mark.skipif(not all(p.exists() for p in _fhir_paths()), reason="FHIR demo / Tier-C manifest not present")
def test_tierc_loop_guard(tmp_path):
    from scopebench.schema import Task, read_jsonl
    from scopebench.tierc.agent import TierCSpec, run_tierc
    from scopebench.tierc.store import load_store
    fhir, manifest, pmap_path = _fhir_paths()
    tasks = [Task.from_dict(r) for r in read_jsonl(manifest / "tasks.jsonl")]
    tasks = [t for t in tasks if t.contract.scope == "pre" and t.metadata["main_table"] == "chartevents" and t.metadata["rt_in_lt"]][:2]
    over = {tasks[0].task_id}
    seen: dict[str, list] = {}

    class Backend:
        name = "scripted"

        def generate_many(self, requests):
            out = {}
            for r in requests:
                seen.setdefault(r.key, []).append(r)
                n_tool = sum(1 for m in r.messages if m["role"] in ("user", "tool") and "<tool_response>" in m["content"])
                if '"context_limit_reached": true' in r.messages[-1]["content"] or (n_tool >= 2 and r.key not in over):
                    t = next(x for x in tasks if x.task_id == r.key)
                    out[r.key] = Generation(text=f'The final answer is: [["2133-12-31 02:00:00"]]\nCitations: {json.dumps(list(t.required_ids))}',
                                            tokens_in=15000, tokens_out=20, latency_seconds=0.01)
                elif n_tool >= 2:
                    out[r.key] = Generation(text="", tokens_in=0, tokens_out=0, latency_seconds=0.0, error=SGLANG_TOTAL)
                else:
                    pid = [m for m in r.messages if m["role"] == "user"][0]["content"].split("Patient FHIR ID: ")[1].split("\n")[0]
                    call = {"name": "get_resources_by_patient_fhir_id", "arguments": {"resource_type": "Observation", "patient_fhir_id": pid}}
                    out[r.key] = Generation(text=json.dumps(call).join(["<tool_call>\n", "\n</tool_call>"]), tokens_in=14000 if n_tool else 2000,
                                            tokens_out=30, latency_seconds=0.01)
            return out

        def metadata(self):
            return {"backend": "scripted"}

    store = load_store(fhir)
    run_tierc(TierCSpec(configuration="prompt_only", adapter="text"), tasks, store, json.load(open(pmap_path)), Backend(), None, tmp_path / "po", manifest_path=manifest)
    rows = {r["task_id"]: r for r in read_jsonl(tmp_path / "po/predictions.jsonl")}
    a, b = rows[tasks[0].task_id], rows[tasks[1].task_id]
    assert not a["generation_errors"] and a["context_guard"] and a["metrics"]["calls"] == 3
    assert seen[tasks[0].task_id][-1].max_tokens == final_max_tokens(1024, 16384, 14000, 30)
    assert not b["generation_errors"] and not b["context_guard"] and b["metrics"]["calls"] == 3
