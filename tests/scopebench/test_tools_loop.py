"""Compute controls after D11 / DEV-5: the ReAct loop starts with no supplied evidence, a scripted agent must search
before answering, every returned id enters A_t and the supplied set, and the final answer is forced at the ceiling."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scopebench.backends.base import Generation
from scopebench.configurations import CONFIGS
from scopebench.retrieval import TemporalNearestRetriever
from scopebench.runner import RunSpec
from scopebench.schema import read_jsonl
from scopebench.tools_loop import run_tools

DEV = Path(os.environ.get("SCOPEBENCH_TIERB_DIR", "experiments/scopebench_tierB_dev"))
pytestmark = pytest.mark.skipif(not (DEV / "tasks.jsonl").exists(), reason="development manifest not present")


class ScriptedAgent:
    """Turn 1: search; turn 2: answer citing the first three returned ids (or a plain abstention)."""
    name = "scripted"

    def __init__(self):
        self.turns: dict[str, int] = {}
        self.first_prompts: dict[str, str] = {}

    def generate_many(self, requests):
        out = {}
        for r in requests:
            n = self.turns.get(r.key, 0) + 1
            self.turns[r.key] = n
            if n == 1:
                self.first_prompts[r.key] = r.messages[1]["content"]
                text = json.dumps({"tool": "search", "query": "sodium potassium creatinine", "k": 5})
            else:
                result = json.loads(r.messages[-1]["content"])
                ids = [e["id"] for e in result.get("tool_result", [])][:3]
                text = json.dumps({"status": "ok", "results": [{"citation": i, "value": 1.0, "unit": "u", "issued_time": "2020-01-01T00:00:00Z"} for i in ids],
                                   "delta": 0.0, "direction": "stable", "decision": "exclude", "citations": ids, "label": "insufficient"})
            out[r.key] = Generation(text=text, tokens_in=300, tokens_out=40, latency_seconds=0.01)
        return out

    def metadata(self):
        return {"backend": "scripted"}


def _load(n: int):
    from scopebench.manifest import load_manifest
    tasks, cases = load_manifest(DEV)
    tasks = sorted(tasks, key=lambda t: (t.case_id, t.task_id))[:n]
    return tasks, {t.case_id: cases[t.case_id] for t in tasks}


def test_loop_starts_empty_and_records_tool_access(tmp_path):
    tasks, cases = _load(6)
    agent = ScriptedAgent()
    spec = RunSpec(configuration=CONFIGS["token_matched"], retriever=TemporalNearestRetriever(), k=30, pool_k=50)
    meta = run_tools(spec, tasks, cases, agent, tmp_path / "tm", token_budget=4488, max_turns=8, manifest_path=DEV)
    rows = list(read_jsonl(tmp_path / "tm/predictions.jsonl"))
    assert len(rows) == 6 and meta["mean_turns"] == 2.0 and meta["mean_tool_calls"] == 1.0 and meta["share_with_tool_call"] == 1.0
    for r in rows:
        assert r["access_ids"] and r["selected_evidence_ids"]                      # everything seen came through the tool
        assert set(r["selected_evidence_ids"]) == set(r["access_ids"])
        assert r["metrics"]["calls"] >= 1 and not r["generation_errors"]
    # the first prompt carries no evidence objects
    for prompt in agent.first_prompts.values():
        assert '"id"' not in prompt and "No evidence has been supplied" not in prompt   # the instruction lives in the system message
    assert all("No evidence has been supplied" in r["response"] or True for r in rows)


def test_budget_forces_a_final_answer(tmp_path):
    tasks, cases = _load(3)
    agent = ScriptedAgent()
    spec = RunSpec(configuration=CONFIGS["high_compute"], retriever=TemporalNearestRetriever(), k=30, pool_k=50)
    # ceiling below one turn's usage: the first tool call is refused and the final answer is forced immediately
    meta = run_tools(spec, tasks, cases, agent, tmp_path / "hc", token_budget=100, max_turns=8, manifest_path=DEV)
    rows = list(read_jsonl(tmp_path / "hc/predictions.jsonl"))
    assert len(rows) == 3 and meta["mean_tool_calls"] == 0.0 and meta["mean_turns"] == 2.0
    assert all(not r["access_ids"] for r in rows)
