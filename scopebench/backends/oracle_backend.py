"""Oracle 'model': answers every request with the task's gold in the task type's output format.

Not a model — a pipeline test and an upper bound: with scoped retrieval it should score 100 % on every
task whose gold citations were supplied, and it exposes retrieval recall (gold citations missing from
the supplied top-k make the oracle answer cite unsupplied evidence). Never used in formal runs."""

from __future__ import annotations

import json
from typing import Any

from ..schema import Task
from .base import Generation, Request


def gold_to_answer(task: Task) -> dict[str, Any]:
    g = task.gold
    tt = task.task_type
    if tt == "lab_trend_k3":
        return {"status": g.get("status", "ok"), "results": g.get("results", []), "delta": g.get("delta"), "direction": g.get("direction")}
    if tt == "outcome_lookup_v1":
        return {"status": g.get("status", "ok"), "value": g.get("value"), "time": g.get("time"), "citation": g.get("citation")}
    if tt == "cohort_v1":
        return {"decision": g.get("decision"), "citations": list(g.get("citations", []))}
    if tt == "claim_v1":
        return {"label": g.get("label"), "citations": list(g.get("line_ids", []))}
    if tt == "reconstruct_v1":
        return {"status": g.get("status", "ok"), "lines": [{"citation": c, "text": None} for c in g.get("line_ids", [])]}
    if tt == "summarize_exams_v1":
        return {"status": g.get("status", "ok"), "exams": list(g.get("exams", []))}
    raise ValueError(tt)


class OracleBackend:
    name = "oracle"

    def __init__(self, tasks: list[Task], cases: dict[str, Any]):
        self.tasks = {t.task_id: t for t in tasks}
        self.cases = cases

    def generate_many(self, requests: list[Request]) -> dict[str, Generation]:
        out = {}
        for r in requests:
            task = self.tasks[r.key]
            answer = gold_to_answer(task)
            if task.task_type == "reconstruct_v1":   # copy the verbatim line text from the case pool
                pool = self.cases[task.case_id].by_id()
                for line in answer["lines"]:
                    line["text"] = pool[line["citation"]].fields.get("text") if line["citation"] in pool else ""
            text = json.dumps(answer, ensure_ascii=False)
            out[r.key] = Generation(text=text, tokens_in=len(json.dumps(r.messages)) // 4, tokens_out=len(text) // 4, latency_seconds=0.0)
        return out

    def metadata(self) -> dict[str, Any]:
        return {"backend": self.name, "note": "gold answers; pipeline test only"}
