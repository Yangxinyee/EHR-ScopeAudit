"""Compute controls (plan §4.2, §8.4; D11 / DEV-5): retrieval-equipped ReAct agents with a token ceiling and no
scope enforcement. The agent starts with NO supplied evidence — only the task, its contract statement and two tools:
search(query, k) over the case pool without any predicate, and get(ids). Everything a tool returns enters A_t and the
supplied set; nothing is filtered. `token_matched` gets the full system's mean token budget (T1); `high_compute`
gets 4.62 × the prompt-only mean (T0). When the ceiling is reached the model is asked for its final JSON once.
DEV-19: when the server rejects a turn because the live conversation no longer fits the serving context, the model is
asked for its final JSON instead (the tool result that did not fit is withheld; `context_guard.py`); before DEV-19 the
episode ended as a generation error. `retry_errors` re-runs rows whose generation failed (DEV-19: it was not passed to
this loop before, so the panel drivers' retry passes did not re-run tool-loop rows).

(Until 2026-09-28 the loop started from the prompt-only evidence with tools optional; every checkpoint answered
directly without calling a tool, so the controls collapsed to prompt_only. DEV-5 records the change.)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import parser
from .backends.base import Backend, Generation, Request
from .context_guard import NOTICE, final_max_tokens, parse_overflow
from .retrieval import AccessLog
from .runner import RunSpec, TaskState, _view, build_messages, write_outputs
from .schema import Case, EvidenceObject, Task, read_jsonl
from .tasktypes import get as get_task_type

TOOL_INSTRUCTIONS = """
No evidence has been supplied with this task. Retrieve it yourself with the tools below, then answer.
To call a tool, return exactly one JSON object and no prose:
  {"tool": "search", "query": "<free text>", "k": <1-20>}   -> returns up to k matching evidence objects
  {"tool": "get", "ids": ["<evidence id>", ...]}          -> returns those evidence objects in full
Each tool result is returned to you as a user message; the evidence objects it contains are the evidence you may
cite. Start with a search. When you have enough evidence, return the final answer as the JSON object described by
the output schema (and nothing else). You have a total token budget of {budget} tokens for this task including
tool results; when told the budget is exhausted, return your final answer immediately."""

DEFAULT_K = 10


def _search(spec: RunSpec, task: Task, case: Case, query: str, k: int) -> list[EvidenceObject]:
    probe = Task(**{**task.__dict__, "question": query})
    return spec.retriever.rank(probe, case.evidence, max(1, min(int(k), 20)), None)


def run_tools(
    spec: RunSpec,
    tasks: list[Task],
    cases: dict[str, Case],
    backend: Backend,
    out_dir: Path,
    *,
    token_budget: int,
    max_turns: int = 8,
    manifest_path: Path | None = None,
    extra_metadata: dict[str, Any] | None = None,
    retry_errors: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    previous = list(read_jsonl(pred_path)) if pred_path.exists() else []
    redo: set[str] = set()
    if retry_errors:
        # as in runner.run: rows whose generation failed are dropped with their access-log and stage rows, then re-run
        redo = {row["task_id"] for row in previous if row.get("generation_errors")}
        for name in ("predictions.jsonl", "access_log.jsonl", "stages.jsonl") if redo else ():
            path = out_dir / name
            if path.exists():
                kept = [row for row in read_jsonl(path) if row.get("task_id") not in redo]
                with open(path, "w", encoding="utf-8") as f:
                    for row in kept:
                        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    done = {row["task_id"] for row in previous} - redo
    todo = sorted((t for t in tasks if t.task_id not in done), key=lambda t: (t.case_id, t.task_id))
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    states: dict[str, TaskState] = {}
    active: list[TaskState] = []
    for task in todo:
        case = cases[task.case_id]
        tt = get_task_type(task.task_type)
        access = AccessLog()
        access.record("prompt", [])   # nothing is supplied up front: every record the agent sees comes through a tool
        st = TaskState(task=task, task_type=tt, case=case, pool=case.by_id(), supplied=[], access=access)
        msgs = build_messages(tt, task, [], None)
        msgs[0] = {"role": "system", "content": msgs[0]["content"] + TOOL_INSTRUCTIONS.replace("{budget}", str(token_budget))}
        st.messages = msgs
        st.initial = Generation(text="", tokens_in=0, tokens_out=0, latency_seconds=0.0)
        states[task.task_id] = st
        active.append(st)
    turns = {st.task.task_id: 0 for st in active}
    tool_calls = {st.task.task_id: 0 for st in active}
    forced: set[str] = set()
    guarded: set[str] = set()                    # DEV-19: episodes asked for the final answer at the context limit
    last_ok: dict[str, tuple[int, int]] = {}     # tokens in/out of the last request that fitted
    max_tok: dict[str, int] = {}
    while active:
        reqs = [Request(st.task.task_id, st.messages, max_tok.get(st.task.task_id, spec.max_tokens_override or st.task_type.max_tokens)) for st in active]
        gens = backend.generate_many(reqs)
        still: list[TaskState] = []
        for st in active:
            g = gens[st.task.task_id]
            tid = st.task.task_id
            overflow = parse_overflow(g.error)
            if overflow and tid not in guarded and tid in last_ok:
                # the live conversation no longer fits: withhold the message that did not fit and ask for the final answer
                guarded.add(tid); forced.add(tid)
                what = "The conversation reached the model's context limit." if st.messages[-1]["content"].startswith('{"budget_exhausted"') else NOTICE
                st.messages = st.messages[:-1] + [{"role": "user", "content": json.dumps({"budget_exhausted": True, "context_limit_reached": True,
                                                                                        "instruction": what + " Return your final answer JSON now."})}]
                max_tok[tid] = final_max_tokens(spec.max_tokens_override or st.task_type.max_tokens, overflow[1], *last_ok[tid])
                st.access.record("context_guard", [])
                still.append(st)
                continue
            turns[tid] += 1
            st.initial = Generation(text=g.text, tokens_in=st.initial.tokens_in + g.tokens_in, tokens_out=st.initial.tokens_out + g.tokens_out,
                                    latency_seconds=st.initial.latency_seconds + g.latency_seconds, error=g.error or st.initial.error,
                                    extra={"turns": turns[tid], "tool_calls": tool_calls[tid], "budget": token_budget, "forced_final": tid in forced,
                                           "context_guard": tid in guarded})
            if g.error:
                continue   # infrastructure error: the row is written with generation_errors and re-run with --retry-errors
            last_ok[tid] = (g.tokens_in, g.tokens_out)
            parsed = parser.extract_json(g.text)
            used = st.initial.tokens_in + st.initial.tokens_out
            is_tool = isinstance(parsed, dict) and parsed.get("tool") in ("search", "get")
            if not is_tool or tid in forced:
                continue  # final answer (or a tool call after the forced-final request: taken as the final text)
            if turns[tid] < max_turns and used < token_budget:
                tool_calls[tid] += 1
                if parsed["tool"] == "search":
                    hits = _search(spec, st.task, st.case, str(parsed.get("query", "")), parsed.get("k", DEFAULT_K))
                else:
                    hits = [st.pool[i] for i in parsed.get("ids", []) if isinstance(i, str) and i in st.pool]
                shown = {e.id for e in st.supplied}
                st.supplied = st.supplied + [e for e in hits if e.id not in shown]
                st.access.record(f"tool_turn{turns[tid]}", [e.id for e in hits])
                result = json.dumps({"tool_result": [_view(st.task_type, st.task, e) for e in hits], "tokens_used": used, "budget": token_budget}, ensure_ascii=False)
                st.messages = st.messages + [{"role": "assistant", "content": g.text}, {"role": "user", "content": result}]
                still.append(st)
            else:
                forced.add(tid)
                st.messages = st.messages + [{"role": "assistant", "content": g.text},
                                             {"role": "user", "content": json.dumps({"budget_exhausted": True, "instruction": "Return your final answer JSON now."})}]
                still.append(st)
        active = still
    for st in states.values():
        st.parsed_initial = parser.extract_json(st.initial.text)
    n = max(1, len(turns))
    return write_outputs(spec, list(states.values()), tasks, done, backend, out_dir, pred_path, manifest_path,
                         {**(extra_metadata or {}), "token_budget": token_budget, "max_turns": max_turns, "mean_turns": sum(turns.values()) / n,
                          "mean_tool_calls": sum(tool_calls.values()) / n, "share_with_tool_call": sum(1 for v in tool_calls.values() if v > 0) / n,
                          "controls_design": "DEV-5: no supplied evidence; search/get over the unscoped pool",
                          "context_guard": "DEV-19", "context_guard_episodes": len(guarded), "retried_error_rows": len(redo)}, started)
