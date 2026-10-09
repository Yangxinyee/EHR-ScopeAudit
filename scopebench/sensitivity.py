"""Sensitivity-suite variants (plan §8.5, Supp. Table S5). Each variant is a pure transformation of the task
list; the manifest on disk is never changed. Retriever variants (top-k 5/20, BM25-only, dense-only, no section
expansion) are CLI flags of the runner; the alternative repair wording is `RunSpec.repair_wording = "b"`; the
five decoding seeds are backend sampling parameters (temperature 0.7, seeds 13/29/47/71/101). Every
sensitivity cell is labelled `cell_kind = sensitivity` in its metadata.

Variants:
  strict_boundaries  every inclusive window edge and the at-index tolerance become exclusive
  at_window_6h       the at-index scope admits records up to 6 h after the index
  extended_rt        the required set R_t is widened: evidence-retrieval tasks require the nearest FIVE results
                     (oracle `lab_trend_k3` with k = 5) instead of three; the question, gold answer and skill
                     parameters are unchanged, so a compliant three-result answer reaches at most 3/5 completeness
                     on those tasks — the row measures whether the sign of the full − prompt-only completeness
                     change survives a stricter denominator. Cohort tasks are unchanged: under SKILL_SPEC v1.2 the
                     reference R_t already cites every satisfied-criterion row (and the whole examined domain of the
                     decisive failed criterion), so "all satisfied-criterion rows" is the reference itself.

Correction before any variant `full` result existed (DEV-14, 2026-09-30): the two contract variants now redefine the
task under the variant contract — the gold answer and R_t of the three oracle-backed task types (lab_trend_k3,
outcome_lookup_v1, cohort_v1) are recomputed with the frozen oracle on the variant's legal set, exactly as the
Tier-B builders compute them; the other task types keep their gold and R_t, which must lie inside the variant's legal
set (checked). Previously only the contract changed, so a few tasks carried an R_t outside the variant scope (5 under
strict_boundaries) or one the variant's own oracle would not produce (8 under at_window_6h), and the verifier's
replacement answer could not satisfy them. `extended_rt` is a scoring denominator only: the tasks the runner sees are
the reference tasks (so the runs equal the reference runs), and `extended_scoring_tasks` gives the tasks with the
widened R_t for re-scoring (`rescore_rows`).

DEV-14 addendum (2026-10-01): a recomputed cohort gold can be "uncertain" (a criterion's domain is empty under the
variant window). The Tier-B builder treats an uncertain gold as an abstention that cannot be scored and samples another
anchor; a variant cannot resample, so such tasks are DROPPED from the variant (strict_boundaries: 4 of 9,600; the
variant cells and their re-scored prompt_only rows have 9,596 tasks, paired on the common ids). Keeping them would pit
the guard's "required evidence is supplied; do not abstain" rule against the verifier's own uncertain answer, which is
what aborted the second strict_boundaries/full run.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .oracle import OPERATIONS as ORACLE
from .predicate import filter_legal, legal
from .schema import Case, Contract, Task

VARIANTS = ("strict_boundaries", "at_window_6h", "extended_rt")
SEEDS = (13, 29, 47, 71, 101)
SEED_TEMPERATURE = 0.7
EXTENDED_K = 5


def variant_contract(contract: Contract, name: str) -> Contract:
    if name == "strict_boundaries":
        return replace(contract, strict_boundaries=True)
    if name == "at_window_6h":
        return replace(contract, at_tolerance_hours=6.0) if contract.scope == "at" else contract
    if name == "extended_rt":
        return contract
    raise ValueError(f"unknown sensitivity variant {name!r}; known: {VARIANTS}")


def extended_required(task: Task, case: Case) -> tuple[str, ...]:
    if task.task_type != "lab_trend_k3":
        return task.required_ids
    result = ORACLE["lab_trend_k3"](filter_legal(case.evidence, task.contract), task.contract, {"k": EXTENDED_K})
    if result.get("status") != "ok":
        return task.required_ids   # fewer than five legal results: the reference set stands
    return tuple(result["ordered_result_ids"])


def _regold(task: Task, case: Case) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Gold answer and R_t of an oracle-backed task under its (variant) contract, as the Tier-B builders compute them."""
    lg = filter_legal(case.evidence, task.contract)
    if task.task_type == "lab_trend_k3":
        g = ORACLE["lab_trend_k3"](lg, task.contract, {"k": 3})
        if g["status"] != "ok":
            raise ValueError(f"{task.task_id}: fewer than three legal results under the variant contract")
        return ({"status": "ok", "results": g["provenance"], "delta": g["delta_last_minus_first"], "direction": g["direction"]}, tuple(g["ordered_result_ids"]))
    if task.task_type == "outcome_lookup_v1":
        g = ORACLE["outcome_lookup"](lg, task.contract, task.metadata["skill_params"])
        if g["status"] != "ok" or g.get("value") is None:
            raise ValueError(f"{task.task_id}: no outcome under the variant contract")
        return ({"status": "ok", "value": g["value"], "time": g["time"], "citation": g["citation"]}, tuple(g["provenance"]))
    if task.task_type == "cohort_v1":
        g = ORACLE["cohort_criteria"](lg, task.contract, {"criteria": task.metadata["criteria"]})
        return ({"decision": g["decision"], "citations": g["provenance"]}, tuple(g["provenance"]))
    raise KeyError(task.task_type)


REGOLD_TYPES = ("lab_trend_k3", "outcome_lookup_v1", "cohort_v1")


def apply_variant(tasks: list[Task], name: str, cases: dict[str, Case] | None = None, dropped: list[str] | None = None) -> list[Task]:
    if name == "extended_rt":
        return list(tasks)   # scoring-only variant: the runner sees the reference tasks
    if cases is None:
        raise ValueError(f"{name} needs the cases (evidence) to redefine gold and R_t under the variant contract")
    out = []
    for t in tasks:
        vt = replace(t, contract=variant_contract(t.contract, name))
        case = cases[t.case_id]
        if t.task_type in REGOLD_TYPES:
            gold, req = _regold(vt, case)
            if gold.get("decision") == "uncertain":
                # an abstention cannot be scored (Tier-B builder rule); the variant cannot resample, so the task is left out
                if dropped is not None:
                    dropped.append(t.task_id)
                continue
            vt = replace(vt, gold=gold, required_ids=req, metadata={**t.metadata, "variant": name, "variant_regold": True,
                                                                     "reference_required_ids": list(t.required_ids), "reference_gold": t.gold})
        else:
            pool = case.by_id()
            outside = [i for i in t.required_ids if i in pool and not legal(pool[i], vt.contract)]
            if outside:
                raise ValueError(f"{t.task_id}: R_t outside the {name} legal set: {outside}")
            vt = replace(vt, metadata={**t.metadata, "variant": name})
        out.append(vt)
    return out


def extended_scoring_tasks(tasks: list[Task], cases: dict[str, Case]) -> list[Task]:
    return [replace(t, required_ids=extended_required(t, cases[t.case_id]), metadata={**t.metadata, "rt_variant": "extended_rt", "reference_required_ids": list(t.required_ids)}) for t in tasks]


def rescore_rows(rows: list[dict[str, Any]], tasks: dict[str, Task], cases: dict[str, Case]) -> list[dict[str, Any]]:
    """Re-score saved rows against (variant) tasks: valid for configurations whose generation does not depend on the
    gold answer or R_t (prompt_only), and for extended_rt, whose runs equal the reference runs."""
    from .scoring import score
    from .tasktypes import get as get_task_type
    out = []
    for row in rows:
        task = tasks.get(row["task_id"])
        if task is None:
            continue   # task dropped from the variant (uncertain gold): its row is dropped too
        m = row["metrics"]
        new = dict(row)
        new["metrics"] = score(row.get("parsed_answer"), task, get_task_type(task.task_type), cases[task.case_id].by_id(), set(row.get("access_ids") or []),
                               tokens_in=m["tokens_in"], tokens_out=m["tokens_out"], calls=m["calls"], latency=m["latency_seconds"])
        new["rescored_against"] = task.metadata.get("variant") or task.metadata.get("rt_variant")
        out.append(new)
    return out
