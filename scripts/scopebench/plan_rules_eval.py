"""DEV-34: evaluation of rule-routed skill-only (scopebench/plan_rules.py), called by plans_offline.py `rules`.

Per task: the rule parser writes a plan from what the model would read; the plan is executed on the permitted
evidence and its answer submitted. Reported: plan validity and accuracy against the reference plan, answer
correctness, and completeness of the plan's citations against R_t — split by covered / uncovered and by split.
Tier-B results are aggregate counts only (MIMIC-derived). Tier-C `valid` is scored leave-one-out (the router's
exemplars are the valid questions).
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from scopebench import plans
from scopebench.plan_rules import TierCRouter, chart_vocabulary, masked, tier_b_plan
from scopebench.tasktypes import get as task_type


class Tally:
    def __init__(self):
        self.n = 0
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, **kw: Any) -> None:
        self.n += 1
        for k, v in kw.items():
            if v is None:
                continue
            self.sums[k] += float(v)
            self.counts[k] += 1

    def summary(self) -> dict[str, Any]:
        return {"n": self.n, **{k: round(self.sums[k] / self.counts[k], 4) for k in sorted(self.sums)}, **{f"n_{k}": self.counts[k] for k in sorted(self.counts) if self.counts[k] != self.n}}


def _completeness(cited: list[str], required: tuple[str, ...]) -> float | None:
    return len(set(cited) & set(required)) / len(set(required)) if required else None


def evaluate_b() -> dict[str, Any]:
    from scopebench.manifest import load_manifest
    from scopebench.runner import legal_all
    from scopebench.skills import OPERATIONS
    from plans_offline import TIER_B

    out: dict[str, Any] = {}
    for split, manifest in TIER_B.items():
        tasks, cases = load_manifest(manifest, None)
        tallies: dict[str, Tally] = defaultdict(Tally)
        for t in tasks:
            tt = task_type(t.task_type)
            pred = tier_b_plan(tt.build_user_payload(t, [], None))
            ref = plans.reference_plan(t)
            acc = plans.plan_accuracy(pred, ref) if ref.operation is not None else {"exact": pred.operation is None}
            answer = None
            res = plans.execute(pred, legal_all(t, cases[t.case_id]), t, OPERATIONS) if pred.operation else None
            if res is not None and res.error is None and tt.skill_answer is not None:
                answer = tt.skill_answer(res.summary, t, legal_all(t, cases[t.case_id]))
            correct = bool(answer is not None and tt.answer_correct(answer, t))
            ev_c = _completeness(tt.citations(answer), t.required_ids) if answer is not None else (0.0 if t.required_ids else None)
            for key in ("all", "covered" if tt.covered else "uncovered", t.task_type):
                tallies[key].add(plan_exact=acc["exact"], operation=acc.get("operation"), args=acc.get("args"), answer_correct=correct, completeness=ev_c,
                                 required_exact=(set(res.required) == set(t.required_ids)) if res is not None and res.error is None else None)
        out[split] = {k: v.summary() for k, v in sorted(tallies.items())}
    return out


def evaluate_c() -> dict[str, Any]:
    from scopebench.tierc.plans import Executor, reference_plan
    from scopebench.tierc.skills import patient_pairs
    from scripts.fhir_benchmark_core import canonical_answer_correct
    from plans_offline import tier_c_context

    tasks, ts, patient_map, _ = tier_c_context()
    ex = Executor(patient_map)
    refs = {t.task_id: reference_plan(t, ex.catalog) for t in tasks}
    vocab: dict[str, dict[str, list[str]]] = {}

    def chart(t) -> dict[str, list[str]]:
        pid = patient_map[t.patient]
        if pid not in vocab:
            vocab[pid] = chart_vocabulary([r for r, _ in patient_pairs(ts.store, pid, t.patient)])
        return vocab[pid]

    valid = [(t.task_id, masked(t.question, chart(t)), refs[t.task_id].operation) for t in tasks if t.split == "valid" and refs[t.task_id].operation]
    router = TierCRouter([(q, tid) for _, q, tid in valid], ex.catalog)
    tallies: dict[tuple[str, str], Tally] = defaultdict(Tally)
    for t in tasks:
        r = router
        if t.split == "valid" and refs[t.task_id].operation:   # leave-one-out on the exemplar split
            r = TierCRouter([(q, tid) for i, q, tid in valid if i != t.task_id], ex.catalog)
        pred = r.plan(t.question, chart(t))
        ref = refs[t.task_id]
        covered = ref.operation is not None
        acc = plans.plan_accuracy(pred, ref) if covered else {"exact": pred.operation is None}
        res = ex(pred, ts, t, t.metadata.get("assumption"))
        correct, ev_c = False, (0.0 if t.required_ids else None)
        if res is not None and res.ok:
            gold = t.gold.get("answer")
            correct = bool(canonical_answer_correct(json.dumps(res.summary["answer"]), gold if isinstance(gold, str) else json.dumps(gold)))
            ev_c = _completeness(res.required, t.required_ids)
        template_ok = pred.operation == ref.operation
        for split in (t.split, "train+test" if t.split in ("train", "test") else None):
            if split is None:
                continue
            for key in ("all", "covered" if covered else "uncovered"):
                tallies[(split, key)].add(plan_exact=acc["exact"], template=template_ok if covered else None, operation=acc.get("operation"), args=acc.get("args"),
                                          answer_correct=correct, completeness=ev_c)
    out: dict[str, Any] = defaultdict(dict)
    for (split, key), v in sorted(tallies.items()):
        out[split][key] = v.summary()
    return dict(out)


def evaluate(tiers: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "B" in tiers:
        result["tier_b"] = evaluate_b()
    if "C" in tiers:
        result["tier_c"] = evaluate_c()
    return result
