"""No-GPU regression: replay the frozen ScopePair LabTrend trajectories through the ScopeBench
parser, guard and scorer and require agreement with the frozen guard outcomes (plan §12 M1 gate).

For every frozen run directory experiments/scopepair_fhir_test_<tag>_frozen the frozen row gives
the model's raw `response`, the evidence ScopePair selected, the guard errors it produced, and
whether it repaired / fell back. ScopeBench must (1) select the same evidence for the same system,
(2) parse the same answer, (3) reproduce the initial guard verdict (error / no error) for the
guard systems, and (4) reproduce the deterministic-fallback flag.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from scopebench import parser  # noqa: E402
from scopebench.compat_scopepair import labtrend_tasks  # noqa: E402
from scopebench.configurations import CONFIGS  # noqa: E402
from scopebench.guard import check  # noqa: E402
from scopebench.predicate import legal  # noqa: E402
from scopebench.retrieval import AccessLog, TemporalNearestRetriever  # noqa: E402
from scopebench.runner import RunSpec, _guard_errors_for, select_evidence  # noqa: E402
from scopebench.schema import read_jsonl  # noqa: E402
from scopebench.tasktypes import LAB_TREND_K3, REFERENCE_SKILLS  # noqa: E402

MANIFEST = ROOT / "experiments/scopepair_fhir/tasks.jsonl"
RUNS = sorted(p for p in ROOT.glob("experiments/scopepair_fhir_test_*_frozen") if (p / "predictions.jsonl").exists())
REPLAY_OUT = ROOT / "experiments/scopebench_m1/replay"
SYSTEM_TO_CONFIG = {
    "unscoped": "sp_unscoped",
    "wrong_time_scope_first": "sp_wrong_time_scope_first",
    "scope_first": "sp_scope_first",
    "guard_refill": "sp_guard_refill",
    "clinehr_os": "sp_clinehr_os",
}

pytestmark = pytest.mark.skipif(not MANIFEST.exists() or not RUNS, reason="frozen ScopePair artifacts not present")


@pytest.fixture(scope="module")
def labtrend():
    tasks, cases = labtrend_tasks(MANIFEST, split="test")
    return {t.task_id: t for t in tasks}, cases


@pytest.mark.parametrize("run_dir", RUNS, ids=[p.name for p in RUNS])
def test_replay_frozen_run(run_dir: Path, labtrend):
    tasks, cases = labtrend
    rows = [r for r in read_jsonl(run_dir / "predictions.jsonl") if r["system"] in SYSTEM_TO_CONFIG]
    assert rows, "no rows"
    mismatches = {"evidence": [], "parse": [], "initial_guard": [], "fallback": []}
    n = 0
    for r in rows:
        task = tasks[r["task_id"]]
        case = cases[task.case_id]
        cfg = CONFIGS[SYSTEM_TO_CONFIG[r["system"]]]
        spec = RunSpec(configuration=cfg, retriever=TemporalNearestRetriever(), k=3, pool_k=3,
                       skills={"lab_trend_k3": REFERENCE_SKILLS["lab_trend_k3"]})
        access = AccessLog()
        supplied = select_evidence(spec, task, case, access)
        if [e.id for e in supplied] != list(r["selected_evidence_ids"]):
            mismatches["evidence"].append(r["task_id"])
            continue
        parsed = parser.extract_json(r["response"])
        if (parsed is None) != (r["parsed_answer"] is None and not r["guard_repaired"] and not r["guard_deterministic_fallback"]) and not r["guard_repaired"]:
            # ScopePair stored the *final* parsed answer; for un-repaired rows it equals the initial parse
            if parsed != r["parsed_answer"] and not r["guard_deterministic_fallback"]:
                mismatches["parse"].append(r["task_id"])
        if cfg.uses_guard:
            pool = case.by_id()
            skill = REFERENCE_SKILLS["lab_trend_k3"](spec.retriever.rank(task, case.evidence, 3, lambda e: legal(e, task.contract))) if cfg.skill_verify else None
            rep = check(parsed, task, LAB_TREND_K3, pool, supplied, access.ids(), check_completeness=False, skill_summary=skill)
            ours = bool(_guard_errors_for(cfg, rep))
            theirs = bool(r["initial_guard_errors"])
            if ours != theirs:
                mismatches["initial_guard"].append((r["task_id"], _guard_errors_for(cfg, rep), r["initial_guard_errors"]))
            # deterministic fallback is decided after the repair; replay the frozen repair text
            if r["guard_deterministic_fallback"] or r["guard_repaired"]:
                final_text = r["repair_response"] if r["guard_repaired"] else r["response"]
                final_parsed = parser.extract_json(final_text)
                final_supplied = spec.retriever.rank(task, case.evidence, 3, lambda e: legal(e, task.contract))
                rep2 = check(final_parsed, task, LAB_TREND_K3, pool, final_supplied, access.ids() | {e.id for e in final_supplied}, check_completeness=False, skill_summary=skill)
                would_replace = bool(_guard_errors_for(cfg, rep2)) and cfg.skill_replace
                if would_replace != bool(r["guard_deterministic_fallback"]):
                    mismatches["fallback"].append((r["task_id"], _guard_errors_for(cfg, rep2)))
        n += 1
    summary = {k: len(v) for k, v in mismatches.items()}
    REPLAY_OUT.mkdir(parents=True, exist_ok=True)
    (REPLAY_OUT / f"{run_dir.name}.json").write_text(json.dumps({"n": n, "mismatches": summary, "examples": {k: v[:5] for k, v in mismatches.items()}}, indent=1, default=str))
    assert summary["evidence"] == 0, summary
    assert summary["initial_guard"] == 0, (summary, mismatches["initial_guard"][:3])
    assert summary["fallback"] == 0, (summary, mismatches["fallback"][:3])
    assert summary["parse"] == 0, (summary, mismatches["parse"][:3])
