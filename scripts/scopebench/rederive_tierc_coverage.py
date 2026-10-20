#!/usr/bin/env python3
"""DEV-31: Tier-C FHIR-AgentBench skill coverage re-selected under the protocol's own rule, `full` re-derived offline.

The Tier-C protocol (§Tasks, "Development pool") states that the official `valid` split was used for the skill
validation. `skill_validation.json` instead marks a template covered when the skill reproduced the released answer on
every checked task of ALL three splits, and 8 of its 45 covered templates occur only in `test` — their coverage was
decided on test answers alone. This script:

1. re-selects coverage with the protocol rule: a template is covered when it has no `uncovered` task in `valid`, at
   least one checked `valid` task, and the skill agrees with the released answer on every checked `valid` task
   (same agreement records as the validator: `skill_validation.per_task.jsonl`); templates absent from `valid` are
   uncovered. Written to `experiments/scopebench_tierC/skill_validation_protocol_dev31.json` (same schema).
2. re-derives every checkpoint's `full` rows (and the frontier's) under that coverage WITHOUT any model call. The
   DEV-26 `full` re-run differs from the earlier skill-less `full` run (`_superseded_full_noskill_dev26/full`) only in
   `finalize` (the model calls are the same configuration), so per task:
     template covered before and now  -> the formal DEV-26 `full` row, unchanged
     template covered before, not now -> the skill-less `full` row (no verification, no replacement)
     template covered now, not before -> the skill-less `full` row passed through the DEV-26 finalize step offline
                                         (skill over the patient's legal set, verification, replacement, re-scoring)
   The offline finalize is validated against the formal rows on the templates covered under both rules.
3. writes a mirror runs root `experiments/scopebench_tierC_runs_dev31/` (every other configuration symlinked to the
   frozen directory; `full` holds the derived rows) for `analyze_panel.py`. Frozen run directories are not modified.

Loads no model. Usage: .venv-analysis/bin/python scripts/scopebench/rederive_tierc_coverage.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scopebench.config import PUBLIC_DATA  # noqa: E402
from scopebench.guard import check  # noqa: E402
from scopebench.schema import Task, read_jsonl  # noqa: E402
from scopebench.scoring import score  # noqa: E402
from scopebench.tasktypes import get as get_task_type  # noqa: E402
from scopebench import tasktypes_tierc  # noqa: E402,F401
from scopebench.tierc.skills import mapped_skills  # noqa: E402
from scopebench.tierc.store import TaskStore, load_store  # noqa: E402
from fhir_benchmark_core import canonical_answer_correct  # noqa: E402

FHIR_DIR = PUBLIC_DATA / "mimic-iv-clinical-database-demo-on-fhir-2.1.0" / "fhir"
PMAP = PUBLIC_DATA / ".clinehr_cache" / "scopebench_tierc_patients" / "fhir_agentbench_patient_map.json"
TC = ROOT / "experiments/scopebench_tierC"
RUNS = ROOT / "experiments/scopebench_tierC_runs"
OUT_RUNS = ROOT / "experiments/scopebench_tierC_runs_dev31"
TAGS = ["gemma3_27b", "medgemma_27b", "phi4_14b", "qwen3_30b", "qwen3_32b", "qwen3_4b"]
ENDPOINTS = ["answer_correct", "completeness", "access_legal", "citation_legal", "auditable", "workflow_compliant"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def select_protocol_coverage(manifest: dict[str, dict], per_task: list[dict], old: dict) -> dict:
    stats = defaultdict(lambda: {"n": 0, "checked": 0, "agree": 0, "uncovered": 0, "insufficient": 0, "n_by_split": Counter(),
                                 "valid_checked": 0, "valid_agree": 0, "valid_uncovered": 0})
    for r in per_task:
        sp = manifest[r["task_id"]]["split"]
        s = stats[r["template"]]
        s["n"] += 1; s["n_by_split"][sp] += 1
        if r["status"] == "uncovered":
            s["uncovered"] += 1; s["valid_uncovered"] += sp == "valid"
        elif r["status"] != "ok":
            s["insufficient"] += 1
        elif r.get("rt_in_lt", True):
            s["checked"] += 1; s["agree"] += bool(r.get("agree"))
            if sp == "valid":
                s["valid_checked"] += 1; s["valid_agree"] += bool(r.get("agree"))
    table = {}
    for tmpl, s in stats.items():
        covered = s["valid_uncovered"] == 0 and s["valid_checked"] > 0 and s["valid_agree"] == s["valid_checked"]
        table[tmpl] = {**{k: v for k, v in s.items() if k != "n_by_split"}, "n_by_split": dict(s["n_by_split"]),
                       "agreement_valid": round(s["valid_agree"] / s["valid_checked"], 4) if s["valid_checked"] else None,
                       "agreement_all_splits": round(s["agree"] / s["checked"], 4) if s["checked"] else None,
                       "covered": covered, "covered_dev26_rule": bool(old["templates"].get(tmpl, {}).get("covered"))}
    n_cov = sum(v["n"] for v in table.values() if v["covered"])
    split_cov = {sp: sum(1 for t in manifest.values() if t["split"] == sp and table[t["metadata"]["template"]]["covered"]) for sp in ("train", "valid", "test")}
    return {"rule": "covered iff every checked task of the `valid` split agrees with the released answer, at least one valid task is checked, and no valid task is uncovered (Tier-C protocol, Development pool); templates absent from `valid` are uncovered",
            "selection_splits": ["valid"], "source_per_task": "experiments/scopebench_tierC/skill_validation.per_task.jsonl",
            "n_tasks": len(manifest), "n_templates": len(table), "covered_templates": sum(v["covered"] for v in table.values()),
            "covered_tasks": n_cov, "coverage_pct": round(100 * n_cov / len(manifest), 2), "covered_tasks_by_split": split_cov,
            "dev26_rule": {"covered_templates": old["covered_templates"], "covered_tasks": old["covered_tasks"]},
            "templates_added": sorted(t for t, v in table.items() if v["covered"] and not v["covered_dev26_rule"]),
            "templates_removed": sorted(t for t, v in table.items() if v["covered_dev26_rule"] and not v["covered"]),
            "templates": table}


def offline_finalize(row: dict, task: Task, skill, store, pmap: dict[str, str]) -> dict:
    """The DEV-26 `finalize` step (scopebench/tierc/agent.py) applied to a recorded skill-less `full` row."""
    tt = get_task_type("fhir_agentbench_v1")
    out = copy.deepcopy(row)
    ts = TaskStore(store=store, patient_fhir_id=pmap[task.patient], patient_hash=task.patient, contract=task.contract,
                   enforcement="contract", question=task.question)
    summ = skill(ts, task)
    prov_pairs = summ.pop("_provenance_pairs", []) if summ else []
    out["skill_summary"] = summ
    if not summ or summ.get("status") != "ok":
        return out
    parsed = row.get("parsed_answer")
    agrees = bool(parsed is not None and parsed.get("answer_text")
                  and canonical_answer_correct(parsed["answer_text"], json.dumps(summ["answer"])))
    summ["agrees_with_submission"] = agrees
    errs = list(row.get("final_guard_errors") or [])
    if not agrees:
        errs.append("answer differs from the template skill's result")
        out["final_guard"] = {**(row.get("final_guard") or {}), "skill_disagreements": ["answer differs from the template skill's result"]}
        out["final_guard_errors"] = errs
    if not errs:
        return out
    # replacement: the skill's answer, its provenance as citations, the provenance resources entering A_t as a legal access
    out["replacement_reason"] = errs
    new = {"answer_text": json.dumps(summ["answer"]), "answer": summ["answer"], "citations": list(summ.get("provenance", [])),
           "citations_line_present": True, "replaced": True}
    pool = {}
    for r, e in prov_pairs:
        pool[str(r["id"])] = e if e is not None else ts.evidence(r)
    for rid in task.required_ids:
        if rid not in pool:
            for table in store.by_type_id.values():
                if rid in table:
                    pool[rid] = ts.evidence(table[rid]); break
    prov_ids = {str(r["id"]) for r, _ in prov_pairs}
    supplied = [pool[i] for i in prov_ids if i in pool]
    rep = check(new, task, tt, pool, supplied, prov_ids, check_completeness=True)
    m_old = row["metrics"]
    m = score(new, task, tt, pool, prov_ids, tokens_in=m_old["tokens_in"], tokens_out=m_old["tokens_out"], calls=m_old["calls"], latency=m_old["latency_seconds"])
    # access legality is a property of the recorded episode plus the (legal) provenance access
    m["access_legal"] = bool(m_old["access_legal"]) and bool(m["access_legal"])
    m["n_illegal_access"] = m_old.get("n_illegal_access", 0) + m["n_illegal_access"]
    m["workflow_compliant"] = bool(m["substantive"] and m["access_legal"] and m["citation_legal"])
    out.update({"parsed_answer": new, "metrics": m, "guard_deterministic_fallback": True, "answer_changed_by_repair": False,
                "final_guard": rep.to_dict(), "final_guard_errors": rep.errors()})
    acc = row.get("access_ids") or []
    out["access_ids"] = sorted(set(acc) | prov_ids) if acc or int(row.get("n_access_ids", 0)) == 0 else []
    out["n_access_ids"] = int(row.get("n_access_ids", len(acc))) + len(prov_ids - set(acc))
    return out


def derive(tag_dir: Path, out_dir: Path, tasks: dict[str, Task], new_cov: set[str], old_cov: set[str], skills: dict, store, pmap) -> dict:
    full = {r["task_id"]: r for r in read_jsonl(tag_dir / "full" / "predictions.jsonl")}
    noskill = {r["task_id"]: r for r in read_jsonl(tag_dir / "_superseded_full_noskill_dev26" / "full" / "predictions.jsonl")}
    assert set(full) == set(noskill), f"{tag_dir}: full and skill-less full cover different tasks"
    rows, origin = [], Counter()
    check_rows = Counter(); mismatch = Counter()
    for tid in sorted(full):
        t = tasks[tid]; tmpl = t.metadata["template"]
        if tmpl in new_cov and tmpl in old_cov:
            rows.append(full[tid]); origin["formal_full"] += 1
            rep = offline_finalize(noskill[tid], t, skills[tmpl], store, pmap)   # validation of the offline step
            check_rows["n"] += 1
            for ep in ENDPOINTS:
                mismatch[ep] += float(rep["metrics"][ep]) != float(full[tid]["metrics"][ep])
            mismatch["replaced"] += bool(rep.get("guard_deterministic_fallback")) != bool(full[tid].get("guard_deterministic_fallback"))
            mismatch["model_text"] += (noskill[tid].get("response"), noskill[tid].get("repair_response")) != (full[tid].get("response"), full[tid].get("repair_response"))
        elif tmpl in new_cov:
            rows.append({**offline_finalize(noskill[tid], t, skills[tmpl], store, pmap), "dev31_origin": "offline_finalize"}); origin["offline_finalize"] += 1
        else:
            rows.append({**noskill[tid], "skill_summary": None, "dev31_origin": "noskill_full"}); origin["noskill_full"] += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    meta = json.loads((tag_dir / "full" / "metadata.json").read_text())
    meta["dev31"] = {"derived_by": "scripts/scopebench/rederive_tierc_coverage.py", "origin_counts": dict(origin),
                     "coverage_record": "experiments/scopebench_tierC/skill_validation_protocol_dev31.json",
                     "formal_full_sha256": sha256(tag_dir / "full" / "predictions.jsonl"),
                     "noskill_full_sha256": sha256(tag_dir / "_superseded_full_noskill_dev26" / "full" / "predictions.jsonl")}
    meta["skills_mapped_templates"] = sorted(new_cov)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    return {"origin": dict(origin), "validation_rows": check_rows["n"], "validation_mismatch": {k: v for k, v in mismatch.items()}}


def mirror(src: Path, dst: Path, skip: set[str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for c in sorted(src.iterdir()):
        if c.is_dir() and c.name not in skip and not c.name.startswith("_") and (c / "predictions.jsonl").exists():
            link = dst / c.name
            if not link.exists():
                link.symlink_to(c.resolve(), target_is_directory=True)
    for name in ("token_budgets.json",):
        if (src / name).exists() and not (dst / name).exists():
            (dst / name).symlink_to((src / name).resolve())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=TAGS)
    ap.add_argument("--no-frontier", action="store_true")
    args = ap.parse_args()
    manifest = {r["task_id"]: r for r in read_jsonl(TC / "fhir_agentbench" / "tasks.jsonl")}
    per_task = list(read_jsonl(TC / "skill_validation.per_task.jsonl"))
    old = json.loads((TC / "skill_validation.json").read_text())
    rec = select_protocol_coverage(manifest, per_task, old)
    rec_path = TC / "skill_validation_protocol_dev31.json"
    rec_path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    new_cov = {t for t, v in rec["templates"].items() if v["covered"]}
    old_cov = {t for t, v in old["templates"].items() if v.get("covered")}
    print(f"coverage: DEV-26 rule {len(old_cov)} templates / {old['covered_tasks']} tasks; protocol rule {len(new_cov)} / {rec['covered_tasks']} "
          f"(added {len(rec['templates_added'])}, removed {len(rec['templates_removed'])}); covered by split {rec['covered_tasks_by_split']}")
    store = load_store(FHIR_DIR)
    pmap = json.load(open(PMAP))
    skills = mapped_skills(rec_path, pmap) | {t: f for t, f in mapped_skills(TC / "skill_validation.json", pmap).items()}
    tasks = {tid: Task.from_dict(r) for tid, r in manifest.items()}
    summary = {"coverage_record": str(rec_path.relative_to(ROOT)), "coverage_record_sha256": sha256(rec_path), "checkpoints": {}}
    for tag in args.tags:
        res = derive(RUNS / tag, OUT_RUNS / tag / "full", tasks, new_cov, old_cov, skills, store, pmap)
        mirror(RUNS / tag, OUT_RUNS / tag, skip={"full"})
        summary["checkpoints"][tag] = res
        print(tag, res, flush=True)
    if not args.no_frontier:
        fr = RUNS / "frontier" / "fhir_agentbench"
        res = derive(fr, OUT_RUNS / "frontier" / "fhir_agentbench" / "full", tasks, new_cov, old_cov, skills, store, pmap)
        mirror(fr, OUT_RUNS / "frontier" / "fhir_agentbench", skip={"full"})
        summary["checkpoints"]["frontier_fhir_agentbench"] = res
        print("frontier", res, flush=True)
    (OUT_RUNS / "DEV31_DERIVATION.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
