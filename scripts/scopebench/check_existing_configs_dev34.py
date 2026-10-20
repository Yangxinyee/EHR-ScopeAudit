#!/usr/bin/env python3
"""DEV-34 regression (experiments/SCOPEBENCH_PLAN_PROTOCOL.md §7.3): every configuration that existed before DEV-34
writes byte-identical rows (predictions, access log, stages) with the DEV-34 code and with the pre-DEV-34 code.
Run once per code root (e.g. a `git worktree` of the parent commit) with the same --data, then compare the two output
trees with `cmp`. Deterministic scripted backends only: Tier-B dev (1,008 tasks; gold or abstention by a fixed hash,
gold on repair — exercises repair, verification and replacement) under 13 configurations; Tier-C valid (first 60;
one tool call, then an uncited answer) under scope_aware / audit / full. Loads no model.
Usage: SCOPEBENCH_PUBLIC_DATA=<public_data> check_existing_configs_dev34.py --root <code root> --out <dir> [--data <repo with the data>]"""
import argparse, hashlib, json, shutil, sys
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[2])
a = ap.parse_args()
DATA = a.data
sys.path[:0] = [str(a.root), str(a.root / "scripts")]
import scopebench  # noqa
assert Path(scopebench.__file__).resolve().is_relative_to(a.root.resolve()), scopebench.__file__
from scopebench.backends.base import Generation
from scopebench.backends.oracle_backend import gold_to_answer
from scopebench.configurations import CONFIGS
from scopebench.manifest import load_manifest
from scopebench.retrieval import TemporalNearestRetriever
from scopebench.runner import RunSpec, run
from scopebench.skills import OPERATIONS
from scopebench import tasktypes_tierb  # noqa

shutil.rmtree(a.out, ignore_errors=True)
tasks, cases = load_manifest(DATA / "experiments/scopebench_tierB_dev", None)
by_id = {t.task_id: t for t in tasks}
ABSTAIN = {"lab_trend_k3": {"status": "insufficient", "results": [], "delta": None, "direction": None},
           "outcome_lookup_v1": {"status": "insufficient", "value": None, "time": None, "citation": None},
           "cohort_v1": {"decision": "uncertain", "citations": []}, "claim_v1": {"label": "insufficient", "citations": []},
           "reconstruct_v1": {"status": "insufficient", "lines": []}, "summarize_exams_v1": {"status": "insufficient", "exams": []}}


class Mixed:
    """gold answer on every other task, abstention on the rest; the repair turn answers with gold (exercises repair,
    verification and replacement paths deterministically)."""
    name = "mixed"
    def generate_many(self, reqs):
        out = {}
        for r in reqs:
            t = by_id[r.key]
            odd = int(hashlib.sha256(t.task_id.encode()).hexdigest(), 16) % 2
            ans = gold_to_answer(t) if (odd or len(r.messages) > 2) else ABSTAIN[t.task_type]
            out[r.key] = Generation(text=json.dumps(ans, sort_keys=True), tokens_in=1, tokens_out=1, latency_seconds=0.0)
        return out
    def metadata(self):
        return {"backend": "mixed"}


cfgs = ["prompt_only", "wrong_basis", "scope_aware", "audit", "full", "retry_generic", "feedback_schema_only", "skill_payload_only",
        "skill_verify_no_replace", "skill_replace_no_payload", "citation_validator", "temporal_sql_view", "abac_pdp"]
for c in cfgs:
    run(RunSpec(configuration=CONFIGS[c], retriever=TemporalNearestRetriever(), k=30, skills=dict(OPERATIONS)), tasks, cases, Mixed(), a.out / "B" / c)

# Tier-C: scope_aware / audit / full on 60 valid tasks with a scripted two-turn backend
from scopebench.schema import Task, read_jsonl
from scopebench.tierc.agent import TierCSpec, run_tierc
from scopebench.tierc.skills import mapped_skills
from scopebench.tierc.store import load_store
from scopebench.config import PUBLIC_DATA
store = load_store(PUBLIC_DATA / "mimic-iv-clinical-database-demo-on-fhir-2.1.0" / "fhir")
pmap = json.load(open(PUBLIC_DATA / ".clinehr_cache/scopebench_tierc_patients/fhir_agentbench_patient_map.json"))
ctasks = [Task.from_dict(r) for r in read_jsonl(DATA / "experiments/scopebench_tierC/fhir_agentbench/tasks.jsonl") if r["split"] == "valid"][:60]


class TwoTurn:
    name = "two-turn"
    def __init__(self): self.n = {}
    def generate_many(self, reqs):
        out = {}
        for r in reqs:
            k = self.n[r.key] = self.n.get(r.key, 0) + 1
            if k == 1:
                pid = [m for m in r.messages if m["role"] == "user"][0]["content"].split("Patient FHIR ID: ")[1].split("\n")[0]
                text = json.dumps({"name": "get_resources_by_patient_fhir_id", "arguments": {"resource_type": "Observation", "patient_fhir_id": pid}}).join(["<tool_call>\n", "\n</tool_call>"])
            else:
                text = 'The final answer is: [["x"]]\nCitations: []'
            out[r.key] = Generation(text=text, tokens_in=1, tokens_out=1, latency_seconds=0.0)
        return out
    def metadata(self):
        return {"backend": "two-turn"}


skills = mapped_skills(DATA / "experiments/scopebench_tierC/skill_validation_protocol_dev31.json", pmap)
for c in ["scope_aware", "audit", "full"]:
    run_tierc(TierCSpec(configuration=c, adapter="text", skills=skills if c == "full" else {}), ctasks, store, pmap, TwoTurn(), None, a.out / "C" / c,
              manifest_path=DATA / "experiments/scopebench_tierC/fhir_agentbench")
print("done", a.root)
