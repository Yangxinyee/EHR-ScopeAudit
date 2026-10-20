#!/usr/bin/env python3
"""DEV-32: assemble the corrected Tier-C runs tree after the rerun, and check the control sample. Loads no model.

Per checkpoint (and the frontier), into experiments/scopebench_tierC_runs_dev32/<tag>/<config>:
  scoped configurations (wrong_basis, scope_aware, audit, full, security_label_consent):
      the DEV-31 rows (scopebench_tierC_runs_dev31: formal rows; `full` under the protocol coverage) with every AFFECTED
      task replaced by its rerun row (scopebench_tierC_runs_dev32_raw). The control tasks keep their original rows; their
      rerun rows only measure run-to-run identity on unaffected episodes (reported in DEV32_ASSEMBLY.json).
  unscoped configurations (prompt_only, token_matched, high_compute, direct_api, specialized_tools):
      the re-scored rows (scopebench_tierC_runs_dev32_rescored; same episodes, corrected legality).
  late_filter: re-derived from the re-scored prompt_only (derive_late_filter.py --loader tierc).
  anything else (agreement, medagentbench, token budgets): symlinked to the formal tree.
Refuses to assemble a scoped configuration whose rerun lacks an affected task (use --dry-run to see what is missing).
Usage: assemble_tierc_dev32.py [--tags ...] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench.schema import read_jsonl  # noqa: E402

E = ROOT / "experiments"
BASE, RAW, RESC, OUT = E / "scopebench_tierC_runs_dev31", E / "scopebench_tierC_runs_dev32_raw", E / "scopebench_tierC_runs_dev32_rescored", E / "scopebench_tierC_runs_dev32"
FORMAL = E / "scopebench_tierC_runs"
SETS = E / "scopebench_tierC/dev32"
TAGS = ["gemma3_27b", "medgemma_27b", "phi4_14b", "qwen3_30b", "qwen3_32b", "qwen3_4b"]
SCOPED = ["wrong_basis", "scope_aware", "audit", "full", "security_label_consent"]
UNSCOPED = ["prompt_only", "token_matched", "high_compute", "direct_api", "specialized_tools"]
FRONTIER_SCOPED = ["wrong_basis", "scope_aware", "audit", "full"]
ENDPOINTS = ["answer_correct", "completeness", "access_legal", "citation_legal", "auditable", "workflow_compliant"]


def rel(tag: str) -> str:
    return "frontier/fhir_agentbench" if tag == "frontier" else tag


def rows_by_id(p: Path) -> dict[str, dict]:
    return {r["task_id"]: r for r in read_jsonl(p)} if p.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=TAGS + ["frontier"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    affected = set(json.loads((SETS / "affected_task_ids.json").read_text()))
    control = set(json.loads((SETS / "control_task_ids.json").read_text()))
    report, missing_any = {}, False
    for tag in args.tags:
        scoped = FRONTIER_SCOPED if tag == "frontier" else SCOPED
        unscoped = ["prompt_only"] if tag == "frontier" else UNSCOPED
        rep = {"scoped": {}, "unscoped": {}}
        for cfg in scoped:
            base = rows_by_id(BASE / rel(tag) / cfg / "predictions.jsonl")
            raw = rows_by_id(RAW / rel(tag) / cfg / "predictions.jsonl")
            need = affected & set(base)
            miss = sorted(need - set(raw))
            ctrl = sorted(control & set(base) & set(raw))
            ident = {"n": len(ctrl), "model_text_identical": sum((raw[i].get("response"), raw[i].get("repair_response")) == (base[i].get("response"), base[i].get("repair_response")) for i in ctrl),
                     **{f"{ep}_identical": sum(float(raw[i]["metrics"][ep]) == float(base[i]["metrics"][ep]) for i in ctrl) for ep in ENDPOINTS}}
            rep["scoped"][cfg] = {"n_base": len(base), "n_affected": len(need), "n_rerun_present": len(need) - len(miss), "missing": len(miss), "control": ident}
            if miss:
                missing_any = True
                continue
            if not args.dry_run:
                d = OUT / rel(tag) / cfg
                d.mkdir(parents=True, exist_ok=True)
                with open(d / "predictions.jsonl", "w") as fh:
                    for tid in sorted(base):
                        r = dict(raw[tid]) if tid in affected else base[tid]
                        if tid in affected:
                            r["dev32_origin"] = "rerun"
                        fh.write(json.dumps(r, default=str) + "\n")
                meta = json.loads((BASE / rel(tag) / cfg / "metadata.json").read_text()) if (BASE / rel(tag) / cfg / "metadata.json").exists() else {}
                meta["dev32"] = {"by": "scripts/scopebench/assemble_tierc_dev32.py", "affected_rows_from": str((RAW / rel(tag) / cfg).relative_to(ROOT)), "n_replaced": len(need)}
                (d / "metadata.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
        for cfg in unscoped:
            src = RESC / rel(tag) / cfg
            ok = (src / "predictions.jsonl").exists()
            rep["unscoped"][cfg] = {"rescored_present": ok}
            missing_any |= not ok
            if ok and not args.dry_run:
                link = OUT / rel(tag) / cfg
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.exists():
                    link.symlink_to(src.resolve(), target_is_directory=True)
        if not args.dry_run and (RESC / rel(tag) / "prompt_only/predictions.jsonl").exists():
            out_lf = OUT / rel(tag) / "late_filter"
            if not (out_lf / "predictions.jsonl").exists():
                subprocess.run([sys.executable, str(ROOT / "scripts/scopebench/derive_late_filter.py"), "--loader", "tierc", "--manifest", str(E / "scopebench_tierC/fhir_agentbench"),
                                "--prompt-only", str(RESC / rel(tag) / "prompt_only"), "--out", str(out_lf)], check=True)
            src_root = FORMAL / rel(tag)
            for c in sorted(src_root.iterdir()):
                link = OUT / rel(tag) / c.name
                if c.name.startswith("_") or link.exists() or c.name in scoped + unscoped + ["late_filter"]:
                    continue
                if c.is_dir() or c.name == "token_budgets.json":
                    link.symlink_to(c.resolve(), target_is_directory=c.is_dir())
        report[tag] = rep
        print(tag, json.dumps({c: (v["missing"], v["control"]["model_text_identical"], v["control"]["n"]) for c, v in rep["scoped"].items()}), flush=True)
    if not args.dry_run:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "DEV32_ASSEMBLY.json").write_text(json.dumps(report, indent=1) + "\n")
    if missing_any:
        print("INCOMPLETE: some scoped reruns or rescored configurations are missing (see the report)")
        sys.exit(3 if not args.dry_run else 0)


if __name__ == "__main__":
    main()
