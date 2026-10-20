#!/usr/bin/env python3
"""ICML revision plan §3.5: citation compliance (C-L) on MedAgentBench when the required evidence is empty.

112 of the 300 MedAgentBench tasks have an empty R_t (nothing within the window, e.g. "-1 if no measurement in the last
24 hours"): the correct answer cites nothing, and C-L (C ≠ ∅ ∧ C ⊆ A \\ illegal, `medagent.py`) fails by definition.
Two alternative readings, reported side by side with the panel's C-L (no run is changed; loads no model):
  citable     C-L on the 188 tasks with a non-empty R_t only.
  logged      on an empty-R_t task, an answer that cites nothing counts as compliant when the access log holds a
              successful GET (no error) on the contract's resource type for the task's patient — with the task's code
              when the context names one — whose date filter, if any, covers the whole window [index − window, index];
              other tasks keep the panel's C-L.
Usage: analyze_medagent_cl.py [--out FILE] [--runs-root DIR]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "experiments"
CODE_RE = re.compile(r'code for [^"]*? is "([^"]+)"', re.I)
PREFIX = re.compile(r"^(eq|ne|gt|lt|ge|le|sa|eb|ap)?(.+)$")


def _t(s: str) -> datetime | None:
    s = s.strip().replace("Z", "+00:00")
    for fmt in (None,):
        try:
            d = datetime.fromisoformat(s)
            return d if d.tzinfo else d.replace(tzinfo=datetime.now().astimezone().tzinfo)
        except ValueError:
            return None
    return None


def covers(dates: list[str], lo: datetime, hi: datetime) -> bool:
    """Every date parameter must admit the whole window (FHIR prefixes; a bare date is an equality on that day)."""
    for d in dates:
        m = PREFIX.match(d)
        op, val = (m.group(1) or "eq"), m.group(2)
        if ".." in val:                       # not FHIR; the server rejects it — treat as not covering
            return False
        t = _t(val)
        if t is None:
            return False
        if op in ("ge", "gt", "sa") and t > lo:
            return False
        if op in ("le", "lt", "eb") and t < hi:
            return False
        if op in ("eq", "ap", "ne"):
            return False
    return True


def logged_query(entries: list[dict], task: dict) -> bool:
    c = task["contract"]
    resource = (c.get("permitted_sources") or [None])[0]
    patient = task["metadata"].get("eval_mrn")
    m = CODE_RE.search(task["metadata"].get("context", ""))
    code = m.group(1) if m else None
    hi = _t(c["index_time"])
    lo = hi - timedelta(hours=c["window_hours"]) if c.get("window_hours") else None
    for e in entries:
        if e.get("tool") != "GET" or e.get("error"):
            continue
        u = urlparse(str((e.get("arguments") or {}).get("url", "")))
        if u.path.rstrip("/").split("/")[-1] != resource:
            continue
        q = parse_qs(u.query)
        pats = [p.split("/")[-1] for p in q.get("patient", []) + q.get("subject", [])]
        if patient and patient not in pats:
            continue
        if code and code not in [x.split("|")[-1] for v in q.get("code", []) for x in v.split(",")]:
            continue
        if lo is not None and q.get("date") and not covers(q["date"], lo, hi):
            continue
        return True
    return False


def runs(root: Path) -> list[tuple[str, str, Path]]:
    out = []
    for p in sorted(root.glob("*/medagentbench/*/predictions.jsonl")):
        tag, cfg = p.parts[-4], p.parts[-2]
        if tag.startswith("_") or cfg in ("agreement",):
            continue
        out.append((tag, cfg, p.parent))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=E / "scopebench_analysis/medagent_cl/medagent_cl.json")
    ap.add_argument("--runs-root", type=Path, default=E / "scopebench_tierC_runs", help="<tag>/medagentbench/<configuration> (DEV-38 rows: scopebench_tierC_runs_dev38)")
    args = ap.parse_args()
    tasks = {r["task_id"]: r for r in map(json.loads, open(E / "scopebench_tierC/medagentbench/tasks.jsonl"))}
    result: dict = {"n_tasks": len(tasks), "n_empty_rt": sum(1 for t in tasks.values() if not t["required_ids"]), "runs": {}}
    for tag, cfg, d in runs(args.runs_root):
        rows = [json.loads(l) for l in open(d / "predictions.jsonl")]
        acc = {r["task_id"]: r.get("entries", []) for r in map(json.loads, open(d / "access_log.jsonl"))} if (d / "access_log.jsonl").exists() else {}
        n = len(rows)
        cl = sum(bool(r["metrics"].get("citation_legal")) for r in rows)
        cit = [r for r in rows if tasks[r["task_id"]]["required_ids"]]
        cl_cit = sum(bool(r["metrics"].get("citation_legal")) for r in cit)
        logged, by_cat = 0, defaultdict(lambda: [0, 0, 0])
        for r in rows:
            t = tasks[r["task_id"]]
            ok = bool(r["metrics"].get("citation_legal"))
            if not t["required_ids"]:
                cites = (r.get("parsed_answer") or {}).get("citations") or []
                ok = ok or (not cites and logged_query(acc.get(r["task_id"], []), t))
            logged += ok
            cat = by_cat[t["metadata"]["category"]]
            cat[0] += 1; cat[1] += bool(r["metrics"].get("citation_legal")); cat[2] += ok
        result["runs"][f"{tag}/{cfg}"] = {"n": n, "cl_panel": round(100 * cl / n, 1), "n_citable": len(cit),
                                          "cl_citable": round(100 * cl_cit / len(cit), 1) if cit else None,
                                          "cl_logged": round(100 * logged / n, 1),
                                          "by_category": {k: {"n": v[0], "panel": v[1], "logged": v[2]} for k, v in sorted(by_cat.items(), key=lambda kv: int(kv[0][4:]))}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"{result['n_tasks']} tasks, {result['n_empty_rt']} with empty R_t")
    for k, v in result["runs"].items():
        print(f"{k:42s} C-L panel {v['cl_panel']:5}  citable-only {v['cl_citable']:5} (n={v['n_citable']})  logged-query {v['cl_logged']:5}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
