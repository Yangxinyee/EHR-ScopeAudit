#!/usr/bin/env python3
"""ICML revision plan §3.2 (the stronger form): for agents that issue their own requests, is out-of-scope access
followed by confident wrong answers? Paired by task: the same model on the same task without enforcement
(`prompt_only`, where it may read out-of-scope resources) and with the contract enforced at the tool layer
(`scope_aware`, where it cannot). A confident wrong answer is a substantive (non-abstaining, parseable) answer that the
benchmark marks incorrect. Tasks are split by whether the unenforced run accessed anything out of scope; the tasks it
did not expose serve as the control for everything else enforcement changes (difference in differences). Patient-
cluster bootstrap. Frontier model on FHIR-AgentBench (DEV-32 rows) and MedAgentBench, the open panel on FHIR-AgentBench
train + test. Loads no model.
Usage: analyze_exposure_confident_wrong.py [--out FILE] [--n-boot N] [--med-runs-root DIR]   (DEV-38 rows: scopebench_tierC_runs_dev38)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/scopebench"))

from analyze_agent_hidden import OPEN, n_illegal  # noqa: E402
from scopebench.schema import read_jsonl  # noqa: E402

E = ROOT / "experiments"


def confident_wrong(r: dict) -> bool:
    m = r["metrics"]
    return not m.get("abstained") and not m.get("parse_failed") and not m["answer_correct"]


def load(path: Path, keep: set[str] | None) -> dict[str, dict]:
    return {r["task_id"]: r for r in read_jsonl(path) if keep is None or r["task_id"] in keep} if path.exists() else {}


def cell(pairs: list[tuple[str, dict, dict]], n_boot: int, seed: int = 2027) -> dict:
    """pairs: (cluster, unenforced row, enforced row). Rates in %, DiD with a cluster bootstrap."""
    def rates(ps):
        if not ps:
            return None
        n = len(ps)
        return {"n": n, "cw_unenforced": 100 * sum(confident_wrong(a) for _, a, _ in ps) / n,
                "cw_enforced": 100 * sum(confident_wrong(b) for _, _, b in ps) / n,
                "correct_unenforced": 100 * sum(bool(a["metrics"]["answer_correct"]) for _, a, _ in ps) / n,
                "correct_enforced": 100 * sum(bool(b["metrics"]["answer_correct"]) for _, _, b in ps) / n,
                "abstain_unenforced": 100 * sum(bool(a["metrics"].get("abstained")) for _, a, _ in ps) / n,
                "abstain_enforced": 100 * sum(bool(b["metrics"].get("abstained")) for _, _, b in ps) / n}

    exp = [p for p in pairs if n_illegal(p[1]) > 0]
    une = [p for p in pairs if n_illegal(p[1]) == 0]

    def did(e, u):
        re_, ru = rates(e), rates(u)
        if not re_ or not ru:
            return None
        return (re_["cw_unenforced"] - re_["cw_enforced"]) - (ru["cw_unenforced"] - ru["cw_enforced"])
    point = did(exp, une)
    by_cluster = defaultdict(list)
    for p in pairs:
        by_cluster[p[0]].append(p)
    keys = sorted(by_cluster)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [p for k in (rng.choice(keys) for _ in keys) for p in by_cluster[k]]
        d = did([p for p in sample if n_illegal(p[1]) > 0], [p for p in sample if n_illegal(p[1]) == 0])
        if d is not None:
            boots.append(d)
    boots.sort()
    ci = [round(boots[int(0.025 * len(boots))], 1), round(boots[int(0.975 * len(boots)) - 1], 1)] if boots else None
    rnd = lambda d: {k: (round(v, 1) if isinstance(v, float) else v) for k, v in d.items()} if d else None  # noqa: E731
    return {"exposed": rnd(rates(exp)), "unexposed": rnd(rates(une)), "did_confident_wrong": round(point, 1) if point is not None else None, "did_ci": ci}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=E / "scopebench_analysis/agent_hidden/confident_wrong.json")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--med-runs-root", type=Path, default=E / "scopebench_tierC_runs")
    args = ap.parse_args()
    keep = set(json.loads((E / "scopebench_tierC/plan_dev34/traintest_task_ids.json").read_text()))
    settings = {"frontier_fhir": (E / "scopebench_tierC_runs_dev32/frontier/fhir_agentbench", None),
                "frontier_med": (args.med_runs_root / "frontier/medagentbench", None)}
    for t in OPEN:
        settings[f"open_fhir:{t}"] = (E / "scopebench_tierC_runs_dev32" / t, keep)
    out, pooled_open = {}, []
    for name, (base, k) in settings.items():
        a, b = load(base / "prompt_only/predictions.jsonl", k), load(base / "scope_aware/predictions.jsonl", k)
        pairs = [(str(a[i].get("patient")), a[i], b[i]) for i in sorted(set(a) & set(b))]
        if not pairs:
            continue
        out[name] = cell(pairs, args.n_boot)
        if name.startswith("open_fhir:"):
            pooled_open += [(f"{name}|{c}", x, y) for c, x, y in pairs]
    if pooled_open:
        out["open_fhir_pooled"] = cell(pooled_open, args.n_boot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for name, c in out.items():
        e, u = c["exposed"] or {}, c["unexposed"] or {}
        print(f"{name:22s} exposed n={e.get('n')}: CW {e.get('cw_unenforced')}→{e.get('cw_enforced')}, correct {e.get('correct_unenforced')}→{e.get('correct_enforced')} | "
              f"unexposed n={u.get('n')}: CW {u.get('cw_unenforced')}→{u.get('cw_enforced')} | DiD {c['did_confident_wrong']} {c['did_ci']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
