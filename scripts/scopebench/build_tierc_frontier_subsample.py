#!/usr/bin/env python3
"""Frontier-arm task subsample for Tier-C FHIR-AgentBench (plan D12b, DEV-7; Xinye 2026-09-29).

Draws N tasks (default 300) from the frozen FHIR-AgentBench manifest with proportional allocation over the
(template × contract scope) cells (largest-remainder rounding) and a deterministic within-cell order
SHA-256(seed | template | scope | task_id), the same device as the Tier-B builder. Rows are copied unchanged,
so every open-weight run on the full manifest can be restricted to exactly these task ids for the
frontier-comparable columns. Writes <out>/tasks.jsonl, <out>/sampling.json and <out>/metadata.json.

Usage: build_tierc_frontier_subsample.py [--manifest experiments/scopebench_tierC/fhir_agentbench]
        [--out experiments/scopebench_tierC/fhir_agentbench_frontier300] [--n 300] [--seed 2027]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def order_key(seed: int, template: str, scope: str, task_id: str) -> str:
    return hashlib.sha256(f"{seed}|{template}|{scope}|{task_id}".encode()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT / "experiments/scopebench_tierC/fhir_agentbench")
    p.add_argument("--out", type=Path, default=ROOT / "experiments/scopebench_tierC/fhir_agentbench_frontier300")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=2027)
    a = p.parse_args()
    rows = [json.loads(l) for l in open(a.manifest / "tasks.jsonl")]
    cells: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for r in rows:
        cells[(r["metadata"]["template"], r["contract"]["scope"])].append(r)
    total = len(rows)
    # largest-remainder allocation of a.n over the cells, proportional to cell size
    quotas = {c: a.n * len(v) / total for c, v in cells.items()}
    alloc = {c: int(q) for c, q in quotas.items()}
    short = a.n - sum(alloc.values())
    for c, _ in sorted(quotas.items(), key=lambda kv: (-(kv[1] - int(kv[1])), kv[0]))[:short]:
        alloc[c] += 1
    chosen: list[dict] = []
    for c, v in cells.items():
        v_sorted = sorted(v, key=lambda r: order_key(a.seed, c[0], c[1], r["task_id"]))
        chosen.extend(v_sorted[: alloc[c]])
    chosen.sort(key=lambda r: r["task_id"])
    assert len(chosen) == a.n and len({r["task_id"] for r in chosen}) == a.n
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "tasks.jsonl", "w") as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # skill coverage per task from the validated registry (validate_tierc_skills.py), not from the presence of skill_params
    cov_file = ROOT / "experiments/scopebench_tierC/skill_validation.per_task.jsonl"
    covered: dict[str, bool] = {}
    if cov_file.exists():
        for l in open(cov_file):
            j = json.loads(l); covered[j.get("task_id")] = j.get("status") == "ok"
    skill_cov = lambda rs: (sum(1 for r in rs if covered.get(r["task_id"])) / len(rs)) if covered else None  # noqa: E731
    stats = {
        "source_manifest": str(a.manifest.relative_to(ROOT)) if a.manifest.is_absolute() else str(a.manifest),
        "source_sha256": hashlib.sha256(open(a.manifest / "tasks.jsonl", "rb").read()).hexdigest(),
        "n": a.n, "seed": a.seed, "method": "proportional allocation over (template × scope) cells, largest-remainder rounding, "
                                             "within-cell order SHA-256(seed|template|scope|task_id)",
        "n_cells": len(cells), "n_cells_sampled": sum(1 for c in alloc if alloc[c] > 0),
        "templates_total": len({c[0] for c in cells}), "templates_covered": len({c[0] for c in alloc if alloc[c] > 0}),
        "scope_counts": dict(collections.Counter(r["contract"]["scope"] for r in chosen)),
        "scope_counts_source": dict(collections.Counter(r["contract"]["scope"] for r in rows)),
        "split_counts": dict(collections.Counter(r["split"] for r in chosen)),
        "leakage_share": sum(1 for r in chosen if r.get("leakage")) / a.n,
        "leakage_share_source": sum(1 for r in rows if r.get("leakage")) / total,
        "skill_covered_share": skill_cov(chosen), "skill_covered_share_source": skill_cov(rows),
        "allocation": {f"{c[0]} | {c[1]}": {"source": len(cells[c]), "sampled": alloc[c]} for c in sorted(cells)},
        "task_ids_sha256": hashlib.sha256("\n".join(r["task_id"] for r in chosen).encode()).hexdigest(),
    }
    (a.out / "sampling.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    meta = json.load(open(a.manifest / "metadata.json")) if (a.manifest / "metadata.json").exists() else {}
    meta.update({"subsample_of": stats["source_manifest"], "subsample_n": a.n, "subsample_seed": a.seed, "purpose": "frontier arm (D12b / DEV-7)"})
    (a.out / "metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {a.out}: n={a.n} cells={stats['n_cells_sampled']}/{stats['n_cells']} templates={stats['templates_covered']}/{stats['templates_total']} "
          f"scopes={stats['scope_counts']} splits={stats['split_counts']} skill_cov={stats['skill_covered_share']:.3f} (source {stats['skill_covered_share_source']:.3f}) "
          f"leakage={stats['leakage_share']:.3f} (source {stats['leakage_share_source']:.3f}) ids_sha={stats['task_ids_sha256'][:12]}")


if __name__ == "__main__":
    main()
