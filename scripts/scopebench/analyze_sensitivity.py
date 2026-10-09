#!/usr/bin/env python3
"""Sensitivity suite and masking replay for one checkpoint (plan §8.5 Supp. S5, §8.6). Loads no model.

S5: for every cell under <runs-dir>/sensitivity/<cell>/, the paired contrast full − prompt_only (patient-cluster
bootstrap) on answer correctness, completeness, auditability and the two scope endpoints, next to the reference
configuration (<runs-dir>/prompt_only, <runs-dir>/full) on the same task ids. Cells that run on a subsample (seeds on
the 1,200-task subsample, D7) are compared with the reference restricted to their task ids. `repair_wording_b` has
no prompt_only run (no repair turn); its pair is the reference prompt_only, as in the sensitivity driver. The five seed
cells are also summarized as mean, SD and range. Descriptive: no multiplicity correction.

Masking replay (diagnostic): for every <runs-dir>/masking/<removal>/<configuration>/, the paired change masked −
unmasked on the same tasks of the reference configuration, per configuration and pooled over prompt_only and full, and
the difference of the two changes (full − prompt_only, paired by task).

Usage: analyze_sensitivity.py --runs-dir experiments/scopebench_tierB_runs/<tag> --out experiments/scopebench_analysis/tierB/<tag>
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench.schema import read_jsonl  # noqa: E402
from scopebench.stats import paired_cluster_contrast  # noqa: E402

S5_ENDPOINTS = ["answer_correct", "completeness", "auditable", "access_legal", "citation_legal"]
MASK_ENDPOINTS = ["answer_correct", "completeness"]
SEEDS = ("seed13", "seed29", "seed47", "seed71", "seed101")
NO_PROMPT_ONLY_CELLS = {"repair_wording_b"}


def load(d: Path) -> dict[str, dict] | None:
    f = d / "predictions.jsonl"
    return {r["task_id"]: r for r in read_jsonl(f)} if f.exists() else None


def cell_kind(d: Path) -> str | None:
    m = d / "metadata.json"
    return json.loads(m.read_text()).get("cell_kind") if m.exists() else None


def ep(rows: dict[str, dict], name: str, ids=None) -> dict[str, float]:
    return {k: float(r["metrics"][name]) for k, r in rows.items() if ids is None or k in ids}


def contrast(a: dict[str, dict], b: dict[str, dict], endpoints, n_boot: int, seed: int, ids=None) -> dict:
    """b − a per endpoint over the common task ids (percentage points)."""
    common = set(a) & set(b) if ids is None else set(a) & set(b) & set(ids)
    clusters = {k: a[k]["patient"] for k in common}
    out = {"n": len(common), "n_patients": len(set(clusters.values()))}
    for e in endpoints:
        r = paired_cluster_contrast(ep(a, e, common), ep(b, e, common), clusters, n_boot=n_boot, seed=seed)
        out[e] = {"delta": round(100 * r["delta"], 2), "ci_low": round(100 * r["ci_low"], 2), "ci_high": round(100 * r["ci_high"], 2), "p_bootstrap": r["p_bootstrap"]}
    return out


def level(rows: dict[str, dict], endpoints, ids=None) -> dict[str, float]:
    keep = [r for k, r in rows.items() if ids is None or k in ids]
    return {e: round(100 * sum(float(r["metrics"][e]) for r in keep) / len(keep), 2) for e in endpoints}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=2027)
    args = p.parse_args()
    runs, out = args.runs_dir, args.out
    out.mkdir(parents=True, exist_ok=True)
    ref_po, ref_full = load(runs / "prompt_only"), load(runs / "full")
    if ref_po is None or ref_full is None:
        sys.exit(f"reference prompt_only / full missing under {runs}")
    res: dict = {"runs_dir": str(runs), "n_boot": args.n_boot, "seed": args.seed,
                 "reference": {"contrast": contrast(ref_po, ref_full, S5_ENDPOINTS, args.n_boot, args.seed),
                               "prompt_only": level(ref_po, S5_ENDPOINTS), "full": level(ref_full, S5_ENDPOINTS)},
                 "cells": {}, "masking": {}}

    # ---- S5 cells
    sens = runs / "sensitivity"
    for cdir in sorted(sens.iterdir()) if sens.exists() else []:
        full = load(cdir / "full")
        if full is None:
            continue
        po = ref_po if cdir.name in NO_PROMPT_ONLY_CELLS else load(cdir / "prompt_only")
        if po is None:
            continue
        ids = set(full) & set(po)
        cell = {"prompt_only_source": "reference" if cdir.name in NO_PROMPT_ONLY_CELLS else "cell",
                "cell_kind": {"prompt_only": cell_kind(cdir / "prompt_only"), "full": cell_kind(cdir / "full")},
                "contrast": contrast(po, full, S5_ENDPOINTS, args.n_boot, args.seed),
                "prompt_only": level(po, S5_ENDPOINTS, ids), "full": level(full, S5_ENDPOINTS, ids),
                "generation_errors": sum(1 for r in list(po.values()) + list(full.values()) if r.get("generation_errors"))}
        if len(ids) < len(ref_full):   # subsample cell: reference on the same task ids
            cell["reference_on_same_tasks"] = contrast(ref_po, ref_full, S5_ENDPOINTS, args.n_boot, args.seed, ids=ids)
        res["cells"][cdir.name] = cell
    seeds = [res["cells"][s] for s in SEEDS if s in res["cells"]]
    if seeds:
        res["seed_summary"] = {"n_seeds": len(seeds), "seeds": [s for s in SEEDS if s in res["cells"]]}
        for e in S5_ENDPOINTS:
            d = [c["contrast"][e]["delta"] for c in seeds]
            res["seed_summary"][e] = {"mean": round(statistics.mean(d), 2), "sd": round(statistics.stdev(d), 2) if len(d) > 1 else None,
                                      "min": min(d), "max": max(d)}

    # ---- masking replay
    mroot = runs / "masking"
    for mdir in sorted(mroot.iterdir()) if mroot.exists() else []:
        entry, per_task = {}, {}
        for cfg, ref in (("prompt_only", ref_po), ("full", ref_full)):
            masked = load(mdir / cfg)
            if masked is None:
                continue
            entry[cfg] = {"cell_kind": cell_kind(mdir / cfg), "change": contrast(ref, masked, MASK_ENDPOINTS, args.n_boot, args.seed),
                          "unmasked": level(ref, MASK_ENDPOINTS, set(masked)), "masked": level(masked, MASK_ENDPOINTS)}
            per_task[cfg] = (ref, masked)
        if len(per_task) == 2:
            # pooled over both configurations (cluster = patient) and the difference of the two changes
            (rp, mp), (rf, mf) = per_task["prompt_only"], per_task["full"]
            a_pool, b_pool, cl = {}, {}, {}
            for cfg, (ref, masked) in per_task.items():
                for k in set(ref) & set(masked):
                    a_pool[f"{cfg}|{k}"], b_pool[f"{cfg}|{k}"], cl[f"{cfg}|{k}"] = float(ref[k]["metrics"]["answer_correct"]), float(masked[k]["metrics"]["answer_correct"]), ref[k]["patient"]
            r = paired_cluster_contrast(a_pool, b_pool, cl, n_boot=args.n_boot, seed=args.seed)
            entry["pooled_answer_correct"] = {"n": r["n"], "delta": round(100 * r["delta"], 2), "ci_low": round(100 * r["ci_low"], 2), "ci_high": round(100 * r["ci_high"], 2)}
            common = set(rp) & set(mp) & set(rf) & set(mf)
            d_po = {k: float(mp[k]["metrics"]["answer_correct"]) - float(rp[k]["metrics"]["answer_correct"]) for k in common}
            d_fu = {k: float(mf[k]["metrics"]["answer_correct"]) - float(rf[k]["metrics"]["answer_correct"]) for k in common}
            r = paired_cluster_contrast(d_po, d_fu, {k: rp[k]["patient"] for k in common}, n_boot=args.n_boot, seed=args.seed)
            entry["full_minus_prompt_only_change"] = {"n": r["n"], "delta": round(100 * r["delta"], 2), "ci_low": round(100 * r["ci_low"], 2), "ci_high": round(100 * r["ci_high"], 2)}
        if entry:
            res["masking"][mdir.name] = entry

    (out / "sensitivity.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    with open(out / "sensitivity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "n", "endpoint", "prompt_only", "full", "delta_full_minus_prompt_only", "ci_low", "ci_high"])
        rows = [("reference", res["reference"]["contrast"], res["reference"]["prompt_only"], res["reference"]["full"])]
        rows += [(name, c["contrast"], c["prompt_only"], c["full"]) for name, c in res["cells"].items()]
        for name, con, lp, lf in rows:
            for e in S5_ENDPOINTS:
                w.writerow([name, con["n"], e, lp[e], lf[e], con[e]["delta"], con[e]["ci_low"], con[e]["ci_high"]])
    with open(out / "masking.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["removal", "configuration", "n", "endpoint", "unmasked", "masked", "change", "ci_low", "ci_high"])
        for name, entry in res["masking"].items():
            for cfg in ("prompt_only", "full"):
                if cfg in entry:
                    c = entry[cfg]
                    for e in MASK_ENDPOINTS:
                        w.writerow([name, cfg, c["change"]["n"], e, c["unmasked"][e], c["masked"][e], c["change"][e]["delta"], c["change"][e]["ci_low"], c["change"][e]["ci_high"]])
    ref = res["reference"]["contrast"]
    print(f"reference full − prompt_only: Δans {ref['answer_correct']['delta']:+.2f}  Δcompl {ref['completeness']['delta']:+.2f}  Δaud {ref['auditable']['delta']:+.2f}")
    for name, c in res["cells"].items():
        print(f"  {name:22s} n={c['contrast']['n']:5d}  Δans {c['contrast']['answer_correct']['delta']:+6.2f}  Δcompl {c['contrast']['completeness']['delta']:+6.2f}  Δaud {c['contrast']['auditable']['delta']:+6.2f}")
    for name, entry in res["masking"].items():
        parts = [f"{cfg} {entry[cfg]['change']['answer_correct']['delta']:+.2f}" for cfg in ("prompt_only", "full") if cfg in entry]
        print(f"  mask {name:22s} Δcorrect: " + ", ".join(parts))
    print(f"written {out}")


if __name__ == "__main__":
    main()
