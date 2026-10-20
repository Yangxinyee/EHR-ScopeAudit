#!/usr/bin/env python3
"""ICML revision plan §6: the checkpoint as a sampling unit. The patient-cluster intervals of the paper condition on the
evaluated checkpoints; this reads the per-checkpoint paired differences already in the frozen analyses and reports, for
each pooled contrast, (a) a t interval over checkpoints (each checkpoint's difference is one observation, k − 1 degrees
of freedom), (b) the leave-one-checkpoint-out range of the pooled difference (every checkpoint runs the same tasks, so
the pooled row difference is the mean of the checkpoint differences), and (c) the number of checkpoints with a positive
difference. It also applies Holm's step-down adjustment across the deployable-auditor family (plan_full − scope_aware and
plan_full − full, answer correctness, per tier) as a robustness check; the four protocol contrasts keep their own Holm
family. Loads no model and no rows.
Usage: analyze_checkpoint_uncertainty.py [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench.stats import holm  # noqa: E402

AN = ROOT / "experiments/scopebench_analysis"
# two-sided 97.5 % Student-t quantiles by degrees of freedom (k − 1 for k checkpoints)
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def summarize(deltas: dict[str, float]) -> dict:
    xs = list(deltas.values())
    k = len(xs)
    m = mean(xs)
    out = {"k": k, "per_checkpoint": {t: round(v, 1) for t, v in sorted(deltas.items())}, "mean": round(m, 1),
           "n_positive": sum(x > 0 for x in xs), "min": round(min(xs), 1), "max": round(max(xs), 1)}
    if k >= 2:
        half = T975[k - 1] * stdev(xs) / math.sqrt(k)
        out["t_ci"] = [round(m - half, 1), round(m + half, 1)]
        loo = [mean(xs[:i] + xs[i + 1:]) for i in range(k)]
        out["loco_range"] = [round(min(loo), 1), round(max(loo), 1)]
    return out


def plan_deltas(files: list[str], contrast: str) -> dict[str, float]:
    out = {}
    for f in files:
        d = json.loads((AN / "plans_dev34" / f).read_text())
        for t, e in d["checkpoints"].items():
            c = e["contrasts"].get(contrast)
            if c and c["all"].get("answer_correct"):
                out[t] = c["all"]["answer_correct"]["delta"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=AN / "checkpoint_uncertainty/checkpoint_uncertainty.json")
    args = ap.parse_args()
    tiers = {"B": ["tierB_formal.json", "tierB_formal_extension.json"], "A": ["tierA_formal.json"],
             "C": ["tierC_formal.json", "tierC_formal_extension.json"]}
    res: dict = {"deployable": {}, "protocol": {}}
    for X, files in tiers.items():
        for name in ("plan_full - scope_aware", "plan_full - full"):
            res["deployable"][f"{X}|{name}"] = summarize(plan_deltas(files, name))
    # the four protocol contrasts of the Tier-B panel (six checkpoints)
    hyp = json.loads((AN / "tierB/panel/panel.json").read_text())["hypotheses"]
    for h in hyp["pooled"]:
        res["protocol"][h] = summarize({t: v[h]["delta_pct"] for t, v in hyp["per_checkpoint"].items()})
    # Holm across the deployable family (pooled patient-cluster bootstrap p of the frozen analyses)
    raw = {}
    for X, files in tiers.items():
        d = json.loads((AN / "plans_dev34" / files[0]).read_text())
        for name in ("plan_full - scope_aware", "plan_full - full"):
            raw[f"{X}|{name}"] = d["pooled"]["contrasts"][name]["all"]["answer_correct"]["p"]
    res["holm_deployable"] = {"raw_p": raw, "p_holm": holm(raw),
                              "note": "bootstrap p floors at 1/10,000; the family is the six pooled answer-correctness contrasts of the protocol checkpoints"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    for k, v in list(res["deployable"].items()) + list(res["protocol"].items()):
        print(f"{k:40s} k={v['k']} mean={v['mean']:>6} t-CI={v.get('t_ci')} LOCO={v.get('loco_range')} pos={v['n_positive']}/{v['k']}")
    print("Holm (deployable):", res["holm_deployable"]["p_holm"])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
