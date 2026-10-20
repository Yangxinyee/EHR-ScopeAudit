#!/usr/bin/env python3
"""WP7 analysis (plan §10; paper §4 "Clinician adjudication"): inter-rater agreement, confusion matrices against the
programmatic labels, the prespecified five-point non-inferiority test, the direct-set sub-study, and the power
calculation that goes into the protocol.

Inputs: one or more packet roots (e.g. experiments/scopebench_human/adjudication and, inside the enclave,
<enclave>/human/adjudication). Under each root, tier directories hold the packets, ANSWER_KEY.jsonl (main cards) and
ANSWER_KEY_direct.jsonl (direct-set cards). Ratings are the files the rating page exports, `ratings_<initials>_<bundle>.jsonl`,
placed anywhere under a root (one row per card: pair_id, packet_type, rater, complete, the three answers or the
allowed_ids / needed_ids lists, comment, seconds). Only rows marked complete are analysed.

Outputs a JSON report (and prints it): per field — Krippendorff's α (nominal, over the cards rated by ≥ 2 raters),
majority label vs programmatic label (sensitivity / specificity / agreement), human–human pairwise agreement, the
one-sided non-inferiority test (margin 5 points, α .05) and per-stratum agreement; direct-set — item-level agreement
of each rater's allowed / needed sets with the programmatic sets (recall, precision, Jaccard), rater–rater Jaccard on
the overlap cards, and the same non-inferiority test at the item level.

Usage: adjudication_analysis.py <packets root> [<packets root> ...] [--margin 5] [--alpha 0.05] [--json report.json]
       adjudication_analysis.py --power [--p0 0.90 --margin 5 --alpha 0.05 --power-target 0.80]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = ("admissible_evidence_used", "required_evidence_cited", "answer_clinically_correct")
PROGRAMMATIC = {"admissible_evidence_used": "citation_legal", "required_evidence_cited": "completeness_full", "answer_clinically_correct": "answer_correct"}


def krippendorff_alpha_nominal(units: dict[str, list[str]]) -> float | None:
    """Krippendorff's α for nominal data; units -> list of ratings (≥ 2 raters per unit are used)."""
    values = [v for vs in units.values() if len(vs) >= 2 for v in vs]
    if not values:
        return None
    n = len(values)
    cats = sorted(set(values))
    o = {c: {d: 0.0 for d in cats} for c in cats}
    for vs in units.values():
        m = len(vs)
        if m < 2:
            continue
        for a, b in itertools.permutations(vs, 2):
            o[a][b] += 1.0 / (m - 1)
    nc = {c: sum(o[c].values()) for c in cats}
    do = sum(o[c][d] for c in cats for d in cats if c != d)
    de = sum(nc[c] * nc[d] for c in cats for d in cats if c != d) / (n - 1) if n > 1 else 0.0
    return None if de == 0 else 1.0 - do / de


def confusion(human: dict[str, str], programmatic: dict[str, bool]) -> dict:
    cm = Counter()
    for pid, h in human.items():
        if pid not in programmatic or h == "unsure":
            continue
        cm[(h == "yes", programmatic[pid])] += 1
    tp, fn, fp, tn = cm[(True, True)], cm[(False, True)], cm[(True, False)], cm[(False, False)]
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "sensitivity": sens, "specificity": spec, "agreement": (tp + tn) / max(1, tp + fn + fp + tn), "n": tp + fn + fp + tn}


def noninferiority(p_prog: float, p_human: float, n: int, margin: float, alpha: float = 0.05) -> dict:
    """One-sided test that the programmatic label's agreement with the human majority is not more than `margin`
    points below the human–human reference (Wald, paired proportions approximated as independent)."""
    d = (p_prog - p_human) * 100.0
    se = 100.0 * math.sqrt(max(1e-12, p_prog * (1 - p_prog) / n + p_human * (1 - p_human) / n))
    z = (d + margin) / se
    from statistics import NormalDist
    p = 1 - NormalDist().cdf(z)
    return {"difference_points": d, "se_points": se, "z": z, "p_one_sided": p, "noninferior_at_alpha": p < alpha, "ci_lower_points": d - 1.645 * se}


def power(p0: float, margin: float, alpha: float, target: float) -> dict:
    """Sample size per arm for a one-sided non-inferiority test of two proportions with true equality (p1 = p0),
    margin in points; normal approximation."""
    from statistics import NormalDist
    z_a, z_b = NormalDist().inv_cdf(1 - alpha), NormalDist().inv_cdf(target)
    delta = margin / 100.0
    n = ((z_a + z_b) ** 2) * 2 * p0 * (1 - p0) / (delta ** 2)
    return {"p0": p0, "margin_points": margin, "alpha_one_sided": alpha, "power": target, "n_per_arm": math.ceil(n),
            "note": "every main card is rated on all three fields, so the per-field n is the number of cards with a majority label; per-stratum estimates are descriptive only"}


# ------------------------------------------------------------------------------------------ loading
def read_jsonl(p: Path):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _set_aside(root: Path, p: Path) -> bool:
    """Directories whose name starts with '_' hold superseded samples (for example `_sample600_20261011/`); skipped."""
    return any(part.startswith("_") for part in p.relative_to(root).parts)


def load_roots(roots: list[Path]):
    key, key_direct, strata, ratings = {}, {}, {}, defaultdict(dict)
    for root in roots:
        for kf in root.rglob("ANSWER_KEY.jsonl"):
            if _set_aside(root, kf):
                continue
            for r in read_jsonl(kf):
                key[r["pair_id"]] = r
            for pf in kf.parent.glob("*.json"):
                if pf.name.startswith("direct_"):
                    continue
                try:
                    p = json.loads(pf.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(p, dict) and "pair_id" in p:
                    strata[p["pair_id"]] = (p.get("tier"), p.get("scope"), p.get("task_group"))
        for kf in root.rglob("ANSWER_KEY_direct.jsonl"):
            if _set_aside(root, kf):
                continue
            for r in read_jsonl(kf):
                key_direct[r["pair_id"]] = r
        for rf in root.rglob("ratings_*.jsonl"):
            if _set_aside(root, rf):
                continue
            for r in read_jsonl(rf):
                if not r.get("complete"):
                    continue
                rater = str(r.get("rater") or rf.stem.split("_")[1]).upper()
                ratings[rater][r["pair_id"]] = r
    return key, key_direct, strata, ratings


# ------------------------------------------------------------------------------------------ main cards
def analyse_fields(key, strata, ratings, margin, alpha, conf_tiers=("B", "C")):
    """Per field: reliability, agreement with the programmatic label (all tiers, and per tier), and the prespecified
    non-inferiority test on the confirmatory tiers only; the other tiers are descriptive."""
    out = {}
    for f in FIELDS:
        units = defaultdict(list)
        for rt, rows in ratings.items():
            for pid, r in rows.items():
                if pid in key and r.get(f) in ("yes", "no", "unsure"):
                    units[pid].append(r[f])
        kalpha = krippendorff_alpha_nominal(units)
        majority = {pid: Counter(v).most_common(1)[0][0] for pid, v in units.items() if len(v) >= 2 and Counter(v).most_common(1)[0][1] > len(v) / 2}
        single = {pid: v[0] for pid, v in units.items() if len(v) == 1}
        prog = {}
        for pid, k in key.items():
            m = k.get("metrics") or {}
            prog[pid] = float(m.get("completeness", 0.0)) >= 1.0 if f == "required_evidence_cited" else bool(m.get(PROGRAMMATIC[f]))
        cm_majority = confusion(majority, prog)
        cm_all = confusion({**single, **majority}, prog)   # every card with a label: majority where ≥ 2 raters, the single rater otherwise
        agree, tot = 0, 0
        for pid, v in units.items():
            for x, y in itertools.combinations(v, 2):
                tot += 1
                agree += (x == y)
        p_human = agree / tot if tot else None
        labelled = {**single, **majority}
        conf = {pid: h for pid, h in labelled.items() if strata.get(pid, (None,))[0] in conf_tiers}
        cm_conf = confusion(conf, prog)
        pairs_conf = [v for pid, v in units.items() if strata.get(pid, (None,))[0] in conf_tiers]
        agree_c = sum((x == y) for v in pairs_conf for x, y in itertools.combinations(v, 2)); tot_c = sum(len(v) * (len(v) - 1) // 2 for v in pairs_conf)
        p_human_conf = agree_c / tot_c if tot_c else None
        ni = noninferiority(cm_conf["agreement"], p_human_conf, max(1, cm_conf["n"]), margin, alpha) if tot_c and cm_conf["n"] else None
        by_tier = {}
        for pid, h in labelled.items():
            if h == "unsure" or pid not in prog:
                continue
            d = by_tier.setdefault(str(strata.get(pid, ("?",))[0]), {"n": 0, "agree": 0}); d["n"] += 1; d["agree"] += int((h == "yes") == prog[pid])
        for d in by_tier.values():
            d["agreement"] = d["agree"] / d["n"] if d["n"] else None
        by_stratum = {}
        for pid, h in {**single, **majority}.items():
            if h == "unsure" or pid not in prog:
                continue
            s = "|".join(str(x) for x in strata.get(pid, ("?", "?", "?")))
            d = by_stratum.setdefault(s, {"n": 0, "agree": 0})
            d["n"] += 1
            d["agree"] += int((h == "yes") == prog[pid])
        for d in by_stratum.values():
            d["agreement"] = d["agree"] / d["n"] if d["n"] else None
        out[f] = {"krippendorff_alpha": kalpha, "n_units_rated": len(units), "n_rated_by_2plus": sum(1 for v in units.values() if len(v) >= 2),
                  "n_majority": len(majority), "unsure_share": (sum(v.count("unsure") for v in units.values()) / max(1, sum(len(v) for v in units.values()))),
                  "confusion_vs_programmatic_majority": cm_majority, "confusion_vs_programmatic_all_labelled": cm_all,
                  "confusion_vs_programmatic_confirmatory_tiers": cm_conf, "confirmatory_tiers": list(conf_tiers),
                  "human_pairwise_agreement": p_human, "human_pairwise_agreement_confirmatory_tiers": p_human_conf,
                  "noninferiority": ni, "by_tier": dict(sorted(by_tier.items())), "by_stratum": dict(sorted(by_stratum.items()))}
    return out


# ------------------------------------------------------------------------------------------ direct-set cards
def _sets(r):
    return set(r.get("allowed_ids") or []), set(r.get("needed_ids") or [])


def gold_spot_check(key, strata, ratings):
    """The reference answers checked through the raters: on cards whose submission the scorer marks correct (it
    reproduces the reference), a clinician's 'No' on answer correctness is a disagreement with the reference itself.
    Reported per tier with the majority label where two or more raters rated the card."""
    units = defaultdict(list)
    for rt, rows in ratings.items():
        for pid, r in rows.items():
            k = key.get(pid)
            if k and bool((k.get("metrics") or {}).get("answer_correct")) and r.get("answer_clinically_correct") in ("yes", "no", "unsure"):
                units[pid].append(r["answer_clinically_correct"])
    out = {"n_cards_auto_correct": sum(1 for k in key.values() if bool((k.get("metrics") or {}).get("answer_correct"))), "n_rated": len(units)}
    by_tier = defaultdict(lambda: {"n": 0, "yes": 0, "no": 0, "unsure": 0})
    for pid, v in units.items():
        lab = Counter(v).most_common(1)[0][0] if (len(v) == 1 or Counter(v).most_common(1)[0][1] > len(v) / 2) else "unsure"
        d = by_tier[str(strata.get(pid, ("?",))[0])]; d["n"] += 1; d[lab] += 1
    tot = {"n": 0, "yes": 0, "no": 0, "unsure": 0}
    for d in by_tier.values():
        for k2 in tot:
            tot[k2] += d[k2]
    out["by_tier"] = dict(sorted(by_tier.items())); out["all"] = tot
    out["reference_disagreement_rate"] = tot["no"] / tot["n"] if tot["n"] else None
    return out


def analyse_direct(key_direct, ratings, margin, alpha):
    per_rater = defaultdict(lambda: {"allowed": [], "needed": []})
    item_agree = {"allowed": [0, 0], "needed": [0, 0]}   # human-vs-programmatic item agreement (agree, total) over all rated cards
    hh = {"allowed": [0, 0], "needed": [0, 0]}           # human-vs-human item agreement on the overlap cards
    jac_rr = {"allowed": [], "needed": []}
    by_card = defaultdict(dict)
    for rt, rows in ratings.items():
        for pid, r in rows.items():
            if pid in key_direct and r.get("packet_type") == "direct_set":
                by_card[pid][rt] = _sets(r)
    for pid, raters in by_card.items():
        k = key_direct[pid]
        prog = {"allowed": set(k.get("programmatic_allowed_ids") or []), "needed": set(k.get("required_ids") or [])}
        universe = set(k.get("programmatic_allowed_ids") or [])
        for rt, (al, nd) in raters.items():
            universe |= al | nd
        for which, idx in (("allowed", 0), ("needed", 1)):
            p = prog[which]
            for rt, sets in raters.items():
                h = sets[idx]
                rec = len(h & p) / len(p) if p else None
                prec = len(h & p) / len(h) if h else None
                jac = len(h & p) / len(h | p) if (h | p) else None
                per_rater[rt][which].append({"pair_id": pid, "recall": rec, "precision": prec, "jaccard": jac, "n_human": len(h), "n_programmatic": len(p)})
                for item in universe:
                    item_agree[which][1] += 1
                    item_agree[which][0] += int((item in h) == (item in p))
            for (r1, s1), (r2, s2) in itertools.combinations(raters.items(), 2):
                h1, h2 = s1[idx], s2[idx]
                jac_rr[which].append(len(h1 & h2) / len(h1 | h2) if (h1 | h2) else None)
                for item in universe:
                    hh[which][1] += 1
                    hh[which][0] += int((item in h1) == (item in h2))
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None
    out = {"n_cards_rated": len(by_card), "n_cards_rated_by_2plus": sum(1 for v in by_card.values() if len(v) >= 2), "sets": {}}
    for which in ("allowed", "needed"):
        p_prog = item_agree[which][0] / item_agree[which][1] if item_agree[which][1] else None
        p_h = hh[which][0] / hh[which][1] if hh[which][1] else None
        ni = noninferiority(p_prog, p_h, max(1, item_agree[which][1]), margin, alpha) if (p_prog is not None and p_h is not None) else None
        out["sets"][which] = {"per_rater": {rt: {"n_cards": len(v[which]), "mean_recall_of_programmatic": mean([x["recall"] for x in v[which]]),
                                                  "mean_precision": mean([x["precision"] for x in v[which]]), "mean_jaccard": mean([x["jaccard"] for x in v[which]])} for rt, v in per_rater.items()},
                              "item_agreement_human_vs_programmatic": p_prog, "item_agreement_human_vs_human": p_h, "mean_rater_rater_jaccard": mean(jac_rr[which]),
                              "noninferiority_item_level": ni}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("packets", nargs="*", type=Path)
    ap.add_argument("--margin", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--p0", type=float, default=0.90)
    ap.add_argument("--power-target", type=float, default=0.80)
    ap.add_argument("--confirmatory-tiers", default="B,C", help="tiers on which the non-inferiority test is confirmatory (default B,C); the rest are descriptive")
    a = ap.parse_args()
    if a.power:
        print(json.dumps(power(a.p0, a.margin, a.alpha, a.power_target), indent=1))
        return 0
    if not a.packets:
        ap.error("at least one packets root is required unless --power")
    key, key_direct, strata, ratings = load_roots(a.packets)
    if not ratings:
        print(json.dumps({"n_main_cards": len(key), "n_direct_cards": len(key_direct), "note": "no completed ratings yet (ratings_*.jsonl); nothing to analyse"}, indent=1))
        return 0
    report = {"roots": [str(p) for p in a.packets], "n_main_cards": len(key), "n_direct_cards": len(key_direct), "raters": sorted(ratings),
              "n_complete_ratings": {rt: len(rows) for rt, rows in ratings.items()},
              "confirmatory_tiers": [x.strip() for x in a.confirmatory_tiers.split(",") if x.strip()],
              "fields": analyse_fields(key, strata, ratings, a.margin, a.alpha, tuple(x.strip() for x in a.confirmatory_tiers.split(",") if x.strip())),
              "gold_spot_check": gold_spot_check(key, strata, ratings),
              "direct_set": analyse_direct(key_direct, ratings, a.margin, a.alpha)}
    print(json.dumps(report, indent=1))
    if a.json:
        a.json.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
