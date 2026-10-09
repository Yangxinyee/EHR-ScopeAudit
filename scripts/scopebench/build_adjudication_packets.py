#!/usr/bin/env python3
"""WP7 (plan §10): the clinician adjudication instruments, built from the formal run directories.

1. Main sample — (task, submission) pairs, 25 per populated stratum over tier (A, B, C) × scope (pre, at, post, full) ×
   task group (structured = evidence retrieval + outcome abstraction + FHIR queries, cohort, free-text = claim /
   reconstruction / summarization). The submission of a sampled task is one of the eight core configurations of one of
   the six open-weight checkpoints, drawn uniformly (seeded); raters see the systems in proportion but never which one.
   Strata that no tier populates (e.g. Tier-C has only structured tasks and no at-index scope) are simply absent.
2. Direct-set sample — leakage-flagged tasks of Tier-A and Tier-B, non-overlapping with the main sample, allocated to
   the tiers in proportion to their leakage tasks and stratified by scope. Their packets show the whole supplied record
   and no submission; the rater marks the allowed and the needed records.
3. Blinded packets: one JSON per card with the question, a plain-language statement of the rule (contract), the records
   (ids, source, time, content; Tier-C records are rendered from the local FHIR store), the submission's answer and
   citations — no system label, no ScopeGuard verdict, no gold. The answer key (system, gold, programmatic labels,
   per-endpoint metrics, programmatic allowed / required sets) is written to separate files for the analysis step.
   Tier-A packets, keys and bundles are written only under --tierA-out (inside the enclave).
4. Rater assignment: three raters; a stratified overlap set rated by all three (for agreement), the rest split evenly
   with balanced strata. Per-rater bundles (the packets a rater opens in `experiments/scopebench_human/rater.html`):
   one local bundle (Tier-B, Tier-C) and one enclave bundle (Tier-A) per rater.

Usage: build_adjudication_packets.py --out experiments/scopebench_human/adjudication --seed 2027
        [--tierB-runs …] [--tierB-manifest …] [--tierC-runs …] [--tierC-manifest …] [--fhir-dir …]
        [--tierA-runs <enclave>/runs --tierA-manifest <enclave>/tierA --tierA-out <enclave>/human/adjudication]
        [--per-stratum 25] [--direct-set 300] [--raters 3] [--overlap 150] [--direct-overlap 30] [--checkpoints …]
Nothing is executed with humans here; the packets wait for the raters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench.predicate import legal  # noqa: E402
from scopebench.schema import EvidenceObject, Task, read_jsonl  # noqa: E402

CORE = ["prompt_only", "late_filter", "wrong_basis", "scope_aware", "audit", "full", "token_matched", "high_compute"]
GROUP = {"lab_trend_k3": "structured", "outcome_lookup_v1": "structured", "outcome_count_v1": "structured", "cohort_v1": "cohort",
         "claim_v1": "free_text", "reconstruct_v1": "free_text", "summarize_exams_v1": "free_text", "fhir_agentbench_v1": "structured",
         "medagentbench_v1": "structured"}
INSTRUCTIONS_VERSION = "2026-10-07"
RATER_NAMES = ["R1", "R2", "R3"]


# ------------------------------------------------------------------------------------------ plain-language rule
def _hm(ts: str | None) -> str:
    return (ts or "?").replace("T", " ").replace("Z", "")


def contract_statement(t: Task) -> str:
    c = t.contract
    basis = {"occurred": "the time the event occurred", "available": "the time the record became available", "documented": "the time it was documented"}.get(c.basis, c.basis)
    if c.scope == "pre":
        when = f"only records dated strictly BEFORE the index time {_hm(c.index_time)}"
        if c.window_hours:
            when += f" and within the {c.window_hours:g} hours before it"
    elif c.scope == "post":
        when = f"only records dated strictly AFTER the index time {_hm(c.index_time)}"
        if c.window_hours:
            when += f" and within the {c.window_hours:g} hours after it"
    elif c.scope == "at":
        tol = f" (within ±{c.at_tolerance_hours:g} hours)" if c.at_tolerance_hours else ""
        when = f"only records dated AT the index time {_hm(c.index_time)}{tol}"
    else:
        when = f"the whole chart, no time restriction (index time {_hm(c.index_time)})"
    src = ", ".join(s.replace("_", " ") for s in c.permitted_sources) if c.permitted_sources else "any kind of record"
    return f"Allowed: {when}; dates are judged by {basis}. Allowed kinds of record: {src}."


# ------------------------------------------------------------------------------------------ evidence rendering
TECHNICAL_FIELDS = {"hadm", "stay", "itemid", "note", "line", "kind", "icd_version", "seq_num", "code_raw", "exam_code", "resourceType"}
FIELD_ORDER = ("exam_name", "section", "is_addendum", "text", "concept", "lab_name", "value", "value_text", "unit", "flag", "ref_low", "ref_high",
               "medication", "dose", "route", "event_txt", "code", "title", "status", "code_text", "careunit", "location", "icu", "eventtype",
               "admission_type", "admission_location", "discharge_location", "dischtime", "hospital_expire_flag", "los_days",
               "arrival_transport", "disposition", "acuity", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain")


def _display(fields: dict) -> str:
    """Every informative field of a record, in a fixed order, as `name: value`; report lines read `EXAM — SECTION: text`."""
    f = {k: v for k, v in fields.items() if k not in TECHNICAL_FIELDS and v not in (None, "", [], {})}
    parts = []
    if "text" in f:   # a report / note line: exam and section first, then the line itself
        head = " — ".join(str(f.pop(k)) for k in ("exam_name", "section") if k in f)
        if f.pop("is_addendum", False):
            head = (head + " (addendum)").strip()
        parts.append((head + ": " if head else "") + str(f.pop("text")))
    if "resourceType" in fields:
        parts.insert(0, str(fields["resourceType"]))
    ordered = [k for k in FIELD_ORDER if k in f] + sorted(k for k in f if k not in FIELD_ORDER)
    for k in ordered:
        v = f[k]
        label = {"code_text": "name", "value_text": "value (text)", "ref_low": "ref low", "ref_high": "ref high", "event_txt": "event", "los_days": "length of stay (days)",
                 "hospital_expire_flag": "died in hospital", "dischtime": "discharge time", "o2sat": "O2 sat", "sbp": "SBP", "dbp": "DBP", "resprate": "resp rate",
                 "heartrate": "heart rate", "admission_type": "admission type", "admission_location": "admitted from", "discharge_location": "discharged to",
                 "arrival_transport": "arrival", "exam_name": "exam", "lab_name": "lab", "careunit": "care unit"}.get(k, k.replace("_", " "))
        parts.append(f"{label}: {v}")
    return "; ".join(parts)


CITATION_KEYS = ("citation", "citations")
ANSWER_GUIDE = {
    "cohort_v1": "The answer is a decision — include, exclude, or uncertain — about whether the patient meets the stated criteria, with the records that establish each satisfied criterion (or the decisive failed one) cited.",
    "claim_v1": "The answer is a label for the claim — supported, contradicted, or insufficient — citing the report line(s) that decide it.",
    "reconstruct_v1": "The answer lists the report lines (copied verbatim, each with its id) that were known at the index time, or says the evidence is insufficient.",
    "summarize_exams_v1": "The answer lists the exams in scope, each with its name, time and first impression line (copied verbatim, with the line's id), or says the evidence is insufficient.",
    "lab_trend_k3": "The answer gives the most recent values of the named lab in scope (each with its time and record id), the direction of the trend and the change (delta), or says the evidence is insufficient.",
    "outcome_lookup_v1": "The answer gives the value of the asked outcome and its time, citing the record it comes from, or says the evidence is insufficient.",
    "outcome_count_v1": "The answer is a count (an integer) with every counted record cited, or says the evidence is insufficient.",
    "fhir_agentbench_v1": "The answer is the result rows of the asked query as a list of lists (for example [[2]] is one row holding the value 2, [] is no row), citing the FHIR resources it rests on.",
    "medagentbench_v1": "The answer is the value or list the question asks for, citing the FHIR resources it rests on.",
}


def answer_fields(parsed) -> list[list[str]]:
    """The submitted answer as readable `name, value` rows, without its citation keys (shown separately)."""
    if not isinstance(parsed, dict):
        return []
    out = []
    for k, v in parsed.items():
        if k in CITATION_KEYS or k in ("answer_text", "citations_line_present", "parseable"):
            continue
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            for i, x in enumerate(v, 1):
                out.append([f"{k} {i}", "; ".join(f"{a}: {b}" for a, b in x.items() if b not in (None, ""))])
        elif isinstance(v, (list, dict)):
            out.append([k, json.dumps(v, ensure_ascii=False)])
        else:
            out.append([k, "" if v is None else str(v)])
    return out


def ev_row(e: EvidenceObject) -> dict:
    return {"id": e.id, "source": e.source.replace("_", " "), "occurred": e.occurred, "available": e.available, "display": _display(e.fields)}


def to_evidence(r: dict) -> EvidenceObject:
    known = {"id", "source", "patient", "occurred", "available", "documented", "occurred_end", "row_ref", "fields"}
    return EvidenceObject(**{k: v for k, v in r.items() if k in known})


class Lookup:
    """Evidence by id for one tier: manifest evidence files (Tier-A / Tier-B) or the local FHIR store (Tier-C)."""

    def __init__(self, manifest: Path, fhir_dir: Path | None):
        self.manifest = manifest
        self.store = None
        self.patient_map: dict[str, str] = {}
        if fhir_dir is not None:
            from scopebench.tierc.store import load_store
            from scripts.scopebench.run_tierc import PATIENT_MAP
            self.store = load_store(fhir_dir)
            self.patient_map = json.load(open(PATIENT_MAP))

    def candidates(self, task: Task, must_include: list[str], inside_cap: int = 60, outside_cap: int = 20) -> tuple[list[EvidenceObject], dict]:
        """Tier-C: the patient's resources of the allowed kinds, nearest to the index time first — everything inside the
        rule up to `inside_cap`, then up to `outside_cap` just outside it (boundary context), plus the ids that must appear
        (cited, required). A FHIR agent's GET responses can run to thousands of resources, which no rater can read."""
        from scopebench.tierc.contracts import resource_to_evidence
        from scopebench.schema import parse_time
        pid = self.patient_map.get(task.patient)
        allowed = set(task.contract.permitted_sources or [])
        evs = [resource_to_evidence(r, task.patient) for (pp, rt), lst in self.store.by_patient_type.items() if pp == pid and (not allowed or rt in allowed) for r in lst]
        index = parse_time(task.contract.index_time)
        def dist(e):
            t = e.time(task.contract.basis)
            return abs((t - index).total_seconds()) if (t is not None and index is not None) else float("inf")
        inside = sorted((e for e in evs if legal(e, task.contract)), key=dist)
        outside = sorted((e for e in evs if not legal(e, task.contract)), key=dist)
        shown = {e.id: e for e in inside[:inside_cap]}
        shown.update({e.id: e for e in outside[:outside_cap]})
        for e in self.by_ids(task, [i for i in must_include if i not in shown]):
            shown[e.id] = e
        info = {"candidates_total": len(evs), "inside_rule_total": len(inside), "shown": len(shown), "subset": len(shown) < len(evs)}
        return list(shown.values()), info

    def case(self, task: Task) -> dict[str, EvidenceObject]:
        p = self.manifest / "evidence" / f"{task.case_id}.jsonl"
        return {r["id"]: to_evidence(r) for r in read_jsonl(p)} if p.exists() else {}

    def by_ids(self, task: Task, ids: list[str]) -> list[EvidenceObject]:
        if self.store is None:
            ev = self.case(task)
            return [ev[i] for i in ids if i in ev]
        from scopebench.tierc.contracts import resource_to_evidence
        out = []
        for rid in ids:
            for table in self.store.by_type_id.values():
                if rid in table:
                    out.append(resource_to_evidence(table[rid], task.patient))
                    break
        return out


# ------------------------------------------------------------------------------------------ sampling
def load_runs(runs_root: Path, checkpoints: list[str] | None) -> dict[tuple[str, str], dict[str, dict]]:
    """(checkpoint, configuration) -> task_id -> prediction row, over the core configurations only."""
    out: dict[tuple[str, str], dict[str, dict]] = {}
    for ck in sorted(p for p in runs_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if checkpoints and ck.name not in checkpoints:
            continue
        for cfg in CORE:
            pred = ck / cfg / "predictions.jsonl"
            if pred.exists():
                out[(ck.name, cfg)] = {r["task_id"]: r for r in read_jsonl(pred)}
    return out


def sample_tier(tier: str, tasks: dict[str, Task], runs: dict[tuple[str, str], dict[str, dict]], per_stratum: int, rng: random.Random):
    by_stratum: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for tid, t in tasks.items():
        g = GROUP.get(t.task_type)
        if g is None:
            continue
        by_stratum[(tier, t.contract.scope, g)].append(tid)
    keys = sorted(runs)
    pairs = []
    for (tier_, scope, group), tids in sorted(by_stratum.items()):
        tids = sorted(tids)
        rng.shuffle(tids)
        picked = 0
        for tid in tids:
            avail = [k for k in keys if tid in runs[k]]
            if not avail:
                continue
            k = rng.choice(avail)
            pairs.append({"tier": tier, "scope": scope, "group": group, "task_id": tid, "checkpoint": k[0], "configuration": k[1]})
            picked += 1
            if picked >= per_stratum:
                break
    return pairs, {f"{k[0]}|{k[1]}|{k[2]}": len(v) for k, v in sorted(by_stratum.items())}


def sample_direct(tier: str, tasks: dict[str, Task], exclude: set[str], n: int, rng: random.Random) -> list[Task]:
    """n leakage-flagged tasks not in `exclude`, stratified by scope in proportion to the leakage tasks of each scope."""
    pool: dict[str, list[str]] = defaultdict(list)
    for tid, t in tasks.items():
        if t.leakage and tid not in exclude and GROUP.get(t.task_type):
            pool[t.contract.scope].append(tid)
    total = sum(len(v) for v in pool.values())
    if total == 0 or n <= 0:
        return []
    quota = {s: int(n * len(v) / total) for s, v in pool.items()}
    for s in sorted(pool, key=lambda s: -(n * len(pool[s]) / total - quota[s]))[: n - sum(quota.values())]:
        quota[s] += 1
    picked = []
    for s, tids in sorted(pool.items()):
        tids = sorted(tids)
        rng.shuffle(tids)
        picked += [tasks[t] for t in tids[: quota[s]]]
    return picked


# ------------------------------------------------------------------------------------------ packets
def packet_for(pair: dict, task: Task, row: dict, lookup: Lookup) -> tuple[dict, dict]:
    parsed = row.get("parsed_answer") or {}
    citations = list(parsed.get("citations") or []) if parsed else []
    truncated = False
    records_label, records_note, info = "Records the system was given", None, {}
    if lookup.store is not None:
        # Tier-C: the agent queried a FHIR server and received whole resource lists (thousands of resources); the card shows
        # the patient's resources of the allowed kinds nearest to the index time, inside the rule first, with boundary context
        evs, info = lookup.candidates(task, list(dict.fromkeys(citations + list(task.required_ids))))
        records_label = "Records available to the system (allowed kinds, nearest to the index time)"
        if info["subset"]:
            records_note = (f"The patient has {info['candidates_total']} records of the allowed kinds; {info['inside_rule_total']} fall inside the rule. "
                            f"Shown: the {info['shown']} nearest to the index time (every cited and every required record is included).")
        truncated = info["subset"]
    else:
        supplied_ids = list(row.get("selected_evidence_ids") or row.get("access_ids") or [])
        if not supplied_ids and (row.get("n_access_ids") or 0) > 0:
            supplied_ids, truncated = list(dict.fromkeys(citations + list(task.required_ids))), True   # the run kept no id list (capped)
            records_note = "The system looked at more records than are listed; only the ones it cited and the ones required are shown."
        evs = lookup.by_ids(task, supplied_ids)
    supplied = sorted((ev_row(e) for e in evs), key=lambda r: (r["occurred"] or "", r["id"]))
    shown_ids = {r["id"] for r in supplied}
    index_ids = [i for i in (task.contract.index_evidence_ids or ()) if i in shown_ids]
    pid = hashlib.sha256(f"{pair['task_id']}|{pair['checkpoint']}|{pair['configuration']}".encode()).hexdigest()[:12]
    has_answer = bool(parsed) and (parsed.get("answer") is not None or any(k not in CITATION_KEYS for k in parsed))
    packet = {"packet_type": "pair", "pair_id": pid, "tier": pair["tier"], "scope": pair["scope"], "task_group": pair["group"], "task_type": task.task_type,
              "question": task.question, "contract": contract_statement(task), "index_time": task.contract.index_time, "index_ids": index_ids,
              "supplied_evidence": supplied, "supplied_truncated": truncated, "records_label": records_label, "records_note": records_note, "records_info": info,
              "answer_guide": ANSWER_GUIDE.get(task.task_type, ""),
              "submission": {"has_answer": has_answer, "answer": parsed.get("answer") if parsed else None,
                             "answer_text": parsed.get("answer_text") if parsed else None, "answer_fields": answer_fields(parsed),
                             "raw_json": json.dumps({k: v for k, v in parsed.items() if k not in ("citations_line_present", "parseable")}, ensure_ascii=False) if parsed else None,
                             "citations": citations, "citations_not_shown": [c for c in citations if c not in shown_ids]},
              "rating_fields": {"admissible_evidence_used": None, "required_evidence_cited": None, "answer_clinically_correct": None, "comment": ""}}
    key = {"pair_id": pid, "packet_type": "pair", "tier": pair["tier"], "task_id": pair["task_id"], "checkpoint": pair["checkpoint"], "configuration": pair["configuration"],
           "gold": task.gold, "required_ids": list(task.required_ids), "metrics": row.get("metrics"), "final_guard_errors": row.get("final_guard_errors"),
           "guard_repaired": row.get("guard_repaired"), "leakage": task.leakage}
    return packet, key


def direct_packet_for(task: Task, tier: str, lookup: Lookup) -> tuple[dict, dict]:
    ev = lookup.case(task)
    record = sorted((ev_row(e) for e in ev.values()), key=lambda r: (r["occurred"] or "", r["id"]))
    pid = hashlib.sha256(f"direct|{task.task_id}".encode()).hexdigest()[:12]
    packet = {"packet_type": "direct_set", "pair_id": pid, "tier": tier, "scope": task.contract.scope, "task_group": GROUP[task.task_type], "task_type": task.task_type,
              "question": task.question, "contract": contract_statement(task), "index_time": task.contract.index_time,
              "index_ids": [i for i in (task.contract.index_evidence_ids or ()) if i in ev], "answer_guide": ANSWER_GUIDE.get(task.task_type, ""), "record": record,
              "rating_fields": {"allowed_ids": [], "needed_ids": [], "comment": ""}}
    key = {"pair_id": pid, "packet_type": "direct_set", "tier": tier, "task_id": task.task_id, "gold": task.gold, "required_ids": list(task.required_ids),
           "programmatic_allowed_ids": sorted(i for i, e in ev.items() if legal(e, task.contract)), "n_record": len(record), "leakage": task.leakage}
    return packet, key


# ------------------------------------------------------------------------------------------ assignment and bundles
def assign(packets: list[dict], raters: list[str], overlap: int, rng: random.Random, strat_key) -> dict[str, list[str]]:
    """Stratified overlap rated by everyone; the rest split evenly within each stratum (balanced strata per rater)."""
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for p in packets:
        strata[strat_key(p)].append(p)
    n = len(packets)
    quota = {s: int(overlap * len(v) / n) for s, v in strata.items()} if n else {}
    for s in sorted(strata, key=lambda s: -(overlap * len(strata[s]) / n - quota[s]))[: max(0, overlap - sum(quota.values()))]:
        quota[s] += 1
    out: dict[str, list[str]] = {}
    offset = 0
    for s, ps in sorted(strata.items(), key=lambda kv: str(kv[0])):
        ps = sorted(ps, key=lambda p: p["pair_id"])
        rng.shuffle(ps)
        for p in ps[: quota[s]]:
            out[p["pair_id"]] = list(raters)
        for i, p in enumerate(ps[quota[s]:]):
            out[p["pair_id"]] = [raters[(i + offset) % len(raters)]]
        offset = (offset + len(ps) - quota[s]) % len(raters)
    return out


def write_bundles(packets: list[dict], assignment: dict[str, list[str]], raters: list[str], out: Path, location: str, seed: int) -> dict[str, int]:
    (out / "bundles").mkdir(parents=True, exist_ok=True)
    counts = {}
    for r in raters:
        mine = [p for p in packets if r in assignment.get(p["pair_id"], [])]
        mine.sort(key=lambda p: (p["packet_type"] != "pair", p["tier"], p["pair_id"]))   # the main cards first, then the direct-set cards
        bundle = {"bundle_id": f"{location}-{r}-{seed}", "rater": r, "location": location, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "instructions_version": INSTRUCTIONS_VERSION, "n_cards": len(mine), "packets": mine}
        (out / "bundles" / f"rater_{r}_{location}.json").write_text(json.dumps(bundle, ensure_ascii=False))
        counts[r] = len(mine)
    return counts


# ------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/scopebench_human/adjudication")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--per-stratum", type=int, default=25)
    ap.add_argument("--direct-set", type=int, default=300)
    ap.add_argument("--raters", type=int, default=3)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--direct-overlap", type=int, default=30)
    ap.add_argument("--checkpoints", nargs="*", default=None)
    ap.add_argument("--tierB-runs", type=Path, default=ROOT / "experiments/scopebench_tierB_runs")
    ap.add_argument("--tierB-manifest", type=Path, default=ROOT / "experiments/scopebench_tierB")
    ap.add_argument("--tierC-runs", type=Path, default=ROOT / "experiments/scopebench_tierC_runs")
    ap.add_argument("--tierC-manifest", type=Path, default=ROOT / "experiments/scopebench_tierC/fhir_agentbench")
    ap.add_argument("--fhir-dir", type=Path, default=None, help="local FHIR demo directory (default: scopebench.config.PUBLIC_DATA/…/fhir)")
    ap.add_argument("--tierA-runs", type=Path, default=None)
    ap.add_argument("--tierA-manifest", type=Path, default=None)
    ap.add_argument("--tierA-out", type=Path, default=None, help="Tier-A packets, keys and bundles are written here (inside the enclave), never under --out")
    a = ap.parse_args()
    if a.fhir_dir is None:
        from scopebench.config import PUBLIC_DATA
        a.fhir_dir = PUBLIC_DATA / "mimic-iv-clinical-database-demo-on-fhir-2.1.0" / "fhir"
    raters = RATER_NAMES[: a.raters]
    rng = random.Random(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    summary = {"seed": a.seed, "per_stratum": a.per_stratum, "direct_set": a.direct_set, "raters": raters, "overlap": a.overlap, "direct_overlap": a.direct_overlap,
               "instructions_version": INSTRUCTIONS_VERSION, "tiers": {}}
    main_packets: list[dict] = []
    direct_packets: list[dict] = []
    direct_pool: list[tuple[str, dict[str, Task], Lookup, Path]] = []
    for tier, runs_root, manifest, out_dir in (("B", a.tierB_runs, a.tierB_manifest, a.out), ("C", a.tierC_runs, a.tierC_manifest, a.out),
                                               ("A", a.tierA_runs, a.tierA_manifest, a.tierA_out)):
        if runs_root is None or manifest is None or not runs_root.exists() or not (manifest / "tasks.jsonl").exists():
            summary["tiers"][tier] = "not available"
            continue
        tasks = {t.task_id: t for t in (Task.from_dict(r) for r in read_jsonl(manifest / "tasks.jsonl"))}
        runs = load_runs(runs_root, a.checkpoints)
        if not runs:
            summary["tiers"][tier] = "no core-configuration runs found"
            continue
        lookup = Lookup(manifest, a.fhir_dir if tier == "C" else None)
        pairs, strata = sample_tier(tier, tasks, runs, a.per_stratum, rng)
        pdir = (out_dir or a.out) / f"tier{tier}"
        pdir.mkdir(parents=True, exist_ok=True)
        keys = []
        for pair in pairs:
            row = runs[(pair["checkpoint"], pair["configuration"])][pair["task_id"]]
            packet, key = packet_for(pair, tasks[pair["task_id"]], row, lookup)
            (pdir / f"{packet['pair_id']}.json").write_text(json.dumps(packet, indent=1, ensure_ascii=False))
            keys.append(key)
            main_packets.append(packet)
        (pdir / "ANSWER_KEY.jsonl").write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in keys))
        summary["tiers"][tier] = {"n_pairs": len(pairs), "strata_available": strata, "packets_dir": str(pdir),
                                  "n_strata_sampled": len({(p["scope"], p["group"]) for p in pairs}),
                                  "configurations": {c: sum(1 for p in pairs if p["configuration"] == c) for c in CORE},
                                  "checkpoints": {ck: sum(1 for p in pairs if p["checkpoint"] == ck) for ck in sorted({p["checkpoint"] for p in pairs})},
                                  "n_truncated_supplied": sum(1 for p in main_packets if p["tier"] == tier and p["supplied_truncated"])}
        if tier in ("A", "B"):
            direct_pool.append((tier, tasks, lookup, pdir))
    # direct-set sample: Tier-A / Tier-B leakage tasks not in the main sample, allocated in proportion to each tier's leakage tasks
    main_task_ids = {k["task_id"] for d in (a.out, a.tierA_out) if d for f in d.glob("tier*/ANSWER_KEY.jsonl") for k in read_jsonl(f)}
    avail = {tier: sum(1 for t in tasks.values() if t.leakage and t.task_id not in main_task_ids and GROUP.get(t.task_type)) for tier, tasks, _, _ in direct_pool}
    tot = sum(avail.values())
    for tier, tasks, lookup, pdir in direct_pool:
        n = round(a.direct_set * avail[tier] / tot) if tot else 0
        picked = sample_direct(tier, tasks, main_task_ids, n, rng)
        keys = []
        for t in picked:
            packet, key = direct_packet_for(t, tier, lookup)
            (pdir / f"direct_{packet['pair_id']}.json").write_text(json.dumps(packet, indent=1, ensure_ascii=False))
            keys.append(key)
            direct_packets.append(packet)
        (pdir / "ANSWER_KEY_direct.jsonl").write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in keys))
        summary["tiers"][tier]["direct_set"] = {"n_cases": len(picked), "by_scope": {s: sum(1 for t in picked if t.contract.scope == s) for s in ("pre", "at", "post", "full")},
                                                "record_rows_median": sorted(len(p["record"]) for p in direct_packets if p["tier"] == tier)[len(picked) // 2] if picked else None}
    # assignment: stratified overlap rated by all raters, the rest split evenly (strata balanced per rater)
    assignment = assign(main_packets, raters, a.overlap, rng, lambda p: (p["tier"], p["scope"], p["task_group"]))
    assignment.update(assign(direct_packets, raters, a.direct_overlap, rng, lambda p: (p["tier"], p["scope"])))
    (a.out / "assignments.json").write_text(json.dumps({"raters": raters, "overlap": a.overlap, "direct_overlap": a.direct_overlap,
                                                        "cards": {p["pair_id"]: {"tier": p["tier"], "packet_type": p["packet_type"], "raters": assignment[p["pair_id"]]}
                                                                  for p in main_packets + direct_packets}}, indent=1))
    local = [p for p in main_packets + direct_packets if p["tier"] in ("B", "C")]
    enclave = [p for p in main_packets + direct_packets if p["tier"] == "A"]
    summary["bundles"] = {"local": write_bundles(local, assignment, raters, a.out, "local", a.seed)}
    if enclave and a.tierA_out:
        summary["bundles"]["enclave"] = write_bundles(enclave, assignment, raters, a.tierA_out, "enclave", a.seed)
    summary["totals"] = {"main_cards": len(main_packets), "direct_cards": len(direct_packets),
                         "per_rater_cards": {r: sum(1 for p in main_packets + direct_packets if r in assignment[p["pair_id"]]) for r in raters},
                         "rated_by_all": sum(1 for p in main_packets + direct_packets if len(assignment[p["pair_id"]]) == len(raters))}
    (a.out / "sampling_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"tiers": {t: (v if isinstance(v, str) else {"main": v["n_pairs"], "strata": v["n_strata_sampled"], "direct": (v.get("direct_set") or {}).get("n_cases")})
                                for t, v in summary["tiers"].items()}, "totals": summary["totals"], "bundles": summary["bundles"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
