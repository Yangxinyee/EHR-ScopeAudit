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
import re
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
INSTRUCTIONS_VERSION = "2026-10-12"
DIRECT_MAX_ROWS = 200   # feasibility cap for a direct-set card (rows of the whole case record)
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
        # the predicate: t <= index (+ tolerance), plus the index event's own records; a window bounds it from below
        when = f"only records dated at or BEFORE the index time {_hm(c.index_time)}"
        if c.at_tolerance_hours:
            when += f" (allowing up to {c.at_tolerance_hours:g} hours after it)"
        if c.window_hours:
            when += f" and within the {c.window_hours:g} hours before it"
        if c.index_evidence_ids:
            when += (", PLUS the index event's own records, tagged 'index event' in the list: those are allowed by identity even though they "
                     "are dated after the index time")
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
    if "value_text" in f and "value" in f and str(f["value_text"]).strip() == str(f["value"]).strip():
        f.pop("value_text")
    parts = []
    if "text" in f:   # a report / note line: exam, section and line number first, then the line itself
        head = " — ".join(str(f.pop(k)) for k in ("exam_name", "section") if k in f)
        if f.pop("is_addendum", False):
            head = (head + " (addendum)").strip()
        if fields.get("line") is not None:
            head = (head + f" [line {fields['line']}]").strip()
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
INTERNAL_KEYS = {"citation", "citations", "answer_text", "citations_line_present", "parseable", "replaced"}   # bookkeeping, never shown to a rater
ANSWER_GUIDE = {
    "cohort_v1": "The answer is a decision — include, exclude, or uncertain — about whether the patient meets the stated criteria, with the records that establish each satisfied criterion (or the decisive failed one) cited.",
    "claim_v1": "The answer is a label for the claim — supported, contradicted, or insufficient — citing the report line(s) that decide it.",
    "reconstruct_v1": "The answer lists the report lines (copied verbatim, each with its id) that were known at the index time, or says the evidence is insufficient.",
    "summarize_exams_v1": "The answer lists the exams in scope, each with its name, time and first impression line (copied verbatim, with the line's id), or says the evidence is insufficient.",
    "lab_trend_k3": "The answer gives the most recent values of the named lab in scope (each with its time and record id), the direction of the trend and the change (delta), or says the evidence is insufficient.",
    "outcome_lookup_v1": "The answer gives the value of the asked outcome and its time, citing the record it comes from, or says the evidence is insufficient.",
    "outcome_count_v1": "The answer is a count (an integer) with every counted record cited, or says the evidence is insufficient.",
    "fhir_agentbench_v1": ("The answer is the result rows of the asked query as a list of lists (for example [[2]] is one row holding the value 2). "
                           "If no record matches, the correct answer is the empty list []; it may cite the records it examined, or none — that is a result, not a refusal. "
                           "The benchmark's prescription time is the start of the validity period ('prescribed' on the card), not the order-entry time. "
                           "Some medications in this dataset carry only an NDC or formulary code and no name; the system saw the same records. "
                           "Cite the FHIR resources the rows rest on."),
    "medagentbench_v1": "The answer is the value or list the question asks for, citing the FHIR resources it rests on.",
}


def skill_ids(summary) -> list[str]:
    """Record ids a deterministic skill payload handed to the model (its citations / provenance), if the row kept one."""
    out: list[str] = []
    def walk(x, key=None):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
        elif isinstance(x, str) and key in ("citation", "citations", "provenance", "evidence_ids", "record_id", "record_ids") and x not in out:
            out.append(x)
    walk(summary)
    return out


def answer_fields(parsed) -> list[list[str]]:
    """The submitted answer as readable `name, value` rows, without its citation keys (shown separately)."""
    if not isinstance(parsed, dict):
        return []
    out = []
    for k, v in parsed.items():
        if k in INTERNAL_KEYS or k.startswith("_"):
            continue
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            for i, x in enumerate(v, 1):
                out.append([f"{k} {i}", "; ".join(f"{a}: {b}" for a, b in x.items() if b not in (None, ""))])
        elif isinstance(v, (list, dict)):
            out.append([k, json.dumps(v, ensure_ascii=False)])
        else:
            out.append([k, "" if v is None else str(v)])
    return out


def ev_row(e: EvidenceObject, display: str | None = None, group: str | None = None) -> dict:
    """One record as the rater sees it: both clocks, the report it belongs to and its line number (so a report's lines can
    be shown together, in their original order) and the rendered content."""
    line = e.fields.get("line")
    text = display if display is not None else _display(e.fields)
    if not text.strip():
        text = f"{e.source.replace('_', ' ')} record, no values recorded"
    return {"id": e.id, "source": e.source.replace("_", " "), "occurred": e.occurred, "available": e.available,
            "report": e.fields.get("note"), "line": line if isinstance(line, int) else None, "group": group, "display": text}


def row_key(r: dict, basis: str):
    """Sort order of the record table: the time the rule uses, then report id and line number, then id."""
    t = r.get("available" if basis == "available" else "occurred") or r.get("occurred") or ""
    return (t, r.get("report") or "", r["line"] if r.get("line") is not None else 10**9, r["id"])


CITE_KEYS = {"citation", "citations", "id", "ids", "record_id", "record_ids", "line_id", "line_ids", "evidence_id", "evidence_ids", "provenance",
             "source_id", "source_ids", "cited", "cites"}


def collect_citations(parsed, known: set[str]) -> list[str]:
    """Every record id the submission points at, in order of appearance: the top-level citation list, the singular
    `citation` key, citation keys nested inside answer rows, and known ids quoted inside free text. The rater is told
    exactly what the submission cited, wherever it put the id."""
    out: list[str] = []
    def add(v):
        if isinstance(v, str) and v and v not in out:
            out.append(v)
    if isinstance(parsed, dict):
        for v in parsed.get("citations") or []:
            add(v)
        add(parsed.get("citation"))
    def walk(x, key=None):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
        elif isinstance(x, str):
            if key in CITE_KEYS:
                add(x)
            elif known:
                for kid in known:
                    if kid in x:
                        add(kid)
    walk(parsed)
    return out


def answer_guide(task: Task) -> str:
    g = ANSWER_GUIDE.get(task.task_type, "")
    target = ((task.metadata or {}).get("skill_params") or {}).get("target")
    if task.task_type == "outcome_lookup_v1" and (target == "first_careunit" or "transferred to" in task.question):
        g += (" Here 'the first care unit the patient was transferred to' means the care unit on the earliest transfer-table record dated "
              "after the index time, whatever its event type (admit, transfer or ED); that record is the one to cite.")
    return g


def to_evidence(r: dict) -> EvidenceObject:
    known = {"id", "source", "patient", "occurred", "available", "documented", "occurred_end", "row_ref", "fields"}
    return EvidenceObject(**{k: v for k, v in r.items() if k in known})


def _cc(x) -> str | None:
    """CodeableConcept -> readable text."""
    if not isinstance(x, dict):
        return None
    if x.get("text"):
        return str(x["text"])
    for c in x.get("coding") or []:
        if c.get("display"):
            return str(c["display"])
    for c in x.get("coding") or []:
        if c.get("code"):
            return str(c["code"])
    return None


def _qty(x) -> str | None:
    if not isinstance(x, dict) or x.get("value") is None:
        return None
    return f"{x['value']} {x.get('unit') or x.get('code') or ''}".strip()


_CODE_SYSTEM = {"mimic-medication-ndc": "NDC", "mimic-medication-formulary-drug-cd": "formulary", "mimic-medication-name": "name",
                "mimic-medication-icu": "ICU item", "mimic-medication-poe-iv": "POE IV"}


def _med_name(cc) -> str | None:
    """A medication CodeableConcept as a rater should read it: its text or a coding's display when the data has a name;
    a coding under the demo's 'medication-name' system (whose code is the name); otherwise the codes with their system
    and a note that the dataset carries no name."""
    if not isinstance(cc, dict):
        return None
    if cc.get("text"):
        return str(cc["text"])
    codings = [c for c in (cc.get("coding") or []) if isinstance(c, dict)]
    for c in codings:
        if c.get("display"):
            return str(c["display"])
    for c in codings:
        if (c.get("system") or "").endswith("mimic-medication-name") and c.get("code"):
            return str(c["code"]).replace("--", " / ")
    for c in codings:
        sysname = (c.get("system") or "").rsplit("/", 1)[-1]; code = str(c.get("code") or "")
        if not code:
            continue
        if sysname == "mimic-medication-formulary-drug-cd":
            return f"{code} (formulary code)"
        if sysname == "mimic-medication-poe-iv":
            return f"{code} (order type)"
    codes = [f"{_CODE_SYSTEM.get((c.get('system') or '').rsplit('/', 1)[-1], (c.get('system') or '').rsplit('/', 1)[-1])} {c['code']}" for c in codings if c.get("code")]
    return (", ".join(codes) + " (no drug name in this dataset)") if codes else None


def _mix_names(v: str) -> str:
    """A medication-mix identifier `name--CODE--NDC_name--CODE--NDC…` -> `name + name`: components are name, code,
    NDC triples, the next component's name glued to the NDC by `_`."""
    toks = [t.strip() for t in re.split(r"--|(?<=\d{4})_", v)]
    return " + ".join(t for t in toks[0::3] if t)


def _med_from_resource(m: dict) -> str | None:
    """A Medication resource as a rater should read it: its coded name, else the name identifier the demo attaches,
    else a mixture's component names, else a formulary code, else the NDC with a note that the data has no name."""
    name = _med_name(m.get("code"))
    if name and "(no drug name" not in name:
        return name
    ids: dict[str, list[str]] = {}
    for i in m.get("identifier") or []:
        if i.get("value"):
            ids.setdefault((i.get("system") or "").rsplit("/", 1)[-1], []).append(str(i["value"]))
    if ids.get("mimic-medication-name"):
        return ids["mimic-medication-name"][0].replace("--", " / ")
    if ids.get("medication-mix"):
        return _mix_names(ids["medication-mix"][0]) + " (mixture)"
    if ids.get("mimic-medication-formulary-drug-cd"):
        return f"{ids['mimic-medication-formulary-drug-cd'][0]} (formulary code)"
    if ids.get("mimic-medication-ndc"):
        return f"NDC {ids['mimic-medication-ndc'][0]} (no drug name in this dataset)"
    return name


def _unless_repeated(text: str | None, *others) -> str | None:
    """A free-text element that only repeats another shown element is dropped (the eGFR comment twice, …)."""
    if not text:
        return None
    t = " ".join(text.split()).lower()
    for o in others:
        if o and (t == " ".join(str(o).split()).lower() or t in " ".join(str(o).split()).lower() or " ".join(str(o).split()).lower() in t):
            return None
    return text


def _ref(x) -> str | None:
    """Reference -> 'Type id-prefix' so a rater can match it against other rows."""
    r = x.get("reference") if isinstance(x, dict) else x
    if not isinstance(r, str) or not r:
        return None
    t, _, i = r.rpartition("/")
    return f"{t} {i[:8]}" if t else i[:8]


def fhir_display(res: dict, store) -> str:
    """Every informative element of a FHIR resource as `name: value`, in a fixed order per resource type. Medication and
    Location references are resolved to their names through the store, so a rater sees the drug and the unit, not a
    pointer."""
    from scopebench.tierc.contracts import wallclock
    rt = str(res.get("resourceType"))
    hm = lambda t: _hm(wallclock(t)) if isinstance(t, str) else None
    per = lambda x: (f"{hm(x.get('start')) or '?'} to {hm(x.get('end')) or '?'}" if isinstance(x, dict) and (x.get("start") or x.get("end")) else None)
    def med(r):
        if isinstance(r.get("medicationCodeableConcept"), dict):
            return _med_name(r["medicationCodeableConcept"])
        ref = (r.get("medicationReference") or {}).get("reference") if isinstance(r.get("medicationReference"), dict) else None
        if ref:
            m = store.by_type_id.get("Medication", {}).get(ref.rpartition("/")[2]) if store is not None else None
            if m:
                name = _med_from_resource(m)
                ings = [_cc(i.get("itemCodeableConcept")) for i in (m.get("ingredient") or []) if isinstance(i, dict)]
                if name and "(no drug name" not in name and any(ings):
                    name += f" (ingredients: {', '.join(x for x in ings if x)})"
                return name or f"Medication {ref.rpartition('/')[2][:8]} (no drug name in this dataset)"
            return f"Medication {ref.rpartition('/')[2][:8]} (not in the store)"
        return None
    def loc(r):
        ref = (r.get("location") or {}).get("reference") if isinstance(r.get("location"), dict) else None
        if ref and store is not None:
            l = store.by_type_id.get("Location", {}).get(ref.rpartition("/")[2])
            if l:
                return str(l.get("name") or _cc(l.get("physicalType")) or ref.rpartition("/")[2][:8])
        return _ref(r.get("location"))
    P: list[tuple[str, object]] = [("", rt)]
    if rt == "Patient":
        P += [("gender", res.get("gender")), ("birth date", res.get("birthDate")), ("deceased", hm(res.get("deceasedDateTime"))), ("marital status", _cc(res.get("maritalStatus")))]
    elif rt == "Encounter":
        cls = res.get("class") or {}
        P += [("class", cls.get("display") or cls.get("code")), ("status", res.get("status")), ("period", per(res.get("period"))),
              ("service type", _cc(res.get("serviceType"))), ("priority", _cc(res.get("priority"))), ("part of", _ref(res.get("partOf"))),
              ("admit source", _cc((res.get("hospitalization") or {}).get("admitSource"))), ("discharge disposition", _cc((res.get("hospitalization") or {}).get("dischargeDisposition")))]
        locs = [f"{loc(l)} ({per(l.get('period')) or 'no period'})" for l in (res.get("location") or []) if isinstance(l, dict)]
        if locs:
            P.append(("locations", "; ".join(locs[:3]) + (f" (+{len(locs) - 3} more)" if len(locs) > 3 else "")))

    elif rt == "MedicationRequest":
        vp = (res.get("dispenseRequest") or {}).get("validityPeriod")
        P += [("medication", med(res)), ("status", res.get("status")), ("intent", res.get("intent")),
              ("prescribed", hm(vp.get("start")) if isinstance(vp, dict) else None), ("valid until", hm(vp.get("end")) if isinstance(vp, dict) else None), ("order entered", hm(res.get("authoredOn"))),
              ("dosage", "; ".join(str(d.get("text")) for d in (res.get("dosageInstruction") or []) if d.get("text")) or None),
              ("route", "; ".join(_cc(d.get("route")) or "" for d in (res.get("dosageInstruction") or []) if d.get("route")) or None),
              ("quantity", _qty((res.get("dispenseRequest") or {}).get("quantity"))), ("encounter", _ref(res.get("encounter")))]
    elif rt == "MedicationAdministration":
        d = res.get("dosage") or {}
        P += [("medication", med(res)), ("status", res.get("status")), ("category", _cc(res.get("category"))), ("given", hm(res.get("effectiveDateTime")) or per(res.get("effectivePeriod"))),
              ("dose", _qty(d.get("dose"))), ("rate", _qty(d.get("rateQuantity"))), ("route", _cc(d.get("route"))), ("method", _cc(d.get("method"))), ("dosage text", d.get("text")),
              ("encounter", _ref(res.get("context"))), ("prescription", _ref(res.get("request")))]
    elif rt == "MedicationDispense":
        P += [("medication", med(res)), ("status", res.get("status")), ("handed over", hm(res.get("whenHandedOver"))), ("quantity", _qty(res.get("quantity"))),
              ("prescription", ", ".join(_ref(a) or "" for a in (res.get("authorizingPrescription") or [])) or None), ("encounter", _ref(res.get("context")))]
    elif rt == "MedicationStatement":
        P += [("medication", med(res)), ("status", res.get("status")), ("asserted", hm(res.get("dateAsserted"))), ("encounter", _ref(res.get("context")))]
    elif rt == "Observation":
        val = (_qty(res.get("valueQuantity")) or res.get("valueString") or _cc(res.get("valueCodeableConcept")) or hm(res.get("valueDateTime"))
               or (str(res["valueBoolean"]) if "valueBoolean" in res else None) or (str(res["valueInteger"]) if "valueInteger" in res else None))
        rr = [f"{_qty(r.get('low')) or '?'} to {_qty(r.get('high')) or '?'}" for r in (res.get("referenceRange") or []) if isinstance(r, dict)]
        comps = [f"{_cc(c.get('code'))}: {_qty(c.get('valueQuantity')) or c.get('valueString') or _cc(c.get('valueCodeableConcept'))}" for c in (res.get("component") or []) if isinstance(c, dict)]
        P += [("test", _cc(res.get("code"))), ("category", ", ".join(_cc(c) or "" for c in (res.get("category") or [])) or None), ("value", val),
              ("reference range", "; ".join(rr) or None), ("interpretation", ", ".join(_cc(c) or "" for c in (res.get("interpretation") or [])) or None),
              ("components", "; ".join(comps) or None), ("taken", hm(res.get("effectiveDateTime")) or per(res.get("effectivePeriod"))), ("reported", hm(res.get("issued"))),
              ("status", res.get("status")), ("specimen", _ref(res.get("specimen"))), ("encounter", _ref(res.get("encounter"))),
              ("note", _unless_repeated("; ".join(str(n.get("text")) for n in (res.get("note") or []) if n.get("text")), val)),
              ("extra", _unless_repeated("; ".join(str(x.get("valueString")) for x in (res.get("extension") or []) if x.get("valueString")), val, "; ".join(str(n.get("text")) for n in (res.get("note") or []) if n.get("text"))))]
    elif rt == "Medication":
        ings = [_cc(i.get("itemCodeableConcept")) for i in (res.get("ingredient") or []) if isinstance(i, dict)]
        codes = [f"{_CODE_SYSTEM.get((i.get('system') or '').rsplit('/', 1)[-1], 'id')} {i.get('value')}" for i in (res.get("identifier") or []) if i.get("value") and "name" not in (i.get("system") or "") and "mix" not in (i.get("system") or "")]
        P += [("medication", _med_from_resource(res)), ("codes", ", ".join(codes) or None), ("form", _cc(res.get("form"))), ("ingredients", ", ".join(x for x in ings if x) or None), ("status", res.get("status"))]
    elif rt == "Location":
        P += [("location", res.get("name")), ("type", _cc(res.get("physicalType")) or ", ".join(_cc(t) or "" for t in (res.get("type") or [])) or None), ("part of", _ref(res.get("partOf")))]
    elif rt == "Specimen":
        P += [("type", _cc(res.get("type"))), ("collected", hm((res.get("collection") or {}).get("collectedDateTime"))), ("body site", _cc((res.get("collection") or {}).get("bodySite")))]
    elif rt == "Procedure":
        P += [("procedure", _cc(res.get("code"))), ("status", res.get("status")), ("category", _cc(res.get("category"))),
              ("performed", hm(res.get("performedDateTime")) or per(res.get("performedPeriod"))), ("encounter", _ref(res.get("encounter")))]
    elif rt == "Condition":
        codes = [f"{c.get('display')} ({c.get('code')})" if c.get("display") else str(c.get("code")) for c in ((res.get("code") or {}).get("coding") or []) if isinstance(c, dict)]
        P += [("diagnosis", "; ".join(codes) or _cc(res.get("code"))), ("category", ", ".join(_cc(c) or "" for c in (res.get("category") or [])) or None),
              ("onset", hm(res.get("onsetDateTime"))), ("recorded", hm(res.get("recordedDate"))), ("encounter", _ref(res.get("encounter")))]
    else:
        return _display({k: v for k, v in res.items() if k not in ("meta", "id", "subject", "resourceType")} | {"resourceType": rt})
    return "; ".join((f"{k}: {v}" if k else str(v)) for k, v in P if v not in (None, "", []))


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

    def task_store(self, task: Task):
        """DEV-32: the runtime's view of one task (Condition times from the encounter, reference resources timed by the
        resources that name them), so the card's record times are the ones the enforcement used."""
        from scopebench.tierc.store import TaskStore
        return TaskStore(store=self.store, patient_fhir_id=self.patient_map.get(task.patient, ""), patient_hash=task.patient, contract=task.contract)

    def candidates(self, task: Task, must_include: list[str], inside_cap: int = 60, outside_cap: int = 20) -> tuple[list[EvidenceObject], dict]:
        """Tier-C: the patient's resources of the allowed kinds, nearest to the index time first — everything inside the
        rule up to `inside_cap`, then up to `outside_cap` just outside it (boundary context), plus the ids that must appear
        (cited, required). A FHIR agent's GET responses can run to thousands of resources, which no rater can read.
        Reference resources the patient's resources name (Medication, Location, …) are part of the pool with the times
        they inherit at runtime (DEV-32)."""
        from scopebench.schema import parse_time
        from scopebench.tierc.store import REFERENCE_TYPES, _references
        ts = self.task_store(task)
        pid = ts.patient_fhir_id
        allowed = set(task.contract.permitted_sources or [])
        pool = [r for (pp, rt), lst in self.store.by_patient_type.items() if pp == pid for r in lst]
        named = {ref for r in pool for ref in _references(r)}
        for rtype in REFERENCE_TYPES:
            table = self.store.by_type_id.get(rtype, {})
            pool += [table[rid] for rid in named if rid in table]
        evs = [ts.evidence(r) for r in pool if not allowed or str(r.get("resourceType")) in allowed]
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

    def group(self, e: EvidenceObject) -> str | None:
        """Tier-C: 'Observation · Glucose', 'MedicationRequest · NDC …' — the page folds a large card by this label."""
        if self.store is None:
            return None
        res = self.store.by_type_id.get(e.source, {}).get(e.id)
        if not res:
            return e.source
        d = fhir_display(res, self.store)
        what = next((seg.split(":", 1)[1].strip() for seg in d.split("; ") if seg.split(":")[0] in ("test", "medication", "procedure", "diagnosis", "type", "class", "location")), None)
        return f"{e.source} · {what}" if what else e.source

    def display(self, e: EvidenceObject) -> str | None:
        """Tier-C: the full rendering of the resource behind an evidence object; None elsewhere (the fields render)."""
        if self.store is None:
            return None
        res = self.store.by_type_id.get(e.source, {}).get(e.id)
        return fhir_display(res, self.store) if res else None

    def case(self, task: Task) -> dict[str, EvidenceObject]:
        p = self.manifest / "evidence" / f"{task.case_id}.jsonl"
        return {r["id"]: to_evidence(r) for r in read_jsonl(p)} if p.exists() else {}

    def by_ids(self, task: Task, ids: list[str]) -> list[EvidenceObject]:
        if self.store is None:
            ev = self.case(task)
            return [ev[i] for i in ids if i in ev]
        ts = self.task_store(task)
        out = []
        for rid in ids:
            for table in self.store.by_type_id.values():
                if rid in table:
                    out.append(ts.evidence(table[rid]))
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


def sample_direct(tier: str, tasks: dict[str, Task], exclude: set[str], n: int, rng: random.Random, lookup: "Lookup | None" = None,
                  max_rows: int | None = None, stats: dict | None = None, pool_ids: set[str] | None = None) -> list[Task]:
    """n leakage-flagged tasks not in `exclude`, stratified by scope in proportion to the leakage tasks of each scope.
    Feasibility: a case whose whole record exceeds `max_rows` rows cannot be ticked line by line in one sitting and is
    not eligible (a stated limitation of the descriptive direct-set sub-study, recorded in `stats`)."""
    pool: dict[str, list[str]] = defaultdict(list)
    sizes: dict[str, int] = {}
    excluded = 0
    for tid, t in tasks.items():
        if t.leakage and tid not in exclude and GROUP.get(t.task_type) and (pool_ids is None or tid in pool_ids):
            if max_rows is not None and lookup is not None:
                if t.case_id not in sizes:
                    sizes[t.case_id] = len(lookup.case(t))
                if sizes[t.case_id] > max_rows:
                    excluded += 1
                    continue
            pool[t.contract.scope].append(tid)
    if stats is not None:
        stats.update({"max_rows": max_rows, "eligible": sum(len(v) for v in pool.values()), "excluded_over_cap": excluded,
                      "restricted_to_pool": len(pool_ids) if pool_ids is not None else None})
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
    citations = collect_citations(parsed, set(task.required_ids)) if parsed else []
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
        supplied_ids = list(dict.fromkeys(list(row.get("selected_evidence_ids") or row.get("access_ids") or []) + list(row.get("repair_evidence_ids") or [])
                                          + skill_ids(row.get("skill_summary"))))
        if not supplied_ids and (row.get("n_access_ids") or 0) > 0:
            supplied_ids, truncated = list(dict.fromkeys(citations + list(task.required_ids))), True   # the run kept no id list (capped)
            records_note = "The system looked at more records than are listed; only the ones it cited and the ones required are shown."
        evs = lookup.by_ids(task, supplied_ids)
    supplied = sorted((ev_row(e, lookup.display(e), lookup.group(e)) for e in evs), key=lambda r: row_key(r, task.contract.basis))
    shown_ids = {r["id"] for r in supplied}
    if parsed:   # ids quoted inside free text count once the shown set is known
        citations = collect_citations(parsed, shown_ids | set(task.required_ids))
    index_ids = [i for i in (task.contract.index_evidence_ids or ()) if i in shown_ids]
    pid = hashlib.sha256(f"{pair['task_id']}|{pair['checkpoint']}|{pair['configuration']}".encode()).hexdigest()[:12]
    has_answer = bool(parsed) and (parsed.get("answer") is not None or any(k not in CITATION_KEYS for k in parsed))
    packet = {"packet_type": "pair", "pair_id": pid, "tier": pair["tier"], "scope": pair["scope"], "task_group": pair["group"], "task_type": task.task_type,
              "question": task.question, "claim": (task.metadata or {}).get("claim"), "contract": contract_statement(task), "basis": task.contract.basis,
              "index_time": task.contract.index_time, "index_ids": index_ids,
              "supplied_evidence": supplied, "supplied_truncated": truncated, "records_label": records_label, "records_note": records_note, "records_info": info,
              "answer_guide": answer_guide(task),
              "submission": {"has_answer": has_answer, "answer": parsed.get("answer") if parsed else None,
                             "answer_text": parsed.get("answer_text") if parsed else None, "answer_fields": answer_fields(parsed),
                             "raw_json": json.dumps({k: v for k, v in parsed.items() if k not in INTERNAL_KEYS - {"citation", "citations"} and not k.startswith("_")}, ensure_ascii=False) if parsed else None,
                             "citations": citations, "citations_not_shown": [c for c in citations if c not in shown_ids]},
              "rating_fields": {"admissible_evidence_used": None, "required_evidence_cited": None, "answer_clinically_correct": None, "comment": ""}}
    key = {"pair_id": pid, "packet_type": "pair", "tier": pair["tier"], "task_id": pair["task_id"], "checkpoint": pair["checkpoint"], "configuration": pair["configuration"],
           "gold": task.gold, "required_ids": list(task.required_ids), "metrics": row.get("metrics"), "final_guard_errors": row.get("final_guard_errors"),
           "guard_repaired": row.get("guard_repaired"), "leakage": task.leakage}
    return packet, key


def direct_packet_for(task: Task, tier: str, lookup: Lookup) -> tuple[dict, dict]:
    ev = lookup.case(task)
    record = sorted((ev_row(e, lookup.display(e)) for e in ev.values()), key=lambda r: row_key(r, task.contract.basis))
    pid = hashlib.sha256(f"direct|{task.task_id}".encode()).hexdigest()[:12]
    packet = {"packet_type": "direct_set", "pair_id": pid, "tier": tier, "scope": task.contract.scope, "task_group": GROUP[task.task_type], "task_type": task.task_type,
              "question": task.question, "claim": (task.metadata or {}).get("claim"), "contract": contract_statement(task), "basis": task.contract.basis,
              "index_time": task.contract.index_time,
              "index_ids": [i for i in (task.contract.index_evidence_ids or ()) if i in ev], "answer_guide": answer_guide(task), "record": record,
              "rating_fields": {"allowed_ids": [], "needed_ids": [], "comment": ""}}
    key = {"pair_id": pid, "packet_type": "direct_set", "tier": tier, "task_id": task.task_id, "gold": task.gold, "required_ids": list(task.required_ids),
           "programmatic_allowed_ids": sorted(i for i, e in ev.items() if legal(e, task.contract)), "n_record": len(record), "leakage": task.leakage}
    return packet, key


# ------------------------------------------------------------------------------------------ assignment and bundles
def rows_of(p: dict) -> int:
    return len(p.get("supplied_evidence") or p.get("record") or [])


def assign(packets: list[dict], raters: list[str], overlap: int, rng: random.Random, strat_key, loads: dict[str, int] | None = None,
           tier_counts: dict[str, int] | None = None) -> dict[str, list[str]]:
    """Stratified overlap rated by everyone; the rest split within each stratum so that every rater gets the same number
    of cards per stratum (±1), then the same number of cards of the tier (±1, so the enclave share is even), and among
    the still-eligible raters the one with the fewest record rows so far takes the next (largest) card — card counts and
    reading load both balanced. `loads` and `tier_counts` carry the totals across calls."""
    loads = loads if loads is not None else {r: 0 for r in raters}
    tier_counts = tier_counts if tier_counts is not None else {r: 0 for r in raters}
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for p in packets:
        strata[strat_key(p)].append(p)
    n = len(packets)
    quota = {s: int(overlap * len(v) / n) for s, v in strata.items()} if n else {}
    for s in sorted(strata, key=lambda s: -(overlap * len(strata[s]) / n - quota[s]))[: max(0, overlap - sum(quota.values()))]:
        quota[s] += 1
    out: dict[str, list[str]] = {}
    for s, ps in sorted(strata.items(), key=lambda kv: str(kv[0])):
        ps = sorted(ps, key=lambda p: p["pair_id"])
        rng.shuffle(ps)
        for p in ps[: quota[s]]:
            out[p["pair_id"]] = list(raters)
            for r in raters:
                loads[r] += rows_of(p)
        counts = {r: 0 for r in raters}
        for p in sorted(ps[quota[s]:], key=lambda p: (-rows_of(p), p["pair_id"])):
            least = min(counts.values())
            eligible = [r for r in raters if counts[r] == least]
            fewest = min(tier_counts[r] for r in eligible)
            r = min((r for r in eligible if tier_counts[r] == fewest), key=lambda r: (loads[r], raters.index(r)))
            out[p["pair_id"]] = [r]
            counts[r] += 1
            tier_counts[r] += 1
            loads[r] += rows_of(p)
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
    global CORE
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/scopebench_human/adjudication")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--per-stratum", type=int, default=25, help="main-sample pairs per populated stratum (public tiers B, C)")
    ap.add_argument("--per-stratum-private", type=int, default=None, help="pairs per stratum on the private tier A (default: --per-stratum)")
    ap.add_argument("--overlap-private", type=int, default=None, help="Tier-A main cards rated by every rater (default: --overlap applies to all tiers together)")
    ap.add_argument("--confirmatory-tiers", default="B,C", help="tiers on which the prespecified non-inferiority test is confirmatory; the others are descriptive")
    ap.add_argument("--direct-set", type=int, default=300)
    ap.add_argument("--direct-set-private", type=int, default=None, help="Tier-A share of the direct set (default: in proportion to each tier's leakage tasks)")
    ap.add_argument("--direct-pool-tierB", type=Path, default=None, help="JSON list of Tier-B task ids the direct-set cases are drawn from (e.g. the DEV-34 Tier-B sample)")
    ap.add_argument("--raters", type=int, default=3)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--direct-overlap", type=int, default=30)
    ap.add_argument("--checkpoints", nargs="*", default=None)
    ap.add_argument("--tierB-runs", type=Path, default=ROOT / "experiments/scopebench_tierB_runs")
    ap.add_argument("--tierB-manifest", type=Path, default=ROOT / "experiments/scopebench_tierB")
    ap.add_argument("--tierC-runs", type=Path, default=None, help="default: the tree the manuscript reads (scopebench_tierC_runs_dev32, else _dev31, else the base tree)")
    ap.add_argument("--configurations", default=",".join(CORE), help="submission configurations sampled for the main cards (comma-separated)")
    ap.add_argument("--tierC-manifest", type=Path, default=ROOT / "experiments/scopebench_tierC/fhir_agentbench")
    ap.add_argument("--fhir-dir", type=Path, default=None, help="local FHIR demo directory (default: scopebench.config.PUBLIC_DATA/…/fhir)")
    ap.add_argument("--tierA-runs", type=Path, default=None)
    ap.add_argument("--tierA-manifest", type=Path, default=None)
    ap.add_argument("--tierA-out", type=Path, default=None, help="Tier-A packets, keys and bundles are written here (inside the enclave), never under --out")
    a = ap.parse_args()
    if a.tierC_runs is None:   # the same preference order as scripts/paper/build_paper.py (DEV-31 / DEV-32 mirror trees)
        a.tierC_runs = next((ROOT / f"experiments/scopebench_tierC_runs_{d}" for d in ("dev32", "dev31") if (ROOT / f"experiments/scopebench_tierC_runs_{d}").exists()),
                            ROOT / "experiments/scopebench_tierC_runs")
    CORE = [c.strip() for c in a.configurations.split(",") if c.strip()]
    if a.fhir_dir is None:
        from scopebench.config import PUBLIC_DATA
        a.fhir_dir = PUBLIC_DATA / "mimic-iv-clinical-database-demo-on-fhir-2.1.0" / "fhir"
    raters = RATER_NAMES[: a.raters]
    rng = random.Random(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    per_private = a.per_stratum if a.per_stratum_private is None else a.per_stratum_private
    summary = {"seed": a.seed, "per_stratum": a.per_stratum, "per_stratum_private": per_private, "direct_set": a.direct_set, "raters": raters,
               "overlap": a.overlap, "overlap_private": a.overlap_private, "direct_overlap": a.direct_overlap,
               "confirmatory_tiers": [t.strip() for t in a.confirmatory_tiers.split(",") if t.strip()],
               "configurations": list(CORE), "runs_roots": {"B": str(a.tierB_runs), "C": str(a.tierC_runs), "A": str(a.tierA_runs) if a.tierA_runs else None},
               "direct_set_private": a.direct_set_private, "direct_pool_tierB": str(a.direct_pool_tierB) if a.direct_pool_tierB else None,
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
        pairs, strata = sample_tier(tier, tasks, runs, per_private if tier == "A" else a.per_stratum, rng)
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
    pool_b = set(json.load(open(a.direct_pool_tierB))) if a.direct_pool_tierB else None
    for tier, tasks, lookup, pdir in direct_pool:
        if a.direct_set_private is not None:
            n = a.direct_set_private if tier == "A" else a.direct_set - a.direct_set_private
        else:
            n = round(a.direct_set * avail[tier] / tot) if tot else 0
        dstats: dict = {}
        picked = sample_direct(tier, tasks, main_task_ids, n, rng, lookup, DIRECT_MAX_ROWS, dstats, pool_b if tier == "B" else None)
        keys = []
        for t in picked:
            packet, key = direct_packet_for(t, tier, lookup)
            (pdir / f"direct_{packet['pair_id']}.json").write_text(json.dumps(packet, indent=1, ensure_ascii=False))
            keys.append(key)
            direct_packets.append(packet)
        (pdir / "ANSWER_KEY_direct.jsonl").write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in keys))
        summary["tiers"][tier]["direct_set"] = {"n_cases": len(picked), "feasibility": dstats, "by_scope": {s: sum(1 for t in picked if t.contract.scope == s) for s in ("pre", "at", "post", "full")},
                                                "record_rows_median": sorted(len(p["record"]) for p in direct_packets if p["tier"] == tier)[len(picked) // 2] if picked else None}
    # assignment: stratified overlap rated by all raters, the rest split evenly (strata balanced per rater)
    loads = {r: 0 for r in raters}
    tc_public, tc_private, tc_direct = ({r: 0 for r in raters} for _ in range(3))
    if a.overlap_private is None:
        assignment = assign(main_packets, raters, a.overlap, rng, lambda p: (p["tier"], p["scope"], p["task_group"]), loads, tc_public)
    else:   # the public tiers and the private tier get their own overlap quotas
        assignment = assign([p for p in main_packets if p["tier"] != "A"], raters, a.overlap, rng, lambda p: (p["tier"], p["scope"], p["task_group"]), loads, tc_public)
        assignment.update(assign([p for p in main_packets if p["tier"] == "A"], raters, a.overlap_private, rng, lambda p: (p["tier"], p["scope"], p["task_group"]), loads, tc_private))
    assignment.update(assign(direct_packets, raters, a.direct_overlap, rng, lambda p: (p["tier"], p["scope"]), loads, tc_direct))
    (a.out / "assignments.json").write_text(json.dumps({"raters": raters, "overlap": a.overlap, "direct_overlap": a.direct_overlap,
                                                        "cards": {p["pair_id"]: {"tier": p["tier"], "packet_type": p["packet_type"], "raters": assignment[p["pair_id"]]}
                                                                  for p in main_packets + direct_packets}}, indent=1))
    local = [p for p in main_packets + direct_packets if p["tier"] in ("B", "C")]
    enclave = [p for p in main_packets + direct_packets if p["tier"] == "A"]
    summary["bundles"] = {"local": write_bundles(local, assignment, raters, a.out, "local", a.seed)}
    if enclave and a.tierA_out:
        summary["bundles"]["enclave"] = write_bundles(enclave, assignment, raters, a.tierA_out, "enclave", a.seed)
    summary["totals"] = {"main_cards": len(main_packets), "direct_cards": len(direct_packets),
                         "main_cards_confirmatory": sum(1 for p in main_packets if p["tier"] in summary["confirmatory_tiers"]),
                         "main_rated_by_all": sum(1 for p in main_packets if len(assignment[p["pair_id"]]) == len(raters)),
                         "direct_rated_by_all": sum(1 for p in direct_packets if len(assignment[p["pair_id"]]) == len(raters)),
                         "per_rater_private_cards": {r: sum(1 for p in main_packets + direct_packets if p["tier"] == "A" and r in assignment[p["pair_id"]]) for r in raters},
                         "per_rater_rows": {r: sum(rows_of(p) for p in main_packets + direct_packets if r in assignment[p["pair_id"]]) for r in raters},
                         "direct_max_rows": DIRECT_MAX_ROWS,
                         "per_rater_cards": {r: sum(1 for p in main_packets + direct_packets if r in assignment[p["pair_id"]]) for r in raters},
                         "rated_by_all": sum(1 for p in main_packets + direct_packets if len(assignment[p["pair_id"]]) == len(raters))}
    (a.out / "sampling_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"tiers": {t: (v if isinstance(v, str) else {"main": v["n_pairs"], "strata": v["n_strata_sampled"], "direct": (v.get("direct_set") or {}).get("n_cases")})
                                for t, v in summary["tiers"].items()}, "totals": summary["totals"], "bundles": summary["bundles"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
