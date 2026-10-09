# SPDX-License-Identifier: CC-BY-4.0
# FHIR-AgentBench adaptation; source, changes, and attribution: THIRD_PARTY_NOTICES.md
"""Tier-C deterministic skills (Tier-C protocol §Skills): typed operations over the LEGAL FHIR resource set of a
task, parameterized by the task's structured specification (`metadata.skill_params`: the benchmark's placeholder
values, operators and time-filter phrases — never the gold answer or its SQL). Every result is provenance-preserving.

Coverage is decided per template by `scripts/scopebench/validate_tierc_skills.py`: a template is covered only when
the skill reproduces the released answer on every task whose R_t lies inside the contract; templates that depend
on fields the FHIR mapping does not carry (admission type strings, marital status on the visit, care-unit names)
or on two separate time windows (change / comparison between two measurements) are uncovered by construction.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..predicate import legal
from ..schema import EvidenceObject
from .contracts import parse_now, resource_to_evidence

Resource = dict[str, Any]
Result = dict[str, Any]

# vital-sign labels that the benchmark names differently from the chartevents item label
VITAL_SYNONYMS = {"weight": ("daily weight", "admission weight (kg)"), "height": ("height (cm)", "height"),
                  "heart rate": ("heart rate",), "respiratory rate": ("respiratory rate",), "temperature": ("temperature celsius", "temperature fahrenheit"),
                  "spo2": ("o2 saturation pulseoxymetry",), "o2 saturation": ("o2 saturation pulseoxymetry",),
                  "systolic blood pressure": ("non invasive blood pressure systolic", "arterial blood pressure systolic"),
                  "diastolic blood pressure": ("non invasive blood pressure diastolic", "arterial blood pressure diastolic"),
                  "mean blood pressure": ("non invasive blood pressure mean", "arterial blood pressure mean")}


# ----------------------------------------------------------------------------------------------- resource helpers
def _coding(r: Resource, field: str = "code") -> dict[str, Any]:
    c = r.get(field) or {}
    cs = c.get("coding") if isinstance(c, dict) else None
    return (cs or [{}])[0] if isinstance(cs, list) else {}


def _display(r: Resource) -> str:
    return str(_coding(r).get("display") or "").strip().lower()


def _system(r: Resource) -> str:
    return str(_coding(r).get("system") or "").rsplit("/", 1)[-1]


def _value(r: Resource) -> float | None:
    q = r.get("valueQuantity")
    if isinstance(q, dict) and isinstance(q.get("value"), (int, float)):
        return float(q["value"])
    return None


def _when(r: Resource) -> str | None:
    rt = r.get("resourceType")
    if rt == "Observation":
        return r.get("effectiveDateTime")
    if rt == "MedicationRequest":
        vp = (r.get("dispenseRequest") or {}).get("validityPeriod") or {}
        return vp.get("start") if isinstance(vp.get("start"), str) else r.get("authoredOn")   # prescriptions.starttime
    if rt == "Procedure":
        return r.get("performedDateTime") or (r.get("performedPeriod") or {}).get("start")
    if rt == "Encounter":
        return (r.get("period") or {}).get("start")
    if rt == "Specimen":
        return (r.get("collection") or {}).get("collectedDateTime")
    return None


def _dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def local_ts(s: str | None) -> str | None:
    """The benchmark's answer form for timestamps: the FHIR local wall-clock time without its offset."""
    if not s:
        return None
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def _ident_system(r: Resource) -> str:
    for i in r.get("identifier") or []:
        if isinstance(i, dict) and i.get("system"):
            return str(i["system"])
    return ""


def _scalar_strings(x: Any) -> list[str]:
    if isinstance(x, dict):
        return [s for v in x.values() for s in _scalar_strings(v)]
    if isinstance(x, list):
        return [s for v in x for s in _scalar_strings(v)]
    return [str(x)] if isinstance(x, (str, int, float)) else []


# ----------------------------------------------------------------------------------------------- domains
class Legal:
    """The task's legal resources, indexed by type, plus the untimed references the skills need (Medication,
    Specimen type, Patient) — all taken from the same legal set."""

    def __init__(self, resources: list[Resource]):
        self.by_type: dict[str, list[Resource]] = {}
        self.by_id: dict[str, Resource] = {}
        for r in resources:
            self.by_type.setdefault(str(r.get("resourceType")), []).append(r)
            self.by_id[str(r.get("id"))] = r

    def ref(self, reference: Any) -> Resource | None:
        if isinstance(reference, dict):
            reference = reference.get("reference")
        if not isinstance(reference, str):
            return None
        return self.by_id.get(reference.split("/")[-1])

    def medication_text(self, req: Resource) -> str:
        med = self.ref(req.get("medicationReference"))
        return " ".join(_scalar_strings({k: v for k, v in med.items() if k != "meta"})).lower() if med else ""


def legal_all_resources(legal: Legal) -> list[Resource]:
    return [r for rs in legal.by_type.values() for r in rs]


def _match_name(display: str, name: str | None, synonyms: dict[str, tuple[str, ...]] | None = None) -> bool:
    if not name:
        return True
    name = name.strip().lower()
    if display == name:
        return True
    if synonyms and name in synonyms and display in synonyms[name]:
        return True
    return False


def _in_visit(r: Resource) -> bool:
    """The benchmark's SQL joins every event table to the patient's admissions (hadm_id) or ICU stays (stay_id);
    events documented outside any visit (no Encounter reference in FHIR) are not part of its answers."""
    return isinstance((r.get("encounter") or {}).get("reference"), str)


def domain(main_table: str, legal: Legal, values: dict[str, str], template: str = "") -> list[Resource]:
    """Resources the template's main table maps to, filtered by the task's entity values."""
    # the visit-link restriction applies to the template's main domain only; helper lookups (organisms via
    # derivedFrom, specimens, medications) use the whole legal set
    obs = [r for r in legal.by_type.get("Observation", []) if _in_visit(r)]
    if main_table == "chartevents":
        name = values.get("vital_name")
        if name is None:
            # weight / height templates carry the item in the template text, not in a placeholder
            t = template.lower()
            name = "weight" if "weight" in t else ("height" if "height" in t else None)
        return [r for r in obs if _system(r) == "mimic-chartevents-d-items" and _match_name(_display(r), name, VITAL_SYNONYMS)]
    if main_table == "labevents":
        return [r for r in obs if _system(r) == "mimic-d-labitems" and _match_name(_display(r), values.get("lab_name"))]
    if main_table == "outputevents":
        def _is_output(r: Resource) -> bool:
            return any(cc.get("code") == "Output" for c in (r.get("category") or []) if isinstance(c, dict) for cc in (c.get("coding") or []))
        return [r for r in obs if _system(r) == "mimic-d-items" and _is_output(r) and _match_name(_display(r), values.get("output_name"))]
    if main_table == "microbiologyevents":
        tests = [r for r in obs if _system(r) == "mimic-microbiology-test"]
        spec = values.get("spec_name")
        if spec:
            tests = [r for r in tests if (lambda s: s is not None and _match_name(str(_coding(s, "type").get("display") or "").lower(), spec))(legal.ref(r.get("specimen")))]
        return tests
    if main_table == "prescriptions":
        reqs = [r for r in legal.by_type.get("MedicationRequest", []) if _in_visit(r)]
        drug = values.get("drug_name")
        drugs = [values[k] for k in ("drug_name1", "drug_name2", "drug_name3") if values.get(k)]
        route = values.get("drug_route")
        out = []
        for r in reqs:
            text = legal.medication_text(r)
            if drug and drug.strip().lower() not in text:
                continue
            if drugs and not any(d.strip().lower() in text for d in drugs):
                continue
            if route:
                rc = str(_coding((r.get("dosageInstruction") or [{}])[0] if r.get("dosageInstruction") else {}, "route").get("code") or "").lower()
                if rc != route.strip().lower():
                    continue
            out.append(r)
        return out
    if main_table == "admissions":
        encs = [r for r in legal.by_type.get("Encounter", []) if "encounter-hosp" in _ident_system(r)]
        loc = values.get("admission_location")
        if loc:
            encs = [r for r in encs if str(_coding((r.get("hospitalization") or {}).get("admitSource") or {}).get("code") or "").lower() == loc.strip().lower()]
        return encs
    if main_table == "icustays":
        return [r for r in legal.by_type.get("Encounter", []) if "encounter-icu" in _ident_system(r)]
    if main_table == "procedures_icd":
        return [r for r in legal.by_type.get("Procedure", []) if _match_name(_display(r), values.get("procedure_name"))]
    if main_table == "diagnoses_icd":
        return [r for r in legal.by_type.get("Condition", []) if _match_name(_display(r), values.get("diagnosis_name"))]
    if main_table == "patients":
        return legal.by_type.get("Patient", [])
    return []


# ----------------------------------------------------------------------------------------------- operations
_CMP = {"<": lambda a, b: a < b, ">": lambda a, b: a > b, "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b, "=": lambda a, b: a == b}


def _apply_comparison(rows: list[Resource], ops: dict, values: dict) -> list[Resource]:
    cmp = (ops.get("comparison") or {}).get("sql")
    thr = values.get("vital_value") or values.get("lab_value")
    if not cmp or thr is None:
        return rows
    fn = _CMP.get(cmp)
    if fn is None:
        return rows
    t = float(thr)
    return [r for r in rows if _value(r) is not None and fn(_value(r), t)]


def _ordered(rows: list[Resource]) -> list[Resource]:
    timed = [r for r in rows if _dt(_when(r)) is not None]
    return sorted(timed, key=lambda r: (_dt(_when(r)), str(r.get("id"))))


def _boundary(rows: list[Resource], which: str) -> list[Resource]:
    """All rows sharing the earliest (first) or latest (last) time."""
    o = _ordered(rows)
    if not o:
        return []
    pick = o[0] if which == "first" else o[-1]
    t = _dt(_when(pick))
    return [r for r in o if _dt(_when(r)) == t]


def _exact(time_ph: dict) -> str | None:
    e = (time_ph.get("time_filter_exact1") or {}).get("nlq") or (time_ph.get("time_filter_exact1") or {}).get("type") or ""
    e = e.lower()
    return "first" if "first" in e else ("last" if "last" in e else None)


def _sort_dir(ops: dict) -> str | None:
    s = ((ops.get("sort") or {}).get("nlq") or (ops.get("sort") or {}).get("sql") or "").lower()
    return "max" if ("max" in s or "desc" in s or "highest" in s) else ("min" if ("min" in s or "asc" in s or "lowest" in s) else None)


def _agg(ops: dict) -> str | None:
    a = ((ops.get("agg_function") or {}).get("nlq") or (ops.get("agg_function") or {}).get("sql") or "").lower()
    for k in ("max", "min", "avg", "average", "mean", "sum"):
        if k in a:
            return {"average": "avg", "mean": "avg"}.get(k, k)
    return None


def _ok(answer: Any, prov: list[Resource], op: str) -> Result:
    return {"status": "ok", "answer": answer, "provenance": [str(r.get("id")) for r in prov], "op": op}


ABSENCE_PROVENANCE_CAP = 20


def solve(task_meta: dict[str, Any], legal_resources: list[Resource]) -> Result:
    """Dispatch on the template (see `_solve`). A negative or empty result — nothing matched, no open visit — carries no
    matching resource; it then cites the resources it examined to establish the absence (the template's table within
    the legal set, first ABSENCE_PROVENANCE_CAP by store order), so a replaced negative answer is not citation-free
    (DEV-26 follow-up, 2026-10-06). Answers are unchanged; only the provenance of negatives is."""
    res = _solve(task_meta, legal_resources)
    if res.get("status") == "ok" and not res.get("provenance"):
        legal = Legal(legal_resources)
        examined = domain(str(task_meta.get("main_table") or ""), legal, {}, str(task_meta.get("template") or ""))
        res["provenance"] = [str(r.get("id")) for r in examined[:ABSENCE_PROVENANCE_CAP]]
        res["provenance_kind"] = "examined"
    return res


def _solve(task_meta: dict[str, Any], legal_resources: list[Resource]) -> Result:
    """Dispatch on the template; returns {'status': 'uncovered'} for templates without a deterministic operation."""
    template = str(task_meta.get("template") or "")
    main_table = str(task_meta.get("main_table") or "")
    sp = task_meta.get("skill_params") or {}
    values, ops, time_ph = sp.get("values") or {}, sp.get("ops") or {}, sp.get("time") or {}
    legal = Legal(legal_resources)
    T = template.lower()
    rows = domain(main_table, legal, values, template)
    now = parse_now(task_meta.get("assumption"))

    # --- uncovered by construction
    if "change in" in T or "[comparison] than the value" in T or "[time_filter_exact2]" in T or "[unit_average]" in T:
        return {"status": "uncovered", "reason": "two windows / per-unit aggregate"}
    if "admission type" in T or "marital status" in T or "careunit" in T:
        return {"status": "uncovered", "reason": "field not carried by the FHIR mapping"}

    # --- patient facts
    if main_table == "patients":
        if "gender" in T:
            p = rows[0] if rows else None
            return _ok([[str(p.get("gender"))]] if p and p.get("gender") else [], rows[:1], "patient_lookup")
        return {"status": "uncovered", "reason": "patient field"}

    # --- existence
    if T.startswith("has_verb") or "has_verb there been" in T:
        if "organism found" in T:
            tests = _boundary(rows, _exact(time_ph) or "last") if _exact(time_ph) else rows
            orgs = [o for t in tests for o in legal.by_type.get("Observation", []) if _system(o) == "mimic-microbiology-organism" and any(legal.ref(d) is t for d in (o.get("derivedFrom") or []))]
            return _ok([[1 if orgs else 0]], orgs or tests, "exists_organism")
        if "[comparison] than" in T or "ever [comparison]" in T:
            rows = _apply_comparison(rows, ops, values)
        return _ok([[1 if rows else 0]], rows, "exists")

    # --- counts
    if T.startswith("count the number of"):
        if main_table == "prescriptions" and "number of drugs" in T:
            names = {legal.medication_text(r) for r in rows}
            return _ok([[len(names)]], rows, "count_distinct")
        if main_table in ("admissions", "icustays"):
            return _ok([[len({str(r.get("id")) for r in rows})]], rows, "count")
        return _ok([[len(rows)]], rows, "count")

    # --- sums and aggregates
    if "total amount of dose" in T:
        doses = [float(((d.get("doseAndRate") or [{}])[0].get("doseQuantity") or {}).get("value")) for r in rows for d in (r.get("dosageInstruction") or [])
                 if isinstance(((d.get("doseAndRate") or [{}])[0].get("doseQuantity") or {}).get("value"), (int, float))]
        return _ok([[sum(doses)]] if doses else [[None]], rows, "sum_dose")
    if "total volume" in T:
        vals = [_value(r) for r in rows if _value(r) is not None]
        return _ok([[sum(vals)]] if vals else [[None]], rows, "sum")
    agg = _agg(ops)
    if "[agg_function]" in T and agg:
        vals = [_value(r) for r in rows if _value(r) is not None]
        if not vals:
            return _ok([[None]], rows, f"agg_{agg}")
        v = {"max": max(vals), "min": min(vals), "avg": sum(vals) / len(vals), "sum": sum(vals)}[agg]
        return _ok([[v]], rows, f"agg_{agg}")

    # --- durations since an event
    if T.startswith("how many [unit_count] have passed"):
        if now is None:
            return {"status": "insufficient", "reason": "no current time"}
        unit = (values.get("unit_count") or "hour").lower()
        div = 3600.0 if unit.startswith("hour") else 86400.0
        # these templates anchor on the CURRENT visit in their body ("on this hospital visit", "admitted ... currently"):
        # the benchmark's SQL requires an open visit (dischtime / outtime IS NULL); without one the answer is []
        kind = "icu" if "icu" in T else "hospital"
        open_visits = [e for e in legal.by_type.get("Encounter", []) if (f"encounter-{'icu' if kind == 'icu' else 'hosp'}" in _ident_system(e))
                       and (e.get("period") or {}).get("start") and not (e.get("period") or {}).get("end")]
        if not open_visits:
            return _ok([], [], "elapsed")
        if "currently" in T:
            pick = open_visits[-1:]
        else:
            pick = _boundary(rows, _exact(time_ph) or "last")
        if not pick:
            return _ok([], rows, "elapsed")
        t = _dt(_when(pick[0]))
        return _ok([[(now - t).total_seconds() / div]], pick, "elapsed")

    # --- lengths of stay
    if "length of icu stay" in T or "length of hospital stay" in T:
        pick = _boundary(rows, _exact(time_ph) or "last")
        if not pick:
            return _ok([], rows, "los")
        p = pick[0].get("period") or {}
        s, e = _dt(p.get("start")), _dt(p.get("end"))
        return _ok([[(e - s).total_seconds() / 86400.0]] if s and e else [], pick, "los")

    # --- first / last events: time, value, name
    exact = _exact(time_ph)
    if T.startswith("when was") or T.startswith("list the hospital admission time"):
        if "[sort] value" in T or "[sort] {vital_name}" in T:
            d = _sort_dir(ops)
            vals = [(r, _value(r)) for r in rows if _value(r) is not None]
            if not vals:
                return _ok([], rows, "sort_time")
            target = max(v for _, v in vals) if d == "max" else min(v for _, v in vals)
            cand = [r for r, v in vals if v == target]
            pick = _boundary(cand, exact or "first")
            return _ok([[local_ts(_when(r))] for r in pick[:1]], pick, "sort_time")
        if "[comparison] than" in T:
            rows = _apply_comparison(rows, ops, values)
        if "discharge time" in T:
            rows = [r for r in rows if (r.get("period") or {}).get("end")]
            o = sorted(rows, key=lambda r: _dt((r.get("period") or {}).get("end")))
            pick = [o[0] if exact == "first" else o[-1]] if o else []
            return _ok([[local_ts((r.get("period") or {}).get("end"))] for r in pick], pick, "discharge_time")
        if T.startswith("list the"):
            o = _ordered(rows)
            return _ok([[local_ts(_when(r))] for r in o], o, "list_times")
        pick = _boundary(rows, exact or "first") if exact else _ordered(rows)
        return _ok([[local_ts(_when(r))] for r in pick[:1]], pick, "boundary_time")

    if T.startswith("what was the") or T.startswith("what is the"):
        if exact is None:
            return {"status": "uncovered", "reason": "no first/last operator"}
        if main_table == "microbiologyevents":
            pick = _boundary(rows, exact)
            if "organism name" in T:
                orgs = [o for t in pick for o in legal.by_type.get("Observation", []) if _system(o) == "mimic-microbiology-organism" and any(legal.ref(d) is t for d in (o.get("derivedFrom") or []))]
                names = sorted({_coding(o).get("display") for o in orgs if _coding(o).get("display")})
                return _ok([[n] for n in names], orgs or pick, "organism_names")
            if "name of the specimen" in T:
                specs = [legal.ref(t.get("specimen")) for t in pick]
                names = sorted({str(_coding(s, "type").get("display")) for s in specs if s and _coding(s, "type").get("display")})
                return _ok([[n] for n in names], pick, "specimen_names")
            if "name of the microbiology test" in T:
                names = sorted({_coding(t).get("display") for t in pick if _coding(t).get("display")})
                return _ok([[n] for n in names], pick, "test_names")
        if main_table == "prescriptions":
            pick = _boundary(rows, exact)
            if "name of the drug" in T:
                names = []
                for r in pick:
                    med = legal.ref(r.get("medicationReference"))
                    nm = (med or {}).get("code", {}).get("text") if med else None
                    names.append(nm or legal.medication_text(r))
                return _ok([[n] for n in sorted({n for n in names if n})], pick, "drug_names")
            if "dose of" in T:
                doses = [((d.get("doseAndRate") or [{}])[0].get("doseQuantity") or {}).get("value") for r in pick for d in (r.get("dosageInstruction") or [])]
                doses = [float(x) for x in doses if isinstance(x, (int, float))]
                return _ok([[x] for x in sorted(set(doses))], pick, "dose")
        if main_table in ("chartevents", "labevents", "outputevents"):
            pick = _boundary(rows, exact)
            vals = sorted({_value(r) for r in pick if _value(r) is not None})
            return _ok([[v] for v in vals], pick, "boundary_value")
        if main_table == "admissions" and "admitted via" in T:
            pick = _boundary(rows, exact)
            return _ok([[local_ts(_when(r))] for r in pick[:1]], pick, "admission_time")
    return {"status": "uncovered", "reason": "template not mapped"}


def legal_resources_for(task, all_resources: list[Resource], to_evidence: Callable[[Resource], Any], legal_fn: Callable) -> list[Resource]:
    """Filter a patient's resources with the task contract (the same predicate as the guard and the scorer)."""
    return [r for r in all_resources if legal_fn(to_evidence(r), task.contract)]


# ----------------------------------------------------------------------------------------------- loop integration (DEV-26)
def patient_pairs(store, patient_fhir_id: str, patient_hash: str) -> list[tuple[Resource, EvidenceObject]]:
    """Every resource of a patient with its evidence object. A Condition without an occurrence time takes its encounter's
    period (the construction the skill validation used; `validate_tierc_skills.py` calls this function)."""
    rs = [r for (pp, _), lst in store.by_patient_type.items() if pp == patient_fhir_id for r in lst]
    enc = {str(r["id"]): ((r.get("period") or {}).get("start"), (r.get("period") or {}).get("end")) for r in rs if r.get("resourceType") == "Encounter"}
    pairs = []
    for r in rs:
        e = resource_to_evidence(r, patient_hash)
        if r.get("resourceType") == "Condition" and e.occurred is None:
            st, en = enc.get(str((r.get("encounter") or {}).get("reference", "")).split("/")[-1], (None, None))
            e = EvidenceObject(id=e.id, source=e.source, patient=e.patient, occurred=st, available=en or st, occurred_end=en, row_ref=e.row_ref, fields=e.fields)
        pairs.append((r, e))
    return pairs


def legal_set_with_references(store, pairs: list[tuple[Resource, EvidenceObject]], contract, patient_fhir_id: str) -> tuple[list[Resource], list[Resource]]:
    """(the resources inside the contract, the untimed references the skills need: Medications referenced by a legal
    request and the patient's own Patient resource) — the validator's construction."""
    legal_rs = [r for r, e in pairs if legal(e, contract)]
    needed = set()
    for r in legal_rs:
        ref = (r.get("medicationReference") or {}).get("reference")
        if isinstance(ref, str):
            needed.add(ref.split("/")[-1])
    extra = [store.by_type_id["Medication"][i] for i in needed if i in store.by_type_id.get("Medication", {})]
    pat = store.by_type_id.get("Patient", {}).get(patient_fhir_id)
    if pat is not None:
        extra.append(pat)
    return legal_rs, extra


def covered_templates(validation_path: Path | str) -> set[str]:
    """The templates the validation record marks covered (coverage rule of the Tier-C protocol §Skills)."""
    d = json.loads(Path(validation_path).read_text())
    return {t for t, v in d["templates"].items() if v.get("covered")}


def mapped_skills(validation_path: Path | str, patient_map: dict[str, str]) -> dict[str, Callable[[Any, Any], Result]]:
    """template -> skill(task_store, task) for every covered template (DEV-26). The skill solves the task over the
    patient's LEGAL resources exactly as the validator did and returns the result with the provenance resources attached
    under `_provenance_pairs` (resource, evidence); the loop records those as a skill access only when it replaces the
    model's answer, so the replacement's citations resolve to accessed, legal objects."""
    covered = covered_templates(validation_path)
    cache: dict[str, list[tuple[Resource, EvidenceObject]]] = {}

    def make(template: str):
        def skill(task_store, task) -> Result:
            pid = patient_map[task.patient]
            if pid not in cache:
                cache[pid] = patient_pairs(task_store.store, pid, task.patient)
            pairs = cache[pid]
            legal_rs, extra = legal_set_with_references(task_store.store, pairs, task.contract, pid)
            res = solve(task.metadata, legal_rs + extra)
            res["template"] = template
            if res.get("status") == "ok":
                ev = {e.id: e for _, e in pairs}
                prov = set(res.get("provenance", []))
                res["_provenance_pairs"] = [(r, ev.get(str(r.get("id")))) for r in legal_rs + extra if str(r.get("id")) in prov]
            return res
        return skill

    return {t: make(t) for t in sorted(covered)}
