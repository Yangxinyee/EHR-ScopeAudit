"""Structured Tier-B families (plan §5.3 items 2–3): outcome abstraction (post / full) and cohort
abstraction (at / full), gold via the oracle package, skill-covered."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from .. import oracle
from ..predicate import filter_legal
from ..schema import Case, Contract, EvidenceObject, Task, parse_time, stable_hex
from .sampling import LAB_ITEMS, Anchor
from .sources import SourceLoader, hash_patient

CODE_PREFIXES = {"I50": "heart failure", "N17": "acute kidney failure", "E11": "type 2 diabetes", "J44": "COPD", "I10": "essential hypertension",
                 "I48": "atrial fibrillation", "E78": "hyperlipidemia", "N18": "chronic kidney disease", "J18": "pneumonia", "A41": "sepsis"}
EXPOSURE_DRUGS = ["Heparin", "Insulin", "Metoprolol", "Furosemide", "Vancomycin", "Acetaminophen", "Warfarin", "Aspirin"]


def _iso(dt) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _t(s: str):
    return parse_time(s.replace(" ", "T") + ("Z" if len(s) == 19 else ""))


def _seeded_sample(items: list, n: int, salt: tuple) -> list:
    if len(items) <= n:
        return items
    return sorted(items, key=lambda e: stable_hex(*salt, e.id, n=16))[:n]


# ------------------------------------------------------------------ F2 outcome abstraction (post / full)
def build_outcome(loader: SourceLoader, anchor: Anchor, split: str) -> tuple[Case, list[Task]] | None:
    index_time = _t(anchor.payload["index_time"])
    hadm_ref = None
    transfers = loader.transfers(anchor.subject_id)
    admissions = loader.admissions(anchor.subject_id)
    adm = [e for e in admissions if e.occurred and _t(anchor.payload["index_time"]) == e.time("occurred")]
    if not adm:
        return None
    hadm_ref = adm[0].fields["hadm"]
    tr = [e for e in transfers if e.fields.get("hadm") == hadm_ref]
    ed = [e for e in loader.edstays(anchor.subject_id) if e.fields.get("hadm") == hadm_ref]
    emar = [e for e in loader.emar(anchor.subject_id) if e.fields.get("hadm") == hadm_ref]
    emar = _seeded_sample(emar, 24, ("outcome_emar", anchor.subject_id))
    pool = tr + adm + ed + emar   # only the index admission's records: the contract's permitted sources define the legal set exactly
    if len(pool) < 4 or len(pool) > 80:
        return None
    case_id = stable_hex("case", "outcome_abstraction", anchor.subject_id, *anchor.key, n=20)
    case = Case(case_id=case_id, patient=hash_patient(anchor.subject_id), index_event={"type": "admission", "index_time": _iso(index_time), "hadm": hadm_ref}, evidence=pool)
    specs = {
        "post": {"target": "first_careunit", "field": "careunit", "basis": "occurred", "drug": None,
                 "question": f"After the index admission at {_iso(index_time)}, what was the first care unit the patient was transferred to, and when? Cite the transfer record.",
                 "sources": ("transfers",)},
        "full": {"target": "discharge_disposition", "field": "discharge_location", "basis": "occurred", "drug": None,
                 "question": "What discharge disposition (discharge location) is recorded for the index hospital admission? Cite the admission record.",
                 "sources": ("admissions",)},
    }
    tasks = []
    for scope, spec in specs.items():
        contract = Contract(index_time=_iso(index_time), scope=scope, basis="available", permitted_sources=spec["sources"])
        legal = filter_legal(pool, contract)
        params = {"target": spec["target"], "field": spec["field"], "basis": spec["basis"], "drug": spec["drug"]}
        gold = oracle.OPERATIONS["outcome_lookup"](legal, contract, params)
        if gold["status"] != "ok" or gold.get("value") is None:
            return None
        tasks.append(Task(task_id=f"{case_id}-{scope}", case_id=case_id, patient=case.patient, tier="B", task_type="outcome_lookup_v1", question=spec["question"],
                          contract=contract, gold={"status": "ok", "value": gold["value"], "time": gold["time"], "citation": gold["citation"]},
                          required_ids=tuple(gold["provenance"]), split=split, leakage=False,
                          metadata={"family": "outcome_abstraction", "skill_params": params, "contract_extra": {"target": spec["target"], "index_admission": hadm_ref}}))
    return case, tasks


# ------------------------------------------------------------------ F3 cohort abstraction (at / full)
def build_cohort(loader: SourceLoader, anchor: Anchor, split: str) -> tuple[Case, list[Task]] | None:
    index_time = _t(anchor.payload["index_time"])
    labs = loader.labevents(anchor.subject_id, list(LAB_ITEMS))
    labs = [e for e in labs if e.time("available") is not None and e.fields.get("value") is not None and e.time("available") <= index_time + timedelta(days=60)]
    dx = loader.diagnoses(anchor.subject_id)
    dx = [e for e in dx if e.time("available") is not None and e.time("available") <= index_time + timedelta(days=60)]
    emar = [e for e in loader.emar(anchor.subject_id) if e.time("occurred") is not None and e.time("occurred") <= index_time + timedelta(days=60)]
    triage = loader.triage(anchor.subject_id)
    labs = _seeded_sample(labs, 36, ("cohort_labs", anchor.subject_id))
    emar = _seeded_sample(emar, 20, ("cohort_emar", anchor.subject_id))
    dx = _seeded_sample(dx, 20, ("cohort_dx", anchor.subject_id))
    triage = _seeded_sample(triage, 4, ("cohort_triage", anchor.subject_id))
    pool = labs + dx + emar + triage
    if len(pool) < 8 or len(pool) > 90:
        return None
    # criteria chosen from what exists in the pool so that the "at" task is decidable
    at_contract = Contract(index_time=_iso(index_time), scope="at", basis="available", permitted_sources=None)
    at_legal = filter_legal(pool, at_contract)
    concepts = sorted({e.fields["concept"] for e in at_legal if e.source == "labevents"})
    codes = sorted({p for p in CODE_PREFIXES for e in dx if str(e.fields.get("code", "")).upper().startswith(p)})
    drugs = sorted({d for d in EXPOSURE_DRUGS for e in emar if str(e.fields.get("medication", "")).strip().lower() == d.lower()})
    if not concepts:
        return None
    h = int(stable_hex("cohort", anchor.subject_id, *anchor.key, n=12), 16)
    concept = concepts[h % len(concepts)]
    values = sorted(float(e.fields["value"]) for e in at_legal if e.source == "labevents" and e.fields["concept"] == concept)
    threshold = values[len(values) // 2] if values else 0.0
    criteria: list[dict[str, Any]] = [{"kind": "lab_threshold", "concept": concept, "op": [">=", "<="][h % 2], "threshold": threshold, "basis": "available"}]
    if codes:
        criteria.append({"kind": "code_present", "prefix": codes[(h // 2) % len(codes)]})
    elif drugs:
        criteria.append({"kind": "exposure", "drug": drugs[(h // 2) % len(drugs)]})
    else:
        criteria.append({"kind": "code_present", "prefix": list(CODE_PREFIXES)[(h // 2) % len(CODE_PREFIXES)]})
    case_id = stable_hex("case", "cohort_abstraction", anchor.subject_id, *anchor.key, n=20)
    case = Case(case_id=case_id, patient=hash_patient(anchor.subject_id), index_event={"type": "admission", "index_time": _iso(index_time)}, evidence=pool)
    text = "; ".join(_criterion_text(c) for c in criteria)
    question = (f"Using only evidence legal for the contract, decide whether the patient meets ALL of the following criteria relative to the index admission at {_iso(index_time)}: {text}. "
                f"Answer include, exclude, or uncertain, citing the evidence that establishes each satisfied criterion (or the decisive failed one).")
    tasks = []
    golds = {}
    for scope in ("at", "full"):
        contract = Contract(index_time=_iso(index_time), scope=scope, basis="available", permitted_sources=None)
        legal = filter_legal(pool, contract)
        gold = oracle.OPERATIONS["cohort_criteria"](legal, contract, {"criteria": criteria})
        golds[scope] = gold["decision"]
        tasks.append(Task(task_id=f"{case_id}-{scope}", case_id=case_id, patient=case.patient, tier="B", task_type="cohort_v1", question=question,
                          contract=contract, gold={"decision": gold["decision"], "citations": gold["provenance"]}, required_ids=tuple(gold["provenance"]), split=split,
                          metadata={"family": "cohort_abstraction", "criteria": criteria, "skill_params": {"criteria": criteria}}))
    if any(d == "uncertain" for d in golds.values()):
        return None   # an uncertain gold is an abstention and cannot be scored; sample another anchor
    leak = golds["at"] != golds["full"]
    for t in tasks:
        t.leakage = leak
    return case, tasks


def _criterion_text(c: dict[str, Any]) -> str:
    if c["kind"] == "lab_threshold":
        return f"any {c['concept']} result {c['op']} {c['threshold']}"
    if c["kind"] == "code_present":
        return f"a coded diagnosis starting with ICD prefix {c['prefix']} ({CODE_PREFIXES.get(c['prefix'], '')})"
    if c["kind"] == "exposure":
        return f"an administration of {c['drug']}"
    if c["kind"] == "acuity":
        return f"an ED triage acuity of {c['max_acuity']} or lower"
    return str(c)
