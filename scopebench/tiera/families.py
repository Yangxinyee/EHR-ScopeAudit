"""Tier-A family builders (plan §7.2): anchor -> Case + four Tasks, one per scope (pre / at / post / full), the same six
task types as Tier-B re-targeted to the CCDA sources, gold via the frozen oracle package.

Conventions shared with Tier-B: every task has a substantive, non-empty R_t; a builder returns None when the anchor
cannot yield four well-formed tasks (the sampler oversamples); the pool is the case's candidate evidence (what
prompt-only sees, what scoped retrieval filters). Tier-A specifics: imaging is often outpatient, so the at-index window
is ±7 days (`AT_HOURS`) for structured families and ±3 days for report families; CCDA reports carry a single timestamp,
so `available == occurred` for narrative objects; medication names are the export's own strings (the oracle matches
exactly). Leakage (plan §5.4): the pre-index gold differs from the full-chart gold (retrieval: the pre answer changes
when availability is replaced by occurrence time; cohort: pre decision ≠ full decision; reconstruction: always, the
index report is an illegal distractor under pre).
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import timedelta
from typing import Any

from .. import oracle
from .. import tasktypes_tiera  # noqa: F401  (registers outcome_count_v1)
from ..predicate import filter_legal
from ..schema import Case, Contract, EvidenceObject, Task, parse_time, stable_hex
from ..tierb.families_notes import ENUM, NEGATIVE_COMPARISON, first_content_line, note_lines, section_lines  # helpers only (frozen module, read-only)
from ..tierb.families_structured import CODE_PREFIXES, _criterion_text
from .loader import TierALoader
from .sampling import AnchorA

AT_HOURS = 168.0          # structured families: at-index = ±7 days
AT_HOURS_REPORTS = 72.0   # report families: ±3 days
OUTCOME_WINDOW_HOURS = 720.0   # outcome family: administrations counted in the 30 days before / after the index
REPORT_POOL_CAP = 2000         # report families: echocardiography reports run to ~120 lines, so the line pool is capped above Tier-B's 600
SUMMARY_POOL_CAP = 2500
SUMMARY_AT_HOURS = 168.0       # summarization at-index window ±7 days (reports are sparse around outpatient imaging)
SUMMARY_MAX_EXAMS = 12
RETRIEVAL_WINDOW_DAYS = 180    # retrieval family: the analyte's results within ±180 days of the index
SCOPES = ("pre", "at", "post", "full")


def _iso(dt) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _case_id(family: str, anchor: AnchorA, patient: str) -> str:
    return stable_hex("case", "A", family, anchor.modality, anchor.cohort, patient, anchor.index_time, n=20)


def _h(*parts: Any, n: int = 12) -> int:
    return int(stable_hex("tiera", *parts, n=n), 16)


def _seeded_sample(items: list[EvidenceObject], n: int, salt: tuple) -> list[EvidenceObject]:
    if len(items) <= n:
        return items
    return sorted(items, key=lambda e: stable_hex(*salt, e.id, n=16))[:n]


def _index_event(anchor: AnchorA, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    d = {"type": f"{anchor.modality}_exam", "modality": anchor.modality, "index_time": anchor.index_time}
    if extra:
        d.update(extra)
    return d


def _contract(anchor: AnchorA, scope: str, sources: tuple[str, ...] | None, at_hours: float, basis: str = "available", **kw) -> Contract:
    if scope == "at":
        return Contract(index_time=anchor.index_time, scope="at", basis=basis, permitted_sources=sources, at_tolerance_hours=at_hours, window_hours=at_hours, **kw)
    return Contract(index_time=anchor.index_time, scope=scope, basis=basis, permitted_sources=sources, **kw)


def _task(case: Case, anchor: AnchorA, family: str, task_type: str, scope: str, question: str, contract: Contract, gold: dict, required: tuple[str, ...],
          split: str, metadata: dict[str, Any], leakage: bool = False) -> Task:
    return Task(task_id=f"{case.case_id}-{scope}", case_id=case.case_id, patient=case.patient, tier="A", task_type=task_type, question=question,
                contract=contract, gold=gold, required_ids=required, split=split, leakage=leakage,
                metadata={"family": family, "modality": anchor.modality, "cohort": anchor.cohort, **metadata})


# ------------------------------------------------------------------ F1 evidence retrieval (lab_trend_k3; pre / at / post / full)
def build_retrieval(loader: TierALoader, anchor: AnchorA, split: str) -> tuple[Case, list[Task]] | None:
    index_time = parse_time(anchor.index_time)
    labs = [e for e in loader.labevents(anchor.mrn) if e.time("available") is not None and e.fields.get("value") is not None]
    lo, hi = index_time - timedelta(days=RETRIEVAL_WINDOW_DAYS), index_time + timedelta(days=RETRIEVAL_WINDOW_DAYS)
    near = [e for e in labs if lo <= e.time("available") <= hi]
    counts = Counter(e.fields["concept"] for e in near if e.fields.get("concept"))
    eligible = sorted([c for c, n in counts.items() if 6 <= n <= 100], key=lambda c: (-counts[c], c))[:5]
    if not eligible:
        return None
    patient = loader.patient(anchor.mrn)
    concept = eligible[_h("concept", patient, anchor.index_time) % len(eligible)]
    pool = [e for e in near if e.fields["concept"] == concept]
    unit = next((e.fields.get("unit") for e in pool if e.fields.get("unit")), None)
    case = Case(case_id=_case_id("evidence_retrieval", anchor, patient), patient=patient, index_event=_index_event(anchor), evidence=pool)
    phrasing = {"pre": "nearest to but strictly before", "post": "nearest to but strictly after", "at": "nearest to (within the at-index window around)", "full": "nearest to (on either side of)"}
    tasks = []
    for scope in SCOPES:
        contract = _contract(anchor, scope, ("labevents",), AT_HOURS)
        gold = oracle.OPERATIONS["lab_trend_k3"](filter_legal(pool, contract), contract, {"k": 3})
        if gold["status"] != "ok":
            return None
        question = (f"Return the three {concept} results whose results became available {phrasing[scope]} the index exam at {anchor.index_time}. "
                    f"Order them chronologically, then report the last-minus-first numeric change and whether it is increasing, decreasing, or stable.")
        tasks.append(_task(case, anchor, "evidence_retrieval", "lab_trend_k3", scope, question, contract,
                           {"status": "ok", "results": gold["provenance"], "delta": gold["delta_last_minus_first"], "direction": gold["direction"]},
                           tuple(gold["ordered_result_ids"]), split, {"concept": concept, "unit": unit, "skill_params": {"k": 3}}))
    wb = Contract(index_time=anchor.index_time, scope="pre", basis="occurred", permitted_sources=("labevents",))
    wb_gold = oracle.OPERATIONS["lab_trend_k3"](filter_legal(pool, wb), wb, {"k": 3})
    leak = wb_gold["status"] == "ok" and tuple(wb_gold.get("ordered_result_ids") or ()) != tuple(tasks[0].required_ids)
    for t in tasks:
        t.leakage = bool(leak)
    return case, tasks


# ------------------------------------------------------------------ F2 outcome abstraction (outcome_lookup; pre / at / post / full)
def build_outcome(loader: TierALoader, anchor: AnchorA, split: str) -> tuple[Case, list[Task]] | None:
    index_time = parse_time(anchor.index_time)
    patient = loader.patient(anchor.mrn)
    lo, hi = index_time - timedelta(days=30), index_time + timedelta(days=30)
    emar_all = [e for e in loader.emar(anchor.mrn) if e.time("occurred") is not None and lo <= e.time("occurred") <= hi]
    transfers = [e for e in loader.transfers(anchor.mrn) if e.time("occurred") is not None and lo <= e.time("occurred") <= hi]
    admissions = loader.admissions(anchor.mrn)
    index_adm = [e for e in admissions if e.time("occurred") is not None and e.time("occurred") <= index_time + timedelta(hours=AT_HOURS)
                 and (e.occurred_end is None or parse_time(e.occurred_end) >= index_time - timedelta(hours=AT_HOURS))]
    index_adm = sorted(index_adm, key=lambda e: abs((e.time("occurred") - index_time).total_seconds()))[:1]
    # the drug: the most frequent administration in the pre-window with at least one in every window we ask about
    def n_in(drug: str, contract: Contract) -> int:
        return sum(1 for e in filter_legal(emar_all, contract) if e.fields.get("medication") == drug)
    pre_c = _contract(anchor, "pre", ("emar",), AT_HOURS, window_hours=OUTCOME_WINDOW_HOURS)
    at_c = _contract(anchor, "at", ("emar",), 24.0)
    post_c = _contract(anchor, "post", ("emar",), AT_HOURS, window_hours=OUTCOME_WINDOW_HOURS)
    full_c = _contract(anchor, "full", ("emar",), AT_HOURS)
    freq = Counter(e.fields["medication"] for e in filter_legal(emar_all, pre_c) if e.fields.get("medication"))
    drugs = [d for d, _ in freq.most_common(6) if n_in(d, at_c) >= 1 and n_in(d, post_c) >= 1]
    if not drugs:
        return None
    drug = drugs[_h("drug", patient, anchor.index_time) % len(drugs)]
    emar = [e for e in emar_all if e.fields.get("medication") == drug]
    others = _seeded_sample([e for e in emar_all if e.fields.get("medication") != drug], 16, ("outcome_emar", patient))
    pool = emar + others + transfers + index_adm
    if len(pool) < 4 or len(pool) > 120:
        return None
    case = Case(case_id=_case_id("outcome_abstraction", anchor, patient), patient=patient, index_event=_index_event(anchor), evidence=pool)
    specs: dict[str, dict[str, Any]] = {
        "pre": {"contract": pre_c, "params": {"target": "administration_count", "field": None, "basis": "occurred", "drug": drug},
                "question": f"How many administrations of \"{drug}\" are recorded in the 30 days before the index exam at {anchor.index_time}? Cite every administration counted."},
        "at": {"contract": at_c, "params": {"target": "administration_count", "field": None, "basis": "occurred", "drug": drug},
               "question": f"How many administrations of \"{drug}\" are recorded within 24 hours of the index exam at {anchor.index_time}? Cite every administration counted."},
    }
    post_tr = [e for e in transfers if e.time("occurred") > index_time]
    if post_tr:
        specs["post"] = {"contract": _contract(anchor, "post", ("transfers",), AT_HOURS, window_hours=24.0 * 30), "params": {"target": "first_careunit", "field": "careunit", "basis": "occurred", "drug": None},
                         "question": f"After the index exam at {anchor.index_time}, what was the first unit or department the patient was transferred to, and when? Cite the transfer record."}
    else:
        specs["post"] = {"contract": post_c, "params": {"target": "administration_count", "field": None, "basis": "occurred", "drug": drug},
                         "question": f"How many administrations of \"{drug}\" are recorded in the 30 days after the index exam at {anchor.index_time}? Cite every administration counted."}
    if index_adm and index_adm[0].fields.get("admission_type"):
        specs["full"] = {"contract": _contract(anchor, "full", ("admissions",), AT_HOURS), "params": {"target": "discharge_disposition", "field": "admission_type", "basis": "occurred", "drug": None},
                         "question": "What hospital visit type is recorded for the encounter that contains the index exam? Cite the encounter record."}
    else:
        specs["full"] = {"contract": full_c, "params": {"target": "administration_count", "field": None, "basis": "occurred", "drug": drug},
                         "question": f"How many administrations of \"{drug}\" are recorded in the whole chart (all legal evidence)? Cite every administration counted."}
    tasks = []
    for scope in SCOPES:
        spec = specs[scope]
        gold = oracle.OPERATIONS["outcome_lookup"](filter_legal(pool, spec["contract"]), spec["contract"], spec["params"])
        if gold["status"] != "ok" or gold.get("value") is None or (spec["params"]["target"] == "administration_count" and gold["value"] < 1):
            return None
        count = spec["params"]["target"] == "administration_count"
        tt_name = "outcome_count_v1" if count else "outcome_lookup_v1"   # count targets cite every counted row (tasktypes_tiera)
        gold_answer = ({"status": "ok", "value": gold["value"], "citations": list(gold["provenance"])} if count
                       else {"status": "ok", "value": gold["value"], "time": gold["time"], "citation": gold["citation"]})
        tasks.append(_task(case, anchor, "outcome_abstraction", tt_name, scope, spec["question"], spec["contract"],
                           gold_answer, tuple(gold["provenance"]), split,
                           {"skill_params": spec["params"], "contract_extra": {"target": spec["params"]["target"], "drug": drug}}))
    return case, tasks


# ------------------------------------------------------------------ F3 cohort abstraction (cohort_criteria; pre / at / post / full)
def build_cohort(loader: TierALoader, anchor: AnchorA, split: str) -> tuple[Case, list[Task]] | None:
    index_time = parse_time(anchor.index_time)
    patient = loader.patient(anchor.mrn)
    hi = index_time + timedelta(days=60)
    labs = [e for e in loader.labevents(anchor.mrn) if e.time("available") is not None and e.fields.get("value") is not None and index_time - timedelta(days=120) <= e.time("available") <= hi]
    dx = [e for e in loader.diagnoses(anchor.mrn) if e.time("available") is not None and e.time("available") <= hi]
    emar = [e for e in loader.emar(anchor.mrn) if e.time("occurred") is not None and index_time - timedelta(days=120) <= e.time("occurred") <= hi]
    labs = _seeded_sample(labs, 36, ("cohort_labs", patient)); emar = _seeded_sample(emar, 20, ("cohort_emar", patient)); dx = _seeded_sample(dx, 20, ("cohort_dx", patient))
    pool = labs + dx + emar
    if len(pool) < 8 or len(pool) > 90:
        return None
    at_c = _contract(anchor, "at", None, AT_HOURS)
    at_legal = filter_legal(pool, at_c)
    concepts = sorted({e.fields["concept"] for e in at_legal if e.source == "labevents" and e.fields.get("concept")})
    if not concepts:
        return None
    h = _h("cohort", patient, anchor.index_time)
    concept = concepts[h % len(concepts)]
    values = sorted(float(e.fields["value"]) for e in at_legal if e.source == "labevents" and e.fields["concept"] == concept)
    threshold = values[len(values) // 2] if values else 0.0
    criteria: list[dict[str, Any]] = [{"kind": "lab_threshold", "concept": concept, "op": [">=", "<="][h % 2], "threshold": threshold, "basis": "available"}]
    codes = sorted({p for p in CODE_PREFIXES for e in dx if str(e.fields.get("code", "")).upper().startswith(p)})
    drugs = sorted({e.fields["medication"] for e in emar if e.fields.get("medication")})
    if codes:
        criteria.append({"kind": "code_present", "prefix": codes[(h // 2) % len(codes)]})
    elif drugs:
        criteria.append({"kind": "exposure", "drug": drugs[(h // 2) % len(drugs)]})
    else:
        criteria.append({"kind": "code_present", "prefix": list(CODE_PREFIXES)[(h // 2) % len(CODE_PREFIXES)]})
    case = Case(case_id=_case_id("cohort_abstraction", anchor, patient), patient=patient, index_event=_index_event(anchor), evidence=pool)
    text = "; ".join(_criterion_text(c) for c in criteria)
    question = (f"Using only evidence legal for the contract, decide whether the patient meets ALL of the following criteria relative to the index exam at {anchor.index_time}: {text}. "
                f"Answer include, exclude, or uncertain, citing the evidence that establishes each satisfied criterion (or the decisive failed one).")
    tasks, golds = [], {}
    for scope in SCOPES:
        contract = _contract(anchor, scope, None, AT_HOURS)
        gold = oracle.OPERATIONS["cohort_criteria"](filter_legal(pool, contract), contract, {"criteria": criteria})
        golds[scope] = gold["decision"]
        tasks.append(_task(case, anchor, "cohort_abstraction", "cohort_v1", scope, question, contract, {"decision": gold["decision"], "citations": gold["provenance"]},
                           tuple(gold["provenance"]), split, {"criteria": criteria, "skill_params": {"criteria": criteria}}))
    if any(d == "uncertain" for d in golds.values()) or any(not t.required_ids for t in tasks):
        return None
    leak = golds["pre"] != golds["full"]
    for t in tasks:
        t.leakage = leak
    return case, tasks


# ------------------------------------------------------------------ report helpers
IMPRESSION_HEADERS = ("IMPRESSIONS?", "CONCLUSIONS?", "SUMMARY RESULTS", "SUMMARY", "INTERPRETATION SUMMARY", "INTERPRETATION")
_HEADER_LINE = re.compile(r"^\s*[A-Z][A-Z /&-]{2,40}:\s*|^\s*[A-Z][A-Z /&-]{2,30}\s*$")   # 'HEADER:' or a bare all-caps line


def _impression_lines(lines: list[EvidenceObject]) -> list[EvidenceObject]:
    """The conclusion section of one report: by section label when the export carries one, else the lines after the first
    header among IMPRESSION_HEADERS (radiology: IMPRESSION; echocardiography: SUMMARY / INTERPRETATION; catheterisation:
    SUMMARY RESULTS / IMPRESSIONS) up to the next header line."""
    imp = section_lines(lines, "IMPRESSION")
    if imp:
        return imp
    if lines and all(e.fields.get("kind") == "cath_findings" for e in lines):
        return lines[:12]      # catheterisation findings sheets: the findings text is the report's conclusion section
    for hdr in IMPRESSION_HEADERS:
        start = re.compile(rf"^\s*{hdr}\s*(:|$)", re.I)
        started = False; out = []
        for e in lines:
            t = e.fields["text"].strip()
            if not started and start.match(t):
                started = True
                rest = start.sub("", t).strip()
                if len(rest.split()) >= 3:
                    out.append(e)
                continue
            if started:
                if _HEADER_LINE.match(t) and not start.match(t):
                    break
                out.append(e)
        if out:
            return out
    return section_lines(lines, "FINDINGS")[:12] if any(e.fields.get("section") == "FINDINGS" for e in lines) else []


def _reports(pool: list[EvidenceObject]) -> dict[str, list[EvidenceObject]]:
    return {n: note_lines(pool, n) for n in sorted({e.fields["note"] for e in pool})}


REPORT_SOURCES = ("radiology_line", "echo_line", "cath_line")


def _report_pool(loader: TierALoader, anchor: AnchorA, days: int, sources=REPORT_SOURCES) -> list[EvidenceObject]:
    index_time = parse_time(anchor.index_time)
    lines = loader.narrative_lines(anchor.mrn, sources)
    lo, hi = index_time - timedelta(days=days), index_time + timedelta(days=days)
    return [e for e in lines if e.time("available") is not None and lo <= e.time("available") <= hi]


def _neighbour_reports(reports: dict[str, list[EvidenceObject]], index_note: str, index_time, need_impression: bool) -> tuple[str | None, str | None]:
    """(latest report strictly before the index, first report strictly after), by availability of their first line."""
    prior, later = [], []
    for n, lines in reports.items():
        if n == index_note or not lines:
            continue
        if need_impression and not _impression_lines(lines):
            continue
        t = lines[0].time("available")
        if t is None:
            continue
        (prior if t < index_time else later if t > index_time else []).append((t, n))
    p = max(prior)[1] if prior else None
    q = min(later)[1] if later else None
    return p, q


# ------------------------------------------------------------------ F4 claim verification (claim_v1; at / full / pre / post)
def build_claim(loader: TierALoader, anchor: AnchorA, split: str) -> tuple[Case, list[Task]] | None:
    index_time = parse_time(anchor.index_time)
    patient = loader.patient(anchor.mrn)
    pool = _report_pool(loader, anchor, 365)
    if not pool or len(pool) > REPORT_POOL_CAP:
        return None
    idx_lines, idx_kind = loader.index_report_lines(anchor.mrn, anchor.modality, anchor.index_time, pool)
    if len(idx_lines) < 3:
        return None
    index_note = idx_lines[0].fields["note"]
    reports = _reports(pool)
    prior, later = _neighbour_reports(reports, index_note, index_time, need_impression=False)
    if prior is None or later is None:
        return None
    exam_names = sorted({e.fields.get("exam_name") for e in pool if e.fields.get("exam_name")})
    if len(exam_names) < 2:
        return None
    h = _h("claim", patient, anchor.index_time)
    case = Case(case_id=_case_id("claim_verification", anchor, patient), patient=patient,
                index_event=_index_event(anchor, {"note": index_note, "exam_name": idx_lines[0].fields.get("exam_name"), "index_report_kind": idx_kind}), evidence=pool)
    idx_ids = tuple(e.id for e in idx_lines)

    def claim_for(lines: list[EvidenceObject], which: str, choice: int) -> tuple[str, str, list[EvidenceObject], str] | None:
        exam = lines[0].fields.get("exam_name")
        if not exam:
            return None
        comp = next((e for e in lines if re.match(r"^\s*COMPARISON\b", e.fields["text"], re.I)), None)
        if choice == 2 and comp is not None:
            body = comp.fields["text"].split(":", 1)[-1]
            label = "contradicted" if NEGATIVE_COMPARISON.search(body) or not body.strip() else "supported"
            return f"{which} states that a prior study was available for comparison.", label, [comp], "comparison"
        if choice == 1:
            other = [x for x in exam_names if x != exam]
            swapped = other[_h("swap", patient, which) % len(other)]
            return f"{which} documents a {swapped} examination.", "contradicted", [lines[0]], "exam_contradicted"
        return f"{which} documents a {exam} examination.", "supported", [lines[0]], "exam_supported"

    targets = {
        "at": (idx_lines, "The index report"), "full": (idx_lines, "The index report"),
        "pre": (reports[prior], f"The most recent report that became available strictly before the index time {anchor.index_time}"),
        "post": (reports[later], f"The first report that became available strictly after the index time {anchor.index_time}"),
    }
    tasks = []
    for i, scope in enumerate(SCOPES):
        lines, which = targets[scope]
        made = claim_for(lines, which, (h + i) % 3)
        if made is None:
            return None
        claim, label, deciding, template = made
        contract = _contract(anchor, scope, REPORT_SOURCES, AT_HOURS_REPORTS, index_evidence_ids=idx_ids if scope in ("at", "full") else ())
        legal_ids = {e.id for e in filter_legal(pool, contract)}
        if any(e.id not in legal_ids for e in deciding):
            return None
        question = (f"Evaluate the claim against the record using only evidence legal for the contract (index exam {idx_lines[0].fields.get('exam_name')} at {anchor.index_time}). "
                    f"Cite the report line(s) that decide the label.")
        tasks.append(_task(case, anchor, "claim_verification", "claim_v1", scope, question, contract, {"label": label, "line_ids": [e.id for e in deciding]},
                           tuple(e.id for e in deciding), split, {"claim": claim, "template": template, "exam_name": lines[0].fields.get("exam_name"),
                                                                    "target_note": lines[0].fields["note"], "index_report_kind": idx_kind, "contract_extra": {"index_report_note": index_note}}))
    return case, tasks


# ------------------------------------------------------------------ F5 report reconstruction (reconstruct_v1; at / pre / post / full)
def build_reconstruction(loader: TierALoader, anchor: AnchorA, split: str) -> tuple[Case, list[Task]] | None:
    index_time = parse_time(anchor.index_time)
    patient = loader.patient(anchor.mrn)
    pool = _report_pool(loader, anchor, 365)
    if not pool or len(pool) > REPORT_POOL_CAP:
        return None
    idx_lines, idx_kind = loader.index_report_lines(anchor.mrn, anchor.modality, anchor.index_time, pool)
    if not idx_lines:
        return None
    idx_imp = _impression_lines(idx_lines)
    if not idx_imp or len(idx_imp) > 12:
        return None
    index_note = idx_lines[0].fields["note"]
    reports = _reports(pool)
    prior, later = _neighbour_reports(reports, index_note, index_time, need_impression=True)
    if prior is None or later is None:
        return None
    p_imp, q_imp = _impression_lines(reports[prior]), _impression_lines(reports[later])
    if len(p_imp) > 12 or len(q_imp) > 12:
        return None
    exam = idx_lines[0].fields.get("exam_name")
    case = Case(case_id=_case_id("report_reconstruction", anchor, patient), patient=patient, index_event=_index_event(anchor, {"note": index_note, "exam_name": exam, "index_report_kind": idx_kind}), evidence=pool)
    idx_ids = tuple(e.id for e in idx_lines)
    targets = {
        "at": (idx_imp, index_note, f"Reconstruct the IMPRESSION section of the index report ({exam}, indexed at {anchor.index_time}): return its lines in order, each with its line id and text copied verbatim."),
        "full": (idx_imp, index_note, f"Reconstruct the IMPRESSION section of the index report ({exam}, indexed at {anchor.index_time}) using any legal evidence: return its lines in order, each with its line id and text copied verbatim."),
        "pre": (p_imp, prior, f"Reconstruct the IMPRESSION section of the most recent report that became available strictly before the index time {anchor.index_time}: return its lines in order, each with its line id and text copied verbatim."),
        "post": (q_imp, later, f"Reconstruct the IMPRESSION section of the first report that became available strictly after the index time {anchor.index_time}: return its lines in order, each with its line id and text copied verbatim."),
    }
    tasks = []
    for scope in SCOPES:
        imp, target_note, question = targets[scope]
        contract = _contract(anchor, scope, REPORT_SOURCES, AT_HOURS_REPORTS, index_evidence_ids=idx_ids if scope in ("at", "full") else ())
        legal_ids = {e.id for e in filter_legal(pool, contract)}
        if any(e.id not in legal_ids for e in imp):
            return None
        gold = {"status": "ok", "line_ids": [e.id for e in imp], "text": " ".join(e.fields["text"] for e in imp)}
        tasks.append(_task(case, anchor, "report_reconstruction", "reconstruct_v1", scope, question, contract, gold, tuple(gold["line_ids"]), split,
                           {"exam_name": exam, "target_note": target_note, "n_impression_lines": len(imp), "n_notes_in_pool": len(reports), "index_report_kind": idx_kind,
                            "contract_extra": {"index_report_note": index_note}}, leakage=True))
    return case, tasks


# ------------------------------------------------------------------ F6 exam summarization (summarize_exams_v1; pre / at / post / full)
def build_summarization(loader: TierALoader, anchor: AnchorA, split: str) -> tuple[Case, list[Task]] | None:
    patient = loader.patient(anchor.mrn)
    pool = _report_pool(loader, anchor, 730)
    if not pool or len(pool) > SUMMARY_POOL_CAP:
        return None
    case = Case(case_id=_case_id("summarization", anchor, patient), patient=patient, index_event=_index_event(anchor), evidence=pool)
    phrasing = {"pre": "became available strictly before", "post": "became available strictly after", "at": "became available within the at-index window (±7 days) around", "full": "is present in the record around"}
    tasks = []
    for scope in SCOPES:
        contract = _contract(anchor, scope, REPORT_SOURCES, SUMMARY_AT_HOURS)
        legal = filter_legal(pool, contract)
        exams = []
        for n, lines in _reports(legal).items():
            imp = _impression_lines(lines)
            line = first_content_line(imp) if imp else None
            if line is None:
                continue
            exams.append({"exam_name": lines[0].fields.get("exam_name"), "time": line.available, "first_impression_line": line.fields["text"], "citation": line.id})
        if not exams or len(exams) > SUMMARY_MAX_EXAMS:
            return None
        exams.sort(key=lambda x: (x["time"], x["citation"]))
        question = (f"List every imaging or procedure report that {phrasing[scope]} {anchor.index_time}: give the exam name, the report's availability time, "
                    f"and the first substantive line of its IMPRESSION section copied verbatim, citing that line.")
        tasks.append(_task(case, anchor, "summarization", "summarize_exams_v1", scope, question, contract, {"status": "ok", "exams": exams},
                           tuple(x["citation"] for x in exams), split, {"n_exams": len(exams)}))
    return case, tasks


BUILDERS = {
    "evidence_retrieval": build_retrieval,
    "outcome_abstraction": build_outcome,
    "cohort_abstraction": build_cohort,
    "claim_verification": build_claim,
    "report_reconstruction": build_reconstruction,
    "summarization": build_summarization,
}
