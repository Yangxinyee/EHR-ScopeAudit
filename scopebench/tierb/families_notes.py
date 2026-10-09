"""Families over MIMIC-IV-Note radiology reports (plan §5.3 items 4–6): claim verification (at / full),
report reconstruction (at / pre), exam summarization (pre / post). Every task has a substantive gold:
under `pre`, the reconstruction target is the latest report available before the index, and the index
report is an illegal distractor (the leakage-decidable case)."""

from __future__ import annotations

import re
from datetime import timedelta

from ..predicate import filter_legal
from ..schema import Case, Contract, EvidenceObject, Task, parse_time, stable_hex
from .sampling import Anchor
from .sources import SourceLoader, eid, hash_patient

NEGATIVE_COMPARISON = re.compile(r"\b(none|no prior|not available|no comparison|no previous|unavailable)\b", re.I)
ENUM = re.compile(r"^\s*(\(?\d+[.)]|[-*•])\s*")


def _iso(dt) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _t(s: str):
    return parse_time(s.replace(" ", "T") + ("Z" if len(s) == 19 else ""))


def note_lines(pool: list[EvidenceObject], note_ref: str) -> list[EvidenceObject]:
    return sorted((e for e in pool if e.fields.get("note") == note_ref), key=lambda e: int(e.fields["line"]))


def section_lines(lines: list[EvidenceObject], section: str) -> list[EvidenceObject]:
    """Lines of a section, excluding the bare header line."""
    out = [e for e in lines if e.fields.get("section") == section]
    if out and out[0].fields["text"].rstrip(":").strip().upper() == section.upper():
        out = out[1:]
    return out


def first_content_line(lines: list[EvidenceObject]) -> EvidenceObject | None:
    """First line that carries at least three words after stripping a list enumerator ('1.', '-', ...)."""
    for e in lines:
        body = ENUM.sub("", e.fields["text"]).strip()
        if len(body.split()) >= 3:
            return e
    return None


def radiology_pool(loader: SourceLoader, subject_id: int, index_time, days: int = 365) -> list[EvidenceObject]:
    lines = loader.radiology_lines(subject_id)
    lo, hi = index_time - timedelta(days=days), index_time + timedelta(days=days)
    return [e for e in lines if e.time("available") is not None and lo <= e.time("available") <= hi]


def notes_with_impression(pool: list[EvidenceObject]) -> dict[str, list[EvidenceObject]]:
    out = {}
    for n in sorted({e.fields["note"] for e in pool}):
        imp = section_lines(note_lines(pool, n), "IMPRESSION")
        if imp:
            out[n] = imp
    return out


# ------------------------------------------------------------------ F4 claim verification (at / full)
def build_claim(loader: SourceLoader, anchor: Anchor, split: str) -> tuple[Case, list[Task]] | None:
    index_time = _t(anchor.payload["index_time"])
    note_ref = eid("rad_note", anchor.payload["note_id"])
    pool = radiology_pool(loader, anchor.subject_id, index_time)
    idx_lines = note_lines(pool, note_ref)
    if len(idx_lines) < 4 or len(pool) > 600:
        return None
    exam_name = anchor.payload.get("exam_name") or idx_lines[0].fields.get("exam_name")
    if not exam_name:
        return None
    choice = int(stable_hex("claim", anchor.subject_id, *anchor.key, n=8), 16) % 3
    exam_line = next((e for e in idx_lines if e.fields.get("section") in ("EXAMINATION", "EXAM") or e.fields["text"].upper().startswith("EXAMINATION")), None)
    comp_line = next((e for e in idx_lines if e.fields.get("section") == "COMPARISON"), None)
    if choice == 0 and exam_line is not None:
        claim, label, deciding, template = f"The index report documents a {exam_name} examination.", "supported", [exam_line], "exam_supported"
    elif choice == 1 and exam_line is not None:
        other = sorted({e.fields.get("exam_name") for e in pool if e.fields.get("exam_name") and e.fields.get("exam_name") != exam_name})
        if not other:
            return None
        swapped = other[int(stable_hex("swap", anchor.subject_id, *anchor.key, n=8), 16) % len(other)]
        claim, label, deciding, template = f"The index report documents a {swapped} examination.", "contradicted", [exam_line], "exam_contradicted"
    else:
        if comp_line is None:
            return None
        body = comp_line.fields["text"].split(":", 1)[-1]
        claim = "The index report states that a prior study was available for comparison."
        label = "contradicted" if NEGATIVE_COMPARISON.search(body) or not body.strip() else "supported"
        deciding, template = [comp_line], "comparison"
    case_id = stable_hex("case", "claim_verification", anchor.subject_id, *anchor.key, n=20)
    case = Case(case_id=case_id, patient=hash_patient(anchor.subject_id), index_event={"type": "radiology_exam", "index_time": _iso(index_time), "note": note_ref, "exam_name": exam_name}, evidence=pool)
    idx_ids = tuple(e.id for e in idx_lines)
    question = f"Evaluate the claim against the index report ({exam_name}, indexed at {_iso(index_time)}) using only evidence legal for the contract. Cite the report line(s) that decide the label."
    tasks = []
    for scope in ("at", "full"):
        contract = Contract(index_time=_iso(index_time), scope=scope, basis="available", permitted_sources=("radiology_line",), index_evidence_ids=idx_ids if scope == "at" else ())
        tasks.append(Task(task_id=f"{case_id}-{scope}", case_id=case_id, patient=case.patient, tier="B", task_type="claim_v1", question=question,
                          contract=contract, gold={"label": label, "line_ids": [e.id for e in deciding]}, required_ids=tuple(e.id for e in deciding), split=split, leakage=False,
                          metadata={"family": "claim_verification", "claim": claim, "template": template, "exam_name": exam_name, "contract_extra": {"index_report_note": note_ref}}))
    return case, tasks


# ------------------------------------------------------------------ F5 report reconstruction (at / pre)
def build_reconstruction(loader: SourceLoader, anchor: Anchor, split: str) -> tuple[Case, list[Task]] | None:
    index_time = _t(anchor.payload["index_time"])
    note_ref = eid("rad_note", anchor.payload["note_id"])
    pool = radiology_pool(loader, anchor.subject_id, index_time)
    if len(pool) > 600:
        return None
    impressions = notes_with_impression(pool)
    idx_imp = impressions.get(note_ref)
    if not idx_imp or len(idx_imp) > 12:
        return None
    idx_lines = note_lines(pool, note_ref)
    exam_name = anchor.payload.get("exam_name") or idx_lines[0].fields.get("exam_name")
    # the pre-index target: the latest report whose lines became available strictly before the index
    prior = [(imp[0].time("available"), n, imp) for n, imp in impressions.items() if n != note_ref and imp[0].time("available") < index_time and len(imp) <= 12]
    if not prior:
        return None
    prior.sort(key=lambda x: (x[0], x[1]))
    p_time, p_note, p_imp = prior[-1]
    p_exam = note_lines(pool, p_note)[0].fields.get("exam_name")
    case_id = stable_hex("case", "report_reconstruction", anchor.subject_id, *anchor.key, n=20)
    case = Case(case_id=case_id, patient=hash_patient(anchor.subject_id), index_event={"type": "radiology_exam", "index_time": _iso(index_time), "note": note_ref, "exam_name": exam_name}, evidence=pool)
    idx_ids = tuple(e.id for e in idx_lines)
    targets = {
        "at": (idx_imp, f"Reconstruct the IMPRESSION section of the index report ({exam_name}, indexed at {_iso(index_time)}): return its lines in order, each with its line id and text copied verbatim.", note_ref),
        "pre": (p_imp, f"Reconstruct the IMPRESSION section of the most recent radiology report that became available strictly before the index time {_iso(index_time)}: return its lines in order, each with its line id and text copied verbatim.", p_note),
    }
    tasks = []
    for scope, (imp, question, target_note) in targets.items():
        contract = Contract(index_time=_iso(index_time), scope=scope, basis="available", permitted_sources=("radiology_line",), index_evidence_ids=idx_ids if scope == "at" else ())
        gold = {"status": "ok", "line_ids": [e.id for e in imp], "text": " ".join(e.fields["text"] for e in imp)}
        tasks.append(Task(task_id=f"{case_id}-{scope}", case_id=case_id, patient=case.patient, tier="B", task_type="reconstruct_v1", question=question,
                          contract=contract, gold=gold, required_ids=tuple(gold["line_ids"]), split=split, leakage=True,   # the case is leakage-decidable: under pre the index report is an illegal distractor
                          metadata={"family": "report_reconstruction", "exam_name": exam_name, "target_note": target_note, "n_impression_lines": len(imp),
                                    "n_notes_in_pool": len({e.fields.get("note") for e in pool}), "contract_extra": {"index_report_note": note_ref}}))
    return case, tasks


# ------------------------------------------------------------------ F6 exam summarization (pre / post)
def build_summarization(loader: SourceLoader, anchor: Anchor, split: str) -> tuple[Case, list[Task]] | None:
    index_time = _t(anchor.payload["index_time"])
    pool = radiology_pool(loader, anchor.subject_id, index_time, days=730)
    if len(pool) > 800:
        return None
    case_id = stable_hex("case", "summarization", anchor.subject_id, *anchor.key, n=20)
    case = Case(case_id=case_id, patient=hash_patient(anchor.subject_id), index_event={"type": "admission", "index_time": _iso(index_time)}, evidence=pool)
    tasks = []
    for scope in ("pre", "post"):
        contract = Contract(index_time=_iso(index_time), scope=scope, basis="available", permitted_sources=("radiology_line",))
        legal = filter_legal(pool, contract)
        exams = []
        for n, imp in notes_with_impression(legal).items():
            line = first_content_line(imp)
            if line is None:
                continue
            exams.append({"exam_name": note_lines(legal, n)[0].fields.get("exam_name"), "time": line.available, "first_impression_line": line.fields["text"], "citation": line.id})
        if not exams or len(exams) > 8:
            return None
        exams.sort(key=lambda x: (x["time"], x["citation"]))
        side = "available strictly before" if scope == "pre" else "available strictly after"
        question = (f"List every radiology examination whose report became {side} {_iso(index_time)}: give the exam name, the report's availability time, "
                    f"and the first substantive line of its IMPRESSION section copied verbatim, citing that line.")
        tasks.append(Task(task_id=f"{case_id}-{scope}", case_id=case_id, patient=case.patient, tier="B", task_type="summarize_exams_v1", question=question,
                          contract=contract, gold={"status": "ok", "exams": exams}, required_ids=tuple(x["citation"] for x in exams), split=split, leakage=False,
                          metadata={"family": "summarization", "n_exams": len(exams)}))
    return case, tasks
