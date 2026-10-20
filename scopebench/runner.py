"""The run loop (plan §4, §8.2): one (tier manifest, configuration, backend) -> one output directory.

Per task:
  1. evidence selection according to the configuration (A_t recorded)
  2. prompt -> backend (wave 1, all tasks, ordered by case for prefix reuse)
  3. parse -> ScopeGuard check
  4. one repair (wave 2) when the configuration repairs and the guard found errors
  5. skill verification / deterministic replacement when the configuration says so
  6. score; write predictions.jsonl (one row per task), access_log.jsonl, stages.jsonl
Resumable: task ids already present in predictions.jsonl are skipped.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import __version__, parser, plans, scoring
from .backends.base import Backend, Generation, Request
from .configurations import Configuration
from .config import ROOT, relpath
from .guard import SELF_REVISE_MESSAGE, GuardReport, check, repair_message
from .predicate import filter_legal, legal, wrong_basis
from .retrieval import AccessLog, TemporalNearestRetriever
from .schema import Case, EvidenceObject, Task, read_jsonl
from .tasktypes import TaskType, get as get_task_type

Payload = dict[str, Any]
SkillFn = Callable[[list[EvidenceObject]], Payload]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


@dataclass
class RunSpec:
    configuration: Configuration
    retriever: Any                      # Retriever
    k: int                              # top-k for scoped/wrong-basis/refill selection
    pool_k: int | None = None           # prompt-only size (None = whole pool)
    mask: str | None = None             # masking replay: 'cited_admissible' | 'uncited_accessed' (plan §8.6)
    mask_source: dict[str, dict] | None = None   # task_id -> source-run row (citations + access ids)
    skills: dict[str, SkillFn] = field(default_factory=dict)  # skill_op -> fn(legal evidence ordered) -> summary
    max_tokens_override: int | None = None
    label: str = ""
    repair_wording: str = "a"           # guard.REPAIR_WORDINGS key; "b" = alternative wording (sensitivity, plan §8.5)
    retrieval_cache: Any = None         # RetrievalCache: memoized evidence selection per (task, contract, retriever, mode)
    inject: str | None = None           # DEV-35 exposure injection: arm name ('future' | 'inscope'), see `_finish_selection`
    inject_ids: dict[str, dict[str, list[str]]] | None = None   # task_id -> arm -> evidence ids appended to the selection


class RetrievalCache:
    """Memoizes the evidence selection of `select_evidence` per task. The selection is a deterministic function of
    the task (id + contract), the retriever (class + parameters + encoder), the configuration's evidence mode,
    query hint and predicate engine, and k / pool_k — all of which form the key — so a cached selection is exactly
    the selection the retriever would return again (the dense encoder is evaluated once instead of once per
    configuration). Entries live under <dir>/<manifest sha>/<retriever signature>.jsonl and are appended as computed.
    Masking replays and skill-cited additions are applied after the lookup, as before."""

    def __init__(self, directory: Path, manifest_sha: str | None, retriever: Any):
        sig = {"retriever": type(retriever).__name__, **{k: v for k, v in vars(retriever).items() if isinstance(v, (str, int, float, bool))},
               "encoder": getattr(getattr(retriever, "encoder", None), "name", None)}
        self.signature = json.dumps(sig, sort_keys=True)
        self.path = directory / (manifest_sha or "no-manifest") / (hashlib.sha256(self.signature.encode()).hexdigest()[:16] + ".jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, list[str]] = {}
        self.hits = self.misses = 0
        if self.path.exists():
            for row in read_jsonl(self.path):
                self._mem[row["key"]] = row["ids"]
            (self.path.parent / "signatures.txt").open("a").write(f"{self.path.name} {self.signature}\n") if not (self.path.parent / "signatures.txt").exists() else None
        else:
            (self.path.parent / "signatures.txt").open("a").write(f"{self.path.name} {self.signature}\n")

    @staticmethod
    def key(task: Task, cfg: Configuration, k: int, pool_k: int | None) -> str:
        payload = json.dumps({"task": task.task_id, "contract": task.contract.to_dict(), "evidence": cfg.evidence, "hint": cfg.query_scope_hint,
                              "engine": cfg.predicate_engine, "k": k, "pool_k": pool_k}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> list[str] | None:
        ids = self._mem.get(key)
        self.hits += ids is not None
        return ids

    def put(self, key: str, ids: list[str]) -> None:
        self.misses += 1
        self._mem[key] = ids
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "ids": ids}) + "\n")


@dataclass
class TaskState:
    task: Task
    task_type: TaskType
    case: Case
    pool: dict[str, EvidenceObject]
    supplied: list[EvidenceObject]
    access: AccessLog
    messages: list[dict[str, str]] = field(default_factory=list)
    initial: Generation | None = None
    repair: Generation | None = None
    parsed_initial: Payload | None = None
    parsed_final: Payload | None = None
    report_initial: GuardReport | None = None
    report_final: GuardReport | None = None
    skill_summary: Payload | None = None
    repaired: bool = False
    replaced: bool = False
    repair_evidence: list[EvidenceObject] = field(default_factory=list)
    plan_initial: plans.PlanReading | None = None          # DEV-34 plan configurations only
    plan_final: plans.PlanReading | None = None
    plan_result_initial: plans.PlanResult | None = None
    plan_result_final: plans.PlanResult | None = None
    replacement_rejected: list[str] | None = None          # plan_full: the executed plan's answer failed its own audit


def _predicate(cfg: Configuration, task: Task, case: Case):
    """The contract predicate, or an S4 policy engine's legal set for the same contract."""
    if cfg.predicate_engine:
        from .policy_engines import ENGINES

        legal_ids = ENGINES[cfg.predicate_engine](case.evidence, task.contract)
        return lambda e: e.id in legal_ids
    return lambda e: legal(e, task.contract)


def _scope_hint_task(task: Task) -> Task:
    c = task.contract
    hint = {"pre": f"only evidence available strictly before {c.index_time}", "post": f"only evidence available strictly after {c.index_time}",
            "at": f"only evidence available up to {c.index_time}", "full": "any evidence in the chart"}[c.scope]
    return Task(**{**task.__dict__, "question": f"{task.question} Scope: {hint}."})


def select_evidence(spec: RunSpec, task: Task, case: Case, access: AccessLog) -> list[EvidenceObject]:
    cfg = spec.configuration
    cache_key = None
    if spec.retrieval_cache is not None:
        cache_key = RetrievalCache.key(task, cfg, spec.k, spec.pool_k)
        cached = spec.retrieval_cache.get(cache_key)
        if cached is not None:
            pool = case.by_id()
            chosen = [pool[i] for i in cached if i in pool]
            if len(chosen) == len(cached):
                return _finish_selection(spec, task, case, access, chosen)
    chosen = _retrieve(spec, task, case)
    if cache_key is not None:
        spec.retrieval_cache.put(cache_key, [e.id for e in chosen])
    return _finish_selection(spec, task, case, access, chosen)


def _retrieve(spec: RunSpec, task: Task, case: Case) -> list[EvidenceObject]:
    cfg = spec.configuration
    contract = task.contract
    if cfg.query_scope_hint:
        task = _scope_hint_task(task)
    if cfg.evidence == "unscoped":
        k = spec.pool_k if spec.pool_k is not None else len(case.evidence)
        chosen = spec.retriever.rank(task, case.evidence, k, None)
    elif cfg.evidence == "scoped":
        chosen = spec.retriever.rank(task, case.evidence, spec.k, _predicate(cfg, task, case))
    elif cfg.evidence == "wrong_basis":
        wb = wrong_basis(contract)
        wb_retriever = spec.retriever
        if isinstance(spec.retriever, TemporalNearestRetriever):
            wb_retriever = TemporalNearestRetriever(basis=wb.basis, order_basis=spec.retriever.order_basis)
        chosen = wb_retriever.rank(task, case.evidence, spec.k, lambda e: legal(e, wb))
    elif cfg.evidence == "refill":
        if not isinstance(spec.retriever, TemporalNearestRetriever):
            raise ValueError("refill selection requires the temporal retriever")
        chosen = spec.retriever.refill(task, case.evidence, spec.k, lambda e: legal(e, contract))
    else:
        raise ValueError(cfg.evidence)
    return chosen


def _finish_selection(spec: RunSpec, task: Task, case: Case, access: AccessLog, chosen: list[EvidenceObject]) -> list[EvidenceObject]:
    contract = task.contract
    if spec.mask and spec.mask_source is not None:
        src = spec.mask_source.get(task.task_id)
        if src is not None:
            # DEV-18: only string ids count as citations, as in scoring (`_str_list`); a malformed answer such as
            # [["id"], ["id"]] cites nothing (it used to raise "unhashable type: 'list'" here)
            guard_cites = (src.get("final_guard") or {}).get("citations", []) or []
            answer_cites = src.get("parsed_answer", {}).get("citations", []) if isinstance(src.get("parsed_answer"), dict) else []
            cited = {c for c in guard_cites if isinstance(c, str)} or {c for c in (answer_cites if isinstance(answer_cites, list) else []) if isinstance(c, str)}
            if not cited:
                tt = get_task_type(task.task_type)
                cited = set(tt.citations(src.get("parsed_answer")))
            accessed = set(src.get("access_ids", []))
            if spec.mask == "cited_admissible":
                remove = {i for i in cited if i in case.by_id() and legal(case.by_id()[i], contract)}
            elif spec.mask == "uncited_accessed":
                remove = {i for i in accessed if i not in cited}
            else:
                raise ValueError(spec.mask)
            chosen = [e for e in chosen if e.id not in remove]
            access.record("masked_out", sorted(remove))
    if spec.inject and spec.inject_ids is not None:
        # DEV-35 (experiments/SCOPEBENCH_EXPOSURE_PROTOCOL.md): the arm's records, fixed per task before any run, are
        # appended after the scoped selection (same position in both arms); they count as accessed like any shown record
        pool, shown = case.by_id(), {e.id for e in chosen}
        added = [pool[i] for i in spec.inject_ids.get(task.task_id, {}).get(spec.inject, []) if i in pool and i not in shown]
        chosen = chosen + added
        access.record("injected", [e.id for e in added])
    access.record("prompt", [e.id for e in chosen])
    return chosen


def legal_topk(spec: RunSpec, task: Task, case: Case) -> list[EvidenceObject]:
    return spec.retriever.rank(task, case.evidence, spec.k, lambda e: legal(e, task.contract))


def legal_all(task: Task, case: Case) -> list[EvidenceObject]:
    """The whole admissible set L_t in chronological order on the contract basis (skills operate on this,
    not on the retrieval top-k: a ChartSkill maps admissible evidence to a typed result)."""
    items = filter_legal(case.evidence, task.contract)
    basis = task.contract.basis
    return sorted(items, key=lambda e: (0, e.time(basis).timestamp(), e.id) if e.time(basis) is not None else (1, 0.0, e.id))


def build_messages(task_type: TaskType, task: Task, evidence: list[EvidenceObject], skill_summary: Payload | None, system_suffix: str = "") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": task_type.system_prompt + system_suffix},
        {"role": "user", "content": task_type.build_user_payload(task, evidence, skill_summary)},
    ]


def run(
    spec: RunSpec,
    tasks: list[Task],
    cases: dict[str, Case],
    backend: Backend,
    out_dir: Path,
    *,
    manifest_path: Path | None = None,
    extra_metadata: dict[str, Any] | None = None,
    retry_errors: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    previous = list(read_jsonl(pred_path)) if pred_path.exists() else []
    redo: set[str] = set()
    if retry_errors:
        # rows whose generation hit an infrastructure error (HTTP error, timeout) are dropped and re-run;
        # their access-log and stage rows go with them
        redo = {row["task_id"] for row in previous if row.get("generation_errors")}
        for name in ("predictions.jsonl", "access_log.jsonl", "stages.jsonl") if redo else ():
            path = out_dir / name
            if path.exists():
                kept = [row for row in read_jsonl(path) if row.get("task_id") not in redo]
                with open(path, "w", encoding="utf-8") as f:
                    for row in kept:
                        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    done = {row["task_id"] for row in previous} - redo
    todo = sorted((t for t in tasks if t.task_id not in done), key=lambda t: (t.case_id, t.task_id))
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    cfg = spec.configuration
    states: list[TaskState] = []

    # --- wave 1 -------------------------------------------------------------------------------
    for task in todo:
        case = cases[task.case_id]
        tt = get_task_type(task.task_type)
        access = AccessLog()
        supplied = select_evidence(spec, task, case, access)
        st = TaskState(task=task, task_type=tt, case=case, pool=case.by_id(), supplied=supplied, access=access)
        if cfg.skill_payload or cfg.skill_verify:
            fn = spec.skills.get(tt.skill_op) if tt.skill_op else None
            if fn is not None:
                st.skill_summary = _call_skill(fn, legal_all(task, case), task)
                if cfg.skill_payload and st.skill_summary is not None:
                    shown = {e.id for e in supplied}
                    extra = [st.pool[i] for i in _provenance_ids(st.skill_summary) if i in st.pool and i not in shown]
                    if extra:
                        supplied = supplied + extra   # the skill's cited records are handed to the model as supplied evidence
                        st.supplied = supplied
                        access.record("skill", [e.id for e in extra])
        st.messages = build_messages(tt, task, supplied, st.skill_summary if cfg.skill_payload else None,
                                     plans.TIER_B_ADDENDUM if cfg.plan != "none" else "")
        states.append(st)
    max_tokens = lambda st: spec.max_tokens_override or st.task_type.max_tokens  # noqa: E731
    gens = backend.generate_many([Request(st.task.task_id, st.messages, max_tokens(st)) for st in states]) if states else {}
    for st in states:
        st.initial = gens[st.task.task_id]
        st.parsed_initial = parser.extract_json(st.initial.text)
        if cfg.plan != "none":
            st.plan_initial, st.plan_result_initial, st.report_initial = plan_check(spec, st, st.parsed_initial, st.supplied)
        elif cfg.uses_guard:
            st.report_initial = check(
                st.parsed_initial, st.task, st.task_type, st.pool, st.supplied, st.access.ids(),
                check_completeness=cfg.check_completeness,
                skill_summary=st.skill_summary if cfg.skill_verify else None,
            )

    # --- wave 2: repairs ----------------------------------------------------------------------
    repair_states: list[TaskState] = []
    for st in states:
        if cfg.review:
            # DEV-36 self_revise: every task gets the same gold-free review turn (no guard result, no evidence)
            st.messages = st.messages + [
                {"role": "assistant", "content": st.initial.text},
                {"role": "user", "content": SELF_REVISE_MESSAGE},
            ]
            repair_states.append(st)
            continue
        if cfg.repair == "none" or st.report_initial is None:
            continue
        errs = _guard_errors_for(cfg, st.report_initial)
        if not errs:
            continue
        st.repaired = True
        # Refill: hand the model exactly the permitted evidence the guard's errors point at — the missing required
        # records (re-shown even when they were already supplied: an empty permitted list reads as "nothing is
        # permitted" and made Gemma-3 abstain on 174/208 repairs in the pilot) plus legal records it cited without
        # being shown — capped, so the repair turn never re-sends the whole supplied set (context overflow).
        wanted = list(st.report_initial.missing_required) + [c for c in st.report_initial.unsupplied_citations if c in st.pool and legal(st.pool[c], st.task.contract)]
        refill = []
        for i in wanted:
            if i in st.pool and st.pool[i] not in refill:
                refill.append(st.pool[i])
        st.repair_evidence = legal_topk(spec, st.task, st.case) if cfg.repair_resupply_topk else refill[:20]
        st.access.record("repair", [e.id for e in st.repair_evidence])
        mode = {"feedback": "feedback", "generic": "generic", "feedback_no_evidence": "feedback_no_evidence"}[cfg.repair]
        if cfg.guard == "schema_only" and mode == "feedback":
            mode = "schema_only"
        elif mode == "feedback" and not cfg.repair_resupply_topk:
            mode = "feedback_refill"   # partial re-supply: the wording must not read as "only these records are permitted"
        view = [_view(st.task_type, st.task, e) for e in st.repair_evidence]
        repair_skill = st.skill_summary if (cfg.skill_payload and st.skill_summary is not None) else None
        message = repair_message(mode, errs, view, repair_skill, wording=spec.repair_wording)
        if cfg.plan == "full" and st.plan_result_initial is not None and st.plan_result_initial.ok:
            # the result of the plan the model itself declared (never a reference result)
            message = json.dumps({**json.loads(message), "evidence_plan_result": st.plan_result_initial.to_dict()["summary"]}, ensure_ascii=False, sort_keys=True)
        st.messages = st.messages + [
            {"role": "assistant", "content": st.initial.text},
            {"role": "user", "content": message},
        ]
        repair_states.append(st)
    if repair_states:
        gens2 = backend.generate_many([Request(st.task.task_id, st.messages, max_tokens(st)) for st in repair_states])
        for st in repair_states:
            st.repair = gens2[st.task.task_id]

    return write_outputs(spec, states, tasks, done, backend, out_dir, pred_path, manifest_path, extra_metadata, started, n_retried=len(redo))


def write_outputs(spec: RunSpec, states: list[TaskState], tasks: list[Task], done: set[str], backend: Backend, out_dir: Path, pred_path: Path,
                  manifest_path: Path | None, extra_metadata: dict[str, Any] | None, started: str, n_retried: int = 0) -> dict[str, Any]:
    cfg = spec.configuration
    # --- final guard, replacement, scoring, output ---------------------------------------------
    rows: list[dict[str, Any]] = []
    access_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for st in states:
        final_text = st.repair.text if st.repair is not None else st.initial.text
        st.parsed_final = parser.extract_json(final_text)
        if st.repaired and cfg.repair_resupply_topk:
            final_supplied = st.repair_evidence
        elif st.repaired:
            shown = {e.id for e in st.supplied}
            final_supplied = st.supplied + [e for e in st.repair_evidence if e.id not in shown]   # re-shown records are not duplicated
        else:
            final_supplied = st.supplied
        if cfg.plan != "none":
            plan_finalize(spec, st, final_supplied)
        elif cfg.uses_guard:
            st.report_final = check(
                st.parsed_final, st.task, st.task_type, st.pool, final_supplied, st.access.ids(),
                check_completeness=cfg.check_completeness,
                skill_summary=st.skill_summary if cfg.skill_verify else None,
            )
            final_errors = _guard_errors_for(cfg, st.report_final)
            if final_errors and cfg.skill_replace and st.skill_summary is not None and st.task_type.skill_answer is not None:
                st.replaced = True
                st.parsed_final = st.task_type.skill_answer(st.skill_summary, st.task, legal_all(st.task, st.case))
                st.access.record("replacement", _provenance_ids(st.skill_summary))
                st.report_final = check(
                    st.parsed_final, st.task, st.task_type, st.pool, legal_all(st.task, st.case), st.access.ids(),
                    check_completeness=cfg.check_completeness, skill_summary=st.skill_summary,
                )
                if _guard_errors_for(cfg, st.report_final):
                    raise AssertionError(f"verifier-generated skill answer failed its own check: {_guard_errors_for(cfg, st.report_final)}")
        tokens_in = st.initial.tokens_in + (st.repair.tokens_in if st.repair else 0)
        tokens_out = st.initial.tokens_out + (st.repair.tokens_out if st.repair else 0)
        latency = st.initial.latency_seconds + (st.repair.latency_seconds if st.repair else 0.0)
        calls = 1 + (1 if st.repair else 0)
        metrics = scoring.score(st.parsed_final, st.task, st.task_type, st.pool, st.access.ids(),
                                tokens_in=tokens_in, tokens_out=tokens_out, calls=calls, latency=latency)
        initial_metrics = scoring.score(st.parsed_initial, st.task, st.task_type, st.pool, st.access.ids(),
                                        tokens_in=st.initial.tokens_in, tokens_out=st.initial.tokens_out, calls=1, latency=st.initial.latency_seconds)
        row = {
            "task_id": st.task.task_id,
            "case_id": st.task.case_id,
            "patient": st.task.patient,
            "tier": st.task.tier,
            "task_type": st.task.task_type,
            "scope": st.task.scope,
            "leakage": st.task.leakage,
            "configuration": cfg.name,
            "selected_evidence_ids": [e.id for e in st.supplied],
            "repair_evidence_ids": [e.id for e in st.repair_evidence],
            "response": st.initial.text,
            "repair_response": st.repair.text if st.repair else "",
            "parsed_initial": st.parsed_initial,
            "parsed_answer": st.parsed_final,
            "initial_guard": st.report_initial.to_dict() if st.report_initial else None,
            "final_guard": st.report_final.to_dict() if st.report_final else None,
            "initial_guard_errors": _guard_errors_for(cfg, st.report_initial) if st.report_initial else [],
            "final_guard_errors": _guard_errors_for(cfg, st.report_final) if st.report_final else [],
            "guard_repaired": st.repaired,
            "guard_deterministic_fallback": st.replaced,
            "skill_summary": st.skill_summary,
            "access_ids": sorted(st.access.ids()),
            "generation_errors": [g.error for g in (st.initial, st.repair) if g is not None and g.error],
            "metrics": metrics,
            "initial_metrics": initial_metrics,
            "answer_changed_by_repair": (st.parsed_initial != st.parsed_final) and not st.replaced,
        }
        if cfg.review:
            row["self_revised"] = st.repair is not None
        if spec.inject:
            row["exposure_arm"] = spec.inject
            row["injected_ids"] = next((e["ids"] for e in st.access.entries if e.get("stage") == "injected"), [])
        if cfg.plan != "none":
            row["evidence_plan_initial"] = st.plan_initial.to_dict() if st.plan_initial else None
            row["evidence_plan_final"] = st.plan_final.to_dict() if st.plan_final else None
            row["plan_result_initial"] = st.plan_result_initial.to_dict() if st.plan_result_initial else None
            row["plan_result_final"] = st.plan_result_final.to_dict() if st.plan_result_final else None
            if st.replacement_rejected is not None:
                row["plan_replacement_rejected"] = st.replacement_rejected
        rows.append(row)
        access_rows.append({"task_id": st.task.task_id, "configuration": cfg.name, "entries": st.access.entries})
        stage_rows.append({
            "task_id": st.task.task_id, "configuration": cfg.name, "covered": st.task_type.covered,
            "initial_valid": bool(st.report_initial and not _guard_errors_for(cfg, st.report_initial)) if st.report_initial else None,
            "repaired": st.repaired, "replaced": st.replaced,
            "final_valid": bool(st.report_final and not _guard_errors_for(cfg, st.report_final)) if st.report_final else None,
            "initial_correct": initial_metrics["answer_correct"], "final_correct": metrics["answer_correct"],
        })
    with open(pred_path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with open(out_dir / "access_log.jsonl", "a", encoding="utf-8") as f:
        for row in access_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with open(out_dir / "stages.jsonl", "a", encoding="utf-8") as f:
        for row in stage_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    task_types = sorted({t.task_type for t in tasks})
    metadata = {
        "scopebench_version": __version__,
        "git_commit": git_commit(),
        "configuration": cfg.__dict__,
        "retriever": getattr(spec.retriever, "name", type(spec.retriever).__name__),
        "k": spec.k, "pool_k": spec.pool_k,
        "task_types": task_types,
        "system_prompt_sha256": {tt: get_task_type(tt).system_prompt_sha256 for tt in task_types},
        "parser_version": parser.PARSER_VERSION,
        "skills": sorted(spec.skills),
        "task_manifest": relpath(manifest_path) if manifest_path else None,
        "task_manifest_sha256": sha256_file(manifest_path / "tasks.jsonl" if manifest_path and manifest_path.is_dir() else manifest_path) if manifest_path else None,
        "n_tasks_total": len(tasks), "n_tasks_this_run": len(rows), "n_tasks_previously_done": len(done), "n_tasks_retried_after_errors": n_retried,
        "retrieval_cache": ({"path": relpath(spec.retrieval_cache.path), "hits": spec.retrieval_cache.hits, "misses": spec.retrieval_cache.misses, "signature": spec.retrieval_cache.signature}
                            if spec.retrieval_cache is not None else None),
        "backend": backend.metadata(),
        "platform": platform.platform(),
        "started_at": started, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "label": spec.label,
        **(extra_metadata or {}),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=1, sort_keys=True) + "\n")
    return metadata


def _provenance_ids(summary: Payload) -> list[str]:
    prov = summary.get("provenance") or []
    return [p["citation"] if isinstance(p, dict) else str(p) for p in prov]


def _call_skill(fn: Callable[..., Payload], evidence: list[EvidenceObject], task: Task) -> Payload:
    """Skill functions come in two shapes: the frozen reference `fn(evidence)` and the SKILL_SPEC
    package `fn(evidence, contract, params)`. Params carry the task type's fixed parameters."""
    params = dict(task.metadata.get("skill_params") or get_task_type(task.task_type).notes.get("skill_params") or {"k": 3})
    try:
        return fn(evidence, task.contract, params)
    except TypeError:
        return fn(evidence)


def _guard_errors_for(cfg: Configuration, report: GuardReport) -> list[str]:
    if cfg.guard == "schema_only":
        return report.errors(schema=True, scope=False, completeness=False, skill=False)
    if cfg.guard == "scope_only":
        return report.errors(schema=True, scope=True, completeness=False, skill=False)
    if cfg.guard == "scope_completeness":
        return report.errors(schema=True, scope=True, completeness=cfg.check_completeness, skill=cfg.checks_agreement)
    # guard none but skill verification requested
    return report.errors(schema=True, scope=True, completeness=cfg.check_completeness, skill=cfg.checks_agreement)


def plan_check(spec: RunSpec, st: TaskState, parsed: Payload | None, supplied: list[EvidenceObject]) -> tuple[plans.PlanReading, plans.PlanResult | None, GuardReport]:
    """DEV-34: read the plan the submission declares, execute it on L_t, and audit the submission against R̂_t (and, for
    plan_full, against the executed plan's result). No reference input is read: completeness uses `required=R̂_t`."""
    cfg = spec.configuration
    reading = plans.read(parsed)
    result = plans.execute(reading.plan, legal_all(st.task, st.case), st.task, spec.skills) if reading.plan is not None else None
    usable = result is not None and result.error is None
    kwargs = dict(check_completeness=usable, required=result.required if usable else ())
    try:
        report = check(parsed, st.task, st.task_type, st.pool, supplied, st.access.ids(),
                       skill_summary=result.summary if (cfg.plan == "full" and usable and result.ok) else None, **kwargs)
    except (KeyError, TypeError, ValueError):
        # Tier-A addendum 2 fix: the declared operation computes a different kind of result than this task type answers
        # (e.g. a cohort decision on a lab-trend task); it cannot agree with any answer. Never reached on Tier-B or Tier-C.
        report = check(parsed, st.task, st.task_type, st.pool, supplied, st.access.ids(), skill_summary=None, **kwargs)
        report.skill_disagreements = ["the declared operation does not compute an answer of this task type"]
    if report.parseable:   # the plan is part of the required output object of a plan configuration
        report.schema_errors = report.schema_errors + reading.errors + (
            [f"evidence_plan could not be executed: {result.error}"] if result is not None and result.error else [])
    return reading, result, report


def plan_finalize(spec: RunSpec, st: TaskState, final_supplied: list[EvidenceObject]) -> None:
    """Final audit of a plan configuration; plan_full replaces a submission that still fails by the answer of the plan
    the model declared (the reference system replaces it by the reference skill's answer)."""
    cfg = spec.configuration
    st.plan_final, st.plan_result_final, st.report_final = plan_check(spec, st, st.parsed_final, final_supplied)
    res = st.plan_result_final
    if cfg.plan == "full" and _guard_errors_for(cfg, st.report_final) and res is not None and res.ok and st.task_type.skill_answer is not None:
        evidence = legal_all(st.task, st.case)
        st.access.record("replacement", plans.provenance_ids(res.summary))
        try:
            candidate = {**st.task_type.skill_answer(res.summary, st.task, evidence), plans.PLAN_KEY: res.plan.to_dict()}
            _, _, cand_report = plan_check(spec, st, candidate, evidence)
            rejected = _guard_errors_for(cfg, cand_report)
        except (KeyError, TypeError, ValueError) as exc:
            rejected = [f"{type(exc).__name__}: {exc}"]
        if rejected:
            # Tier-A addendum 2 fix: an operation whose result is not an answer of this task type (e.g. a lookup declared on
            # a count task) cannot replace the submission; the submission stands with its final audit and the replacement
            # access is withdrawn. Never reached on Tier-B or Tier-C (those runs asserted here and completed).
            st.access.entries.pop()
            st.replacement_rejected = rejected
            return
        st.replaced = True
        st.parsed_final = candidate
        st.report_final = cand_report


def _view(task_type: TaskType, task: Task, e: EvidenceObject) -> dict[str, Any]:
    """The evidence dictionary as the prompt shows it (task-type specific view when defined)."""
    view_fn = task_type.notes.get("evidence_view")
    if callable(view_fn):
        return view_fn(e)
    return {"id": e.id, "source": e.source, "occurred": e.occurred, "available": e.available, **e.fields}
