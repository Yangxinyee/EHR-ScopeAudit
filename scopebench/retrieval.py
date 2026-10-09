"""Retrieval (plan §4.2): the predicate is applied BEFORE ranking in scope-aware configurations.

Two retrievers:
  TemporalNearestRetriever - rank by |t - index| on a basis (ScopePair semantics; used by
                             the ScopePair regression and by nearest-k task families).
  HybridRetriever          - BM25 + dense (bge-m3) with reciprocal-rank fusion, n_candidates,
                             top-k (paper Supp. S6: 50 candidates, top-k 10, RRF).
Both return the ranked evidence and the caller records A_t from what was returned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol

from .predicate import legal
from .schema import Contract, EvidenceObject, Task

Predicate = Callable[[EvidenceObject], bool]


def predicate_for(contract: Contract) -> Predicate:
    return lambda e: legal(e, contract)


def evidence_text(e: EvidenceObject) -> str:
    parts = [e.source]
    for key, value in sorted(e.fields.items()):
        if value is None:
            continue
        parts.append(f"{key}: {value}")
    return " ".join(parts)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


class Retriever(Protocol):
    name: str

    def rank(self, task: Task, pool: list[EvidenceObject], k: int, predicate: Predicate | None) -> list[EvidenceObject]: ...


@dataclass
class TemporalNearestRetriever:
    """ScopePair's `nearest`, verbatim semantics: select the k evidence objects closest to the index
    on `basis` (ties by id), then return them in chronological order of `order_basis` (ties by id)."""

    basis: str = "available"
    order_basis: str = "available"
    name: str = "temporal_nearest"

    def rank(self, task: Task, pool: list[EvidenceObject], k: int, predicate: Predicate | None) -> list[EvidenceObject]:
        index = task.contract.index
        items = [e for e in pool if (predicate is None or predicate(e))]
        timed = [e for e in items if e.time(self.basis) is not None and e.time(self.order_basis) is not None]
        selected = sorted(timed, key=lambda e: (abs((e.time(self.basis) - index).total_seconds()), e.id))[:k]
        return sorted(selected, key=lambda e: (e.time(self.order_basis), e.id))

    def refill(self, task: Task, pool: list[EvidenceObject], k: int, predicate: Predicate) -> list[EvidenceObject]:
        """ScopePair's guard_refill evidence: the legal members of the unscoped top-k, refilled with the
        nearest legal objects not already present, returned chronologically."""
        unscoped = self.rank(task, pool, k, None)
        legal_items = [e for e in unscoped if predicate(e)]
        seen = {e.id for e in legal_items}
        for e in self.rank(task, pool, len(pool), predicate):
            if len(legal_items) >= k:
                break
            if e.id not in seen:
                legal_items.append(e)
                seen.add(e.id)
        return sorted(legal_items, key=lambda e: (e.time(self.order_basis), e.id))


@dataclass
class BM25Index:
    docs: list[list[str]]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        from rank_bm25 import BM25Okapi  # lazy: analysis venv dependency

        self._bm25 = BM25Okapi(self.docs, k1=self.k1, b=self.b) if self.docs else None

    def scores(self, query: list[str]) -> list[float]:
        if self._bm25 is None:
            return []
        return list(self._bm25.get_scores(query))


class DenseEncoder:
    """bge-m3 through sentence-transformers; loaded once per process."""

    def __init__(self, model_dir: str, device: str = "cuda"):
        from sentence_transformers import SentenceTransformer  # lazy

        from .config import remap_model_path

        self.model = SentenceTransformer(remap_model_path(model_dir), device=device)
        self.name = f"bge-m3@{model_dir}"   # the logical path: part of the retrieval-cache signature on every machine

    def encode(self, texts: list[str]):
        return self.model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for pos, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + pos + 1)
    return [item for item, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def retrieval_query(task: Task) -> str:
    """The retriever's query: the question plus task-type hints declared in the task metadata
    (concept, exam name, claim text, target section). Declared per task type in the protocol."""
    parts = [task.question]
    md = task.metadata
    for key in ("concept", "exam_name", "claim", "retrieval_hint"):
        if md.get(key):
            parts.append(str(md[key]))
    if task.task_type in ("reconstruct_v1", "summarize_exams_v1"):
        parts.append("IMPRESSION")
    if task.task_type == "cohort_v1":
        for c in md.get("criteria", []):
            parts.append(" ".join(str(v) for k, v in c.items() if k in ("concept", "prefix", "drug")))
    return " ".join(parts)


@dataclass
class HybridRetriever:
    """BM25 + dense + temporal proximity with reciprocal-rank fusion (plan §4.2). The predicate (if any)
    is applied to the pool first. `mode` selects the components: hybrid = bm25+dense+temporal,
    bm25_temporal, bm25, dense, temporal."""

    encoder: DenseEncoder | None = None
    n_candidates: int = 50
    rrf_k: int = 60
    mode: str = "hybrid"
    expand_sections: int = 0      # for line-level evidence: add every sibling line (same note + section) of the top-N hit notes
    expand_cap: int = 60          # hard cap on the expanded result size
    name: str = "hybrid_bm25_dense_temporal"
    _cache: dict[str, object] = field(default_factory=dict, repr=False)

    def rank(self, task: Task, pool: list[EvidenceObject], k: int, predicate: Predicate | None) -> list[EvidenceObject]:
        items = [e for e in pool if (predicate is None or predicate(e))]
        if not items:
            return []
        texts = [evidence_text(e) for e in items]
        query = retrieval_query(task)
        rankings: list[list[str]] = []
        if self.mode in ("hybrid", "bm25_temporal", "temporal"):
            index = task.contract.index
            basis = task.contract.basis
            timed = [i for i, e in enumerate(items) if e.time(basis) is not None]
            order = sorted(timed, key=lambda i: (abs((items[i].time(basis) - index).total_seconds()), items[i].id))
            rankings.append([items[i].id for i in order[: self.n_candidates]])
        if self.mode in ("hybrid", "bm25_temporal", "bm25"):
            bm25 = BM25Index([tokenize(t) for t in texts])
            scores = bm25.scores(tokenize(query))
            order = sorted(range(len(items)), key=lambda i: (-scores[i], items[i].id))
            rankings.append([items[i].id for i in order[: self.n_candidates]])
        if self.mode in ("hybrid", "dense"):
            if self.encoder is None:
                raise RuntimeError("dense retrieval requested without an encoder")
            import numpy as np

            doc_vecs = self.encoder.encode(texts)
            q_vec = self.encoder.encode([query])[0]
            sims = np.asarray(doc_vecs) @ np.asarray(q_vec)
            order = sorted(range(len(items)), key=lambda i: (-float(sims[i]), items[i].id))
            rankings.append([items[i].id for i in order[: self.n_candidates]])
        fused = rrf(rankings, self.rrf_k) if len(rankings) > 1 else rankings[0]
        by_id = {e.id: e for e in items}
        top = [by_id[i] for i in fused[:k]]
        if self.expand_sections and any(e.source.endswith("_line") for e in top):
            # section expansion: the top-N distinct (note, section) blocks among the hits are returned whole,
            # in document order, so a hit inside an IMPRESSION brings the complete section with it
            blocks: list[tuple] = []
            for e in top:
                key = (e.fields.get("note"), e.fields.get("section"))
                if e.source.endswith("_line") and key not in blocks:
                    blocks.append(key)
                if len(blocks) >= self.expand_sections:
                    break
            expanded = [e for e in items if e.source.endswith("_line") and (e.fields.get("note"), e.fields.get("section")) in blocks]
            expanded.sort(key=lambda e: (blocks.index((e.fields.get("note"), e.fields.get("section"))), int(e.fields.get("line", 0))))
            seen = {e.id for e in expanded}
            merged = expanded + [e for e in top if e.id not in seen]
            return merged[: max(k, self.expand_cap)]
        return top


@dataclass
class AccessLog:
    """A_t: every evidence id returned by a retriever/tool or inserted into the prompt."""

    entries: list[dict] = field(default_factory=list)

    def record(self, stage: str, ids: Iterable[str]) -> None:
        self.entries.append({"stage": stage, "ids": list(ids)})

    def ids(self) -> set[str]:
        out: set[str] = set()
        for entry in self.entries:
            out.update(entry["ids"])
        return out
