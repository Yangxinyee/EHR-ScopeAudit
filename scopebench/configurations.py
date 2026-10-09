"""Configurations (plan §4.2): the eight paper configurations, the attribution arms, and baselines,
expressed as a small set of orthogonal switches. Late filtering is derived post hoc from
prompt_only outputs (no extra generation), see scoring.late_filter.

Evidence selection
  unscoped   : retriever over the whole pool, no predicate, k = pool_k (whole pool for Tier-B prompt-only)
  scoped     : predicate before ranking, top-k
  wrong_basis: predicate with the basis swapped, top-k
  refill     : ScopePair guard_refill — legal members of the unscoped top-k, refilled with the nearest
               legal objects (equals the scoped top-k for temporal retrieval; kept for the regression)
Guard / repair
  guard      : none | schema_only | scope_only | scope_completeness   (what the guard reports back)
  repair     : none | feedback | feedback_no_evidence | generic    (second call content; issued only when
               the guard finds errors, so all repair arms use equal calls)
Skill
  skill      : none | payload_only | verify_no_replace | replace_no_payload | full
Compute controls (tools=True) are ReAct loops implemented in M2.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Configuration:
    name: str
    evidence: str            # unscoped | scoped | wrong_basis
    guard: str = "none"      # none | schema_only | scope_completeness
    repair: str = "none"     # none | feedback | generic
    skill: str = "none"      # none | payload_only | verify_no_replace | replace_no_payload | full
    check_completeness: bool = True
    tools: bool = False
    token_budget_factor: float | None = None  # for token_matched / high_compute (M2)
    predicate_engine: str | None = None       # None = scopebench.predicate; 'temporal_sql_view' | 'abac_pdp' (S4 baselines)
    query_scope_hint: bool = False            # reranker baseline: the scope is stated in the retrieval query, never enforced
    repair_resupply_topk: bool = False        # ScopePair-compat: the repair turn re-sends the legal top-k (frozen 2026-07 behaviour)
    paper_row: str = ""

    @property
    def uses_guard(self) -> bool:
        return self.guard != "none" or self.skill in {"verify_no_replace", "replace_no_payload", "full"}

    @property
    def skill_payload(self) -> bool:
        return self.skill in {"payload_only", "verify_no_replace", "full"}

    @property
    def skill_verify(self) -> bool:
        return self.skill in {"verify_no_replace", "replace_no_payload", "full"}

    @property
    def skill_replace(self) -> bool:
        return self.skill in {"replace_no_payload", "full"}


CONFIGS: dict[str, Configuration] = {}


def _add(c: Configuration) -> Configuration:
    CONFIGS[c.name] = c
    return c


# --- Table 2 rows -------------------------------------------------------------------------------
_add(Configuration("prompt_only", evidence="unscoped", paper_row="Prompt-only"))
# late_filter: derived from prompt_only outputs (scoring.late_filter)
_add(Configuration("wrong_basis", evidence="wrong_basis", paper_row="Wrong-basis scope"))
_add(Configuration("scope_aware", evidence="scoped", paper_row="Scope-aware retrieval"))
_add(Configuration("audit", evidence="scoped", guard="scope_completeness", repair="feedback", paper_row="+ evidence audit"))
_add(Configuration("full", evidence="scoped", guard="scope_completeness", repair="feedback", skill="full", paper_row="+ skill grounding"))
_add(Configuration("token_matched", evidence="unscoped", tools=True, token_budget_factor=None, paper_row="Token-matched control"))
_add(Configuration("high_compute", evidence="unscoped", tools=True, token_budget_factor=4.62, paper_row="High-compute control"))

# --- Table 3 attribution arms -------------------------------------------------------------------
_add(Configuration("retry_generic", evidence="scoped", guard="scope_completeness", repair="generic", paper_row="+ equal-call generic retry"))
_add(Configuration("feedback_schema_only", evidence="scoped", guard="schema_only", repair="feedback", paper_row="+ schema feedback and retry"))
# "+ scope/completeness feedback and retry" == audit
_add(Configuration("skill_payload_only", evidence="scoped", guard="scope_completeness", repair="feedback", skill="payload_only", paper_row="+ skill payload only"))
_add(Configuration("skill_verify_no_replace", evidence="scoped", guard="scope_completeness", repair="feedback", skill="verify_no_replace", paper_row="+ payload, verify, no replacement"))
_add(Configuration("skill_replace_no_payload", evidence="scoped", guard="scope_completeness", repair="feedback", skill="replace_no_payload", paper_row="+ replacement, no payload"))

# --- Supp. Table S4 Tier-B baselines ---------------------------------------------------------
_add(Configuration("citation_validator", evidence="unscoped", guard="scope_only", repair="feedback_no_evidence", check_completeness=False, paper_row="Output citation validator"))
_add(Configuration("reranker", evidence="unscoped", query_scope_hint=True, paper_row="Evidence reranker"))
_add(Configuration("temporal_sql_view", evidence="scoped", predicate_engine="temporal_sql_view", paper_row="Temporal SQL view"))
_add(Configuration("abac_pdp", evidence="scoped", predicate_engine="abac_pdp", paper_row="Generic ABAC policy engine"))

# --- ScopePair regression (frozen semantics: 3 nearest, ScopePair guard without R_t completeness) ---
_add(Configuration("sp_unscoped", evidence="unscoped", check_completeness=False, paper_row="ScopePair unscoped"))
_add(Configuration("sp_wrong_time_scope_first", evidence="wrong_basis", check_completeness=False, paper_row="ScopePair wrong_time_scope_first"))
_add(Configuration("sp_scope_first", evidence="scoped", check_completeness=False, paper_row="ScopePair scope_first"))
_add(Configuration("sp_guard_refill", repair_resupply_topk=True, evidence="refill", guard="scope_completeness", repair="feedback", check_completeness=False, paper_row="ScopePair guard_refill"))
_add(Configuration("sp_clinehr_os", repair_resupply_topk=True, evidence="scoped", guard="scope_completeness", repair="feedback", skill="full", check_completeness=False, paper_row="ScopePair clinehr_os"))


def get(name: str) -> Configuration:
    return CONFIGS[name]
