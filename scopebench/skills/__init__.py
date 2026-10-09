"""ScopeBench inference-time ChartSkills (implementer B), scopebench/SKILL_SPEC.md v1.2 §1–§7.

Public surface:
    SPEC_VERSION  - the spec version this package implements
    OPERATIONS    - name -> fn(evidence, contract, params) -> dict
"""

from .operations import (
    OPERATIONS,
    SPEC_VERSION,
    cohort_criteria,
    first_after,
    lab_trend_k3,
    last_before,
    med_dose_k3,
    nearest_k,
    outcome_lookup,
    window_count,
    window_mean,
    window_sum,
)

__all__ = [
    "OPERATIONS",
    "SPEC_VERSION",
    "cohort_criteria",
    "first_after",
    "lab_trend_k3",
    "last_before",
    "med_dose_k3",
    "nearest_k",
    "outcome_lookup",
    "window_count",
    "window_mean",
    "window_sum",
]
