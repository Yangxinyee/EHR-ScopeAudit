"""scopebench.oracle: gold-generation implementation of scopebench/SKILL_SPEC.md (implementer A).

Public surface (spec §0): `SPEC_VERSION` and `OPERATIONS[name](evidence, contract, params) -> dict`.
"""

from __future__ import annotations

from .operations import OPERATIONS

SPEC_VERSION = "1.2"

__all__ = ["SPEC_VERSION", "OPERATIONS"]
