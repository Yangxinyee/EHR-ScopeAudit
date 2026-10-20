"""Versioned output parser (plan §4.3). Unparseable -> None -> no credit, and it is a reported stage.

v1 is the frozen ScopePair algorithm, verbatim in semantics (scripts/run_scopepair_fhir.py,
`extract_json`): strip one leading ```/```json fence and one trailing ``` fence, try the whole
text, else the substring from the first '{' to the last '}'. Prose around a single object is
therefore ignored; two objects, or none, make the submission unparseable (docstring corrected
with DEV-38; the code is unchanged).
"""

from __future__ import annotations

import json
import re
from typing import Any

PARSER_VERSION = "v1"


def extract_json(text: str | None) -> dict[str, Any] | None:
    if text is None:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None
