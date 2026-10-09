from __future__ import annotations

import sys
from pathlib import Path
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_medagentbench_corrected import (
    extract_official_prompt,
    parse_post,
    strip_model_wrappers,
    validate_get_url,
    OFFICIAL_TASK_MODULE,
)


@pytest.mark.skipif(not OFFICIAL_TASK_MODULE.exists(), reason="Clone the pinned MedAgentBench source for this integration check")
def test_official_prompt_is_loaded_from_source() -> None:
    prompt = extract_official_prompt(OFFICIAL_TASK_MODULE)
    assert "FINISH([answer1, answer2, ...])" in prompt
    assert "{functions}" in prompt


def test_action_parsing_and_local_url_guard() -> None:
    base = "http://127.0.0.1:8080/fhir/"
    url = validate_get_url(
        "http://127.0.0.1:8080/fhir/Observation?patient=S1&code=MG",
        base,
    )
    assert "_format=json" in url
    post_url, payload = parse_post(
        'POST http://127.0.0.1:8080/fhir/Observation\n'
        '{"resourceType":"Observation"}',
        base,
    )
    assert post_url.endswith("/Observation")
    assert payload["resourceType"] == "Observation"
    try:
        validate_get_url("https://example.com/fhir/Patient?identifier=S1", base)
    except ValueError:
        pass
    else:
        raise AssertionError("external URL was not blocked")


def test_strip_wrappers() -> None:
    assert strip_model_wrappers("```tool_code\nFINISH([1])\n```<|im_end|>") == "FINISH([1])"
