"""Tier-C configuration table: the eight shared configurations plus the three Tier-C baselines exist, and the
`specialized_tools` baseline pins FHIR-AgentBench's focused retrieval mode while every other configuration keeps
the requested mode (no data needed)."""

from __future__ import annotations

from scopebench.tierc.agent import CONFIGS, effective_mode

SHARED = ["prompt_only", "wrong_basis", "scope_aware", "audit", "full", "token_matched", "high_compute"]
TIERC_BASELINES = ["direct_api", "specialized_tools", "security_label_consent"]


def test_configuration_table_is_complete():
    for name in SHARED + TIERC_BASELINES:
        assert name in CONFIGS, name
    assert CONFIGS["direct_api"]["raw_view"] and CONFIGS["direct_api"]["enforcement"] == "none"
    assert CONFIGS["security_label_consent"]["enforcement"] == "security_label"
    assert CONFIGS["specialized_tools"]["enforcement"] == "none" and not CONFIGS["specialized_tools"]["guard"]
    assert CONFIGS["token_matched"]["budget"] and CONFIGS["high_compute"]["budget"]


def test_specialized_tools_pins_focused_mode():
    assert effective_mode("specialized_tools", "broad") == "focused"
    assert effective_mode("specialized_tools", "focused") == "focused"
    for name in SHARED + ["direct_api", "security_label_consent"]:
        assert effective_mode(name, "broad") == "broad"
        assert effective_mode(name, "focused") == "focused"
