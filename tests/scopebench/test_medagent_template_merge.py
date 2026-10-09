"""DEV-29: the MedAgentBench repair prompt under a chat template that requires strict user/assistant alternation. An
episode that ended on the environment's message (task limit) and then receives the guard's repair request carries two
consecutive user turns; a template that rejects them (Gemma 3) gets the turns merged into one, in order, and the episode
records the merge; a template that accepts them is rendered unchanged and records nothing. No server or model needed."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scopebench.tierc.medagent import Episode, merge_consecutive_user_turns, render  # noqa: E402


class StrictTokenizer:
    """Rejects consecutive same-role turns, as the Gemma 3 template does."""
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        for a, b in zip(messages, messages[1:]):
            if a["role"] == b["role"]:
                raise ValueError("Conversation roles must alternate user/assistant/user/assistant/...")
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<assistant>"


class TolerantTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<assistant>"


def episode(messages):
    return Episode(task=None, case={}, store=None, messages=messages)  # type: ignore[arg-type]


ENDED_ON_ENVIRONMENT = [{"role": "system", "content": "S"}, {"role": "user", "content": "task"},
                        {"role": "assistant", "content": "GET /Observation"}, {"role": "user", "content": "{\"entry\": []}"},
                        {"role": "user", "content": "{\"scopeguard_errors\": [\"schema\"]}"}]


def test_merge_joins_runs_of_user_turns_in_order_and_copies():
    merged = merge_consecutive_user_turns(ENDED_ON_ENVIRONMENT)
    assert [m["role"] for m in merged] == ["system", "user", "assistant", "user"]
    assert merged[-1]["content"] == "{\"entry\": []}\n\n{\"scopeguard_errors\": [\"schema\"]}"
    assert len(ENDED_ON_ENVIRONMENT) == 5 and ENDED_ON_ENVIRONMENT[-1]["role"] == "user"   # the input is not mutated
    assert merge_consecutive_user_turns(ENDED_ON_ENVIRONMENT[:4]) == ENDED_ON_ENVIRONMENT[:4]


def test_strict_template_gets_the_merged_turns_and_the_episode_records_it():
    ep = episode(list(ENDED_ON_ENVIRONMENT))
    out = render(ep, StrictTokenizer())
    assert out.count("<user>") == 2 and "{\"entry\": []}\n\n{\"scopeguard_errors\"" in out
    assert ep.template_merges == 1 and ep.trace[-1]["role"] == "template_merge"
    assert len(ep.messages) == 5   # the message list (and so the history the grader sees) is unchanged


def test_tolerant_template_is_rendered_unchanged():
    ep = episode(list(ENDED_ON_ENVIRONMENT))
    out = render(ep, TolerantTokenizer())
    assert out.count("<user>") == 3 and ep.template_merges == 0 and not ep.trace


def test_other_template_errors_still_raise():
    class Broken:
        def apply_chat_template(self, messages, **kw):
            raise RuntimeError("unrelated")
    with pytest.raises(RuntimeError):
        render(episode(list(ENDED_ON_ENVIRONMENT[:4])), Broken())
