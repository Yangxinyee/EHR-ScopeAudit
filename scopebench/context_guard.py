"""Final-answer guard for tool loops at the serving context limit (DEV-19).

The tool loops (Tier-A/B compute controls in `tools_loop.py`; Tier-C FHIR-AgentBench in `tierc/agent.py`; Tier-C
MedAgentBench in `tierc/medagent.py`) budget by cumulative tokens, while the server limits the live conversation. When
the next turn's prompt no longer fits, sglang rejects the request with HTTP 400 ("The input (N tokens) is longer than
the model's context length (C tokens)" or "Requested token count exceeds the model's maximum context length of C tokens
... N tokens from the input messages and M tokens for the completion"). Until DEV-19 such an episode ended as a
generation error and was scored as a failure. Now the loop asks for the final answer instead: the message that did not
fit (the last tool result / environment feedback) is replaced by a short notice, and the completion length of that one
request is capped so that it fits after the last request that did fit.

The guard is triggered only by the server's own rejection, so an episode in which every request fitted is unchanged
(byte-identical under deterministic serving). It fires at most once per episode; if the guarded request is rejected
too, the episode keeps its generation error as before.
"""

from __future__ import annotations

import re

_INPUT_LONGER = re.compile(r"The input \((\d+) tokens\) is longer than the model's context length \((\d+) tokens\)")
_TOTAL_EXCEEDS = re.compile(r"maximum context length of (\d+) tokens\. You requested a total of \d+ tokens: (\d+) tokens from the input messages")

NOTICE = ("The last tool result could not be shown: the conversation reached the model's context limit. "
          "Return your final answer now from the evidence you already have.")
NOTICE_TOKENS = 192          # notice + chat-template overhead of the two re-sent messages, generous upper bound
MIN_FINAL_TOKENS = 64


def parse_overflow(error: str | None) -> tuple[int, int] | None:
    """(input tokens of the rejected request, context length) if `error` is sglang's context-length rejection."""
    if not error:
        return None
    m = _INPUT_LONGER.search(error)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _TOTAL_EXCEEDS.search(error)
    if m:
        return int(m.group(2)), int(m.group(1))
    return None


def final_max_tokens(default: int, context: int, last_ok_tokens_in: int, last_ok_tokens_out: int) -> int:
    """Completion cap for the guarded request: what is left after the last request that fitted, its reply and the notice."""
    room = context - (last_ok_tokens_in + last_ok_tokens_out) - NOTICE_TOKENS
    return max(MIN_FINAL_TOKENS, min(default, room))
