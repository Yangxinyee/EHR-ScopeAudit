"""DEV-36 addendum 2: both SGLang request paths carry the decoding seed (the raw `/generate` path used by the Tier-C
tool loop did not, so seed cells at temperature > 0 repeated one sample)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scopebench.backends.base import Request  # noqa: E402
from scopebench.backends.sglang_http import SGLangBackend  # noqa: E402


class _Resp:
    status_code = 200

    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._p


class _Client:
    def __init__(self):
        self.bodies = []

    async def post(self, url, json):
        self.bodies.append((url, json))
        if url.endswith("/generate"):
            return _Resp({"text": "ok", "meta_info": {"prompt_tokens": 1, "completion_tokens": 1}})
        return _Resp({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


def test_raw_and_chat_paths_send_the_seed():
    b = SGLangBackend(base_url="http://x", concurrency=1, seed=47, temperature=0.7, enable_thinking=False)
    c = _Client()
    asyncio.run(b._one_raw(c, SimpleNamespace(key="k", prompt="p", max_tokens=8, stop=())))
    asyncio.run(b._one(c, Request("k", [{"role": "user", "content": "p"}], 8)))
    raw = next(body for url, body in c.bodies if url.endswith("/generate"))
    chat = next(body for url, body in c.bodies if not url.endswith("/generate"))
    assert raw["sampling_params"]["sampling_seed"] == 47 and raw["sampling_params"]["temperature"] == 0.7
    assert chat["seed"] == 47
