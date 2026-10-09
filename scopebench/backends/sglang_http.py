"""SGLang OpenAI-compatible backend (plan §3.5): async client, bounded concurrency, retries,
per-request token accounting, server-info capture. Requests are sent in the order given
(the runner orders them by case for prefix reuse) and results are keyed."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from .base import Generation, Request


@dataclass
class SGLangBackend:
    base_url: str = "http://127.0.0.1:30000"
    concurrency: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 2027
    enable_thinking: bool = False
    timeout_seconds: float = 900.0
    max_retries: int = 3
    name: str = "sglang"
    _server_info: dict[str, Any] | None = field(default=None, repr=False)

    def server_info(self) -> dict[str, Any]:
        import httpx

        if self._server_info is None:
            r = httpx.get(f"{self.base_url}/get_server_info", timeout=30)
            r.raise_for_status()
            self._server_info = r.json()
        return self._server_info

    def metadata(self) -> dict[str, Any]:
        info = self.server_info()
        keys = [
            "model_path", "version", "attention_backend", "prefill_attention_backend", "decode_attention_backend",
            "disable_radix_cache", "chunked_prefill_size", "max_running_requests", "max_total_num_tokens",
            "mem_fraction_static", "context_length", "dtype", "kv_cache_dtype", "disable_cuda_graph",
            "enable_deterministic_inference", "random_seed", "tp_size",
        ]
        return {
            "backend": self.name,
            "base_url": self.base_url,
            "concurrency": self.concurrency,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "enable_thinking": self.enable_thinking,
            "server_info": {k: info.get(k) for k in keys},
        }

    def generate_many(self, requests: list[Request]) -> dict[str, Generation]:
        return asyncio.run(self._run(requests))

    def generate_many_raw(self, requests: list) -> dict[str, Generation]:
        """Raw-text generation (`/generate`) for prompts rendered by the caller's chat template (Tier-C tool loop)."""
        return asyncio.run(self._run_raw(requests))

    async def _run_raw(self, requests: list) -> dict[str, Generation]:
        import httpx

        sem = asyncio.Semaphore(self.concurrency)
        out: dict[str, Generation] = {}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            async def one(req) -> None:
                async with sem:
                    out[req.key] = await self._one_raw(client, req)

            await asyncio.gather(*(one(r) for r in requests))
        return out

    async def _one_raw(self, client, req) -> Generation:
        body = {
            "text": req.prompt.replace("<image_soft_token>", "<image soft token>"),
            "sampling_params": {"max_new_tokens": req.max_tokens, "temperature": self.temperature, "top_p": self.top_p, "stop": list(req.stop) or None,
                                "skip_special_tokens": False},
        }
        last_error = None
        for attempt in range(self.max_retries):
            started = time.perf_counter()
            try:
                r = await client.post(f"{self.base_url}/generate", json=body)
                if r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code == 400:   # deterministic rejection (e.g. context length, DEV-19): reported with the server's message, not retried
                    return Generation(text="", tokens_in=0, tokens_out=0, latency_seconds=time.perf_counter() - started, error=f"HTTP 400: {r.text[:600]}")
                r.raise_for_status()
                j = r.json()
                info = j.get("meta_info", {})
                return Generation(text=j.get("text", ""), tokens_in=int(info.get("prompt_tokens", 0)), tokens_out=int(info.get("completion_tokens", 0)),
                                  latency_seconds=time.perf_counter() - started, cached_tokens=info.get("cached_tokens"),
                                  finish_reason=(info.get("finish_reason") or {}).get("type") if isinstance(info.get("finish_reason"), dict) else info.get("finish_reason"))
            except Exception as exc:  # noqa: BLE001 - retried, then reported
                last_error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(2.0 * (attempt + 1))
        return Generation(text="", tokens_in=0, tokens_out=0, latency_seconds=0.0, error=last_error)

    async def _run(self, requests: list[Request]) -> dict[str, Generation]:
        import httpx

        sem = asyncio.Semaphore(self.concurrency)
        out: dict[str, Generation] = {}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            async def one(req: Request) -> None:
                async with sem:
                    out[req.key] = await self._one(client, req)

            await asyncio.gather(*(one(r) for r in requests))
        return out

    async def _one(self, client, req: Request) -> Generation:
        # Gemma-3 tokenizers map the literal '<image_soft_token>' to id 262144, which has no embedding row in
        # MedGemma; a prompt containing it kills the server. Real chart text never needs it.
        messages = [{**m, "content": m["content"].replace("<image_soft_token>", "<image soft token>")} if isinstance(m.get("content"), str) else m for m in req.messages]
        body = {
            "model": "default",
            "messages": messages,
            "max_tokens": req.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        last_error = None
        for attempt in range(self.max_retries):
            started = time.perf_counter()
            try:
                r = await client.post(f"{self.base_url}/v1/chat/completions", json=body)
                if r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code == 400:   # deterministic rejection (e.g. context length, DEV-19): reported with the server's message, not retried
                    return Generation(text="", tokens_in=0, tokens_out=0, latency_seconds=time.perf_counter() - started, error=f"HTTP 400: {r.text[:600]}")
                r.raise_for_status()
                j = r.json()
                choice = j["choices"][0]
                usage = j.get("usage", {})
                details = usage.get("prompt_tokens_details") or {}
                return Generation(
                    text=choice["message"]["content"] or "",
                    tokens_in=int(usage.get("prompt_tokens", 0)),
                    tokens_out=int(usage.get("completion_tokens", 0)),
                    latency_seconds=time.perf_counter() - started,
                    cached_tokens=details.get("cached_tokens") if isinstance(details, dict) else None,
                    finish_reason=choice.get("finish_reason"),
                )
            except Exception as exc:  # noqa: BLE001 - retried, then reported
                last_error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(2.0 * (attempt + 1))
        return Generation(text="", tokens_in=0, tokens_out=0, latency_seconds=0.0, error=last_error)
