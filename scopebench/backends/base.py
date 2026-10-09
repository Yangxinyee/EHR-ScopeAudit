from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

Messages = list[dict[str, str]]


@dataclass
class Request:
    key: str
    messages: Messages
    max_tokens: int


@dataclass
class RawRequest:
    """A fully rendered prompt (chat template already applied, e.g. with tools) for backends that accept raw text."""
    key: str
    prompt: str
    max_tokens: int
    stop: list[str] = field(default_factory=list)


@dataclass
class Generation:
    text: str
    tokens_in: int
    tokens_out: int
    latency_seconds: float
    cached_tokens: int | None = None
    finish_reason: str | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    name: str

    def generate_many(self, requests: list[Request]) -> dict[str, Generation]:
        """Generate for every request; returns key -> Generation. Order of execution may be batched."""
        ...

    def metadata(self) -> dict[str, Any]: ...
