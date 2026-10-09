"""LLM backends (plan §4.5): one interface, three implementations (sglang HTTP, transformers, Claude batch files)."""

from .base import Backend, Generation, Request  # noqa: F401
