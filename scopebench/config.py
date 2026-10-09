"""Paths and environment for ScopeBench (plan §3.8: no absolute author paths in metadata)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


PUBLIC_DATA = env_path("SCOPEBENCH_PUBLIC_DATA", ROOT / "local_data")
LLM_ROOT = env_path("SCOPEBENCH_LLM_ROOT", ROOT / "models")
EXPERIMENTS = ROOT / "experiments"
SEED = int(os.environ.get("SCOPEBENCH_SEED", "2027"))


def remap_model_path(path: Path | str) -> str:
    """File-access path of model weights on another machine (DEV-15, the H200 node): SCOPEBENCH_MODEL_ROOT_REMAP=
    "<old root>=<new root>" rewrites a path under <old root>; unset, the path is returned unchanged. Only file access is
    remapped — the logical (panel) path stays the identity wherever it is recorded, e.g. in the retrieval-cache signature."""
    s = str(path)
    spec = os.environ.get("SCOPEBENCH_MODEL_ROOT_REMAP", "")
    if "=" in spec:
        old, new = (x.rstrip("/") for x in spec.split("=", 1))
        if old and (s == old or s.startswith(old + "/")):
            return new + s[len(old):]
    return s


def relpath(path: Path | str) -> str:
    """Repo-relative string for metadata; paths outside the repo keep only their last two parts."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return "/".join(p.parts[-2:])
