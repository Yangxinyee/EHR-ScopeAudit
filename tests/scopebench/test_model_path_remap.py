"""DEV-15: model weights under another root on another machine (H200 node). Only file access is remapped; unset, every
path is unchanged, so the workstation's behaviour and the retrieval-cache signature (logical encoder path) are identical."""

from __future__ import annotations

from scopebench.config import remap_model_path


def test_unset_is_identity(monkeypatch):
    monkeypatch.delenv("SCOPEBENCH_MODEL_ROOT_REMAP", raising=False)
    assert remap_model_path("/opt/models/BAAI__bge-m3") == "/opt/models/BAAI__bge-m3"


def test_remap_under_root_only(monkeypatch):
    monkeypatch.setenv("SCOPEBENCH_MODEL_ROOT_REMAP", "/opt/models=/srv/models/")
    assert remap_model_path("/opt/models/Qwen__Qwen3-4B-Instruct-2507") == "/srv/models/Qwen__Qwen3-4B-Instruct-2507"
    assert remap_model_path("/opt/models") == "/srv/models"
    assert remap_model_path("/opt/models2/x") == "/opt/models2/x"      # a prefix that is not the root
    assert remap_model_path("/other/models/x") == "/other/models/x"


def test_encoder_name_keeps_the_logical_path(monkeypatch):
    import types
    import scopebench.retrieval as r
    captured = {}
    fake = types.SimpleNamespace(SentenceTransformer=lambda path, device: captured.setdefault("path", path))
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", fake)
    monkeypatch.setenv("SCOPEBENCH_MODEL_ROOT_REMAP", "/opt/models=/srv/models")
    enc = r.DenseEncoder("/opt/models/BAAI__bge-m3", device="cpu")
    assert captured["path"] == "/srv/models/BAAI__bge-m3"
    assert enc.name == "bge-m3@/opt/models/BAAI__bge-m3"
