"""The frontier file-queue backend after DEV-21: paged rows, size packing, the route ceiling (DEV-19 guard text) and
request replay. No subagent is involved: completion files are written by the test."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scopebench.backends.base import Request
from scopebench.backends.claude_batch import (CHARS_PER_TOKEN, PAGE_CHARS, ROUTE_CONTEXT_TOKENS, ClaudeBatchBackend, read_batch, row_lines)
from scopebench.context_guard import parse_overflow

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/scopebench"))
import frontier_pending  # noqa: E402


def _text(n: int, seed: str) -> str:
    """FHIR-like filler with quotes and newlines (the characters JSON escaping lengthens)."""
    unit = '{\n  "resourceType": "Observation",\n  "id": "%s",\n  "valueQuantity": { "value": 1.5, "unit": "mg/dL" }\n},\n' % seed
    return (unit * (n // len(unit) + 1))[:n]


def _short(key: str = "t1") -> Request:
    return Request(key, [{"role": "system", "content": "Answer."}, {"role": "user", "content": _text(2000, key)}], 64)


def _long(key: str, sizes: tuple[int, ...]) -> Request:
    roles = ["system", "user", "assistant", "user", "assistant", "user", "assistant", "user"]
    return Request(key, [{"role": roles[i], "content": _text(n, f"{key}-{i}")} for i, n in enumerate(sizes)], 1024)


def _write_out(batch: Path, completions: dict[str, str]) -> None:
    with open(batch.with_suffix(".out.jsonl"), "w", encoding="utf-8") as f:
        for k, v in completions.items():
            f.write(json.dumps({"task_id": k, "completion": v}) + "\n")


def test_short_rows_keep_the_pre_dev21_line_format(tmp_path):
    r = _short()
    lines = row_lines(r)
    assert lines == [json.dumps({"task_id": "t1", "messages": r.messages, "max_tokens": 64}, ensure_ascii=False)]
    b = tmp_path / "x.jsonl"
    b.write_text(lines[0] + "\n")
    assert read_batch(b) == [{"task_id": "t1", "messages": r.messages, "max_tokens": 64}]


def test_long_rows_are_paged_and_reassembled(tmp_path):
    r = _long("big", (5_000, 90_000, 800, 40_000))
    lines = row_lines(r)
    assert len(lines) > 1 and all(len(line.encode()) <= int(PAGE_CHARS * 1.4) for line in lines)
    first, *rest = (json.loads(line) for line in lines)
    assert first["continued"] is True and sum(len(m["content"]) for m in first["messages"]) == PAGE_CHARS
    assert first["messages"][2]["content"] == r.messages[2]["content"]            # the short message stays whole in the first line
    assert all("append" in d and d["task_id"] == "big" for d in rest) and {d["message_index"] for d in rest} == {1, 3}
    assert [d["continued"] for d in rest][-1] is False and all(d["continued"] for d in rest[:-1])
    assert [d["message_index"] for d in rest] == sorted(d["message_index"] for d in rest)      # chunks in message order
    # a MedAgentBench-shaped row: long system prompt, short question, short GET, long response, short GET, long response, short GET
    med = _long("med", (15_000, 100, 120, 25_000, 120, 20_000, 110))
    med_lines = [json.loads(line) for line in row_lines(med)]
    assert len(med_lines) == 3 and {d["message_index"] for d in med_lines[1:]} <= {3, 5}      # only the long responses are split
    b2 = tmp_path / "med.jsonl"
    b2.write_text("".join(line + "\n" for line in row_lines(med)))
    assert read_batch(b2) == [{"task_id": "med", "messages": med.messages, "max_tokens": 1024}]
    b = tmp_path / "x.jsonl"
    b.write_text("".join(line + "\n" for line in lines))
    assert read_batch(b) == [{"task_id": "big", "messages": r.messages, "max_tokens": 1024}]
    # a mixed file: paged row between short rows
    b.write_text("".join(line + "\n" for req in (_short("a"), r, _short("c")) for line in row_lines(req)))
    rows = read_batch(b)
    assert [x["task_id"] for x in rows] == ["a", "big", "c"] and rows[1]["messages"] == r.messages
    assert frontier_pending.n_rows(b) == 3


def test_files_are_packed_by_row_count_and_by_size(tmp_path):
    be = ClaudeBatchBackend(batch_dir=tmp_path, batch_size=8, max_file_chars=400_000)
    paths = be.write_batches([_short(f"s{i}") for i in range(10)], "p")
    assert [len(read_batch(p)) for p in paths] == [8, 2]
    big = [_long(f"L{i}", (120_000,)) for i in range(5)]
    paths = be.write_batches(big, "q")
    assert [len(read_batch(p)) for p in paths] == [3, 2]
    assert all(sum(len(m["content"]) for row in read_batch(p) for m in row["messages"]) <= 400_000 for p in paths)
    assert [row["task_id"] for p in paths for row in read_batch(p)] == [r.key for r in big]      # order kept


def test_requests_above_the_ceiling_get_the_rejection_the_guard_parses(tmp_path):
    be = ClaudeBatchBackend(batch_dir=tmp_path, poll_seconds=0.01, wait_limit_seconds=0.05)
    r = _long("huge", (10_000, ROUTE_CONTEXT_TOKENS * CHARS_PER_TOKEN))
    out = be.generate_many([r])
    g = out["huge"]
    assert g.error and g.extra.get("route_ceiling") and g.text == ""
    assert parse_overflow(g.error) == (sum(len(m["content"]) for m in r.messages) // CHARS_PER_TOKEN, ROUTE_CONTEXT_TOKENS)
    assert not list(tmp_path.glob("*.jsonl")) and be.n_ceiling == 1          # nothing written for it
    assert be.metadata()["route_ceiling_requests"] == 1 and be.metadata()["route"]["context_tokens"] == ROUTE_CONTEXT_TOKENS
    # just under the ceiling: delivered (paged), not rejected
    r2 = _long("under", (ROUTE_CONTEXT_TOKENS * CHARS_PER_TOKEN - 10,))
    out = be.generate_many([r2])
    assert out["under"].error == "no completion produced" and len(list(tmp_path.glob("*.jsonl"))) == 1


def test_replay_reuses_completions_and_issues_only_new_requests(tmp_path):
    first = ClaudeBatchBackend(batch_dir=tmp_path, batch_size=8)
    reqs = [_short("a"), _long("b", (5_000, 70_000)), _short("c")]
    (path,) = first.write_batches(reqs, "b100")
    _write_out(path, {"a": "answer a", "b": "answer b"})                     # c never answered
    second = ClaudeBatchBackend(batch_dir=tmp_path, batch_size=8, poll_seconds=0.01, wait_limit_seconds=0.05)
    changed_b = Request("b", reqs[1].messages + [{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}], 1024)
    out = second.generate_many([reqs[0], reqs[1], changed_b.__class__("b2", changed_b.messages, 1024), reqs[2]])
    assert out["a"].text == "answer a" and out["a"].extra["replayed_from"] == path.name
    assert out["b"].text == "answer b" and out["b"].extra["replayed_from"] == path.name
    assert out["b2"].error == "no completion produced" and out["c"].error == "no completion produced"
    # c was issued before and is still unanswered: it is waited for in its existing file, not re-issued (DEV-22)
    new = sorted(p for p in tmp_path.glob("*.jsonl") if not p.name.endswith(".out.jsonl") and p != path)
    assert len(new) == 1 and [row["task_id"] for row in read_batch(new[0])] == ["b2"]
    assert second.n_replayed == 2 and second.n_attached == 1 and second.metadata()["replayed_requests"] == 2 and second.metadata()["attached_requests"] == 1
    # the pending listing counts rows, not lines, and lists only the unfinished files
    pend = frontier_pending.pending([tmp_path])
    assert [(Path(x["batch"]).name, x["n_rows"], x["rows_done"]) for x in pend] == [(path.name, 3, 2), (new[0].name, 1, 0)]
    # once c is answered in its old file, a third run replays all of a, b, c and issues nothing
    _write_out(path, {"a": "answer a", "b": "answer b", "c": "answer c"})
    third = ClaudeBatchBackend(batch_dir=tmp_path, batch_size=8, poll_seconds=0.01, wait_limit_seconds=0.05)
    out = third.generate_many(reqs)
    assert [out[k].text for k in ("a", "b", "c")] == ["answer a", "answer b", "answer c"] and third.n_replayed == 3 and third.n_attached == 0
    assert len([p for p in tmp_path.glob("*.jsonl") if not p.name.endswith(".out.jsonl")]) == 2
    # collect() limited to the wanted rows ignores rows of a shared file that this call did not ask for
    got, missing = third.collect([path], wanted={"a"})
    assert set(got) == {"a"} and missing == []
