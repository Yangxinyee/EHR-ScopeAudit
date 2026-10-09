"""Frontier checkpoint via Claude Code subagents (plan §3.7): a file-queue backend.

`generate_many` writes batch files and then waits for the corresponding `.out.jsonl` files to
appear (the executing agent spawns `frontier-llm` subagents on them). It validates every row and
returns Generations. Token counts are estimated from character counts (chars/4) because the
subagent route reports none; the estimate is flagged.

DEV-21 (2026-10-03): the route has delivery limits of its own. The subagent reads a batch file with a
tool that returns at most ~25k tokens of any one line and cannot page inside a line, so
  (a) a row longer than one page is written as a first line plus continuation lines, each carrying
      a chunk (`append`) of one message's content; the subagent definition says how to reassemble;
  (b) files are packed by size as well as by row count;
  (c) a request whose estimated size exceeds the open-weight serving context (32,768 tokens; chars/4
      on this route) is not delivered: the backend returns the rejection text an sglang server gives,
      so the loops apply the DEV-19 final-answer guard exactly as for an open-weight checkpoint;
  (d) a request already answered in an earlier batch file of the same run directory (identical task
      id, messages and max_tokens) is replayed from that completion instead of being re-issued, so a
      restarted harness continues where the stopped one was;
  (e) DEV-22: a request already written to a batch file that has no completion yet is waited for in
      that file instead of being written again, so a harness resumed from the loop's checkpoint
      leaves no orphaned batch files behind.
Rows that fit one page are written exactly as before DEV-21.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import Generation, Request

ROUTE_CONTEXT_TOKENS = 32768   # the open-weight serving context (Tier-C protocol), applied to this route too (DEV-21)
CHARS_PER_TOKEN = 4            # the route's declared token estimate
PAGE_CHARS = 32_000            # content characters per batch-file line (≈ 36 KB after JSON escaping; the reading tool shows ≈ 50 KB of a line)
MAX_FILE_CHARS = 400_000       # content characters per batch file (files of this size were read and answered whole before DEV-21)
SMALL_MESSAGE_CHARS = 2_000    # messages up to this size stay whole in a paged row's first line; only longer ones are split (DEV-22 follow-up)


def _chars(messages: list[dict[str, Any]]) -> int:
    return sum(len(m.get("content") or "") for m in messages)


def estimated_input_tokens(messages: list[dict[str, Any]]) -> int:
    return _chars(messages) // CHARS_PER_TOKEN


def row_lines(r: Request) -> list[str]:
    """The batch-file lines of one request: a single line when its contents fit one page; otherwise a first line holding
    the head of every message's content and `"continued": true`, followed by continuation lines
    `{"task_id", "message_index", "append", "continued"}`, each appending a chunk to `messages[message_index].content`.
    Short messages (tool calls, questions) stay whole in the first line; the page budget that remains goes to the long
    messages in order, so a row needs as few lines as its long messages require (DEV-22 follow-up: the earlier
    sequential fill gave every short message after a long one its own continuation line — six lines per row)."""
    if _chars(r.messages) <= PAGE_CHARS:
        return [json.dumps({"task_id": r.key, "messages": r.messages, "max_tokens": r.max_tokens}, ensure_ascii=False)]
    contents = [m.get("content") or "" for m in r.messages]
    small = [len(c) <= SMALL_MESSAGE_CHARS for c in contents]
    if sum(len(c) for c, s in zip(contents, small) if s) > PAGE_CHARS // 2:   # pathological: fall back to the sequential fill
        small = [False] * len(contents)
    head, tail = [], []
    budget = PAGE_CHARS - sum(len(c) for c, s in zip(contents, small) if s)
    for i, (m, c) in enumerate(zip(r.messages, contents)):
        if small[i]:
            head.append({**m, "content": c})
            continue
        take = min(len(c), budget)
        budget -= take
        head.append({**m, "content": c[:take]})
        rest = c[take:]
        tail.extend((i, rest[j : j + PAGE_CHARS]) for j in range(0, len(rest), PAGE_CHARS))
    lines = [json.dumps({"task_id": r.key, "messages": head, "max_tokens": r.max_tokens, "continued": True}, ensure_ascii=False)]
    lines.extend(json.dumps({"task_id": r.key, "message_index": i, "append": chunk, "continued": n < len(tail) - 1}, ensure_ascii=False)
                 for n, (i, chunk) in enumerate(tail))
    return lines


def read_batch(path: Path) -> list[dict[str, Any]]:
    """The rows of a batch file with continuation lines folded back into their row (inverse of `row_lines`)."""
    rows: list[dict[str, Any]] = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "append" in d:
            if not rows or rows[-1]["task_id"] != d["task_id"]:
                raise ValueError(f"{path}: continuation line of {d['task_id']} does not follow its row")
            m = rows[-1]["messages"][d["message_index"]]
            m["content"] = (m.get("content") or "") + d["append"]
        else:
            rows.append(d)
    for r in rows:
        r.pop("continued", None)
    return rows


def read_completions(out_path: Path) -> dict[str, str]:
    got: dict[str, str] = {}
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "task_id" in row and isinstance(row.get("completion"), str):
                got[row["task_id"]] = row["completion"]
    return got


def _replay_key(task_id: str, messages: list[dict[str, Any]], max_tokens: int) -> tuple[str, str, int]:
    return task_id, json.dumps(messages, ensure_ascii=False), max_tokens


@dataclass
class ClaudeBatchBackend:
    batch_dir: Path
    model_id: str = "claude-opus-5-5"
    batch_size: int = 25
    poll_seconds: float = 30.0
    wait_limit_seconds: float = 6 * 3600
    route_context_tokens: int = ROUTE_CONTEXT_TOKENS
    max_file_chars: int = MAX_FILE_CHARS
    name: str = "claude_subagent"
    n_ceiling: int = 0             # requests not delivered because they exceed the route ceiling (DEV-19 guard applies)
    n_replayed: int = 0            # requests answered from an earlier batch file of this run directory
    n_attached: int = 0            # requests found in an earlier, still unanswered batch file and waited for there (DEV-22)

    def metadata(self) -> dict[str, Any]:
        return {"backend": self.name, "model_id": self.model_id, "batch_size": self.batch_size,
                "agent_definition": ".claude/agents/frontier-llm.md", "token_counts": "estimated (chars/4)",
                "route": {"context_tokens": self.route_context_tokens, "chars_per_token": CHARS_PER_TOKEN, "page_chars": PAGE_CHARS,
                          "max_file_chars": self.max_file_chars, "deviation": "DEV-21"},
                "route_ceiling_requests": self.n_ceiling, "replayed_requests": self.n_replayed, "attached_requests": self.n_attached}

    def write_batches(self, requests: list[Request], prefix: str) -> list[Path]:
        """Pack requests into files of at most `batch_size` rows and `max_file_chars` content characters, in order."""
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        files: list[list[Request]] = []
        cur: list[Request] = []
        cur_chars = 0
        for r in requests:
            n = _chars(r.messages)
            if cur and (len(cur) >= self.batch_size or cur_chars + n > self.max_file_chars):
                files.append(cur)
                cur, cur_chars = [], 0
            cur.append(r)
            cur_chars += n
        if cur:
            files.append(cur)
        paths = []
        for i, chunk in enumerate(files):
            path = self.batch_dir / f"{prefix}_{i:05d}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for r in chunk:
                    for line in row_lines(r):
                        f.write(line + "\n")
            paths.append(path)
        return paths

    def issued_index(self) -> dict[tuple[str, str, int], tuple[str | None, Path]]:
        """Every request issued in this run directory: (task id, messages, max_tokens) -> (completion or None, batch file)."""
        idx: dict[tuple[str, str, int], tuple[str | None, Path]] = {}
        if not self.batch_dir.exists():
            return idx
        for b in sorted(self.batch_dir.glob("*.jsonl")):
            if b.name.endswith(".out.jsonl"):
                continue
            got = read_completions(b.with_suffix(".out.jsonl"))
            for row in read_batch(b):
                key = _replay_key(row["task_id"], row["messages"], row["max_tokens"])
                if key not in idx or got.get(row["task_id"]) is not None:       # an answered copy wins over an unanswered one
                    idx[key] = (got.get(row["task_id"]), b)
        return idx

    def replay_index(self) -> dict[tuple[str, str, int], tuple[str, str]]:
        """Completions already produced in this run directory: (task id, messages, max_tokens) -> (completion, batch file name)."""
        return {k: (c, b.name) for k, (c, b) in self.issued_index().items() if c is not None}

    def collect(self, batch_paths: list[Path], wanted: set[str] | None = None) -> tuple[dict[str, Generation], list[str]]:
        out: dict[str, Generation] = {}
        missing: list[str] = []
        for path in batch_paths:
            expected = [row["task_id"] for row in read_batch(path) if wanted is None or row["task_id"] in wanted]
            got = read_completions(path.with_suffix(".out.jsonl"))
            for key in expected:
                if key in got:
                    out[key] = Generation(text=got[key], tokens_in=0, tokens_out=max(1, len(got[key]) // 4), latency_seconds=0.0, extra={"estimated_tokens": True})
                else:
                    missing.append(key)
        return out, missing

    def _fresh_prefix(self) -> str:
        t = int(time.time())
        while any(self.batch_dir.glob(f"b{t}_*.jsonl")):
            t += 1
        return f"b{t}"

    def generate_many(self, requests: list[Request]) -> dict[str, Generation]:
        out: dict[str, Generation] = {}
        issued = self.issued_index()
        todo: list[Request] = []
        attach: dict[Path, set[str]] = {}          # requests already written to a batch file that has no completion yet: wait there
        for r in requests:
            chars = _chars(r.messages)
            est = chars // CHARS_PER_TOKEN
            if est > self.route_context_tokens:
                # DEV-21: not deliverable on this route and beyond the common serving context — the same rejection an
                # sglang server returns, so the loop applies the DEV-19 guard
                self.n_ceiling += 1
                out[r.key] = Generation(text="", tokens_in=0, tokens_out=0, latency_seconds=0.0,
                                        error=f"The input ({est} tokens) is longer than the model's context length ({self.route_context_tokens} tokens). "
                                              f"[frontier route ceiling, DEV-21: {chars} characters of messages / {CHARS_PER_TOKEN}; not delivered]",
                                        extra={"estimated_tokens": True, "route_ceiling": True})
                continue
            hit = issued.get(_replay_key(r.key, r.messages, r.max_tokens))
            if hit is not None and hit[0] is not None:
                self.n_replayed += 1
                out[r.key] = Generation(text=hit[0], tokens_in=0, tokens_out=max(1, len(hit[0]) // 4), latency_seconds=0.0,
                                        extra={"estimated_tokens": True, "replayed_from": hit[1].name})
                continue
            if hit is not None:
                self.n_attached += 1
                attach.setdefault(hit[1], set()).add(r.key)
                continue
            todo.append(r)
        paths = list(attach)
        wanted: set[str] = set().union(*attach.values()) if attach else set()
        if todo:
            self.batch_dir.mkdir(parents=True, exist_ok=True)
            paths += self.write_batches(todo, self._fresh_prefix())
            wanted |= {r.key for r in todo}
        if not paths:
            return out
        deadline = time.time() + self.wait_limit_seconds
        while True:
            got, missing = self.collect(paths, wanted)
            if not missing or time.time() > deadline:
                break
            time.sleep(self.poll_seconds)
        out.update(got)
        for key in missing:
            out[key] = Generation(text="", tokens_in=0, tokens_out=0, latency_seconds=0.0, error="no completion produced")
        return out
