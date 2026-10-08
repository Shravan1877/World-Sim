"""SQLite cache of model answers (CLAUDE.md §11.4), data/cache.db.

Key = sha256(provider, model_id, temperature, system prompt, the user messages (the briefing, plus the
retry message on a retry), sample_index). Re-running a seed replays identical answers at no quota
cost; a new sample_index draws fresh samples. Only answers that came back from the provider are
stored (valid or not), never errors.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

CACHE_DB = Path("data/cache.db")


@dataclass(frozen=True)
class CachedAnswer:
    raw_text: str
    parsed_json: str | None  # TurnDecision JSON as the provider parsed it, None if parsing failed
    parse_error: str | None
    tokens_in: int
    tokens_out: int
    tokens_reasoning: int
    latency_s: float


def cache_key(
    provider: str, model_id: str, temperature: float, messages: Sequence[tuple[str, str]], sample_index: int
) -> str:
    payload = json.dumps([provider, model_id, temperature, [list(m) for m in messages], sample_index])
    return hashlib.sha256(payload.encode()).hexdigest()


class LLMCache:
    def __init__(self, path: str | Path = CACHE_DB) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, provider TEXT, model_id TEXT, "
            "raw_text TEXT, parsed_json TEXT, parse_error TEXT, tokens_in INTEGER, tokens_out INTEGER, "
            "tokens_reasoning INTEGER, latency_s REAL, created_at TEXT)"
        )
        self.conn.commit()

    def get(self, key: str) -> CachedAnswer | None:
        row = self.conn.execute(
            "SELECT raw_text, parsed_json, parse_error, tokens_in, tokens_out, tokens_reasoning, latency_s "
            "FROM cache WHERE key=?",
            (key,),
        ).fetchone()
        return CachedAnswer(*row) if row else None

    def put(self, key: str, provider: str, model_id: str, a: CachedAnswer) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    key, provider, model_id, a.raw_text, a.parsed_json, a.parse_error, a.tokens_in,
                    a.tokens_out, a.tokens_reasoning, a.latency_s, dt.datetime.now(dt.UTC).isoformat(),
                ),
            )  # fmt: skip

    def __len__(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0])
