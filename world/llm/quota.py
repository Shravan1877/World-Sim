"""Quota guard (CLAUDE.md §11.7, D19): never lose a run in the middle.

1. Ledger: a SQLite table (WAL mode) of requests and tokens per (counter, quota window), updated after
   every call from real usage. The counter is the model's `quota_group` if it has one (models that
   share a provider pool), else the model key. Windows come from models.yaml: daily at a time in a
   time zone (Gemini: midnight America/Los_Angeles) or monthly (Mistral, UTC calendar month).
2. Pre-call check at `safety` (80%) of every cap: requests per day (rpd), tokens per day (tpd),
   tokens per month. A call that would pass 80% raises QuotaPause with the reset time.
3. Admission control: before a run, compare its estimated cost with what is left in the window; at
   most one half-finished (paused) run per lane.
4. Header parsing: retry-after and the x-ratelimit-* headers (Mistral names; others tolerated).
Unknown caps (null) are skipped, never crash.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

from world.config import ModelCfg, ModelsConfig
from world.llm.limiter import QuotaPause

QUOTA_DB = Path("data/quota.db")
UTC = dt.UTC

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    counter TEXT, window TEXT, requests INTEGER, tokens INTEGER, updated_at TEXT,
    PRIMARY KEY (counter, window)
);
CREATE TABLE IF NOT EXISTS calls (
    at TEXT, model_key TEXT, counter TEXT, window TEXT, requests INTEGER, tokens INTEGER
);
"""


# ------------------------------------------------------------------------------ headers


@dataclass(frozen=True)
class RateInfo:
    retry_after_s: float | None = None
    limit_req_minute: float | None = None
    remaining_req_minute: float | None = None
    limit_tokens_minute: float | None = None
    remaining_tokens_minute: float | None = None
    query_cost: float | None = None


def _num(v: str | None) -> float | None:
    if v is None:
        return None
    try:
        return float(str(v).strip().rstrip("s"))
    except ValueError:
        return None


def parse_rate_headers(headers: Mapping[str, str]) -> RateInfo:
    h = {k.lower(): v for k, v in headers.items()}
    return RateInfo(
        retry_after_s=_num(h.get("retry-after")),
        limit_req_minute=_num(h.get("x-ratelimit-limit-req-minute")),
        remaining_req_minute=_num(h.get("x-ratelimit-remaining-req-minute")),
        limit_tokens_minute=_num(h.get("x-ratelimit-limit-tokens-minute")),
        remaining_tokens_minute=_num(h.get("x-ratelimit-remaining-tokens-minute")),
        query_cost=_num(h.get("x-ratelimit-tokens-query-cost")),
    )


# ------------------------------------------------------------------------------ windows


def window_of(model: ModelCfg, now: dt.datetime) -> tuple[str, dt.datetime]:
    """(window id, next reset as a UTC datetime) for the model's quota window at `now` (UTC)."""
    w = model.quota_window
    if w.kind == "monthly":
        start = now.astimezone(UTC)
        nxt = (
            start.replace(day=1, hour=0, minute=0, second=0, microsecond=0) + dt.timedelta(days=32)
        ).replace(day=1)
        return f"{start:%Y-%m}", nxt
    tz = ZoneInfo(w.tz or "UTC")
    hh, mm = (int(x) for x in (w.at or "00:00").split(":"))
    local = now.astimezone(tz)
    reset_today = local.replace(hour=hh, minute=mm, second=0, microsecond=0)
    day_start = reset_today if local >= reset_today else reset_today - dt.timedelta(days=1)
    # next reset: the same wall-clock time on the next calendar day (handles DST changes)
    nxt_local = (day_start + dt.timedelta(days=1, hours=12)).replace(hour=hh, minute=mm)
    return f"{day_start:%Y-%m-%d}", nxt_local.astimezone(UTC)


# ------------------------------------------------------------------------------ guard


@dataclass(frozen=True)
class Usage:
    requests: int
    tokens: int


@dataclass(frozen=True)
class Caps:
    rpd: float | None
    tpd: float | None
    tokens_per_month: float | None


class QuotaGuard:
    def __init__(
        self,
        models: ModelsConfig,
        path: str | Path = QUOTA_DB,
        *,
        now: Callable[[], dt.datetime] | None = None,
    ) -> None:
        self.models = models
        self.safety = models.defaults.safety
        self.now = now or (lambda: dt.datetime.now(UTC))
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        if str(path) != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def counter(self, key: str) -> str:
        m = self.models.models[key]
        return m.quota_group or key

    def caps(self, key: str) -> Caps:
        lim = self.models.models[key].limits
        if self.models.models[key].quota_window.kind == "monthly":
            return Caps(None, None, lim.tokens_per_month)
        return Caps(lim.rpd, lim.tpd, None)

    def window(self, key: str) -> tuple[str, dt.datetime]:
        return window_of(self.models.models[key], self.now())

    def used(self, key: str) -> Usage:
        win, _ = self.window(key)
        row = self.conn.execute(
            "SELECT requests, tokens FROM usage WHERE counter=? AND window=?", (self.counter(key), win)
        ).fetchone()
        return Usage(*(row or (0, 0)))

    def remaining(self, key: str) -> dict[str, float | None]:
        """What is left at the safety level (None = no known cap)."""
        u, c = self.used(key), self.caps(key)

        def left(cap: float | None, used: int) -> float | None:
            return None if cap is None else max(0.0, self.safety * cap - used)

        tok_cap = c.tpd if c.tpd is not None else c.tokens_per_month
        return {"requests": left(c.rpd, u.requests), "tokens": left(tok_cap, u.tokens)}

    def check(self, key: str, est_tokens: int) -> None:
        """Raise QuotaPause if one more call of `est_tokens` would pass `safety` of any cap."""
        u, c = self.used(key), self.caps(key)
        _, reset = self.window(key)
        why = None
        if c.rpd is not None and u.requests + 1 > self.safety * c.rpd:
            why = f"{key}: {u.requests} requests in this window, cap {self.safety:.0%} of {c.rpd:.0f}"
        tok_cap = c.tpd if c.tpd is not None else c.tokens_per_month
        if why is None and tok_cap is not None and u.tokens + est_tokens > self.safety * tok_cap:
            why = f"{key}: {u.tokens} tokens in this window, cap {self.safety:.0%} of {tok_cap:.0f}"
        if why is not None:
            raise QuotaPause(why, resume_at=reset.isoformat(timespec="seconds"))

    def record(self, key: str, *, requests: int, tokens: int) -> None:
        win, _ = self.window(key)
        ctr = self.counter(key)
        at = self.now().isoformat(timespec="seconds")
        with self.conn:
            self.conn.execute(
                "INSERT INTO usage VALUES (?,?,?,?,?) ON CONFLICT(counter, window) DO UPDATE SET "
                "requests = requests + excluded.requests, tokens = tokens + excluded.tokens, "
                "updated_at = excluded.updated_at",
                (ctr, win, requests, tokens, at),
            )
            self.conn.execute("INSERT INTO calls VALUES (?,?,?,?,?,?)", (at, key, ctr, win, requests, tokens))

    # ------------------------------------------------------------------------- admission
    def admit(self, key: str, est_calls: int, est_tokens_per_call: int) -> Admission:
        """Can a run of `est_calls` calls start now? `fits` = it can finish inside this window;
        a run that does not fit may still start (it pauses at a seat boundary and resumes after
        the reset) when at least one call fits; `never` = not even one call fits in a fresh window."""
        c = self.caps(key)
        rem = self.remaining(key)
        tokens = est_calls * est_tokens_per_call
        _, reset = self.window(key)
        fresh_req = None if c.rpd is None else self.safety * c.rpd
        tok_cap = c.tpd if c.tpd is not None else c.tokens_per_month
        fresh_tok = None if tok_cap is None else self.safety * tok_cap
        fits = (rem["requests"] is None or rem["requests"] >= est_calls) and (
            rem["tokens"] is None or rem["tokens"] >= tokens
        )
        can_start = (rem["requests"] is None or rem["requests"] >= 1) and (
            rem["tokens"] is None or rem["tokens"] >= est_tokens_per_call
        )
        # it can never fit if not even one call fits in a fresh window (a run may span windows)
        never = (fresh_req is not None and fresh_req < 1) or (
            fresh_tok is not None and est_tokens_per_call > fresh_tok
        )
        return Admission(key, est_calls, tokens, fits, can_start and not never, never, reset)

    def usage_table(self) -> list[dict[str, object]]:
        out = []
        for key, m in self.models.models.items():
            win, reset = self.window(key)
            u, c = self.used(key), self.caps(key)
            out.append({
                "key": key, "provider": m.provider, "model_id": m.model_id, "counter": self.counter(key),
                "window": win, "requests": u.requests, "tokens": u.tokens, "rpd": c.rpd, "tpd": c.tpd,
                "tokens_per_month": c.tokens_per_month, "remaining": self.remaining(key),
                "next_reset_utc": reset.isoformat(timespec="minutes"),
            })  # fmt: skip
        return out


@dataclass(frozen=True)
class Admission:
    key: str
    calls: int
    tokens: int
    fits: bool
    can_start: bool
    never: bool
    next_reset: dt.datetime


@dataclass(frozen=True)
class QueuedRun:
    run_id: str
    lane: str  # the model key (one process per lane)
    est_calls: int
    paused: bool = False  # a half-finished run


def order_runs(queue: Iterable[QueuedRun]) -> list[QueuedRun]:
    """Admission order per lane: a half-finished run first (resume it), then new runs in queue order.
    The runner takes runs one at a time per lane, so a lane never has two half-finished runs: a new
    run starts only after the paused one has finished."""
    q = list(queue)
    paused_lanes: dict[str, int] = {}
    for r in q:
        if r.paused:
            paused_lanes[r.lane] = paused_lanes.get(r.lane, 0) + 1
    bad = [lane for lane, n in paused_lanes.items() if n > 1]
    if bad:
        raise ValueError(f"more than one half-finished run in lanes {bad}")
    return sorted(q, key=lambda r: (r.lane, not r.paused))
