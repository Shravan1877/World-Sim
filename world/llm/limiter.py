"""Client-side pacing for one model key (CLAUDE.md §11.7, D23).

One ModelLimiter per model KEY, shared by every caller in the process (leader calls, retries, later
deep-mode sub-agents and the referee). Effective limits are `safety` (0.8) x the published ones.
Before each call ALL of these must pass, otherwise the limiter sleeps until they do:
  1. spacing: now - last_start >= max(1/eff_rps, 60/eff_rpm)
  2. sliding 60 s request window: calls started in the last 60 s < eff_rpm
  3. sliding 60 s token window: tokens charged in the last 60 s + this call's estimate <= eff_tpm
     (estimate = prompt chars / 4 x 1.2 + rolling p95 of completion tokens, start 500, capped at
     max_output_tokens)
  4. daily / monthly caps: through the QuotaGuard (world/llm/quota.py), which raises QuotaPause.
After each call the real usage replaces the estimate, and server headers correct it: if
remaining-req-minute <= 2 or remaining-tokens-minute < the next estimate, wait to the next minute
boundary + 1 s. A temporary 429 halves every effective rate for 5 minutes (adaptive slow-down).
If the wait needed is longer than `max_block_s` (60 s + the 1 s boundary slack), the limiter raises
QuotaPause instead of sleeping: the run pauses at a seat boundary and resumes later.
A null limit means "no known limit" and is skipped.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from world.config import ModelCfg

WINDOW_S = 60.0
SLOWDOWN_S = 300.0
SLOWDOWN_FACTOR = 0.5
START_COMPLETION_TOKENS = 500
P95_SAMPLES = 50


class Clock(Protocol):
    def time(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class RealClock:
    def time(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class QuotaPause(Exception):  # noqa: N818 - a control signal, not an error
    """Progress is blocked for longer than allowed (daily cap, long wait): pause the run and resume
    at `resume_after` (seconds on the limiter clock, or None if unknown; the quota guard gives a UTC
    wall-clock time in `resume_at`)."""

    def __init__(
        self, reason: str, resume_after_s: float | None = None, resume_at: str | None = None
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.resume_after_s = resume_after_s
        self.resume_at = resume_at


@dataclass(frozen=True)
class Effective:
    rps: float | None
    rpm: float | None
    tpm: float | None

    def scaled(self, f: float) -> Effective:
        def s(x: float | None) -> float | None:
            return None if x is None else x * f

        return Effective(s(self.rps), s(self.rpm), s(self.tpm))

    @property
    def spacing(self) -> float:
        gaps = [1.0 / self.rps if self.rps else 0.0, WINDOW_S / self.rpm if self.rpm else 0.0]
        return max(gaps)


def estimate_prompt_tokens(chars: int) -> int:
    """chars / 4 x 1.2 until a tokenizer count is available (§11.7)."""
    return math.ceil(chars / 4 * 1.2)


class ModelLimiter:
    def __init__(
        self,
        key: str,
        model: ModelCfg,
        *,
        safety: float,
        max_output_tokens: int,
        clock: Clock | None = None,
        quota=None,  # QuotaGuard | None (daily and monthly caps)
        max_block_s: float = WINDOW_S + 1.5,
    ) -> None:
        self.key = key
        lim = model.limits
        self.base = Effective(
            None if lim.rps is None else safety * lim.rps,
            None if lim.rpm is None else safety * lim.rpm,
            None if lim.tpm is None else safety * lim.tpm,
        )
        self.max_output_tokens = max_output_tokens
        self.clock = clock or RealClock()
        self.quota = quota
        self.max_block_s = max_block_s
        self.starts: deque[float] = deque()
        self.charged: deque[list[float]] = deque()  # [time, tokens] (tokens corrected after the call)
        self.completions: deque[int] = deque(maxlen=P95_SAMPLES)
        self.last_start: float | None = None
        self.blocked_until = -math.inf
        self.slow_until = -math.inf
        self.slowdowns = 0

    # ------------------------------------------------------------------------------ limits
    def effective(self, now: float | None = None) -> Effective:
        now = self.clock.time() if now is None else now
        return self.base.scaled(SLOWDOWN_FACTOR) if now < self.slow_until else self.base

    def completion_p95(self) -> int:
        if not self.completions:
            return min(START_COMPLETION_TOKENS, self.max_output_tokens)
        xs = sorted(self.completions)
        return min(xs[min(len(xs) - 1, math.ceil(0.95 * len(xs)) - 1)], self.max_output_tokens)

    def estimate(self, prompt_chars: int) -> int:
        return estimate_prompt_tokens(prompt_chars) + self.completion_p95()

    def _prune(self, now: float) -> None:
        while self.starts and self.starts[0] <= now - WINDOW_S:
            self.starts.popleft()
        while self.charged and self.charged[0][0] <= now - WINDOW_S:
            self.charged.popleft()

    def wait_needed(self, est_tokens: int, now: float | None = None) -> float:
        """Seconds to wait before a call charging `est_tokens` may start (0 = go now)."""
        now = self.clock.time() if now is None else now
        self._prune(now)
        eff = self.effective(now)
        waits = [self.blocked_until - now]
        if self.last_start is not None:
            waits.append(self.last_start + eff.spacing - now)
        if eff.rpm is not None and len(self.starts) + 1 > eff.rpm:
            # the oldest calls must leave the window until one more fits
            k = len(self.starts) + 1 - math.floor(eff.rpm)
            waits.append(self.starts[k - 1] + WINDOW_S - now)
        if eff.tpm is not None:
            used = sum(t for _, t in self.charged)
            if used + est_tokens > eff.tpm:
                if est_tokens > eff.tpm:
                    raise ValueError(f"{self.key}: one call ({est_tokens} tokens) is above the token window")
                for ts, tok in self.charged:
                    used -= tok
                    if used + est_tokens <= eff.tpm:
                        waits.append(ts + WINDOW_S - now)
                        break
        return max(0.0, *waits)

    # ------------------------------------------------------------------------------ calls
    def acquire(self, est_tokens: int) -> float:
        """Block until a call may start, then charge it. Returns the seconds waited.
        Raises QuotaPause if a daily/monthly cap is reached or the wait would be too long."""
        if self.quota is not None:
            self.quota.check(self.key, est_tokens)  # raises QuotaPause
        waited = 0.0
        while True:
            w = self.wait_needed(est_tokens)
            if w <= 0:
                break
            if w > self.max_block_s:
                raise QuotaPause(f"{self.key}: rate limits block progress for {w:.0f} s", resume_after_s=w)
            self.clock.sleep(w)
            waited += w
        now = self.clock.time()
        self.starts.append(now)
        self.charged.append([now, float(est_tokens)])
        self.last_start = now
        return waited

    def record(
        self,
        tokens_in: int,
        tokens_out: int,
        headers: Mapping[str, str] | None = None,
        next_estimate: int | None = None,
    ) -> None:
        """Replace the last call's estimate with its real usage; learn the completion size; apply
        server-header corrections; count it in the quota ledger."""
        total = tokens_in + tokens_out
        if self.charged:
            self.charged[-1][1] = float(total)
        if tokens_out > 0:
            self.completions.append(tokens_out)
        if self.quota is not None:
            self.quota.record(self.key, requests=1, tokens=total)
        if headers:
            from world.llm.quota import parse_rate_headers

            info = parse_rate_headers(headers)
            nxt = next_estimate if next_estimate is not None else self.completion_p95()
            low_req = info.remaining_req_minute is not None and info.remaining_req_minute <= 2
            low_tok = info.remaining_tokens_minute is not None and info.remaining_tokens_minute < nxt
            if low_req or low_tok:
                self.block_to_next_minute()

    def record_failed_attempt(self) -> None:
        """A call that failed before usage was known still counts as a request (§11.7 table)."""
        if self.quota is not None:
            self.quota.record(self.key, requests=1, tokens=0)

    def block_to_next_minute(self) -> None:
        now = self.clock.time()
        self.blocked_until = max(self.blocked_until, (math.floor(now / WINDOW_S) + 1) * WINDOW_S + 1.0)

    def slow_down(self, retry_after_s: float | None) -> float:
        """A temporary 429: halve every effective rate for 5 minutes and wait retry-after x 1.2
        (at least 1 s). Returns the wait applied."""
        now = self.clock.time()
        self.slow_until = now + SLOWDOWN_S
        self.slowdowns += 1
        wait = max(1.0, 1.2 * (retry_after_s or 0.0))
        self.blocked_until = max(self.blocked_until, now + wait)
        return wait


LimiterFactory = Callable[[str], ModelLimiter]
