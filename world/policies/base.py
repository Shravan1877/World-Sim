"""LeaderPolicy protocol: decide(briefing) -> DecisionResult (CLAUDE.md §11.1).

Every leader, LLM or bot, implements this. DecisionResult = the parsed TurnDecision (or None if the
policy could not produce a valid one) plus what the logs need: raw text, parse error, tokens,
latency, retries, and one CallLog per model call. Bots fill only `decision`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from world.actions import TurnDecision
from world.briefing import Briefing


@dataclass(frozen=True)
class CallLog:
    """One model call (an attempt). Every call is logged to llm_calls (§11.4), cached ones included."""

    provider: str
    model_id: str
    model_key: str
    method: str  # structured-output method
    attempt: int  # 0 = first call, 1 = the retry after an invalid answer
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_reasoning: int = 0
    latency_s: float = 0.0
    transport_retries: int = 0  # 429 / 5xx retries inside this call
    raw_text: str = ""
    parsed_json: str | None = None
    error: str | None = None
    cached: bool = False
    sample_index: int = 0


@dataclass(frozen=True)
class DecisionResult:
    decision: TurnDecision | None
    raw_text: str = ""
    parse_error: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    retries: int = 0
    calls: tuple[CallLog, ...] = ()


class LeaderPolicy(Protocol):
    name: str

    def decide(self, briefing: Briefing) -> DecisionResult: ...

    def on_leader_change(self, country: str) -> None:
        """Memory wipe hook (§6.11): called when `country`'s leader is replaced, before the new
        leader's first decision. Lite policies drop their recent-decision memory, deep policies
        archive their notes. The briefing already hides the earlier leaders' turns."""


def notify_leader_change(policy: LeaderPolicy, country: str) -> None:
    """Call the policy's memory-wipe hook if it has one."""
    hook = getattr(policy, "on_leader_change", None)
    if callable(hook):
        hook(country)
