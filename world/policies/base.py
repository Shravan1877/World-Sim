"""LeaderPolicy protocol: decide(briefing) -> DecisionResult (CLAUDE.md §11.1).

Every leader, LLM or bot, implements this. DecisionResult = the parsed TurnDecision (or None if the
policy could not produce a valid one) plus what the logs need: raw text, parse error, tokens,
latency and retries. Bots fill only `decision`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from world.actions import TurnDecision
from world.briefing import Briefing


@dataclass(frozen=True)
class DecisionResult:
    decision: TurnDecision | None
    raw_text: str = ""
    parse_error: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    retries: int = 0


class LeaderPolicy(Protocol):
    name: str

    def decide(self, briefing: Briefing) -> DecisionResult: ...
