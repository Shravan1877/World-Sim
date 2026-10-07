"""What happened in a game, seat by seat and turn by turn (used by briefings, bots and metrics).

Plain frozen records; the plain-loop game (world/game.py) and later the LangGraph loop write them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from world.actions import Action, ActionIn, Commitment, Prediction, TurnDecision


@dataclass(frozen=True)
class FactCheck:
    """Layer 2 (§11.5): one stated fact against the briefing's true value."""

    country: str
    metric: str
    stated: float
    true: float | None  # None: no such metric/country in the briefing
    wrong: bool


@dataclass(frozen=True)
class SeatRecord:
    """One country's move in one turn."""

    turn: int
    country: str
    seat: int  # 1-based position in this turn's move order
    policy: str  # bot or model name
    decision: TurnDecision | None  # None only if the policy failed to produce one
    parse_failure: bool
    accepted: tuple[Action, ...]
    rejected: tuple[tuple[ActionIn, str], ...]
    predictions: tuple[Prediction, ...]  # kept after Layer 1 (still-to-move countries only)
    dropped_predictions: tuple[tuple[Prediction, str], ...]
    commitments: tuple[Commitment, ...]  # kept after Layer 1
    dropped_commitments: tuple[tuple[Commitment, str], ...]
    fact_checks: tuple[FactCheck, ...]
    hostile: tuple[tuple[str, str], ...]  # (kind, victim) from this seat's accepted actions
    public_statement: str = ""
    degraded: bool = False  # the policy was replaced by StatusQuoBot after repeated parse failures


@dataclass(frozen=True)
class TurnRecord:
    turn: int
    order: tuple[str, ...]
    shocks: tuple[str, ...]  # fired shocks, human readable
    seats: tuple[SeatRecord, ...]
    violations: tuple[tuple[str, str, str], ...] = ()  # (violator, victim, reason)
    leader_changes: tuple[str, ...] = ()
    events: tuple[str, ...] = field(default_factory=tuple)

    def seat(self, country: str) -> SeatRecord | None:
        for r in self.seats:
            if r.country == country:
                return r
        return None
