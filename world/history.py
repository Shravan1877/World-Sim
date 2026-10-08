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


def leader_death_text(country: str) -> str:
    """How a leader_death shock appears in TurnRecord.shocks (see game.shock_text)."""
    return f"leader_death ({country})"


def tenure_start(country: str, history: tuple[TurnRecord, ...], shocks: tuple[str, ...], turn: int) -> int:
    """First turn played by the country's current leader (0 = the first leader).

    A leader who falls in step() of turn t (unrest, coup, election loss) is replaced from turn t + 1;
    a leader_death shock at the start of turn t replaces the leader for turn t itself. The briefing
    hides everything the earlier leaders did ("memory is wiped", §6.11).
    """
    start = 0
    for rec in history:
        if leader_death_text(country) in rec.shocks:
            start = max(start, rec.turn)
        if country in rec.leader_changes:
            start = max(start, rec.turn + 1)
    if leader_death_text(country) in shocks:
        start = max(start, turn)
    return start
