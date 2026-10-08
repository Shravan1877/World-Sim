"""A full game as a plain Python loop, without LangGraph (Phase 4 item 7; calibration and bot tests).

    for each turn (hidden horizon T ~ Uniform{10..14}, HORIZON stream, or a fixed `turns`):
        turn_start: shocks, scripted incidents, move order (EVERMERE's random slot, ORDER stream)
        for each seat in order:
            briefing (state after the earlier seats' actions) -> policy.decide -> Layer 1 + 2
            accepted actions change policy fields at once (later seats see them)
        step: resolve the quarter (treaties execute inside step)

Parse failures (decision None) fall back to `wait` (§11.4); after `parse_failure_streak_limit` (3) in
a row the seat is played by StatusQuoBot for the rest of the run and the run is `degraded` (§11.5).
An invariant failure stops the run with status `invariant_failed` and keeps every state so far.
The LangGraph loop (world/graph.py) runs the same seat and turn helpers below; both give the same
states for the same seed (tests/test_graph_bots.py).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from world.actions import wait_decision
from world.briefing import Briefing, build_briefing
from world.config import COUNTRIES, SECTORS, Config, ScenarioCfg, load_config
from world.engine.burn_in import settled_state
from world.engine.invariants import InvariantError
from world.engine.policy import PolicyEffects, apply_actions, hostile_acts
from world.engine.state import WorldState
from world.engine.step import TurnLog, step, turn_start
from world.history import SeatRecord, TurnRecord, leader_death_text
from world.policies.base import DecisionResult, LeaderPolicy, notify_leader_change
from world.policies.bots import AggressorBot, CooperativeBot, GreedyBot, StatusQuoBot, TitForTatBot
from world.policies.random_bot import RandomBot
from world.rng import RngBundle, Stream
from world.validator import Context, fact_check, validate

BOTS: dict[str, type] = {
    "status_quo": StatusQuoBot,
    "tit_for_tat": TitForTatBot,
    "greedy": GreedyBot,
    "cooperative": CooperativeBot,
    "random": RandomBot,
    "aggressor": AggressorBot,
}


def make_bot(name: str, cfg: Config | None = None) -> LeaderPolicy:
    return BOTS[name](cfg=cfg or load_config())


def draw_horizon(seed: int, cfg: Config) -> int:
    """§8: the hidden game length T ~ Uniform{min..max} from the HORIZON stream (never told to agents)."""
    h = cfg.world.horizon
    return int(RngBundle(seed).gen(0, Stream.HORIZON).integers(h.min_, h.max_ + 1))


@dataclass
class GameResult:
    seed: int
    horizon: int
    states: list[WorldState]  # states[0] = start, states[k] = after turn k
    logs: list[TurnLog]
    records: list[TurnRecord]
    status: str = "ok"  # ok | degraded | invariant_failed
    error: str = ""
    degraded_seats: list[str] = field(default_factory=list)

    @property
    def final(self) -> WorldState:
        return self.states[-1]


def shock_text(e) -> str:
    where = COUNTRIES[e.country] if e.country >= 0 else "world"
    sector = f" {SECTORS[e.sector]}" if e.sector >= 0 else ""
    return f"{e.shock_id} ({where}{sector})"


def policy_name(policy: LeaderPolicy) -> str:
    return getattr(policy, "name", type(policy).__name__)


# ----------------------------------------------------------------------------------------------
# One seat and one turn. The plain loop below and the LangGraph nodes (world/graph.py) both call
# these, so the two loops run exactly the same code and give the same states.


def leader_deaths(shocks: tuple[str, ...]) -> list[str]:
    """Countries whose leader died in this turn's shocks (memory-wipe hook before they move)."""
    return [c for c in COUNTRIES if leader_death_text(c) in shocks]


def seat_briefing(
    state: WorldState,
    country: str,
    order: tuple[str, ...],
    shocks: tuple[str, ...],
    done: tuple[SeatRecord, ...],
    history: tuple[TurnRecord, ...],
    cfg: Config,
) -> Briefing:
    return build_briefing(state, country, order, shocks, done, history, cfg)


@dataclass(frozen=True)
class SeatOutcome:
    state: WorldState  # after this seat's accepted actions
    record: SeatRecord
    streak: int  # consecutive parse failures of this country after this seat
    degrade: bool  # the seat is handed to StatusQuoBot from now on (§11.5 streak rule)


def apply_seat(
    state: WorldState,
    seat: int,
    briefing: Briefing,
    out: DecisionResult,
    policy: str,
    effects: PolicyEffects,
    streak: int,
    degraded_already: bool,
    cfg: Config,
) -> SeatOutcome:
    """Layer 1 + 2 on one seat's answer, then apply the accepted actions (they change policy fields
    at once, so later seats see them). `effects` collects what step() needs and is updated in place."""
    country = briefing.country
    i = COUNTRIES.index(country)
    tol = cfg.world.agents.fact_check_rel_tolerance
    decision, failed = out.decision, out.decision is None
    degrade = False
    if failed:
        decision = wait_decision()
        streak += 1
        if streak >= cfg.world.agents.parse_failure_streak_limit and not degraded_already:
            degrade = True
            policy = StatusQuoBot.name
    else:
        streak = 0
    assert decision is not None
    vr = validate(decision, Context(state, i, briefing.still_to_move, cfg))
    s = apply_actions(state, i, vr.actions, cfg, effects)
    record = SeatRecord(
        turn=s.turn,
        country=country,
        seat=seat,
        policy=policy,
        decision=None if failed else decision,
        parse_failure=failed,
        accepted=tuple(vr.actions),
        rejected=tuple((a, why) for _, a, why in vr.rejected),
        predictions=vr.predictions,
        dropped_predictions=vr.dropped_predictions,
        commitments=vr.commitments,
        dropped_commitments=vr.dropped_commitments,
        fact_checks=fact_check(decision.facts_used, briefing.facts(), tol),
        hostile=tuple((kind, COUNTRIES[j]) for kind, j in hostile_acts(state, s, i)),
        public_statement=decision.public_statement,
        degraded=degrade,
        action_notes=vr.notes,
    )
    return SeatOutcome(s, record, streak, degrade)


def turn_record(
    turn: int, order: tuple[str, ...], shocks: tuple[str, ...], seats: tuple[SeatRecord, ...], log: TurnLog
) -> TurnRecord:
    return TurnRecord(
        turn=turn,
        order=order,
        shocks=shocks,
        seats=seats,
        violations=tuple((COUNTRIES[v.violator], COUNTRIES[v.victim], v.reason) for v in log.violations),
        leader_changes=tuple(COUNTRIES[e.country] for e in log.events if e.kind == "leader_change"),
        events=tuple(f"{e.kind} {COUNTRIES[e.country] if e.country >= 0 else ''} {e.detail}".strip()
                     for e in log.events),
    )  # fmt: skip


def run_game(
    policies: Mapping[str, LeaderPolicy],
    *,
    seed: int,
    cfg: Config | None = None,
    turns: int | None = None,
    start: WorldState | None = None,
    scenario: ScenarioCfg | None = None,
    shocks_on: bool = True,
) -> GameResult:
    """Play one game. `policies` maps every country name to its leader."""
    cfg = cfg or load_config()
    missing = set(COUNTRIES) - set(policies)
    if missing:
        raise ValueError(f"no policy for {sorted(missing)}")
    seats = dict(policies)
    horizon = turns if turns is not None else draw_horizon(seed, cfg)
    s = settled_state(cfg, seed) if start is None else start
    rng = RngBundle(seed)
    res = GameResult(seed=seed, horizon=horizon, states=[s], logs=[], records=[])
    streak = dict.fromkeys(COUNTRIES, 0)
    for _ in range(horizon):
        s, ts = turn_start(s, rng, cfg, scenario=scenario, shocks_on=shocks_on)
        shocks = tuple(shock_text(e) for e in ts.fired)
        for c in leader_deaths(shocks):
            notify_leader_change(seats[c], c)
        effects = PolicyEffects()
        done: list[SeatRecord] = []
        for k, country in enumerate(ts.order):
            b = seat_briefing(s, country, ts.order, shocks, tuple(done), tuple(res.records), cfg)
            out = seats[country].decide(b)
            so = apply_seat(
                s, k + 1, b, out, policy_name(seats[country]), effects, streak[country],
                country in res.degraded_seats, cfg,
            )  # fmt: skip
            s, streak[country] = so.state, so.streak
            if so.degrade:
                seats[country] = StatusQuoBot(cfg=cfg)
                res.degraded_seats.append(country)
                res.status = "degraded"
            done.append(so.record)
        try:
            s, log = step(s, effects.turn_inputs(), rng, cfg)
        except InvariantError as e:
            res.status, res.error = "invariant_failed", str(e)
            return res
        res.states.append(s)
        res.logs.append(log)
        rec = turn_record(s.turn, ts.order, shocks, tuple(done), log)
        res.records.append(rec)
        for c in rec.leader_changes:
            notify_leader_change(seats[c], c)
    return res


def bot_seats(spec: str | Mapping[str, str], cfg: Config | None = None) -> dict[str, LeaderPolicy]:
    """'greedy' -> every seat a GreedyBot; {'DORNE': 'aggressor', ...} -> per seat (others status_quo)."""
    cfg = cfg or load_config()
    if isinstance(spec, str):
        return {c: make_bot(spec, cfg) for c in COUNTRIES}
    return {c: make_bot(spec.get(c, "status_quo"), cfg) for c in COUNTRIES}
