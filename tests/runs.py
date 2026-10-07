"""Shared helpers for whole-game tests.

- run(): turn_start() + step() with fixed (status-quo) policies, from the settled state (D38).
- run_bots(): a full game with bots through the plain loop (world/game.py), from the same state.
- play_turn(): one turn with scripted moves, through the validator and policy application.
Every game starts from the settled state after burn-in, loaded from tests/fixtures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from world.actions import ActionIn, Forecast, Stance, TurnDecision
from world.config import COUNTRIES, Config, ScenarioCfg, load_config
from world.engine.burn_in import load_fixture
from world.engine.policy import PolicyEffects, apply_actions
from world.engine.state import WorldState
from world.engine.step import TurnLog, step, turn_start
from world.game import GameResult, bot_seats, run_game
from world.rng import RngBundle
from world.validator import Context, ValidationResult, validate

CFG = load_config()


def run(
    seed: int,
    turns: int,
    *,
    shocks_on: bool = False,
    scenario: ScenarioCfg | None = None,
    cfg: Config = CFG,
) -> tuple[WorldState, list[WorldState], list[TurnLog]]:
    """Fixed policies (nobody acts). Returns (final state, state after each turn, logs).

    step() raises InvariantError the moment any §6.14 invariant fails.
    """
    s = load_fixture(seed)
    rng = RngBundle(seed)
    states, logs = [s], []
    for _ in range(turns):
        s, _ = turn_start(s, rng, cfg, scenario=scenario, shocks_on=shocks_on)
        s, log = step(s, None, rng, cfg)
        states.append(s)
        logs.append(log)
    return s, states, logs


def run_bots(
    seats: str | Mapping[str, str],
    seed: int,
    turns: int,
    *,
    shocks_on: bool = True,
    scenario: ScenarioCfg | None = None,
    cfg: Config = CFG,
) -> GameResult:
    """A bot game (seats: one bot name for all, or {country: bot}, others status_quo)."""
    return run_game(
        bot_seats(seats, cfg),
        seed=seed,
        cfg=cfg,
        turns=turns,
        start=load_fixture(seed),
        scenario=scenario,
        shocks_on=shocks_on,
    )


def decision(actions: Sequence[ActionIn] = (), **over: object) -> TurnDecision:
    """A minimal valid TurnDecision around `actions`."""
    fields: dict[str, object] = dict(
        situation_read="",
        facts_used=[],
        predictions=[],
        stance=Stance(power=0.4, citizens=0.3, world=0.3),
        private_plan="",
        public_statement="",
        commitments=[],
        actions=list(actions),
        forecast=Forecast(my_gdp_growth_pct=0.0, my_stability_next=60.0),
    )
    fields.update(over)
    return TurnDecision(**fields)  # type: ignore[arg-type]


@dataclass
class Played:
    before_step: WorldState  # after all moves, before resolution
    after: WorldState
    log: TurnLog
    results: dict[str, ValidationResult]


def play_turn(
    state: WorldState,
    moves: Sequence[tuple[str, Sequence[ActionIn]]],
    *,
    seed: int = 1,
    cfg: Config = CFG,
    shocks_on: bool = False,
) -> Played:
    """turn_start, then each (country, actions) in the given order through Layer 1 and policy.py,
    then step. Countries not listed do nothing. Predictions are not used here."""
    rng = RngBundle(seed)
    s, _ = turn_start(state, rng, cfg, shocks_on=shocks_on)
    fx = PolicyEffects()
    results = {}
    for country, acts in moves:
        i = COUNTRIES.index(country)
        vr = validate(decision(acts), Context(s, i, (), cfg))
        results[country] = vr
        s = apply_actions(s, i, vr.actions, cfg, fx)
    after, log = step(s, fx.turn_inputs(), rng, cfg)
    return Played(s, after, log, results)
