"""Shared helper for whole-game tests: run turn_start() + step() with fixed (status-quo) policies.

Every game starts from the settled state after burn-in (D38), loaded from tests/fixtures.
An optional `policy(state) -> (state, TurnInputs | None)` hook sets policy fields before each step
(a stand-in for scripted bots until the Phase 4 action layer exists).
"""

from __future__ import annotations

from collections.abc import Callable

from world.config import Config, ScenarioCfg, load_config
from world.engine.burn_in import load_fixture
from world.engine.state import WorldState
from world.engine.step import TurnInputs, TurnLog, step, turn_start
from world.rng import RngBundle

CFG = load_config()

Policy = Callable[[WorldState], tuple[WorldState, TurnInputs | None]]


def run(
    seed: int,
    turns: int,
    *,
    shocks_on: bool = False,
    scenario: ScenarioCfg | None = None,
    cfg: Config = CFG,
    policy: Policy | None = None,
) -> tuple[WorldState, list[WorldState], list[TurnLog]]:
    """Fixed policies (nobody acts) unless `policy` is given. Returns (final state, state after each
    turn, logs).

    step() raises InvariantError the moment any §6.14 invariant fails.
    """
    s = load_fixture(seed)
    rng = RngBundle(seed)
    states, logs = [s], []
    for _ in range(turns):
        s, _ = turn_start(s, rng, cfg, scenario=scenario, shocks_on=shocks_on)
        inputs = None
        if policy is not None:
            s, inputs = policy(s)
        s, log = step(s, inputs, rng, cfg)
        states.append(s)
        logs.append(log)
    return s, states, logs
