"""Shared helper for whole-game tests: run turn_start() + step() with fixed (status-quo) policies."""

from __future__ import annotations

from world.config import Config, ScenarioCfg, load_config
from world.engine.state import WorldState, initial_state
from world.engine.step import TurnLog, step, turn_start
from world.rng import RngBundle

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
    s = initial_state(cfg, seed)
    rng = RngBundle(seed)
    states, logs = [s], []
    for _ in range(turns):
        s, _ = turn_start(s, rng, cfg, scenario=scenario, shocks_on=shocks_on)
        s, log = step(s, None, rng, cfg)
        states.append(s)
        logs.append(log)
    return s, states, logs
