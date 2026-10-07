"""D43: per-country food floor f_min_i = 0.7 x settled FOOD demand per person.

It must not bind at the settled state (status quo, no shocks) and must bind in a harvest failure.
"""

import numpy as np

from tests.runs import CFG, run
from world.engine import shocks
from world.engine.burn_in import load_fixture
from world.engine.step import step, turn_start
from world.rng import RngBundle

HARVEST = next(s for s in CFG.shocks.shocks if s.id == "harvest_failure")


def test_floor_does_not_bind_without_shocks() -> None:
    _, _, logs = run(1, 14)
    for log in logs:
        assert not log.extra["food_floor_binds"].any(), f"t{log.turn}"


def test_floor_binds_in_world_harvest_failure() -> None:
    """Harvest failure (A_FOOD x 0.6 for 2 turns) in every country at turn 1."""
    s = load_fixture(1)
    rng = RngBundle(1)
    binds = []
    for t in range(6):
        s, _ = turn_start(s, rng, CFG, shocks_on=False)
        if t == 0:
            for i in range(6):
                shocks.apply_effect(s, HARVEST, i, -1, CFG)
            shocks.recompute_multipliers(s)
        s, log = step(s, None, rng, CFG)
        binds.append(log.extra["food_floor_binds"])
    assert np.any(binds), "the food floor never bound during a world-wide harvest failure"
