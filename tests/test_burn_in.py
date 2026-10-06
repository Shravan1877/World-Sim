"""Burn-in (§6.15, D38): the saved settled state is current, seed-free and reset correctly."""

import numpy as np

from world.config import load_config
from world.engine.burn_in import load_fixture, run_burn_in, settled_state
from world.engine.state import initial_state
from world.ledger import firms as firm_account

CFG = load_config()


def test_fixture_matches_a_fresh_burn_in() -> None:
    """If this fails, the engine or config changed: regenerate with
    `uv run python -m world.engine.burn_in` and review docs/calibration.md."""
    assert load_fixture(0).state_hash() == settled_state(CFG, 0).state_hash()


def test_burn_in_draws_no_random_numbers() -> None:
    a, b = settled_state(CFG, 1), settled_state(CFG, 2)
    b.seed = a.seed
    assert a.state_hash() == b.state_hash()


def test_stability_held_during_burn_in_and_reset_after() -> None:
    settled, history = run_burn_in(CFG, 0)
    start = initial_state(CFG, 0).stability
    for s in history:
        np.testing.assert_array_equal(s.stability, start)
    np.testing.assert_array_equal(settled.stability, start)


def test_history_discarded() -> None:
    s = load_fixture(3)
    assert s.turn == 0 and s.seed == 3
    assert s.ledger.log == []
    s.ledger.check_conservation()
    np.testing.assert_array_equal(s.gdp_start, s.gdp)
    np.testing.assert_array_equal(s.gdp_hist, np.tile(s.gdp[:, None], (1, s.gdp_hist.shape[1])))
    assert np.all(s.leader_changes == 0) and s.active_shocks == ()
    assert all(s.ledger.balance(firm_account(i, g)) == 0.0 for i in range(6) for g in range(5))
