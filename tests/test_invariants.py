"""§6.14 invariants over whole games: 14 turns, fixed policies, no shocks, 20 seeds.

step() itself raises InvariantError on any failure (money, trade balance, stock-flow,
non-negativity, bounds, shares). This file also re-checks the money rules from outside, turn by turn.
(The same 20 seeds with shocks and the energy_crunch scenario are in test_step.py.)
"""

import numpy as np
import pytest

from tests.runs import CFG, run
from world.engine.invariants import check_firm_accounts_zero, check_money


@pytest.mark.parametrize("seed", range(1, 21))
def test_all_invariants_every_turn(seed: int) -> None:
    final, states, logs = run(seed, 14)
    assert final.turn == 14
    tol = CFG.world.invariants.rel_tol
    for s, log in zip(states[1:], logs, strict=True):
        assert check_money(s.ledger, tol) == []
        assert check_firm_accounts_zero(s.ledger, float(np.abs(s.gdp).sum()), tol) == []
        assert np.all(s.household_cash >= 0.0)
        assert np.all(s.treasury >= 0.0)
        np.testing.assert_allclose(log.exports.sum(axis=0), log.imports.sum(axis=0), rtol=1e-9, atol=1e-12)
        assert log.extra["household_backstop"].sum() == 0.0  # D35 backstop never needed here
