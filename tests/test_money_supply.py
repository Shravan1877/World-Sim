"""D48: government borrowing creates money (a known limitation). It must stay slow.

After burn-in, 60 turns, no shocks, status quo, 20 seeds: world money supply
M = household cash + treasuries grows by less than 1% per turn, every turn.
"""

import pytest

from tests.runs import run

TURNS = 60
MAX_GROWTH = 0.01


def money_supply(s) -> float:
    return float(s.household_cash.sum() + s.treasury.sum())


@pytest.mark.parametrize("seed", range(1, 21))
def test_money_supply_grows_slowly(seed: int) -> None:
    _, states, _ = run(seed, TURNS)
    m = [money_supply(s) for s in states]
    growth = [b / a - 1.0 for a, b in zip(m[:-1], m[1:], strict=True)]
    worst = max(growth)
    assert worst < MAX_GROWTH, f"money supply grew {100 * worst:.2f}% in turn {growth.index(worst) + 1}"
