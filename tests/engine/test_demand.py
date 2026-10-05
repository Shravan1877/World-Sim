"""Demand (§6.3, D28c/d): shares, food floor, wealth term, government GOODS purchases."""

import numpy as np
import pytest

from world.engine.demand import (
    government_goods_demand,
    household_demand,
    spendable_income,
    total_demand,
    update_household_cash,
)
from world.engine.prices import update_prices

SHARES = np.array([[0.2, 0.1, 0.4, 0.1, 0.2]])


def test_share_times_income_over_price() -> None:
    D, binds = household_demand(SHARES, np.array([1000.0]), np.full((1, 5), 8.0), 0.0, np.array([1.0]))
    assert D[0, 2] == pytest.approx(50.0)  # 0.4 * 1000 / 8
    assert not binds[0]


def test_price_doubling_halves_quantity() -> None:
    p = np.full((1, 5), 8.0)
    D1, _ = household_demand(SHARES, np.array([1000.0]), p, 0.0, np.array([1.0]))
    p2 = p.copy()
    p2[0, 2] = 16.0
    D2, _ = household_demand(SHARES, np.array([1000.0]), p2, 0.0, np.array([1.0]))
    assert D2[0, 2] == pytest.approx(D1[0, 2] / 2)
    np.testing.assert_allclose(np.delete(D2, 2), np.delete(D1, 2))


def test_food_floor_binds_and_rescales_the_rest() -> None:
    # Floor 1.0 * 100 people = 100 units at P=8 -> 800 credits; share-based food spend is only 200.
    p = np.full((1, 5), 8.0)
    D, binds = household_demand(SHARES, np.array([1000.0]), p, 1.0, np.array([100.0]))
    assert binds[0]
    assert D[0, 0] == pytest.approx(100.0)
    spend = D * p
    assert spend.sum() == pytest.approx(1000.0)  # total spending == Y_spend
    # the other goods keep their relative shares, scaled by 200 / 800
    np.testing.assert_allclose(spend[0, 1:], SHARES[0, 1:] * 1000.0 * 0.25)


def test_food_floor_costing_more_than_income_caps_spending() -> None:
    p = np.full((1, 5), 8.0)
    D, binds = household_demand(SHARES, np.array([1000.0]), p, 2.0, np.array([100.0]))
    assert binds[0]
    assert (D * p).sum() == pytest.approx(1000.0)
    assert D[0, 0] == pytest.approx(125.0)
    np.testing.assert_allclose(D[0, 1:], 0.0)


def test_government_goods_added_to_total_demand() -> None:
    house = np.ones((2, 5))
    gov = government_goods_demand(np.array([30.0, 0.0]), np.array([2.0, 1.0]))
    np.testing.assert_allclose(gov, [15.0, 0.0])
    total = total_demand(house, gov)
    np.testing.assert_allclose(total[:, 2], [16.0, 1.0])
    np.testing.assert_allclose(np.delete(total, 2, axis=1), 1.0)


def test_household_cash_identity() -> None:
    """Y_disp = Y_spend + (H' - H): income is either spent or added to cash."""
    y, s, H, c_w = np.array([100.0]), np.array([0.1]), np.array([50.0]), 0.1
    y_spend = spendable_income(y, s, H, c_w)
    H_new = update_household_cash(H, y, s, c_w)
    assert y_spend[0] == pytest.approx(95.0)
    assert H_new[0] == pytest.approx(55.0)
    assert (y_spend + (H_new - H))[0] == pytest.approx(y[0])


def _closed_toy(c_w: float, turns: int = 60) -> np.ndarray:
    """One closed country, one good, fixed supply 100, no taxes.

    Households spend Y_spend; firms receive it as revenue and pay ALL of it out as next turn's
    income (wages + profits). The price follows the gradual rule.
    """
    s, Q = np.array([0.1]), np.array([[100.0]])
    y, H, P = np.array([8.6]), np.array([0.0]), np.array([[1.0]])
    incomes = [y[0]]
    for _ in range(turns):
        y_spend = spendable_income(y, s, H, c_w)
        H = update_household_cash(H, y, s, c_w)
        D = y_spend[:, None] / P
        P = update_prices(P, D, Q, 0.3, 0.2, 1e-9)
        y = y_spend  # revenue paid out in full as income
        incomes.append(y[0])
    return np.array(incomes)


def test_wealth_term_keeps_income_alive() -> None:
    """Why c_w exists (D28c): without it, savings leak and nominal income dies."""
    alive = _closed_toy(c_w=0.10)
    dead = _closed_toy(c_w=0.0)
    assert dead[-1] < 0.01 * dead[0]  # 8.6 * 0.9^60 ~ 0.015
    assert alive[-1] > 0.4 * alive[0]  # converges to c_w / (s + c_w) * money = 4.3
    assert alive[-1] == pytest.approx(8.6 * 0.10 / (0.10 + 0.10), rel=1e-3)
