"""Fiscal (§6.7): GDP, interest, premium, borrowing, default."""

import numpy as np
import pytest

from world.engine.fiscal import (
    apply_default,
    disposable_income,
    gdp_value_added,
    interest_due,
    risk_premium,
    settle_budget,
    taxes,
)

DEFAULT = dict(
    threshold=1.5,
    haircut=0.5,
    stability_hit=20,
    premium_add=0.05,
    premium_turns=8,
    periods_per_year=4,
    eps=1e-9,
)


def test_claude_md_budget_check() -> None:
    # GDP 1000, tax 20% -> 200; outlays excl. interest 250; debt 500 at 12% annual -> 15 -> change -65
    tax = taxes(np.array([0.2]), np.array([1000.0]))
    interest = interest_due(np.array([500.0]), np.array([0.12]), np.array([0.0]), 4)
    assert interest[0] == pytest.approx(15.0)
    r = settle_budget(np.array([100.0]), np.array([500.0]), tax, np.array([250.0]) + interest)
    assert r.treasury[0] - 100.0 == pytest.approx(-65.0)
    assert r.borrowing[0] == 0.0


def test_borrowing_when_treasury_goes_negative() -> None:
    r = settle_budget(np.array([30.0]), np.array([500.0]), np.array([200.0]), np.array([265.0]))
    assert r.treasury[0] == 0.0
    assert r.borrowing[0] == pytest.approx(35.0)
    assert r.debt[0] == pytest.approx(535.0)


def test_gdp_is_value_added() -> None:
    P = np.array([[1.0, 2.0, 1.0, 1.0, 1.0]])
    Q = np.array([[10.0, 5.0, 4.0, 0.0, 6.0]])
    E = np.array([[1.0, 0.5, 2.0, 0.0, 0.5]])
    assert gdp_value_added(P, Q, E)[0] == pytest.approx(30.0 - 2.0 * 4.0)


def test_risk_premium() -> None:
    p = risk_premium(
        np.array([400.0, 1000.0]),
        np.array([250.0, 250.0]),
        np.array([0.0, 0.05]),
        slope=0.05,
        threshold=0.6,
        periods_per_year=4,
        eps=1e-9,
    )
    # ratios 0.4 and 1.0 -> 0 and 0.05*0.4=0.02 (+0.05 default premium)
    np.testing.assert_allclose(p, [0.0, 0.07])


def test_default_triggers_above_threshold_only() -> None:
    debt = np.array([1490.0, 1510.0])  # GDP 250/quarter -> annual 1000 -> ratios 1.49, 1.51
    r = apply_default(
        debt, np.full(2, 250.0), np.full(2, 60.0), np.zeros(2), np.zeros(2, dtype=int), **DEFAULT
    )
    np.testing.assert_array_equal(r.defaulted, [False, True])
    np.testing.assert_allclose(r.debt, [1490.0, 755.0])
    np.testing.assert_allclose(r.stability, [60.0, 40.0])
    np.testing.assert_allclose(r.default_premium, [0.0, 0.05])
    np.testing.assert_array_equal(r.default_turns_left, [0, 8])


def test_default_premium_expires_after_8_turns_and_no_double_default() -> None:
    debt, prem, turns, stab = np.array([2000.0]), np.zeros(1), np.zeros(1, dtype=int), np.array([60.0])
    r = apply_default(debt, np.array([250.0]), stab, prem, turns, **DEFAULT)
    debt, prem, turns = np.array([5000.0]), r.default_premium, r.default_turns_left  # still huge debt
    for _ in range(7):
        r = apply_default(debt, np.array([250.0]), stab, prem, turns, **DEFAULT)
        assert not r.defaulted[0]
        prem, turns = r.default_premium, r.default_turns_left
    assert prem[0] == pytest.approx(0.05)
    r = apply_default(debt, np.array([250.0]), stab, prem, turns, **DEFAULT)  # 8th turn: premium ends
    assert r.default_premium[0] == pytest.approx(0.0)
    assert r.default_turns_left[0] == 0


def test_disposable_income() -> None:
    y = disposable_income(
        np.array([100.0]), np.array([20.0]), np.array([0.25]), np.array([10.0]), np.array([2.0])
    )
    assert y[0] == pytest.approx(120 * 0.75 + 12)
