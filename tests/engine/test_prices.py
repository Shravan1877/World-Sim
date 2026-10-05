"""Prices (§6.5, D28b): gradual rule, cap, convergence, exporter/importer cases, CPI, inflation."""

import numpy as np
import pytest

from world.engine.prices import (
    annualize,
    cpi,
    inflation_quarterly,
    market_balance,
    price_change_rate,
    update_prices,
)

SIG, CAP, EPS = 0.3, 0.2, 1e-9


def _p(P: float, D: float, S: float) -> float:
    return float(update_prices(np.array([P]), np.array([D]), np.array([S]), SIG, CAP, EPS)[0])


def test_claude_md_price_checks() -> None:
    assert _p(10, 150, 100) == pytest.approx(11.5)
    assert _p(10, 300, 100) == pytest.approx(12.0)  # capped at +20%
    assert _p(10, 0, 100) == pytest.approx(8.0)  # capped at -20%


def test_converges_to_spending_over_supply() -> None:
    P = 1.0
    for _ in range(10_000):
        new = _p(P, 1000.0 / P, 100.0)
        if abs(new - P) < 1e-6:
            P = new
            break
        P = new
    else:
        pytest.fail("price did not converge")
    assert pytest.approx(10.0, abs=1e-4) == P


def _one_country(demand_food: float, s_dom_food: float, x_req: float, imports: float):
    D = np.zeros((1, 5))
    S = np.zeros((1, 5))
    D[0, 0], S[0, 0] = demand_food, s_dom_food
    X = np.zeros((1, 4))
    M = np.zeros((1, 4))
    X[0, 0], M[0, 0] = x_req, imports
    return market_balance(D, S, np.zeros((1, 5)), X, M)


def test_exporter_price_rises_when_sold_out() -> None:
    # Makes 100, home demand 40, foreign buyers ask for 100 -> excess (140-100)/100 = 0.40 -> +12%
    d_eff, s_eff = _one_country(40, 100, 100, 0)
    assert d_eff[0, 0] == 140 and s_eff[0, 0] == 100
    assert price_change_rate(d_eff, s_eff, SIG, CAP, EPS)[0, 0] == pytest.approx(0.12)
    # The old rule (S = S_dom - exports, exports = the 60 surplus) saw a "balanced" market: 0%.
    old_rate = price_change_rate(np.array([40.0]), np.array([100.0 - 60.0]), SIG, CAP, EPS)[0]
    assert old_rate == pytest.approx(0.0)


def test_importer_price_rises_on_unmet_need() -> None:
    # Need 100, no own supply, receives 60 -> (100-60)/60 * 0.3 = 0.2 -> +20% (at the cap)
    d_eff, s_eff = _one_country(100, 0, 0, 60)
    assert price_change_rate(d_eff, s_eff, SIG, CAP, EPS)[0, 0] == pytest.approx(0.20)


def test_market_balance_without_trade_and_services_rule() -> None:
    D = np.arange(1, 11, dtype=float).reshape(2, 5)
    S = np.full((2, 5), 7.0)
    Q = np.full((2, 5), 3.0)
    zeros = np.zeros((2, 4))
    d_eff, s_eff = market_balance(D, S, Q, zeros, zeros)
    np.testing.assert_array_equal(d_eff, D)
    np.testing.assert_array_equal(s_eff[:, :4], S[:, :4])
    np.testing.assert_array_equal(s_eff[:, 4], Q[:, 4])  # SERVICES: S_eff = Q


def test_cpi_and_inflation() -> None:
    shares = np.array([[0.22, 0.13, 0.25, 0.12, 0.28]])
    assert cpi(shares, np.ones((1, 5)))[0] == pytest.approx(1.0)
    assert cpi(shares, np.full((1, 5), 2.0))[0] == pytest.approx(2.0)
    pi_q = inflation_quarterly(np.array([1.02]), np.array([1.0]))
    assert pi_q[0] == pytest.approx(0.02)
    assert annualize(pi_q, 4)[0] == pytest.approx(1.02**4 - 1)
