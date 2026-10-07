"""Prices (§6.5, D28b, D45): gradual rule, cap, convergence, seller cases, CPI, inflation."""

import numpy as np
import pytest

from world.engine.prices import (
    annualize,
    cpi,
    inflation_quarterly,
    price_change_rate,
    update_prices,
)
from world.engine.trade import allocate_trade
from world.ledger import Ledger, households

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


def _market(stock: np.ndarray, demand: np.ndarray, home_bias: float = 1.0):
    """D45 market for one good among 2 countries (FOOD column only)."""
    n = stock.shape[0]
    return allocate_trade(
        stock=stock[:, None],
        output=np.zeros((n, 1)),
        demand=demand[:, None],
        price=np.ones((n, 1)),
        levy=np.zeros((n, 1)),
        tariff=np.zeros((n, n, 1)),
        export_cap=np.ones((n, n, 1)),
        sanction=np.zeros((n, n), dtype=bool),
        trust=np.full((n, n), 0.7),
        kappa=0.0,
        sigma=3.0,
        passes=2,
        contracts=(),
        ledger=Ledger.with_opening({households(i): 1e6 for i in range(n)}),
        home_bias=home_bias,
    )


def test_exporter_price_rises_when_sold_out() -> None:
    # Country 0 has 100, home demand 40; country 1 has nothing and asks for 100.
    # D45: R_0 = 40 + 100 = 140 requests against S = 100 -> (140-100)/100 * 0.3 = +12%.
    r = _market(np.array([100.0, 0.0]), np.array([40.0, 100.0]))
    assert r.requests[0, 0] == pytest.approx(140.0) and r.supply[0, 0] == 100.0
    assert price_change_rate(r.requests, r.supply, SIG, CAP, EPS)[0, 0] == pytest.approx(0.12)
    # The old rule (S = S_dom - exports, exports = the 60 surplus) saw a "balanced" market: 0%.
    old_rate = price_change_rate(np.array([40.0]), np.array([100.0 - 60.0]), SIG, CAP, EPS)[0]
    assert old_rate == pytest.approx(0.0)


def test_seller_rations_home_and_foreign_buyers_alike() -> None:
    """D45: requests 140 against supply 100 -> everyone gets 100/140 of what they asked."""
    r = _market(np.array([100.0, 0.0]), np.array([40.0, 100.0]))
    assert r.home[0, 0] == pytest.approx(40.0 * 100 / 140)
    assert r.imports[1, 0] == pytest.approx(100.0 * 100 / 140)
    assert r.unmet[:, 0].sum() == pytest.approx(40.0)


def test_home_bias_shifts_requests_home() -> None:
    """Equal prices and trust (kappa = 0): theta = 2 -> home gets 2/3 of a 2-seller choice."""
    r = _market(np.array([1000.0, 1000.0]), np.array([90.0, 0.0]), home_bias=2.0)
    assert r.home[0, 0] == pytest.approx(60.0)
    assert r.imports[0, 0] == pytest.approx(30.0)


def test_seller_without_supply_or_requests_keeps_its_price() -> None:
    assert price_change_rate(np.zeros((1, 1)), np.zeros((1, 1)), SIG, CAP, EPS)[0, 0] == 0.0


def test_cpi_and_inflation() -> None:
    shares = np.array([[0.22, 0.13, 0.25, 0.12, 0.28]])
    assert cpi(shares, np.ones((1, 5)))[0] == pytest.approx(1.0)
    assert cpi(shares, np.full((1, 5), 2.0))[0] == pytest.approx(2.0)
    pi_q = inflation_quarterly(np.array([1.02]), np.array([1.0]))
    assert pi_q[0] == pytest.approx(0.02)
    assert annualize(pi_q, 4)[0] == pytest.approx(1.02**4 - 1)
