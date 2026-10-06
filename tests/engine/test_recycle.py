"""Money loop (§6.7, D37): treasury surplus repays debt or goes home; bond surplus goes to households."""

import numpy as np
import pytest

from world.engine.recycle import recycle
from world.ledger import BOND_MARKET, Ledger, government, households

POP = np.full(6, 10.0)


def test_treasury_above_buffer_repays_debt_then_lump_sum() -> None:
    led = Ledger.with_opening({government(0): 100.0, government(1): 100.0, government(2): 10.0})
    debt = np.array([50.0, 0.0, 30.0, 0.0, 0.0, 0.0])
    outlays = np.full(6, 40.0)  # buffer = 0.5 * 40 = 20
    r = recycle(led, debt=debt, outlays=outlays, buffer_quarters=0.5, population=POP)
    # country 0: excess 80 -> repay 50, lump sum 30; country 1: no debt -> lump 80; country 2: below buffer
    np.testing.assert_allclose(r.repaid[:3], [50.0, 0.0, 0.0])
    np.testing.assert_allclose(r.lump_sum[:3], [30.0, 80.0, 0.0])
    np.testing.assert_allclose(r.debt[:3], [0.0, 0.0, 30.0])
    assert r.ledger.balance(government(0)) == pytest.approx(20.0)
    assert r.ledger.balance(government(2)) == pytest.approx(10.0)
    r.ledger.check_conservation()


def test_positive_bond_market_paid_pro_rata_to_cash() -> None:
    led = Ledger.with_opening({BOND_MARKET: 30.0, households(0): 10.0, households(1): 20.0})
    r = recycle(led, debt=np.zeros(6), outlays=np.zeros(6), buffer_quarters=0.5, population=POP)
    np.testing.assert_allclose(r.bond_payout, [10.0, 20.0, 0, 0, 0, 0])
    assert r.ledger.balance(BOND_MARKET) == pytest.approx(0.0)
    r.ledger.check_conservation()


def test_debt_repayment_recycled_same_turn_and_negative_bond_market_left_alone() -> None:
    led = Ledger.with_opening({government(0): 50.0, households(1): 5.0, BOND_MARKET: -100.0})
    r = recycle(
        led, debt=np.array([200.0, 0, 0, 0, 0, 0]), outlays=np.zeros(6), buffer_quarters=0.5, population=POP
    )
    assert r.repaid[0] == pytest.approx(50.0)
    assert r.ledger.balance(BOND_MARKET) == pytest.approx(-50.0)  # still negative: nothing paid out
    assert r.bond_payout.sum() == 0.0


def test_all_cash_zero_pays_by_population() -> None:
    led = Ledger.with_opening({BOND_MARKET: 12.0})
    pop = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    r = recycle(led, debt=np.zeros(6), outlays=np.zeros(6), buffer_quarters=0.5, population=pop)
    np.testing.assert_allclose(r.bond_payout, [2.0, 4.0, 6.0, 0, 0, 0])
