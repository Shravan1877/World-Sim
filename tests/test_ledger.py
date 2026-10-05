"""Ledger (§6.12): random transfers conserve money exactly to tolerance; accounts reconcile."""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from world.ledger import (
    BOND_MARKET,
    Ledger,
    LedgerError,
    all_accounts,
    firms,
    government,
    households,
)
from world.rng import Stream, make_rng

ACCOUNTS = all_accounts()


def _opening(seed: int) -> Ledger:
    rng = make_rng(seed, 0, Stream.BOT, 1)
    values = rng.uniform(0, 1e6, size=len(ACCOUNTS))
    return Ledger({acc: float(v) for acc, v in zip(ACCOUNTS, values, strict=True)})


@settings(max_examples=15, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_10000_random_transfers_conserve_money(seed: int) -> None:
    ledger = _opening(seed)
    rng = make_rng(seed, 0, Stream.BOT, 2)
    src = rng.integers(0, len(ACCOUNTS), size=10_000)
    shift = rng.integers(1, len(ACCOUNTS), size=10_000)
    dst = (src + shift) % len(ACCOUNTS)  # never equal to src
    amounts = rng.lognormal(mean=5.0, sigma=3.0, size=10_000)  # spans many magnitudes
    for s, d, a in zip(src, dst, amounts, strict=True):
        ledger.transfer(ACCOUNTS[s], ACCOUNTS[d], float(a), "random")
    assert len(ledger.log) == 10_000
    assert ledger.conservation_error() <= 1e-9
    ledger.check_conservation()
    ledger.check_reconciliation()


transfer_strategy = st.tuples(
    st.integers(0, len(ACCOUNTS) - 1),
    st.integers(1, len(ACCOUNTS) - 1),
    st.floats(min_value=0, max_value=1e9, allow_nan=False, allow_infinity=False),
)


@settings(max_examples=200, deadline=None)
@given(transfers=st.lists(transfer_strategy, max_size=200))
def test_any_transfer_sequence_reconciles(transfers: list[tuple[int, int, float]]) -> None:
    ledger = Ledger.with_opening({households(0): 100.0, government(5): 50.0})
    for s, shift, a in transfers:
        ledger.transfer(ACCOUNTS[s], ACCOUNTS[(s + shift) % len(ACCOUNTS)], a, "h")
    ledger.check_conservation()
    ledger.check_reconciliation()
    for acc, diff in ledger.reconcile().items():
        assert math.isclose(diff, 0.0, abs_tol=1e-9 * max(1.0, ledger._scale())), acc


def test_reconciliation_detects_tampering() -> None:
    ledger = Ledger.with_opening({households(0): 100.0})
    ledger.transfer(households(0), firms(0, 2), 40.0, "buy goods")
    ledger.balances[firms(0, 2)] += 1.0  # money appears from nowhere
    with pytest.raises(LedgerError):
        ledger.check_reconciliation()
    with pytest.raises(LedgerError):
        ledger.check_conservation()


def test_basic_transfer_and_bond_market_may_go_negative() -> None:
    ledger = Ledger.with_opening()
    ledger.transfer(BOND_MARKET, government(1), 500.0, "borrow")
    assert ledger.balance(BOND_MARKET) == -500.0
    assert ledger.balance(government(1)) == 500.0
    assert ledger.total() == 0.0
    assert ledger.log[-1].reason == "borrow"


@pytest.mark.parametrize("amount", [-1.0, float("nan"), float("inf")])
def test_bad_amounts_rejected(amount: float) -> None:
    ledger = Ledger.with_opening()
    with pytest.raises(LedgerError):
        ledger.transfer(households(0), households(1), amount, "bad")


def test_bad_accounts_rejected() -> None:
    ledger = Ledger.with_opening()
    with pytest.raises(LedgerError):
        ledger.transfer(households(0), households(0), 1.0, "self")
    with pytest.raises(LedgerError):
        households(6)
    with pytest.raises(LedgerError):
        firms(0, 5)
    with pytest.raises(LedgerError):
        Ledger({households(0): 1.0})  # missing accounts


def test_copy_is_independent() -> None:
    a = Ledger.with_opening({households(0): 10.0})
    b = a.copy()
    b.transfer(households(0), households(1), 5.0, "x")
    assert a.balance(households(0)) == 10.0
    assert len(a.log) == 0
