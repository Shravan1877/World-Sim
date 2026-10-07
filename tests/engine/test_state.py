"""State (§4.3, §5): initial state from config, shapes, money views, copy, deterministic hash."""

import numpy as np
import pytest

from world.config import load_config
from world.engine.state import ENERGY, GOODS, SERVICES, WorldState, initial_state

CFG = load_config()


@pytest.fixture
def state() -> WorldState:
    return initial_state(CFG, seed=1)


def test_shapes(state: WorldState) -> None:
    grid = ("productivity", "price", "stock", "output", "revenue", "markup", "n_firms", "consumption_shares")
    for name in grid:
        assert getattr(state, name).shape == (6, 5), name
    for name in ("wage", "tax_rate", "debt", "gdp", "stability", "military", "population"):
        assert getattr(state, name).shape == (6,), name
    assert state.trust.shape == state.sanction.shape == (6, 6)
    assert state.tariff.shape == state.export_cap.shape == state.trade.shape == (6, 6, 5)
    assert len(state.firms) == 30


def test_starting_values_follow_config(state: WorldState) -> None:
    assert np.all(state.price == 1.0)
    np.testing.assert_allclose(state.population, state.labor_force / 0.6)
    np.testing.assert_allclose(state.treasury[[0, 2, 4]], state.gdp[[0, 2, 4]] * np.array([0.25, 0.05, 2.0]))
    np.testing.assert_allclose(state.debt / (4 * state.gdp), [0.3, 0.5, 0.8, 0.9, 0.6, 0.7])
    np.testing.assert_allclose(state.trust[:, 3][[0, 1, 2, 4, 5]], 0.5)  # everyone -> FALKEN
    assert state.trust[0, 1] == 0.7
    assert state.markup[0, ENERGY] == 0.5 and state.markup[5, 3] == 0.5  # dominant firms
    assert state.markup[3, 0] == 0.5 and state.markup[1, GOODS] == pytest.approx(1 / 3)
    assert len(state.firms_of(0, ENERGY)) == 1
    assert np.all(state.stock[:, SERVICES] == 0)
    assert np.all(state.stock[:, ENERGY] >= state.energy_in.sum(axis=1))  # turn 1 can produce
    np.testing.assert_allclose(state.consumption_shares.sum(axis=1), 1.0)
    assert np.all(np.isnan(state.policy_rate_override))


def test_all_finite_and_non_negative(state: WorldState) -> None:
    names = ("price", "stock", "output", "revenue", "wage", "gdp", "debt", "military")
    for name in (*names, "household_cash", "treasury"):
        v = getattr(state, name)
        assert np.all(np.isfinite(v)) and np.all(v >= 0), name
    assert np.all(state.gdp > 0)


def test_ledger_opens_balanced(state: WorldState) -> None:
    state.ledger.check_conservation()
    state.ledger.check_reconciliation()


def test_hash_is_deterministic_and_sensitive(state: WorldState) -> None:
    again = initial_state(CFG, seed=1)
    assert state.state_hash() == again.state_hash()
    assert initial_state(CFG, seed=2).state_hash() != state.state_hash()
    again.price[2, 3] += 1e-12
    assert again.state_hash() != state.state_hash()


def test_copy_is_deep(state: WorldState) -> None:
    c = state.copy()
    assert c.state_hash() == state.state_hash()
    c.price[0, 0] = 99.0
    from world.ledger import households

    c.ledger.transfer(households(0), households(1), 1.0, "x")
    assert state.price[0, 0] == 1.0
    assert state.ledger.balance(households(1)) != c.ledger.balance(households(1))


def test_country_view(state: WorldState) -> None:
    v = state.country(4)
    assert v.name == "AURELIA"
    assert v.treasury == pytest.approx(2.0 * v.gdp)
    assert v.n_firms == (4, 4, 4, 4, 4)
