"""Production (§6.2): Cobb-Douglas checks, factor demand, rationing, energy taken from stock once."""

import numpy as np
import pytest

from world.engine.production import (
    cobb_douglas,
    effective_productivity,
    factor_demand,
    produce,
    ration_to_limit,
)

B1 = np.array([0.5])
G1 = np.array([0.3])


def test_cobb_douglas_claude_md_check() -> None:
    A = np.array([[2.0]])
    E = np.array([[100.0]])
    assert cobb_douglas(A, np.array([[100.0]]), E, B1, G1)[0, 0] == pytest.approx(79.6, abs=0.05)
    assert cobb_douglas(A, np.array([[200.0]]), E, B1, G1)[0, 0] == pytest.approx(112.6, abs=0.05)
    # +41%: diminishing returns to labor
    ratio = cobb_douglas(A, np.array([[200.0]]), E, B1, G1) / cobb_douglas(A, np.array([[100.0]]), E, B1, G1)
    assert ratio[0, 0] == pytest.approx(2**0.5)


def test_factor_demand_formula() -> None:
    # R~ = 100 + 0.5*10 = 105; (1-mu) R~ = 78.75; L_d = 78.75*0.5/2; E_d = 78.75*0.3/4
    L_d, E_d = factor_demand(
        revenue_last=np.array([[100.0]]),
        subsidy=np.array([[10.0]]),
        eta=np.array([0.5]),
        mu=np.array([[0.25]]),
        beta=B1,
        gamma=G1,
        wage=np.array([2.0]),
        p_energy=np.array([4.0]),
    )
    assert L_d[0, 0] == pytest.approx(19.6875)
    assert E_d[0, 0] == pytest.approx(5.90625)


def test_monopoly_markup_cuts_hiring() -> None:
    args = dict(
        revenue_last=np.array([[100.0, 100.0]]),
        subsidy=np.zeros((1, 2)),
        eta=np.array([0.5]),
        beta=np.array([0.5, 0.5]),
        gamma=np.array([0.3, 0.3]),
        wage=np.array([1.0]),
        p_energy=np.array([1.0]),
    )
    L_d, _ = factor_demand(mu=np.array([[0.25, 0.5]]), **args)
    assert L_d[0, 1] < L_d[0, 0]


def test_rationing_scales_proportionally() -> None:
    demand = np.array([[100.0, 200.0], [10.0, 20.0]])
    limit = np.array([150.0, 1000.0])
    out = ration_to_limit(demand, limit)
    np.testing.assert_allclose(out[0], [50.0, 100.0])  # scaled by 150/300
    np.testing.assert_allclose(out[1], [10.0, 20.0])  # under the limit: unchanged
    assert out[0].sum() == pytest.approx(150.0)


def test_rationing_zero_limit_and_zero_demand() -> None:
    out = ration_to_limit(np.array([[5.0, 5.0], [0.0, 0.0]]), np.array([0.0, 0.0]))
    np.testing.assert_allclose(out, 0.0)


def test_effective_productivity() -> None:
    A = np.array([[2.0, 2.0]])
    out = effective_productivity(A, np.array([[0.5, 1.0]]), np.array([[False, True]]), 0.15)
    np.testing.assert_allclose(out, [[1.0, 1.7]])


def _produce_inputs(energy_stock: float) -> dict:
    n_s = 5
    return dict(
        A=np.full((2, n_s), 1.0),
        shock_mult=np.ones((2, n_s)),
        nationalized=np.zeros((2, n_s), dtype=bool),
        delta_nat=0.15,
        revenue_last=np.full((2, n_s), 100.0),
        subsidy=np.zeros((2, n_s)),
        eta=np.array([0.5, 0.5]),
        mu=np.full((2, n_s), 0.25),
        beta=np.array([0.6, 0.5, 0.5, 0.6, 0.7]),
        gamma=np.array([0.2, 0.1, 0.35, 0.2, 0.1]),
        wage=np.array([1.0, 1.0]),
        price=np.ones((2, n_s)),
        labor_force=np.array([1000.0, 100.0]),
        stock=np.array([[0, energy_stock, 0, 0, 0], [0, 1000.0, 0, 0, 0]], dtype=float),
    )


def test_produce_rations_labor_and_energy_and_removes_energy_once() -> None:
    inputs = _produce_inputs(energy_stock=30.0)
    r = produce(**inputs)
    # country 0: energy demand 0.75*100*sum(gamma) = 71.25 > stock 30 -> scaled to 30
    assert r.energy_demand[0].sum() == pytest.approx(71.25)
    assert r.energy[0].sum() == pytest.approx(30.0)
    np.testing.assert_allclose(r.energy[0] / r.energy_demand[0], 30.0 / 71.25)
    assert r.stock_after_inputs[0, 1] == pytest.approx(0.0)
    # country 1: labor demand 0.75*100*sum(beta) = 217.5 > LF 100 -> scaled to 100
    assert r.labor[1].sum() == pytest.approx(100.0)
    # energy not rationed in country 1: stock falls by exactly the energy used
    assert r.stock_after_inputs[1, 1] == pytest.approx(1000.0 - r.energy[1].sum())
    # non-energy stocks untouched
    np.testing.assert_array_equal(r.stock_after_inputs[:, [0, 2, 3, 4]], inputs["stock"][:, [0, 2, 3, 4]])
