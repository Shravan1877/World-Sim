"""Purity: every pure engine function (Phase 1-3) gives identical output twice and never mutates inputs."""

import copy
from dataclasses import fields, is_dataclass

import numpy as np
import pytest

from world.config import load_config
from world.engine import demand, firms, military, prices, production, trust
from world.rng import Stream, make_rng

R = make_rng(0, 0, Stream.BOT)  # test data only
P65 = R.uniform(0.5, 2.0, (6, 5))
Q65 = R.uniform(10, 100, (6, 5))
V6 = R.uniform(50, 500, 6)
SH = np.tile([0.22, 0.13, 0.25, 0.12, 0.28], (6, 1))
BETA = np.array([0.6, 0.5, 0.5, 0.6, 0.7])
GAMMA = np.array([0.2, 0.1, 0.35, 0.2, 0.1])
X64 = R.uniform(0, 20, (6, 4))

CALLS = {
    "effective_productivity": (production.effective_productivity, (Q65, P65, Q65 > 50, 0.15), {}),
    "factor_demand": (
        production.factor_demand,
        (Q65, P65, V6 / 500, P65 / 4, BETA, GAMMA, V6, P65[:, 1]),
        {},
    ),
    "ration_to_limit": (production.ration_to_limit, (Q65, V6), {}),
    "cobb_douglas": (production.cobb_douglas, (P65, Q65, Q65, BETA, GAMMA), {}),
    "produce": (
        production.produce,
        (),
        dict(
            A=P65,
            shock_mult=np.ones((6, 5)),
            nationalized=Q65 > 90,
            delta_nat=0.15,
            revenue_last=Q65,
            subsidy=Q65 / 10,
            eta=V6 / 500,
            mu=P65 / 4,
            beta=BETA,
            gamma=GAMMA,
            wage=V6 / 100,
            price=P65,
            labor_force=V6,
            stock=Q65,
        ),
    ),
    "spendable_income": (demand.spendable_income, (V6, V6 / 1000, V6, 0.1), {}),
    "update_household_cash": (demand.update_household_cash, (V6, V6, V6 / 1000, 0.1), {}),
    "household_demand": (demand.household_demand, (SH, V6, P65, 0.15, V6), {}),
    "government_goods_demand": (demand.government_goods_demand, (V6, P65[:, 2]), {}),
    "total_demand": (demand.total_demand, (Q65, V6), {}),
    "price_change_rate": (prices.price_change_rate, (Q65, P65 * 50, 0.3, 0.2, 1e-9), {}),
    "update_prices": (prices.update_prices, (P65, Q65, P65 * 50, 0.3, 0.2, 1e-9), {}),
    "cpi": (prices.cpi, (SH, P65), {}),
    "inflation_quarterly": (prices.inflation_quarterly, (V6, V6 * 0.99), {}),
    "annualize": (prices.annualize, (V6 / 10000, 4), {}),
    "base_markup": (firms.base_markup, (np.array([[1, 2, 3, 4, 10]]), 0.5), {}),
    "effective_markup": (
        firms.effective_markup,
        (np.full((6, 5), 2), Q65 > 90, np.ones((6, 5), dtype=int), V6 / 1000, 0.5),
        {},
    ),
    "entry_hazard": (firms.entry_hazard, (P65 / 4, np.ones((6, 5)), 1.0, 1.0, load_config().world.firms), {}),
    "update_military": (military.update_military, (V6, V6 / 10, V6 / 500, 0.05), {}),
    "update_trust": (
        trust.update_trust,
        (R.uniform(0, 1, (6, 6)), trust.TrustEvents(violations=((0, 1),), honored=((2, 3),))),
        {"p": load_config().world.trust},
    ),
}


def _equal(a: object, b: object) -> bool:
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)
    if is_dataclass(a):
        return all(_equal(getattr(a, f.name), getattr(b, f.name)) for f in fields(a))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, tuple | list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


@pytest.mark.parametrize("name", sorted(CALLS))
def test_same_inputs_same_outputs_and_no_mutation(name: str) -> None:
    fn, args, kwargs = CALLS[name]
    before = copy.deepcopy((args, kwargs))
    out1 = fn(*args, **kwargs)
    out2 = fn(*args, **kwargs)
    assert _equal(out1, out2)
    assert _equal(before, (args, kwargs)), f"{name} mutated its inputs"
