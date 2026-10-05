"""Stability (§6.11): formula, clip, unrest and leader-fall draws on the SHOCK stream."""

import numpy as np
import pytest

from world.config import load_config
from world.engine.stability import stability_change, unrest_and_leader_fall, update_stability
from world.rng import RngBundle

P = load_config().world.stability


def _delta(**over) -> float:
    args = dict(
        stability=np.array([70.0]),
        u=np.array([0.05]),
        pi_annual=np.array([0.02]),
        food_shortage_frac=np.zeros(1),
        energy_shortage_frac=np.zeros(1),
        welfare_to_gdp=np.array([0.10]),
        sanction_cost=np.zeros(1),
        shock_effects=np.zeros(1),
        u_n=0.05,
        pi_target=0.02,
        p=P,
    )
    args.update(over)
    return float(stability_change(**args)[0])


def test_neutral_conditions_give_zero_change() -> None:
    assert _delta() == pytest.approx(0.0)


def test_each_term() -> None:
    assert _delta(u=np.array([0.10])) == pytest.approx(-5.0)
    assert _delta(pi_annual=np.array([0.07])) == pytest.approx(-0.8 * 5)
    assert _delta(food_shortage_frac=np.array([0.1])) == pytest.approx(-30.0)
    assert _delta(energy_shortage_frac=np.array([0.1])) == pytest.approx(-15.0)
    assert _delta(welfare_to_gdp=np.array([0.15])) == pytest.approx(5.0)
    assert _delta(stability=np.array([50.0])) == pytest.approx(2.0)
    assert _delta(sanction_cost=np.array([0.5])) == pytest.approx(-0.5)


def test_clip() -> None:
    np.testing.assert_allclose(
        update_stability(np.array([5.0, 98.0]), np.array([-20.0, 10.0]), P), [0.0, 100.0]
    )


def test_no_draws_above_thresholds_and_deterministic_below() -> None:
    r = unrest_and_leader_fall(np.full(6, 50.0), np.full(6, 0.5), P, RngBundle(1), turn=3)
    assert not r.unrest.any() and not r.leader_falls.any()
    stab = np.array([0.0, 5.0, 10.0, 14.0, 20.0, 29.0])
    a = unrest_and_leader_fall(stab, np.full(6, 0.8), P, RngBundle(1), turn=3)
    b = unrest_and_leader_fall(stab, np.full(6, 0.8), P, RngBundle(1), turn=3)
    np.testing.assert_array_equal(a.unrest, b.unrest)
    np.testing.assert_array_equal(a.leader_falls, b.leader_falls)
    np.testing.assert_allclose(a.stability, np.where(a.unrest, np.maximum(stab - 5, 0), stab))


def test_unrest_frequency_matches_probability() -> None:
    hits = sum(
        unrest_and_leader_fall(np.full(6, 0.0), np.zeros(6), P, RngBundle(s), turn=1).unrest.sum()
        for s in range(500)
    )
    assert hits / 3000 == pytest.approx(0.5, abs=0.04)  # (30 - 0) / 60
