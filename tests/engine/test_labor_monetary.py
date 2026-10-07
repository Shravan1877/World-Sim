"""Labor (§6.6) and monetary policy (§6.8)."""

import numpy as np
import pytest

from world.engine.labor import unemployment_rate, update_wage
from world.engine.monetary import (
    policy_rate,
    saving_rate,
    taylor_rate,
)

TAYLOR = dict(r_n=0.02, pi_target=0.02, a=0.5, b=0.5, u_n=0.05, floor=0.0)


def test_taylor_claude_md_check() -> None:
    assert taylor_rate(np.array([0.06]), np.array([0.04]), **TAYLOR)[0] == pytest.approx(0.065)


def test_taylor_floor_and_override() -> None:
    assert taylor_rate(np.array([-0.10]), np.array([0.20]), **TAYLOR)[0] == 0.0
    r = policy_rate(np.array([0.05, 0.05]), np.array([np.nan, 0.12]))
    np.testing.assert_allclose(r, [0.05, 0.12])


def test_saving_rate_response_and_clip() -> None:
    s = saving_rate(np.array([0.02, 0.06, 0.50]), s0=0.10, k_r=1.5, r_n=0.02, s_min=0.0, s_max=0.4)
    np.testing.assert_allclose(s, [0.10, 0.16, 0.4])


def test_unemployment_and_wage_rule() -> None:
    lf = np.array([100.0, 100.0, 100.0])
    L_d = np.array([[30.0, 30.0], [60.0, 60.0], [50.0, 50.0]])  # 60, 120, 100
    np.testing.assert_allclose(unemployment_rate(L_d, lf), [0.4, 0.0, 0.0])
    w = update_wage(np.ones(3), L_d, lf, psi=0.5, cap=0.10)
    np.testing.assert_allclose(w, [0.90, 1.10, 1.0])  # -0.2 and +0.1 are capped at +/-10%
    w2 = update_wage(np.ones(3), L_d * 0.95, lf, psi=0.5, cap=0.10)
    assert w2[1] == pytest.approx(1 + 0.5 * 0.14)


def test_phillips_sign() -> None:
    """Higher unemployment -> lower wage growth."""
    lf = np.full(5, 100.0)
    L_d = np.linspace(80, 110, 5)[:, None] * np.array([[0.5, 0.5]])
    u = unemployment_rate(L_d, lf)
    growth = update_wage(np.ones(5), L_d, lf, 0.5, 0.1) - 1
    assert np.all(np.diff(u) <= 0) and np.all(np.diff(growth) >= 0)
