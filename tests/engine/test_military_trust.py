"""Military stock (§6.10) and trust updates (§6.13)."""

import numpy as np
import pytest

from world.config import load_config
from world.engine.military import update_military
from world.engine.trust import TrustEvents, drift, update_trust

CFG = load_config()
T = CFG.world.trust


def test_military_stock_update() -> None:
    mil = update_military(np.array([100.0, 100.0]), np.array([10.0, 10.0]), np.array([1.0, 1.5]), 0.05)
    np.testing.assert_allclose(mil, [95 + 10, 95 + 15])


def _flat(v: float = 0.7) -> np.ndarray:
    t = np.full((6, 6), v)
    np.fill_diagonal(t, 1.0)
    return t


def test_violation_hits_victim_and_third_parties() -> None:
    t = update_trust(_flat(), TrustEvents(violations=((3, 1),)), T)  # FALKEN violates a treaty with BRONTIA
    for k in (0, 2, 4, 5):
        assert t[k, 3] == pytest.approx(0.6 + 0.02 * 0.1)
    assert t[1, 3] == pytest.approx(0.5 + 0.02 * 0.2)  # victim: 0.7 - 0.2, then drift toward 0.7
    assert t[3, 1] == pytest.approx(0.7)  # the violator's own trust is unchanged


def test_sanction_honored_and_renounce() -> None:
    t = update_trust(_flat(0.5), TrustEvents(sanctions=((0, 2),), honored=((4, 5),), renounced=((3, 0),)), T)
    assert t[2, 0] == pytest.approx(0.4 + 0.02 * 0.3)
    assert t[4, 5] == pytest.approx(0.52 + 0.02 * 0.18)
    assert t[5, 4] == pytest.approx(t[4, 5])
    assert t[0, 3] == pytest.approx(0.5 + 0.02 * 0.2)  # renounce: no penalty, only drift


def test_drift_clip_and_diagonal() -> None:
    t = _flat(0.0)
    t[0, 1] = 1.0
    d = drift(t, 0.02, 0.7)
    assert d[2, 3] == pytest.approx(0.014) and d[0, 1] == pytest.approx(0.994)
    assert np.all(np.diag(d) == 1.0)
    low = update_trust(_flat(0.05), TrustEvents(violations=((0, 1),) * 3), T)
    assert low.min() >= 0.0 and low.max() <= 1.0
