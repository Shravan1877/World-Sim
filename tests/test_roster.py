"""Roster-character test (Phase 4 calibration, D52). Required; do not loosen.

At the settled state after burn-in (status quo, no shocks), trade averaged over the last 3 burn-in
turns (world.engine.burn_in.settled_trade_values, the same numbers scripts/calibrate_balance.py uses):
- each roster country is the top exporter of its good by export value: DORNE ENERGY, CERES FOOD,
  BRONTIA GOODS, EVERMERE TECH, AURELIA SERVICES (CLAUDE.md §4.1);
- every country's net exports are within +/-8% of its GDP.
"""

import numpy as np
import pytest

from tests.runs import CFG
from world.config import COUNTRIES, SECTORS
from world.engine.burn_in import run_burn_in, settled_trade_values

TOP_EXPORTER = {
    "ENERGY": "DORNE",
    "FOOD": "CERES",
    "GOODS": "BRONTIA",
    "TECH": "EVERMERE",
    "SERVICES": "AURELIA",
}
NX_BAND = 0.08


@pytest.fixture(scope="module")
def settled() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s, history = run_burn_in(CFG, 0)
    export_value, net = settled_trade_values(history)
    return export_value, net, s.gdp


@pytest.mark.parametrize("good", list(TOP_EXPORTER))
def test_top_exporter_by_export_value(settled: tuple[np.ndarray, np.ndarray, np.ndarray], good: str) -> None:
    export_value, _, _ = settled
    col = export_value[:, SECTORS.index(good)]
    ranking = {COUNTRIES[i]: round(float(col[i]), 3) for i in np.argsort(-col)}
    assert COUNTRIES[int(np.argmax(col))] == TOP_EXPORTER[good], f"{good} export value: {ranking}"


def test_net_exports_within_band(settled: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    _, net, gdp = settled
    nx = net.sum(axis=1) / gdp
    bad = {COUNTRIES[i]: f"{100 * nx[i]:+.1f}%" for i in range(len(COUNTRIES)) if abs(nx[i]) > NX_BAND}
    assert not bad, f"net exports outside +/-{100 * NX_BAND:.0f}% of GDP: {bad}"
