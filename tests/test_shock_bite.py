"""Shock-bite tests (Phase 4 calibration, D53). Required; do not loosen.

1. energy_crunch (turn 4: energy_disaster on DORNE), status-quo policies, no random shocks: within 3
   turns of the shock (turns 4-6) at least two energy-importing countries lose at least 10 stability
   points against their level after turn 3, and unemployment or inflation visibly moves in at least
   two of them (unemployment +2 points, or annualized inflation moved by 3 points).
   Energy importers = countries whose ENERGY imports exceed their ENERGY exports at the settled state.
2. One AggressorBot sanctions every other country and sets military spending to the maximum (0.4
   of GDP), the others are StatusQuoBots, random shocks on: for every choice of aggressor, some seed
   in 1-10 sees a leader fall (stability-driven, §6.11) in at least one country within 14 turns.
"""

import numpy as np
import pytest

from tests.runs import run, run_bots
from world.config import COUNTRIES, load_scenario
from world.engine.burn_in import load_fixture
from world.engine.state import ENERGY

SCN = load_scenario("energy_crunch")
SHOCK_TURN = SCN.incidents[0].turn  # 4
WINDOW = 3  # turns 4, 5, 6
MIN_STAB_DROP = 10.0
MIN_U_RISE = 0.02
MIN_PI_MOVE = 0.03
SEEDS = range(1, 11)


def energy_importers() -> list[int]:
    trade = load_fixture(0).trade[:, :, ENERGY]  # (importer, exporter), diagonal 0
    return [i for i in range(len(COUNTRIES)) if trade[i].sum() > trade[:, i].sum()]


def test_there_are_energy_importers() -> None:
    assert len(energy_importers()) >= 2, energy_importers()


@pytest.mark.parametrize("seed", SEEDS)
def test_energy_crunch_bites(seed: int) -> None:
    _, states, _ = run(seed, SHOCK_TURN + WINDOW - 1, scenario=SCN)
    before, after = states[SHOCK_TURN - 1], states[SHOCK_TURN : SHOCK_TURN + WINDOW]
    stab = np.array([s.stability for s in after])
    u = np.array([s.unemployment for s in after])
    pi = np.array([(1 + s.inflation_q) ** 4 - 1 for s in after])
    pi_before = (1 + before.inflation_q) ** 4 - 1
    drop = before.stability - stab.min(axis=0)
    u_rise = u.max(axis=0) - before.unemployment
    pi_move = np.abs(pi - pi_before).max(axis=0)
    imp = energy_importers()
    hit = [COUNTRIES[i] for i in imp if drop[i] >= MIN_STAB_DROP]
    moved = [COUNTRIES[i] for i in imp if u_rise[i] >= MIN_U_RISE or pi_move[i] >= MIN_PI_MOVE]
    detail = {
        COUNTRIES[i]: f"stab -{drop[i]:.1f}, u +{100 * u_rise[i]:.1f}pt, pi moved {100 * pi_move[i]:.1f}pt"
        for i in imp
    }
    assert len(hit) >= 2, f"importers losing >= {MIN_STAB_DROP} stability: {hit}; {detail}"
    assert len(moved) >= 2, f"importers with visible u/pi moves: {moved}; {detail}"


@pytest.mark.parametrize("country", COUNTRIES)
def test_aggressor_can_topple_a_leader(country: str) -> None:
    """AggressorBot (world/policies/bots.py) in one seat, status-quo bots elsewhere."""
    falls = {}
    for seed in SEEDS:
        r = run_bots({country: "aggressor"}, seed, 14, shocks_on=True)
        assert r.status == "ok", r.error
        who = [(rec.turn, c) for rec in r.records for c in rec.leader_changes]
        if who:
            falls[seed] = who
    assert falls, f"no leader fall in seeds {list(SEEDS)} with {country} as aggressor"
