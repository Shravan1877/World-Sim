"""Shocks (§7): each effect, timing of multipliers, scheduled incidents, and common random numbers."""

import math

import numpy as np
import pytest

from world.config import ScenarioCfg, check_scenario, load_config, load_scenario
from world.engine import shocks
from world.engine.state import ENERGY, FOOD, TECH, ActiveShock, WorldState, initial_state
from world.engine.step import turn_start
from world.rng import RngBundle

CFG = load_config()
SPEC = {s.id: s for s in CFG.shocks.shocks}
DORNE, BRONTIA, CERES, FALKEN, AURELIA, EVERMERE = range(6)


def _state(turn: int = 1) -> WorldState:
    s = initial_state(CFG, seed=1)
    s.turn = turn
    return s


def _always(shock_id: str):
    spec = SPEC[shock_id]
    return spec.model_copy(update={"hazard_default": 1.0, "hazard_by_country": {}})


def test_harvest_failure_lasts_two_turns() -> None:
    s, ev = shocks.harvest_failure(_state(), RngBundle(1), _always("harvest_failure"), CFG)
    assert all(e.fired for e in ev) and len(ev) == 6
    shocks.recompute_multipliers(s)
    np.testing.assert_allclose(s.shock_mult[:, FOOD], 0.6)
    s.active_shocks = shocks.age_shocks(s.active_shocks)
    shocks.recompute_multipliers(s)
    np.testing.assert_allclose(s.shock_mult[:, FOOD], 0.6)  # still on for the 2nd turn
    s.active_shocks = shocks.age_shocks(s.active_shocks)
    shocks.recompute_multipliers(s)
    np.testing.assert_allclose(s.shock_mult, 1.0)  # gone


def test_energy_disaster_hazard_only_dorne_but_all_dice_rolled() -> None:
    s0 = _state()
    _, ev = shocks.energy_disaster(s0, RngBundle(1), SPEC["energy_disaster"], CFG)
    assert len(ev) == 6
    assert [e.hazard for e in ev] == [0.03, 0, 0, 0, 0, 0]


def test_industry_collapse_cuts_a_firm_min_one() -> None:
    s, _ = shocks.industry_collapse(_state(), RngBundle(1), _always("industry_collapse"), CFG)
    shocks.recompute_multipliers(s)
    np.testing.assert_allclose(s.shock_mult, 0.5)
    assert s.n_firms[DORNE, ENERGY] == 1 and s.n_firms[BRONTIA, 0] == 3
    assert all(len(fl) == n for fl, n in zip(s.firms, s.n_firms.ravel(), strict=True))
    assert s.markup[BRONTIA, 0] == pytest.approx(1 / 3)
    assert {sh.turns_left for sh in s.active_shocks} == {4}


def test_pandemic_cuts_labor_force_for_three_turns() -> None:
    s, ev = shocks.pandemic(_state(), RngBundle(1), _always("pandemic"), CFG)
    assert len(ev) == 1 and ev[0].country == -1
    shocks.recompute_multipliers(s)
    np.testing.assert_allclose(s.labor_force, 0.9 * s.labor_force_base)
    for _ in range(3):
        s.active_shocks = shocks.age_shocks(s.active_shocks)
    shocks.recompute_multipliers(s)
    np.testing.assert_allclose(s.labor_force, s.labor_force_base)


def test_leader_death_changes_leader() -> None:
    s0 = _state(turn=3)
    s, _ = shocks.leader_death(s0, RngBundle(1), _always("leader_death"), CFG)
    np.testing.assert_allclose(s.stability, np.minimum(s0.stability + 15, 100))
    assert np.all(s.leader_changes == 1) and np.all(s.leader_changed_turn == 3)


def test_industry_leader_death_only_hits_dominant_sectors() -> None:
    s, ev = shocks.industry_leader_death(_state(), RngBundle(1), _always("industry_leader_death"), CFG)
    fired = {(e.country, e.sector) for e in ev if e.fired}
    assert len(ev) == 30  # every die rolled
    assert (DORNE, ENERGY) in fired and (EVERMERE, TECH) in fired and (FALKEN, FOOD) in fired
    assert (BRONTIA, FOOD) not in fired  # n = 4
    shocks.recompute_multipliers(s)
    assert s.shock_mult[DORNE, ENERGY] == pytest.approx(0.8)
    assert shocks.entry_boost(s)[EVERMERE, TECH] == 2.0 and shocks.entry_boost(s)[BRONTIA, FOOD] == 1.0
    assert {sh.turns_left for sh in s.active_shocks if sh.kind == "entry_hazard"} == {4}


def test_resource_discovery_is_permanent() -> None:
    s0 = _state()
    s, _ = shocks.resource_discovery(s0, RngBundle(1), _always("resource_discovery"), CFG)
    np.testing.assert_allclose(s.productivity[:, ENERGY], 1.3 * s0.productivity[:, ENERGY])
    assert s.active_shocks == ()


def test_age_shocks_keeps_permanent() -> None:
    perm = ActiveShock("x", 0, 0, 1.3, -1)
    assert shocks.age_shocks((perm, ActiveShock("y", 0, 0, 0.5, 1))) == (perm,)


def test_shock_functions_do_not_mutate_input() -> None:
    s0 = _state()
    h = s0.state_hash()
    for sid, fn in shocks.SHOCK_FUNCTIONS.items():
        fn(s0, RngBundle(1), _always(sid), CFG)
    assert s0.state_hash() == h


# ------------------------------------------------------------------------------- scenarios


def test_energy_crunch_fires_on_turn_4_in_every_run() -> None:
    scn = load_scenario("energy_crunch")
    for seed in range(1, 21):
        s = initial_state(CFG, seed)
        rng = RngBundle(seed)
        for turn in range(1, 6):
            s, ts = turn_start(s, rng, CFG, scenario=scn)
            sched = [e for e in ts.shock_events if e.scheduled]
            if turn == 4:
                assert [(e.shock_id, e.country) for e in sched] == [("energy_disaster", DORNE)]
                assert s.shock_mult[DORNE, ENERGY] <= 0.5 + 1e-12
            else:
                assert sched == []


def test_scenario_validation() -> None:
    bad = ScenarioCfg(name="x", incidents=[{"turn": 1, "shock": "meteor", "country": "DORNE"}])
    with pytest.raises(ValueError, match="unknown shock"):
        check_scenario(bad, CFG.shocks)
    with pytest.raises(ValueError, match="needs a country"):
        check_scenario(ScenarioCfg(name="x", incidents=[{"turn": 1, "shock": "harvest_failure"}]), CFG.shocks)
    with pytest.raises(ValueError, match="needs a sector"):
        check_scenario(
            ScenarioCfg(name="x", incidents=[{"turn": 1, "shock": "industry_collapse", "country": "CERES"}]),
            CFG.shocks,
        )
    with pytest.raises(ValueError, match="state-dependent"):
        check_scenario(ScenarioCfg(name="x", incidents=[{"turn": 1, "shock": "unrest"}]), CFG.shocks)


def test_scheduled_events_have_no_dice() -> None:
    s = _state(turn=4)
    _, ev = shocks.apply_scenario(s, load_scenario("energy_crunch"), CFG)
    assert len(ev) == 1 and math.isnan(ev[0].u) and ev[0].fired
