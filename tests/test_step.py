"""Full turns through turn_start() + step() with no agents (status-quo policies): §6.1, §6.14, §7, §8."""

import numpy as np
import pytest

from world.config import load_config, load_scenario
from world.engine.burn_in import load_fixture
from world.engine.state import GOODS, WorldState
from world.engine.step import TurnInputs, step, turn_start
from world.ledger import firms as firm_account
from world.ledger import government
from world.rng import RngBundle

CFG = load_config()
SCN = load_scenario("energy_crunch")
FALKEN = 3


def _run(seed: int, turns: int, policy=None) -> tuple[WorldState, list, list]:
    s = load_fixture(seed)  # the settled state after burn-in (D38)
    rng = RngBundle(seed)
    starts, logs = [], []
    for _ in range(turns):
        s, ts = turn_start(s, rng, CFG, scenario=SCN)
        if policy is not None:
            policy(s)
        s, log = step(s, None, rng, CFG)
        starts.append(ts)
        logs.append(log)
    return s, starts, logs


def _aggressive(s: WorldState) -> None:
    """A very different policy mix: 80% tariffs everywhere, FALKEN sanctions everyone."""
    s.tariff = np.full_like(s.tariff, 0.8)
    s.sanction = s.sanction.copy()
    s.sanction[FALKEN, :] = True
    s.sanction[FALKEN, FALKEN] = False


@pytest.mark.parametrize("seed", range(1, 21))
def test_invariants_hold_14_turns_with_shocks(seed: int) -> None:
    """§6.14: step() raises InvariantError if anything breaks; 14 turns, shocks on, 20 seeds."""
    s, _, _ = _run(seed, 14)
    assert s.turn == 14


def test_determinism_same_seed_same_hash() -> None:
    a, _, _ = _run(5, 6)
    b, _, _ = _run(5, 6)
    assert a.state_hash() == b.state_hash()
    c, _, _ = _run(6, 6)
    assert c.state_hash() != a.state_hash()


def test_common_random_numbers_across_policies() -> None:
    """Same seed, very different policies: every raw shock die and firm die is identical.

    Six turns: the runs stop before turn 7, where energy importers have collapsed (see summary).
    """
    _, starts_a, logs_a = _run(9, 6)
    end_b, starts_b, logs_b = _run(9, 6, policy=_aggressive)
    assert end_b.state_hash() != _run(9, 6)[0].state_hash()  # the policies did change the world
    for ta, tb in zip(starts_a, starts_b, strict=True):
        dice_a = [(e.shock_id, e.country, e.sector, e.u) for e in ta.shock_events if not e.scheduled]
        dice_b = [(e.shock_id, e.country, e.sector, e.u) for e in tb.shock_events if not e.scheduled]
        assert dice_a == dice_b and len(dice_a) == 6 + 6 + 30 + 1 + 6 + 30 + 6
        assert ta.order == tb.order
    for la, lb in zip(logs_a, logs_b, strict=True):
        np.testing.assert_array_equal(la.firm_dice, lb.firm_dice)


def test_nationalized_profits_go_to_treasury() -> None:
    s = load_fixture(1)
    rng = RngBundle(1)
    s, _ = turn_start(s, rng, CFG, shocks_on=False)
    s.nationalized = s.nationalized.copy()
    s.nationalized[FALKEN, GOODS] = True
    out, log = step(s, None, rng, CFG)
    new = out.ledger.log[len(s.ledger.log) :]
    acc, gov = firm_account(FALKEN, GOODS), government(FALKEN)
    signed = sum(t.amount for t in new if (t.src, t.dst) == (acc, gov))
    signed -= sum(
        t.amount for t in new if (t.src, t.dst) == (gov, acc) and t.reason == "loss covered by owner"
    )
    assert signed == pytest.approx(log.profit[FALKEN, GOODS])  # profit (or loss) is the treasury's
    assert not [
        t
        for t in new
        if t.reason.startswith(("profit", "loss")) and acc in (t.src, t.dst) and gov not in (t.src, t.dst)
    ]
    # delta_nat: A_eff is 15% lower; mu = 0 so all revenue is spent on factors or paid out
    assert out.markup[FALKEN, GOODS] == 0.0


def test_antitrust_input_cuts_markup_this_turn() -> None:
    s = load_fixture(1)
    rng = RngBundle(1)
    s, _ = turn_start(s, rng, CFG, shocks_on=False)
    k = np.zeros((6, 5), dtype=int)
    k[5, 3] = 1  # EVERMERE antitrust on TECH (e = 0.30)
    _, base = step(s, None, rng, CFG)
    _, cut = step(s, TurnInputs(antitrust=k), rng, CFG)
    b, c = base.extra["labor_demand"], cut.extra["labor_demand"]
    assert c[5, 3] / b[5, 3] == pytest.approx((1 - 0.5 * 0.7) / (1 - 0.5))  # (1 - mu) goes 0.5 -> 0.65
    c, b = c.copy(), b.copy()
    c[5, 3] = b[5, 3]
    np.testing.assert_array_equal(c, b)  # nothing else changes in this turn's hiring plans


def test_military_uses_falken_efficiency() -> None:
    s = load_fixture(1)
    rng = RngBundle(1)
    s, _ = turn_start(s, rng, CFG, shocks_on=False)
    out, log = step(s, None, rng, CFG)
    goods = log.extra["military_goods"]
    np.testing.assert_allclose(out.military, s.military * 0.95 + np.array([1, 1, 1, 1.5, 1, 1]) * goods)


def test_leader_change_flag_and_bonus() -> None:
    from world.engine.stability import leader_change

    lc = leader_change(np.array([10.0, 95.0]), np.array([0, 2]), np.array([-1, 1]), 1, 7, CFG.world.stability)
    np.testing.assert_allclose(lc.stability, [10.0, 100.0])
    assert lc.leader_changes.tolist() == [0, 3] and lc.leader_changed_turn.tolist() == [-1, 7]
