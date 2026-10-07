"""Policy application (§9) and every country's special powers (§4.1), end to end through step()."""

import numpy as np
import pytest

from tests.runs import CFG, play_turn
from world.actions import ActionIn
from world.engine.burn_in import load_fixture
from world.engine.params import country_params
from world.engine.policy import PolicyEffects, apply_actions, hostile_acts
from world.engine.state import ENERGY, FOOD, GOODS

A = ActionIn
D, B, C, F, AU, E = range(6)
S0 = load_fixture(1)
CONTROL = play_turn(S0, [])


def test_shared_domestic_policies_set_fields() -> None:
    p = play_turn(S0, [("CERES", [A(type="set_tax", rate=0.3),
                                  A(type="set_spending", welfare=0.15, military=0.05, subsidy=0.03),
                                  A(type="set_subsidy_target", sector="FOOD")])])  # fmt: skip
    s = p.before_step
    assert (s.tax_rate[C], s.welfare_share[C], s.military_share[C], s.subsidy_share[C]) == (
        0.3,
        0.15,
        0.05,
        0.03,
    )
    assert s.subsidy_target[C] == FOOD
    sub = p.log.extra["subsidy"][C]
    assert sub == pytest.approx(0.03 * S0.gdp[C])


def test_tariff_all_goods_and_revenue_to_importer() -> None:
    p = play_turn(S0, [("AURELIA", [A(type="set_tariff", target="CERES", good="ALL", rate=0.3)])])
    assert np.all(p.before_step.tariff[AU, C] == 0.3) and np.all(p.before_step.tariff[AU, D] == 0)
    tariffs = [t for t in p.after.ledger.log if t.reason.startswith("tariff") and t.dst.country == AU]
    assert tariffs and all(t.dst.kind == "government" for t in tariffs)


def test_sanction_blocks_trade_both_ways_and_is_new_only_once() -> None:
    fx = PolicyEffects()
    s = apply_actions(S0, D, [_strict(A(type="set_sanction", target="CERES", on=True))], CFG, fx)
    s = apply_actions(s, D, [_strict(A(type="set_sanction", target="CERES", on=True))], CFG, fx)
    assert fx.sanctions_imposed == [(D, C)]
    p = play_turn(S0, [("DORNE", [A(type="set_sanction", target="CERES", on=True)])])
    assert np.all(p.after.trade[D, C] == 0) and np.all(p.after.trade[C, D] == 0)


def _free_sanctions():
    """CFG with every sanction self-cost set to 0 (same trade effects, no stability cost)."""
    trade = CFG.world.trade.model_copy(update={"sanction_self_cost_default": 0.0})
    countries = [
        c.model_copy(update={"params": c.params.model_copy(update={"sanction_self_cost": None})})
        for c in CFG.countries.countries
    ]
    return CFG.model_copy(
        update={
            "world": CFG.world.model_copy(update={"trade": trade}),
            "countries": CFG.countries.model_copy(update={"countries": countries}),
        }
    )


@pytest.mark.parametrize(("country", "i"), [("DORNE", D), ("AURELIA", AU)])
def test_sanction_self_cost_and_aurelia_cheap_sanctions(country: str, i: int) -> None:
    """Same sanction with and without the self-cost: the stability gap is exactly that cost."""
    moves = [(country, [A(type="set_sanction", target="CERES", on=True)])]
    paid, free = play_turn(S0, moves), play_turn(S0, moves, cfg=_free_sanctions())
    cost = country_params(CFG).sanction_self_cost[i]
    assert free.after.stability[i] - paid.after.stability[i] == pytest.approx(cost)
    if country == "AURELIA":
        assert cost == pytest.approx(0.3 * CFG.world.trade.sanction_self_cost_default)


def test_dorne_quota_and_levy() -> None:
    p = play_turn(S0, [("DORNE", [A(type="set_energy_export_quota", target="ALL", rate=0.2),
                                  A(type="set_energy_export_levy", rate=0.3)])])  # fmt: skip
    s = p.before_step
    assert np.allclose(np.delete(s.export_cap[:, D, ENERGY], D), 0.2) and s.levy[D, ENERGY] == 0.3
    assert s.export_cap[B, D, FOOD] == 1.0  # other goods untouched
    exp_share = p.after.trade[:, D, ENERGY].sum() / CONTROL.after.trade[:, D, ENERGY].sum()
    assert exp_share < 1.0
    levies = sum(t.amount for t in p.after.ledger.log if t.reason.startswith("export levy"))
    assert levies > 0


def test_dorne_quota_on_one_target() -> None:
    p = play_turn(S0, [("DORNE", [A(type="set_energy_export_quota", target="BRONTIA", rate=0.0)])])
    assert p.after.trade[B, D, ENERGY] == 0.0
    assert p.before_step.export_cap[C, D, ENERGY] == 1.0


def test_ceres_food_ban_on_and_off() -> None:
    p = play_turn(S0, [("CERES", [A(type="set_food_export_ban", target="ALL", on=True)])])
    assert np.all(p.after.trade[:, C, FOOD] == 0)
    q = play_turn(p.after, [("CERES", [A(type="set_food_export_ban", target="AURELIA", on=False)])])
    assert q.before_step.export_cap[AU, C, FOOD] == CFG.world.trade.export_cap_default
    assert q.before_step.export_cap[B, C, FOOD] == 0.0


def test_brontia_subsidize_industry_full_efficiency() -> None:
    p = play_turn(S0, [("BRONTIA", [A(type="subsidize_industry", sector="GOODS", amount=0.05)])])
    extra = p.log.extra["subsidy"][B] - CONTROL.log.extra["subsidy"][B]
    assert extra == pytest.approx(0.05 * S0.gdp[B])
    eta = country_params(CFG).subsidy_efficiency
    assert eta[B] == 1.0 and eta[D] == CFG.world.production.subsidy_efficiency_default == 0.5
    assert p.after.labor[B, GOODS] > CONTROL.after.labor[B, GOODS]


def test_falken_nationalize() -> None:
    p = play_turn(S0, [("FALKEN", [A(type="nationalize", sector="ENERGY")])])
    s = p.before_step
    assert s.nationalized[F, ENERGY]
    assert all(f.owner == "state" for f in s.firms_of(F, ENERGY))
    assert all(f.owner == "private" for f in s.firms_of(F, GOODS))
    assert p.after.markup[F, ENERGY] == 0.0
    assert p.log.extra["state_profit"][F] != 0


def test_aurelia_policy_rate_override_and_back() -> None:
    p = play_turn(S0, [("AURELIA", [A(type="set_policy_rate", rate=0.12)])])
    assert p.after.policy_rate[AU] == pytest.approx(0.12)
    q = play_turn(p.after, [("AURELIA", [A(type="set_policy_rate")])])
    assert np.isnan(q.before_step.policy_rate_override[AU])
    assert q.after.policy_rate[AU] != pytest.approx(0.12)


def test_antitrust_counts_and_evermere_strength() -> None:
    fx = PolicyEffects()
    apply_actions(S0, E, [_strict(A(type="antitrust", sector="TECH"))] * 2, CFG, fx)
    assert fx.turn_inputs().antitrust[E, 3] == 2
    cp = country_params(CFG)
    assert cp.antitrust_strength[E] == pytest.approx(2 * cp.antitrust_strength[D])
    assert (cp.breakup_omega[E], cp.entry_mult[E]) == (0.8, 2.0)
    p = play_turn(S0, [("EVERMERE", [A(type="antitrust", sector="TECH")])])
    assert p.after.labor[E, 3] > CONTROL.after.labor[E, 3]  # lower markup -> more hiring


def test_falken_military_efficiency() -> None:
    cp = country_params(CFG)
    assert cp.military_efficiency[F] == 1.5 and cp.military_efficiency[D] == 1.0


def test_hostile_acts() -> None:
    fx = PolicyEffects()
    s = apply_actions(
        S0,
        D,
        [
            _strict(A(type="set_tariff", target="CERES", good="FOOD", rate=0.1)),
            _strict(A(type="set_sanction", target="FALKEN", on=True)),
            _strict(A(type="set_energy_export_quota", target="BRONTIA", rate=0.5)),
        ],
        CFG,
        fx,
    )
    assert hostile_acts(S0, s, D) == [("cut_exports", B), ("raise_tariff", C), ("sanction", F)]
    assert hostile_acts(s, S0, D) == []  # undoing them is not hostile


def test_policy_does_not_mutate_its_input() -> None:
    before = S0.state_hash()
    apply_actions(S0, D, [_strict(A(type="set_tax", rate=0.5))], CFG, PolicyEffects())
    assert S0.state_hash() == before


def _strict(a: ActionIn):
    from world.validator import convert

    out = convert(a)
    assert not isinstance(out, str), out
    return out
