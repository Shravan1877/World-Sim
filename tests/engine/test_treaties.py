"""Treaties (§10): each kind's life cycle, execution inside step(), and every violation path.

Moves go through the validator and policy.py (tests.runs.play_turn), so these are end-to-end.
"""

import numpy as np
import pytest

from tests.runs import CFG, play_turn
from world.actions import ActionIn
from world.config import COUNTRIES
from world.engine import treaties as tr
from world.engine.burn_in import load_fixture
from world.ledger import BOND_MARKET, government

A = ActionIn
D, B, C, F, AU, E = range(6)  # DORNE, BRONTIA, CERES, FALKEN, AURELIA, EVERMERE
S0 = load_fixture(1)
DRIFT = CFG.world.trust.drift_rate


def propose(target: str, kind: str, duration: int = 3, terms: str | None = None) -> ActionIn:
    return A(type="propose_treaty", target=target, kind=kind, duration=duration, terms=terms)  # type: ignore[arg-type]


def accept(tid: str) -> ActionIn:
    return A(type="accept_treaty", treaty_id=tid)


def status(state, tid: str) -> str:
    t = tr.by_id(state.treaties, tid)
    assert t is not None
    return t.status


# ------------------------------------------------------------------------------ life cycle


def test_propose_then_accept_same_turn() -> None:
    p = play_turn(
        S0, [("DORNE", [propose("BRONTIA", "no_sanction_pact")]), ("BRONTIA", [accept("T1-DORNE-1")])]
    )
    assert not p.results["BRONTIA"].rejected
    t = tr.by_id(p.after.treaties, "T1-DORNE-1")
    assert t is not None and t.status == "active" and t.start_turn == 1
    assert t.proposer == D and t.addressee == B


def test_accept_next_turn_then_expire_after_one_full_turn() -> None:
    p1 = play_turn(S0, [("DORNE", [propose("BRONTIA", "no_sanction_pact")])])
    assert status(p1.after, "T1-DORNE-1") == "proposed"
    p2 = play_turn(p1.after, [("BRONTIA", [accept("T1-DORNE-1")])])  # turn 2: still open
    assert status(p2.after, "T1-DORNE-1") == "active"
    q2 = play_turn(p1.after, [])  # nobody answers in turn 2
    q3 = play_turn(q2.after, [("BRONTIA", [accept("T1-DORNE-1")])])  # turn 3: expired at turn start
    assert status(q3.after, "T1-DORNE-1") == "expired"
    assert q3.results["BRONTIA"].rejected


def test_reject() -> None:
    p = play_turn(S0, [("DORNE", [propose("BRONTIA", "tariff_cap", terms="max_rate=0.1")]),
                       ("BRONTIA", [A(type="reject_treaty", treaty_id="T1-DORNE-1")])])  # fmt: skip
    assert status(p.after, "T1-DORNE-1") == "rejected"


def test_active_treaty_ends_after_its_duration() -> None:
    p = play_turn(S0, [("DORNE", [propose("BRONTIA", "no_sanction_pact", duration=2)]),
                       ("BRONTIA", [accept("T1-DORNE-1")])])  # fmt: skip
    assert status(p.after, "T1-DORNE-1") == "active"
    p2 = play_turn(p.after, [])
    t = tr.by_id(p2.after.treaties, "T1-DORNE-1")
    assert t.status == "ended" and t.note == "completed" and t.end_turn == 2


def test_honored_treaty_raises_trust_both_ways() -> None:
    base = play_turn(S0, [])
    p = play_turn(
        S0, [("DORNE", [propose("BRONTIA", "no_sanction_pact")]), ("BRONTIA", [accept("T1-DORNE-1")])]
    )
    bonus = CFG.world.trust.treaty_honored * (1 - DRIFT)
    assert p.after.trust[D, B] - base.after.trust[D, B] == pytest.approx(bonus)
    assert p.after.trust[B, D] - base.after.trust[B, D] == pytest.approx(bonus)


def test_ids_are_unique_per_proposer_and_turn() -> None:
    p = play_turn(
        S0, [("DORNE", [propose("BRONTIA", "no_sanction_pact"), propose("CERES", "no_sanction_pact")])]
    )
    assert [t.id for t in p.after.treaties] == ["T1-DORNE-1", "T1-DORNE-2"]


# ------------------------------------------------------------------------------- execution


def test_supply_contract_delivers_at_contract_price() -> None:
    terms = "role=seller; good=ENERGY; quantity=0.3; price=0.9"
    p = play_turn(S0, [("DORNE", [propose("BRONTIA", "supply_contract", terms=terms)]),
                       ("BRONTIA", [accept("T1-DORNE-1")])])  # fmt: skip
    (d,) = p.log.deliveries
    assert d.delivered == pytest.approx(0.3) and d.shortfall == 0 and not d.policy_caused
    assert p.log.violations == ()
    paid = [t for t in p.after.ledger.log if t.reason == "contract T1-DORNE-1"]
    assert sum(t.amount for t in paid) == pytest.approx(0.27)


def test_buyer_proposed_supply_contract_swaps_roles() -> None:
    terms = "role=buyer; good=FOOD; quantity=0.2; price=0.5"
    p = play_turn(S0, [("AURELIA", [propose("CERES", "supply_contract", terms=terms)])])
    t = tr.by_id(p.after.treaties, "T1-AURELIA-1")
    assert (t.term("seller"), t.term("buyer")) == (C, AU)


def test_loan_disbursed_then_repaid_then_completed() -> None:
    p1 = play_turn(S0, [("AURELIA", [propose("CERES", "loan", duration=2, terms="amount=0.2; rate=0.08")]),
                        ("CERES", [accept("T1-AURELIA-1")])])  # fmt: skip
    (pay,) = p1.log.loans
    assert (pay.kind, pay.paid) == ("disbursement", pytest.approx(0.2))
    installment = 0.2 / 2 + 0.2 * 0.08 / 4  # equal principal + simple quarterly interest (D57)
    p2 = play_turn(p1.after, [])
    assert p2.log.loans[0].kind == "installment" and p2.log.loans[0].due == pytest.approx(installment)
    p3 = play_turn(p2.after, [])
    assert p3.log.loans[0].paid == pytest.approx(installment)
    t = tr.by_id(p3.after.treaties, "T1-AURELIA-1")
    assert t.status == "ended" and t.note == "completed"
    p4 = play_turn(p3.after, [])
    assert p4.log.loans == ()


def test_installment_without_funds_is_not_a_violation() -> None:
    p1 = play_turn(S0, [("AURELIA", [propose("CERES", "loan", duration=2, terms="amount=1; rate=0.0")]),
                        ("CERES", [accept("T1-AURELIA-1")])])  # fmt: skip
    s = p1.after.copy()
    s.ledger.transfer(government(C), BOND_MARKET, s.treasury[C], "test: empty the treasury")
    p2 = play_turn(s, [])
    assert p2.log.loans[0].paid < p2.log.loans[0].due
    assert p2.log.violations == ()


# ------------------------------------------------------------------------------ violations


def _agreed(kind: str, terms: str | None = None, a: str = "DORNE", b: str = "BRONTIA"):
    return play_turn(S0, [(a, [propose(b, kind, duration=4, terms=terms)]), (b, [accept(f"T1-{a}-1")])]).after


def _violated(p, tid: str, violator: int, victim: int) -> None:
    assert [(v.treaty_id, v.violator, v.victim) for v in p.log.violations] == [(tid, violator, victim)]
    t = tr.by_id(p.after.treaties, tid)
    assert t.status == "suspended" and t.note == f"violated by {COUNTRIES[violator]}"
    assert any(e.kind == "treaty_violation" for e in p.log.events)


def test_tariff_above_cap_is_a_violation() -> None:
    s = _agreed("tariff_cap", "max_rate=0.1")
    p = play_turn(s, [("BRONTIA", [A(type="set_tariff", target="DORNE", good="TECH", rate=0.15)])])
    _violated(p, "T1-DORNE-1", B, D)


def test_tariff_at_cap_is_fine() -> None:
    s = _agreed("tariff_cap", "max_rate=0.1")
    p = play_turn(s, [("BRONTIA", [A(type="set_tariff", target="DORNE", good="ALL", rate=0.1)])])
    assert p.log.violations == ()


def test_sanction_under_pact_is_a_violation_and_costs_trust() -> None:
    s = _agreed("no_sanction_pact")
    control = play_turn(s, [])
    p = play_turn(s, [("DORNE", [A(type="set_sanction", target="BRONTIA", on=True)])])
    _violated(p, "T1-DORNE-1", D, B)
    t = CFG.world.trust
    # victim: violation + sanctioned; a third party: violation only (control: honored treaty, drift)
    victim = (t.violation_victim + t.sanctioned - t.treaty_honored) * (1 - DRIFT)
    assert p.after.trust[B, D] - control.after.trust[B, D] == pytest.approx(victim)
    assert p.after.trust[C, D] - control.after.trust[C, D] == pytest.approx(
        t.violation_third_party * (1 - DRIFT)
    )


@pytest.mark.parametrize(
    ("seller", "good", "cut"),
    [
        ("DORNE", "ENERGY", A(type="set_energy_export_quota", target="BRONTIA", rate=0.0)),
        ("CERES", "FOOD", A(type="set_food_export_ban", target="ALL", on=True)),
        ("DORNE", "ENERGY", A(type="set_sanction", target="BRONTIA", on=True)),
    ],
    ids=["quota", "food_ban", "sanction"],
)
def test_supply_cut_by_policy_is_a_violation(seller: str, good: str, cut: ActionIn) -> None:
    s = _agreed("supply_contract", f"role=seller; good={good}; quantity=0.2; price=1.0", a=seller)
    p = play_turn(s, [(seller, [cut])])
    _violated(p, f"T1-{seller}-1", COUNTRIES.index(seller), B)


def test_loan_installment_blocked_by_borrowers_sanction_is_a_violation() -> None:
    s = play_turn(S0, [("AURELIA", [propose("CERES", "loan", duration=3, terms="amount=1; rate=0.05")]),
                       ("CERES", [accept("T1-AURELIA-1")])]).after  # fmt: skip
    p = play_turn(s, [("CERES", [A(type="set_sanction", target="AURELIA", on=True)])])
    assert p.log.loans[0].paid == 0 and p.log.loans[0].blocked_by == (C,)
    _violated(p, "T1-AURELIA-1", C, AU)


def test_renounce_is_legal_exit_costing_falken_stability_not_trust() -> None:
    s = _agreed("no_sanction_pact", a="FALKEN", b="CERES")
    control = play_turn(s, [])
    p = play_turn(s, [("FALKEN", [A(type="renounce_treaty", treaty_id="T1-FALKEN-1")])])
    t = tr.by_id(p.after.treaties, "T1-FALKEN-1")
    assert t.status == "ended" and t.note == "renounced"
    assert p.log.violations == ()
    cost = CFG.countries.by_name("FALKEN").params.renounce_stability_cost
    assert control.after.stability[F] - p.after.stability[F] == pytest.approx(cost)
    # no trust penalty: the only difference is the honored bonus the control run still earned
    bonus = CFG.world.trust.treaty_honored * (1 - DRIFT)
    assert control.after.trust[C, F] - p.after.trust[C, F] == pytest.approx(bonus)
    assert np.allclose(
        np.delete(control.after.trust, [C, F], axis=0), np.delete(p.after.trust, [C, F], axis=0)
    )
