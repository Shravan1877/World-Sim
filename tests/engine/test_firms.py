"""Firms (§6.9): Cournot numbers, markup rule, breakup (analytic + Monte Carlo), entry, HHI, profits."""

import numpy as np
import pytest

from world.config import COUNTRIES, SECTORS, load_config
from world.engine import firms
from world.engine.state import Firm
from world.ledger import Ledger, government, households
from world.ledger import firms as firm_account
from world.rng import RngBundle

CFG = load_config()
P = CFG.world.firms


@pytest.mark.parametrize(
    ("n", "q", "price", "mu"), [(2, 50.0, 1.00, 0.50), (4, 37.5, 0.67, 0.25), (10, 18.0, 0.56, 0.10)]
)
def test_cournot_check_values(n: int, q: float, price: float, mu: float) -> None:
    q_i, p_i, mu_i = firms.cournot(a=2.0, d=100.0, n=n)
    assert q_i == pytest.approx(q)
    assert p_i == pytest.approx(price, abs=0.005)
    assert mu_i == pytest.approx(mu)


def test_markup_rule() -> None:
    n = np.array([[1, 2, 4, 10, 3]] * 6)
    nat = np.zeros((6, 5), dtype=bool)
    nat[3, 4] = True
    k = np.zeros((6, 5), dtype=int)
    k[5, 0] = 2
    e = np.array([0.15] * 5 + [0.30])
    mu = firms.effective_markup(n, nat, k, e, mu_dominant=0.5)
    np.testing.assert_allclose(mu[0], [0.5, 0.5, 0.25, 0.1, 1 / 3])
    assert mu[3, 4] == 0.0  # nationalized
    assert mu[5, 0] == pytest.approx(0.5 * 0.7**2)  # EVERMERE antitrust x2, e = 0.30


@pytest.mark.parametrize(("o", "expected"), [(1, 0.10), (5, 0.41), (20, 0.88)])
def test_breakup_probability_analytic(o: int, expected: float) -> None:
    assert firms.breakup_probability(0.9, o) == pytest.approx(expected, abs=0.005)


@pytest.mark.parametrize("o", [1, 5, 20])
def test_breakup_monte_carlo(o: int) -> None:
    """All 30 sectors are monopolies with dominance age o-1; one update makes it o. Count breakups."""
    mono = tuple((Firm(f"f{k}", 1.0),) for k in range(30))
    turns = 400
    hits = 0
    for t in range(1, turns + 1):
        fu = firms.update_firms(
            firm_lists=mono,
            nationalized=np.zeros((6, 5), dtype=bool),
            dominance_age=np.full((6, 5), o - 1),
            mu=np.full((6, 5), 0.5),
            entry_boost=np.ones((6, 5)),
            breakup_omega=np.full(6, 0.9),
            entry_mult=np.ones(6),
            p=P,
            rng=RngBundle(7),
            turn=t,
            country_names=COUNTRIES,
            sector_names=SECTORS,
        )
        hits += sum(e.kind == "antitrust_breakup" for e in fu.events)
    trials = turns * 30
    expected = 1 - 0.9**o
    se = np.sqrt(expected * (1 - expected) / trials)
    assert abs(hits / trials - expected) < 4 * se


def test_breakup_splits_largest_and_entry_breaks_monopoly() -> None:
    fl = (Firm("a", 1.0),)
    split = firms.split_largest(fl, "b")
    assert [f.share for f in split] == [0.5, 0.5]
    entered = firms.add_entrant(fl, "s")
    assert [f.share for f in entered] == [0.5, 0.5]
    four = firms.add_entrant(tuple(Firm(str(k), 1 / 3) for k in range(3)), "x")
    np.testing.assert_allclose([f.share for f in four], 0.25)


def test_entry_hazard_formula() -> None:
    # h = 0.02 (1 + 2 * 0.5)(1 + 8/8) * 2 = 0.16 ; with boost 2 -> 0.32
    assert firms.entry_hazard(0.5, 8, 2.0, 1.0, P) == pytest.approx(0.16)
    assert firms.entry_hazard(0.5, 8, 2.0, 2.0, P) == pytest.approx(0.32)
    assert firms.entry_hazard(0.25, 0, 1.0, 1.0, P) == pytest.approx(0.03)


@pytest.mark.parametrize(("shares", "expected"), [([0.5, 0.3, 0.2], 0.38), ([0.25] * 4, 0.25), ([1.0], 1.0)])
def test_hhi(shares: list[float], expected: float) -> None:
    assert firms.hhi(shares) == pytest.approx(expected)


def test_remove_firm_renormalizes_and_keeps_one() -> None:
    fl = (Firm("a", 0.5), Firm("b", 0.3), Firm("c", 0.2))
    out = firms.remove_smallest(fl)
    assert [f.id for f in out] == ["a", "b"]
    assert sum(f.share for f in out) == pytest.approx(1.0)
    assert firms.remove_smallest((Firm("a", 1.0),)) == (Firm("a", 1.0),)


def test_nationalized_sector_has_no_entry_or_breakup() -> None:
    mono = tuple((Firm(f"f{k}", 1.0, "state"),) for k in range(30))
    fu = firms.update_firms(
        firm_lists=mono,
        nationalized=np.ones((6, 5), dtype=bool),
        dominance_age=np.full((6, 5), 50),
        mu=np.zeros((6, 5)),
        entry_boost=np.full((6, 5), 100.0),
        breakup_omega=np.full(6, 0.1),
        entry_mult=np.ones(6),
        p=P,
        rng=RngBundle(1),
        turn=1,
        country_names=COUNTRIES,
        sector_names=SECTORS,
    )
    assert fu.events == () and np.all(fu.n_firms == 1) and np.all(fu.markup == 0.0)


def test_firm_dice_do_not_depend_on_state() -> None:
    kwargs = dict(
        entry_boost=np.ones((6, 5)),
        breakup_omega=np.full(6, 0.9),
        entry_mult=np.ones(6),
        p=P,
        rng=RngBundle(3),
        turn=5,
        country_names=COUNTRIES,
        sector_names=SECTORS,
    )
    a = firms.update_firms(
        firm_lists=tuple((Firm("x", 1.0),) for _ in range(30)),
        nationalized=np.zeros((6, 5), dtype=bool),
        dominance_age=np.zeros((6, 5), dtype=int),
        mu=np.full((6, 5), 0.5),
        **kwargs,
    )
    b = firms.update_firms(
        firm_lists=tuple(tuple(Firm(str(k), 0.25) for k in range(4)) for _ in range(30)),
        nationalized=np.ones((6, 5), dtype=bool),
        dominance_age=np.full((6, 5), 9),
        mu=np.zeros((6, 5)),
        **kwargs,
    )
    np.testing.assert_array_equal(a.dice, b.dice)


def test_profit_payout_routes_to_owner_and_empties_accounts() -> None:
    led = Ledger.with_opening({households(0): 100.0, government(0): 100.0})
    led.transfer(households(0), firm_account(0, 0), 10.0, "sales")  # private, profit +10
    led.transfer(government(0), firm_account(0, 1), 4.0, "sales")  # state, profit +4
    led.transfer(firm_account(0, 2), households(0), 3.0, "wages")  # private, loss -3
    nat = np.zeros((6, 5), dtype=bool)
    nat[0, 1] = True
    out = firms.pay_out_profits(led, nat)
    assert out.private[0] == pytest.approx(7.0)
    assert out.state[0] == pytest.approx(4.0)
    assert all(out.ledger.balance(firm_account(i, g)) == 0.0 for i in range(6) for g in range(5))
    out.ledger.check_conservation()
