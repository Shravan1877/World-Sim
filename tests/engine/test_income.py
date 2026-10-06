"""Income step (§6.1 step 7, §6.7, §6.9, D28e, D31, D33): payouts, taxes, losses, the income identity."""

import numpy as np
import pytest

from world.config import load_config
from world.engine.income import settle_income
from world.engine.state import initial_state
from world.engine.step import step, turn_start
from world.ledger import BOND_MARKET, Ledger, government, households
from world.ledger import firms as firm_account
from world.rng import RngBundle

CFG = load_config()
Z65 = np.zeros((6, 5))
NO_TAX = np.zeros(6)


def _settle(led: Ledger, *, wages=Z65, energy=Z65, subsidy=Z65, nat=None, tax=NO_TAX):
    nat = np.zeros((6, 5), dtype=bool) if nat is None else nat
    return settle_income(
        led, wages_paid=wages, energy_cost=energy, subsidy=subsidy, nationalized=nat, tax_rate=tax
    )


def _all_firm_accounts_zero(led: Ledger) -> bool:
    return all(led.balance(firm_account(i, g)) == 0.0 for i in range(6) for g in range(5))


def test_profit_routes_to_owner_and_empties_accounts() -> None:
    led = Ledger.with_opening({households(0): 100.0, government(0): 100.0})
    led.transfer(households(0), firm_account(0, 0), 10.0, "sales")  # private, profit +10
    led.transfer(government(0), firm_account(0, 1), 4.0, "sales")  # state, profit +4
    nat = np.zeros((6, 5), dtype=bool)
    nat[0, 1] = True
    wages = Z65.copy()
    wages[0, 2] = 3.0  # private, no sales: loss -3
    out = _settle(led, wages=wages, nat=nat)
    assert out.private_profit[0] == pytest.approx(10.0 - 3.0)
    assert out.state_profit[0] == pytest.approx(4.0)
    assert out.loss_households[0] == pytest.approx(3.0) and out.loss_bond_market[0] == 0.0
    assert _all_firm_accounts_zero(out.ledger)
    out.ledger.check_conservation()


def test_wages_and_energy_inputs_paid_and_revenue_measured() -> None:
    led = Ledger.with_opening({households(0): 100.0})
    led.transfer(households(0), firm_account(0, 2), 50.0, "domestic sales")  # GOODS sells 50
    wages, energy = Z65.copy(), Z65.copy()
    wages[0, 2], energy[0, 2] = 20.0, 10.0  # GOODS pays 20 wages, buys 10 of energy
    out = _settle(led, wages=wages, energy=energy)
    assert out.revenue[0, 2] == pytest.approx(50.0)
    assert out.revenue[0, 1] == pytest.approx(10.0)  # ENERGY sold 10 of inputs
    assert out.profit[0, 2] == pytest.approx(20.0) and out.profit[0, 1] == pytest.approx(10.0)
    assert out.ledger.balance(households(0)) == pytest.approx(100.0 - 50.0 + 20.0 + 30.0)
    assert _all_firm_accounts_zero(out.ledger)


def test_taxes_on_wages_plus_private_profit_only() -> None:
    """D31: tax = rate * (wages + private profits). State profits are untaxed."""
    led = Ledger.with_opening({households(0): 1000.0})
    led.transfer(households(0), firm_account(0, 0), 100.0, "sales")  # private sector
    led.transfer(households(0), firm_account(0, 4), 60.0, "sales")  # state sector
    nat = np.zeros((6, 5), dtype=bool)
    nat[0, 4] = True
    wages = Z65.copy()
    wages[0, 0], wages[0, 4] = 30.0, 20.0
    tax = np.full(6, 0.2)
    out = _settle(led, wages=wages, nat=nat, tax=tax)
    # wages 50 + private profit (100 - 30) = 120 -> 24; state profit 40 to the treasury untaxed
    assert out.taxes[0] == pytest.approx(24.0)
    assert out.ledger.balance(government(0)) == pytest.approx(24.0 + 40.0)
    assert out.profits_to_households[0] == pytest.approx(70.0)


def _loss_case(cash_before: float, loss: float, tax: float = 0.0):
    """TECH pays `loss` in wages and sells nothing; households start the step with cash_before."""
    led = Ledger.with_opening({})
    if cash_before > 0:
        led.transfer(BOND_MARKET, households(0), cash_before, "test")
    elif cash_before < 0:
        led.transfer(households(0), government(0), -cash_before, "test")  # spent ahead of income
    wages = Z65.copy()
    wages[0, 3] = loss
    return _settle(led, wages=wages, tax=np.full(6, tax))


@pytest.mark.parametrize(
    ("cash_before", "from_households", "from_bond"),
    [(5.0, 12.0, 0.0), (-5.0, 7.0, 5.0), (-20.0, 0.0, 12.0)],
)
def test_private_loss_households_first_then_bond_market(
    cash_before: float, from_households: float, from_bond: float
) -> None:
    """D33: owners absorb losses from their cash (never below 0); the bond market covers the rest."""
    out = _loss_case(cash_before, 12.0, tax=0.5)
    assert out.taxes[0] == 0.0  # base = wages 12 + private profit -12 = 0
    assert out.loss_households[0] == pytest.approx(from_households)
    assert out.loss_bond_market[0] == pytest.approx(from_bond)
    assert out.ledger.balance(households(0)) == pytest.approx(cash_before + 12.0 - from_households)
    assert _all_firm_accounts_zero(out.ledger)
    out.ledger.check_conservation()


def test_state_loss_paid_by_treasury() -> None:
    led = Ledger.with_opening({government(0): 10.0})
    nat = np.zeros((6, 5), dtype=bool)
    nat[0, 2] = True
    wages = Z65.copy()
    wages[0, 2] = 4.0
    out = _settle(led, wages=wages, nat=nat)
    assert out.state_profit[0] == pytest.approx(-4.0)
    assert out.ledger.balance(government(0)) == pytest.approx(6.0)
    assert _all_firm_accounts_zero(out.ledger)


def _sum(log, reason, src_kind=None, dst_kind=None, country=None, startswith=False) -> float:
    total = 0.0
    for t in log:
        ok = t.reason.startswith(reason) if startswith else t.reason == reason
        if not ok:
            continue
        if src_kind and t.src.kind != src_kind:
            continue
        if dst_kind and t.dst.kind != dst_kind:
            continue
        if country is not None:
            side = t.dst if dst_kind else t.src
            if side.country != country:
                continue
        total += t.amount
    return total


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_income_identity_full_turns(seed: int) -> None:
    """Household income + government revenue = value added + transfers, per country and turn.

    Everything except y_disp is read back from the ledger log of the turn (not from engine arrays):
      y_disp_i + (taxes + tariffs + levies + state profits)_i
        = sales value added_i + subsidies_i + welfare_i + savings interest_i
          + tariffs_i + levies_i + losses the bond market covered_i
    Sales value added = what the country's firms took in from outside (sales, exports) minus what
    they paid outside (energy imported for stock). Energy sold between home firms nets out.
    No money is stuck in firm accounts, and none is created.
    """
    s = initial_state(CFG, seed)
    rng = RngBundle(seed)
    for _ in range(3):
        s, _ = turn_start(s, rng, CFG, shocks_on=False)
        n0 = len(s.ledger.log)
        s, log = step(s, None, rng, CFG)
        new = s.ledger.log[n0:]
        for i in range(6):
            firm_in = sum(
                t.amount
                for t in new
                if t.dst.kind == "firms"
                and t.dst.country == i
                and not (t.src.kind == "firms" and t.src.country == i)
                and t.reason not in ("subsidy",)
                and not t.reason.startswith("loss covered")
            )
            firm_out = sum(
                t.amount
                for t in new
                if t.src.kind == "firms"
                and t.src.country == i
                and not (t.dst.kind == "firms" and t.dst.country == i)
                and t.reason not in ("wages", "profit")
            )
            va = firm_in - firm_out
            subsidy = _sum(new, "subsidy", dst_kind="firms", country=i)
            welfare = _sum(new, "welfare", dst_kind="households", country=i)
            sav = _sum(new, "savings interest", dst_kind="households", country=i)
            tariffs = _sum(new, "tariff", dst_kind="government", country=i, startswith=True)
            levies = _sum(new, "export levy", dst_kind="government", country=i, startswith=True)
            taxes = _sum(new, "taxes", dst_kind="government", country=i)
            state_profit = _sum(new, "profit", dst_kind="government", country=i) - sum(
                t.amount for t in new if t.reason == "loss covered by owner" and t.src == government(i)
            )
            bond_loss = _sum(new, "loss covered by bond market", dst_kind="firms", country=i)
            lhs = s.y_disp[i] + taxes + tariffs + levies + state_profit
            rhs = va + subsidy + welfare + sav + tariffs + levies + bond_loss
            assert lhs == pytest.approx(rhs, rel=1e-9, abs=1e-9)
            assert taxes == pytest.approx(log.taxes[i])
