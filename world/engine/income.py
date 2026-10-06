"""The income step (CLAUDE.md §6.1 step 7, §6.7, §6.9, D28e, D31, D33).

After this turn's sales are in the firm accounts, settle_income() moves every remaining credit of
firm income through the ledger, in this order:

  1. wages        firms[i,g] -> households[i]                w_i * L[i,g]
  2. energy       firms[i,g] -> firms[i,ENERGY]              P[i,ENERGY] * E[i,g]   (g != ENERGY)
  3. profit       residual = whatever is left in the firm account (may be negative, D33)
                  residual > 0 -> owner (households[i] if private, government[i] if state)
                  residual < 0, state owner   -> the treasury covers it
                  residual < 0, private owner -> covered in step 5
  4. taxes        households[i] -> government[i]: tax_rate * max(wages + private profits, 0)   (D31)
                  private profits are net: positive residuals minus private losses.
                  State-firm profits go to the treasury untaxed.
  5. losses       households cover private losses up to their (non-negative) cash; the
                  bond_market covers the rest (D33).
Every firm account ends the step at exactly 0.

Sales revenue R^ (the base of next turn's factor demand, §6.2) is measured after steps 1-2:
    R^[i,g] = max(balance + wages paid + energy paid to the ENERGY sector - subsidy, 0)
which is all sales (home, exports, contracts, military, energy inputs sold to other sectors) minus
energy imported for stock (only the ENERGY account buys that, D34). For an energy importer R^ is the
value its own ENERGY sector sold, not the resale of imported energy.

firms.py only decides who owns what (the `nationalized` mask); this module moves the money.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world.ledger import BOND_MARKET, Ledger, government, households
from world.ledger import firms as firm_account

ENERGY = 1


@dataclass(frozen=True)
class IncomeResult:
    wages: np.ndarray  # (6,): wages paid
    profit: np.ndarray  # (6, 5): residual per firm account (negative = loss)
    private_profit: np.ndarray  # (6,): net residual of private firms (the profit part of the tax base)
    state_profit: np.ndarray  # (6,): net residual of state firms (treasury revenue, untaxed)
    taxes: np.ndarray  # (6,)
    loss_households: np.ndarray  # (6,): private losses paid by households
    loss_bond_market: np.ndarray  # (6,): private losses the bond market had to cover
    profits_to_households: np.ndarray  # (6,): private profits received - private losses households paid
    revenue: np.ndarray  # (6, 5): R^, sales revenue for next turn's factor demand
    ledger: Ledger


def balances(ledger: Ledger, n_c: int, n_s: int) -> np.ndarray:
    return np.array([[ledger.balance(firm_account(i, g)) for g in range(n_s)] for i in range(n_c)])


def settle_income(
    ledger: Ledger,
    *,
    wages_paid: np.ndarray,
    energy_cost: np.ndarray,
    subsidy: np.ndarray,
    nationalized: np.ndarray,
    tax_rate: np.ndarray,
) -> IncomeResult:
    """Steps 1-5 above. Returns a new ledger; the input ledger is not changed.

    wages_paid, energy_cost, subsidy and nationalized are (6, 5); tax_rate is (6,).
    """
    led = ledger.copy()
    n_c, n_s = nationalized.shape

    # 1-2 factor payments
    for i in range(n_c):
        for g in range(n_s):
            acc = firm_account(i, g)
            if wages_paid[i, g] > 0:
                led.transfer(acc, households(i), wages_paid[i, g], "wages")
            if g != ENERGY and energy_cost[i, g] > 0:
                led.transfer(acc, firm_account(i, ENERGY), energy_cost[i, g], "energy inputs")

    # 3 profits to owners; state losses to the treasury; private losses wait for step 5
    profit = balances(led, n_c, n_s)
    energy_paid = np.where(np.arange(n_s)[None, :] == ENERGY, 0.0, energy_cost)
    revenue = np.maximum(profit + wages_paid + energy_paid - subsidy, 0.0)
    for i in range(n_c):
        for g in range(n_s):
            acc, bal = firm_account(i, g), profit[i, g]
            owner = government(i) if nationalized[i, g] else households(i)
            if bal > 0:
                led.transfer(acc, owner, bal, "profit")
            elif bal < 0 and nationalized[i, g]:
                led.transfer(owner, acc, -bal, "loss covered by owner")
    private = np.where(nationalized, 0.0, profit)
    private_profit = private.sum(axis=1)
    state_profit = np.where(nationalized, profit, 0.0).sum(axis=1)

    # 4 taxes on household income (D31)
    wages = wages_paid.sum(axis=1)
    tax = tax_rate * np.maximum(wages + private_profit, 0.0)
    for i in range(n_c):
        if tax[i] > 0:
            led.transfer(households(i), government(i), tax[i], "taxes")

    # 5 private losses: households first (never below 0 cash), then the bond market (D33)
    loss_h, loss_b = np.zeros(n_c), np.zeros(n_c)
    for i in range(n_c):
        for g in range(n_s):
            loss = -private[i, g]
            if loss <= 0:
                continue
            acc = firm_account(i, g)
            from_h = min(loss, max(led.balance(households(i)), 0.0))
            if from_h > 0:
                led.transfer(households(i), acc, from_h, "loss covered by owner")
            if loss - from_h > 0:
                led.transfer(BOND_MARKET, acc, loss - from_h, "loss covered by bond market")
            loss_h[i] += from_h
            loss_b[i] += loss - from_h

    profits_received = np.where(private > 0, private, 0.0).sum(axis=1)
    return IncomeResult(
        wages=wages,
        profit=profit,
        private_profit=private_profit,
        state_profit=state_profit,
        taxes=tax,
        loss_households=loss_h,
        loss_bond_market=loss_b,
        profits_to_households=profits_received - loss_h,
        revenue=revenue,
        ledger=led,
    )
