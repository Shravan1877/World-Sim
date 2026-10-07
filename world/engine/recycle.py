"""The money loop (CLAUDE.md §6.7, D37): money that would sit idle goes back into circulation.

Run every turn at the end of the fiscal step (after borrowing and the default check), in this order:
  (b) treasury cash above a buffer of `buffer_quarters` x this turn's outlays repays debt
      (government -> bond_market); once debt is 0, the rest is returned to the country's
      households as a lump sum (government -> households).
  (a) any positive bond_market balance is paid to households of all countries pro rata to their
      population (D52; `weights="cash"` = the original D37 rule: pro rata to cash H, by population
      if every H is 0). The bond market may stay negative: it is the issuer.
      Why D52: with cash weights the richest households (DORNE's) collected about 30% of every
      government's interest and repayments, which paid for a trade deficit of -15% of GDP.
(b) runs first so that interest and repayments received this turn are recycled in the same turn.
All moves are logged ledger transfers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world.ledger import BOND_MARKET, Ledger, government, households


@dataclass(frozen=True)
class RecycleResult:
    debt: np.ndarray
    repaid: np.ndarray  # (6,): debt repaid from treasury surplus
    lump_sum: np.ndarray  # (6,): treasury surplus returned to households (debt already 0)
    bond_payout: np.ndarray  # (6,): bond-market surplus paid to each country's households
    ledger: Ledger


def recycle(
    ledger: Ledger,
    *,
    debt: np.ndarray,
    outlays: np.ndarray,
    buffer_quarters: float,
    population: np.ndarray,
    weights: str = "population",
) -> RecycleResult:
    led = ledger.copy()
    n = debt.shape[0]
    debt = debt.copy()
    repaid, lump, payout = np.zeros(n), np.zeros(n), np.zeros(n)

    for i in range(n):
        excess = led.balance(government(i)) - buffer_quarters * max(outlays[i], 0.0)
        if excess <= 0:
            continue
        repaid[i] = min(excess, debt[i])
        if repaid[i] > 0:
            led.transfer(government(i), BOND_MARKET, repaid[i], "debt repayment")
            debt[i] -= repaid[i]
        lump[i] = excess - repaid[i]
        if lump[i] > 0:
            led.transfer(government(i), households(i), lump[i], "treasury surplus to households")

    surplus = led.balance(BOND_MARKET)
    if surplus > 0:
        cash = np.maximum([led.balance(households(i)) for i in range(n)], 0.0)
        w = cash if weights == "cash" and cash.sum() > 0 else population.astype(float)
        payout = surplus * w / w.sum()
        for i in range(n):
            if payout[i] > 0:
                led.transfer(BOND_MARKET, households(i), payout[i], "bond market surplus to households")
    return RecycleResult(debt, repaid, lump, payout, led)
