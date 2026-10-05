"""Taylor rule, AURELIA's policy-rate override, saving response, savings interest (CLAUDE.md §6.8).

  r_i = max(r_n + pi_t + a (pi_i^annual - pi_t) + b (u_n - u_i), floor)
  s_i = clip(s0 + k_r (r_i - r_n), s_min, s_max)
Savings stay as household cash and earn r_i / 4 per quarter from the bond_market account.
"""

from __future__ import annotations

import numpy as np

from world.ledger import BOND_MARKET, Ledger, households


def taylor_rate(
    pi_annual: np.ndarray,
    u: np.ndarray,
    *,
    r_n: float,
    pi_target: float,
    a: float,
    b: float,
    u_n: float,
    floor: float,
) -> np.ndarray:
    return np.maximum(r_n + pi_target + a * (pi_annual - pi_target) + b * (u_n - u), floor)


def policy_rate(taylor: np.ndarray, override: np.ndarray) -> np.ndarray:
    """Use the override where it is set (not NaN), else the Taylor rate (AURELIA's power, §9)."""
    return np.where(np.isnan(override), taylor, override)


def saving_rate(
    rate: np.ndarray, *, s0: float, k_r: float, r_n: float, s_min: float, s_max: float
) -> np.ndarray:
    return np.clip(s0 + k_r * (rate - r_n), s_min, s_max)


def savings_interest(savings: np.ndarray, rate: np.ndarray, periods_per_year: int) -> np.ndarray:
    """Quarterly interest on household savings: r / 4 * H (negative savings earn nothing)."""
    return rate / periods_per_year * np.maximum(savings, 0.0)


def pay_savings_interest(ledger: Ledger, interest: np.ndarray) -> Ledger:
    """bond_market -> households[i]. Returns a new ledger."""
    ledger = ledger.copy()
    for i, amount in enumerate(interest):
        if amount > 0:
            ledger.transfer(BOND_MARKET, households(i), float(amount), "savings interest")
    return ledger
