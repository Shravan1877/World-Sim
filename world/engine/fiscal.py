"""GDP, government budget, interest, borrowing and default (CLAUDE.md §6.7). Pure functions.

  GDP_i      = sum_g P[i,g] Q[i,g] - sum_g P[i,ENERGY] E[i,g]      (value added)
  revenue_i  = taxes + tariffs + levies + state firm profits
  outlays_i  = welfare + military + subsidy + interest
  interest_i = (policy_rate_i + premium_i) / 4 * debt_i
  premium_i  = max(0, slope * (debt / (4 GDP) - threshold)) + default_premium_i
  treasury'  = treasury + revenue - outlays; if < 0, borrow the gap (debt += gap, treasury = 0)
  default    : debt / (4 GDP) > 1.5 -> debt x 0.5, stability -20, premium +0.05 for 8 turns

NOTE (Phase 2, open question A): the tax BASE is not wired here yet. CLAUDE.md says
taxes = tax_rate * GDP, but Y_disp = (wages + private profits)(1 - tax_rate) taxes income, and the
two differ whenever output is stocked or sold from stock. `taxes()` takes the base as an input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ENERGY = 1


def gdp_value_added(price: np.ndarray, output: np.ndarray, energy_in: np.ndarray) -> np.ndarray:
    return (price * output).sum(axis=1) - price[:, ENERGY] * energy_in.sum(axis=1)


def taxes(tax_rate: np.ndarray, base: np.ndarray) -> np.ndarray:
    return tax_rate * base


def debt_to_annual_gdp(debt: np.ndarray, gdp_q: np.ndarray, periods_per_year: int, eps: float) -> np.ndarray:
    return debt / (periods_per_year * np.maximum(gdp_q, eps))


def risk_premium(
    debt: np.ndarray,
    gdp_q: np.ndarray,
    default_premium: np.ndarray,
    *,
    slope: float,
    threshold: float,
    periods_per_year: int,
    eps: float,
) -> np.ndarray:
    ratio = debt_to_annual_gdp(debt, gdp_q, periods_per_year, eps)
    return np.maximum(0.0, slope * (ratio - threshold)) + default_premium


def interest_due(
    debt: np.ndarray, rate: np.ndarray, premium: np.ndarray, periods_per_year: int
) -> np.ndarray:
    return (rate + premium) / periods_per_year * debt


@dataclass(frozen=True)
class BudgetResult:
    treasury: np.ndarray
    debt: np.ndarray
    borrowing: np.ndarray  # credits borrowed from the bond market this turn


def settle_budget(
    treasury: np.ndarray, debt: np.ndarray, revenue: np.ndarray, outlays: np.ndarray
) -> BudgetResult:
    """treasury' = treasury + revenue - outlays; a negative result is borrowed (debt absorbs it)."""
    raw = treasury + revenue - outlays
    borrowing = np.maximum(-raw, 0.0)
    return BudgetResult(treasury=np.maximum(raw, 0.0), debt=debt + borrowing, borrowing=borrowing)


@dataclass(frozen=True)
class DefaultResult:
    defaulted: np.ndarray  # bool: newly defaulted this turn
    debt: np.ndarray
    stability: np.ndarray
    default_premium: np.ndarray
    default_turns_left: np.ndarray


def apply_default(
    debt: np.ndarray,
    gdp_q: np.ndarray,
    stability: np.ndarray,
    default_premium: np.ndarray,
    default_turns_left: np.ndarray,
    *,
    threshold: float,
    haircut: float,
    stability_hit: float,
    premium_add: float,
    premium_turns: int,
    periods_per_year: int,
    eps: float,
) -> DefaultResult:
    """Default rule. A country already in default (turns_left > 0) cannot default again.

    The extra premium lasts `premium_turns` turns: when the counter runs out it is removed.
    The haircut is a write-down of the debt record (the bond market absorbs it; no cash moves).
    """
    ticking = default_turns_left > 0
    turns = np.where(ticking, default_turns_left - 1, default_turns_left)
    premium = np.where(
        ticking & (turns == 0), np.maximum(default_premium - premium_add, 0.0), default_premium
    )

    ratio = debt_to_annual_gdp(debt, gdp_q, periods_per_year, eps)
    new = (ratio > threshold) & ~ticking
    return DefaultResult(
        defaulted=new,
        debt=np.where(new, debt * haircut, debt),
        stability=np.where(new, np.maximum(stability - stability_hit, 0.0), stability),
        default_premium=np.where(new, premium + premium_add, premium),
        default_turns_left=np.where(new, premium_turns, turns).astype(np.int64),
    )


def in_default(default_turns_left: np.ndarray) -> np.ndarray:
    """While in default, the validator forbids spending plans with outlays > revenue (§6.7)."""
    return default_turns_left > 0


def disposable_income(
    wages: np.ndarray,
    private_profits: np.ndarray,
    tax_rate: np.ndarray,
    welfare: np.ndarray,
    savings_interest: np.ndarray,
) -> np.ndarray:
    """Y_disp = (wages + private profits)(1 - tax) + welfare + interest on savings."""
    return (wages + private_profits) * (1.0 - tax_rate) + welfare + savings_interest
