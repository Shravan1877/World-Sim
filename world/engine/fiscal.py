"""GDP, government budget, interest, borrowing and default (CLAUDE.md §6.7, D31, D32). Pure functions.

  GDP_i      = sum_g P[i,g] Q[i,g] - sum_g P[i,ENERGY] E[i,g]      (value added)
  taxes_i    = tax_rate_i * max(wages_i + private_profits_i, 0)     (D31: household income, not GDP)
  revenue_i  = taxes + tariffs + levies + state firm profits (untaxed)
  outlays_i  = welfare + military + subsidy + interest
  GDP_ref_i  = max(mean of the last 4 quarterly GDPs, 0.10 * starting GDP)   (D32)
  interest_i = (policy_rate_i + premium_i) / 4 * debt_i
  premium_i  = min(max(0, slope * (debt / (4 GDP_ref) - threshold)) + default_premium_i, cap)
  treasury'  = treasury + revenue - outlays; if < 0, borrow the gap (debt += gap, treasury = 0)
  default    : debt / (4 GDP_ref) > 1.5 -> debt x 0.5, stability -20, premium +0.05 for 8 turns
GDP itself is used only for spending shares and (through GDP_ref) debt ratios.
Taxes are collected in the income step (income.py), right after profits are paid out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ENERGY = 1


def gdp_value_added(price: np.ndarray, output: np.ndarray, energy_in: np.ndarray) -> np.ndarray:
    return (price * output).sum(axis=1) - price[:, ENERGY] * energy_in.sum(axis=1)


def taxes(tax_rate: np.ndarray, base: np.ndarray) -> np.ndarray:
    """tax_rate * max(base, 0). The base is household income: wages + private profits (D31)."""
    return tax_rate * np.maximum(base, 0.0)


def push_gdp(gdp_hist: np.ndarray, gdp_q: np.ndarray) -> np.ndarray:
    """Drop the oldest quarter of the GDP window and append this quarter (newest last)."""
    return np.concatenate([gdp_hist[:, 1:], gdp_q[:, None]], axis=1)


def gdp_reference(gdp_hist: np.ndarray, gdp_start: np.ndarray, floor_share: float) -> np.ndarray:
    """GDP_ref = max(moving average of the window, floor_share * starting GDP) (D32)."""
    return np.maximum(gdp_hist.mean(axis=1), floor_share * gdp_start)


def debt_to_annual_gdp(debt: np.ndarray, gdp_q: np.ndarray, periods_per_year: int, eps: float) -> np.ndarray:
    return debt / (periods_per_year * np.maximum(gdp_q, eps))


def risk_premium(
    debt: np.ndarray,
    gdp_q: np.ndarray,
    default_premium: np.ndarray,
    *,
    slope: float,
    threshold: float,
    cap: float,
    periods_per_year: int,
    eps: float,
) -> np.ndarray:
    """Total annual premium, capped at `cap` (D32). Pass GDP_ref as gdp_q."""
    ratio = debt_to_annual_gdp(debt, gdp_q, periods_per_year, eps)
    return np.minimum(np.maximum(0.0, slope * (ratio - threshold)) + default_premium, cap)


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
    taxes_paid: np.ndarray,
    transfers: np.ndarray,
) -> np.ndarray:
    """Y_disp = wages + private profits - taxes + transfers (D31, D42).

    With taxes = tax_rate * (wages + private profits) this is (wages + private profits)(1 - tax)
    + transfers. private_profits is what households actually got: profits paid out minus the losses
    they covered (D33). Transfers = welfare + the D37 money-loop payments (treasury surplus, and the
    bond-market sweep, which is how households receive interest; there is no separate savings
    interest, D42).
    """
    return wages + private_profits - taxes_paid + transfers
