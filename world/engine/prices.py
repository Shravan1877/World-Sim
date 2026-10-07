"""Gradual price rule, CPI and inflation (CLAUDE.md §6.5, D45). Pure functions.

Every seller's price moves with the requests it received against its supply (D45):
    R_s = all quantity requested from seller s (its own buyers + foreign buyers, first pass)
    S_s = S_dom = stock after inputs + Q        (SERVICES has no stock: S_s = Q)
    P'  = P * (1 + clip(sigma_p * (R_s - S_s) / (S_s + eps), -cap, +cap))
"""

from __future__ import annotations

import numpy as np


def excess_demand(requests: np.ndarray, supply: np.ndarray, eps: float) -> np.ndarray:
    """(R_s - S_s) / (S_s + eps): the relative excess demand a seller faces."""
    return (requests - supply) / (supply + eps)


def smooth_excess(excess: np.ndarray, previous: np.ndarray, weight: float) -> np.ndarray:
    """D50: exponential average of excess demand; weight 1.0 = no smoothing."""
    return weight * excess + (1.0 - weight) * previous


def update_prices_from_excess(
    price: np.ndarray, excess: np.ndarray, sigma_p: float, cap: float
) -> np.ndarray:
    """P' = P * (1 + clip(sigma_p * excess, -cap, +cap))."""
    return price * (1.0 + np.clip(sigma_p * excess, -cap, cap))


def price_change_rate(
    d_eff: np.ndarray, s_eff: np.ndarray, sigma_p: float, cap: float, eps: float
) -> np.ndarray:
    """clip(sigma_p * (D_eff - S_eff) / (S_eff + eps), -cap, +cap)."""
    return np.clip(sigma_p * (d_eff - s_eff) / (s_eff + eps), -cap, cap)


def update_prices(
    price: np.ndarray, d_eff: np.ndarray, s_eff: np.ndarray, sigma_p: float, cap: float, eps: float
) -> np.ndarray:
    """P' = P * (1 + capped change rate)."""
    return price * (1.0 + price_change_rate(d_eff, s_eff, sigma_p, cap, eps))


def cpi(shares: np.ndarray, price: np.ndarray) -> np.ndarray:
    """CPI_i = sum_g share[i,g] * P[i,g] with fixed base-period shares."""
    return (shares * price).sum(axis=1)


def inflation_quarterly(cpi_now: np.ndarray, cpi_prev: np.ndarray) -> np.ndarray:
    """pi_i = CPI_i(t) / CPI_i(t-1) - 1."""
    return cpi_now / cpi_prev - 1.0


def annualize(rate_quarterly: np.ndarray, periods_per_year: int) -> np.ndarray:
    """(1 + pi)^4 - 1: what the Taylor rule and briefings use."""
    return np.power(1.0 + rate_quarterly, periods_per_year) - 1.0
