"""Gradual price rule, CPI and inflation (CLAUDE.md §6.5, D28b).

All functions are pure. Shapes: (6, 5) country x sector, (6, 4) for traded goods only.

The price of good g in country i moves with post-trade excess demand:
    D_eff = D + X_req          own demand + what foreign buyers asked from i (before rationing)
    S_eff = S_dom + imports    own supply (stock after inputs + Q) + what i actually received
    P'    = P * (1 + clip(sigma_p * (D_eff - S_eff) / (S_eff + eps), -cap, +cap))
Every good is traded (D40). SERVICES is perishable and has no stock, so its S_dom = Q and
S_eff = Q + imports. Export requests and imports may cover fewer goods than demand (older tests).
"""

from __future__ import annotations

import numpy as np


def market_balance(
    demand: np.ndarray,
    supply_domestic: np.ndarray,
    output: np.ndarray,
    export_requests: np.ndarray,
    imports: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build D_eff = D + X_req and S_eff = S_dom + imports (6, 5).

    `output` is kept in the signature for callers; with no SERVICES stock, S_dom already equals Q there.
    """
    del output
    n_tr = export_requests.shape[1]
    d_eff = demand.astype(float).copy()
    s_eff = supply_domestic.astype(float).copy()
    d_eff[:, :n_tr] += export_requests
    s_eff[:, :n_tr] += imports
    return d_eff, s_eff


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
