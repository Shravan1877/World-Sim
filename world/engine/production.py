"""Factor demand, labor and energy rationing, and Cobb-Douglas production (CLAUDE.md §6.2).

All functions are pure: they take arrays and return new arrays. Shapes: country x sector
arrays are (6, 5); per-country arrays are (6,); per-sector exponents are (5,).

Energy used as an input is taken out of the ENERGY stock HERE, in step 1, and nowhere else
(D28a). produce() returns the stock after inputs; later steps must not subtract it again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ENERGY = 1


def effective_productivity(
    A: np.ndarray, shock_mult: np.ndarray, nationalized: np.ndarray, delta_nat: float
) -> np.ndarray:
    """A_eff = A x (active shock multipliers) x (1 - delta_nat if nationalized)."""
    return A * shock_mult * np.where(nationalized, 1.0 - delta_nat, 1.0)


def factor_demand(
    revenue_last: np.ndarray,
    subsidy: np.ndarray,
    eta: np.ndarray,
    mu: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    wage: np.ndarray,
    p_energy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Labor and energy demand from last turn's revenue plus effective subsidy.

    R~[i,g]  = R^[i,g] + eta_i * subsidy[i,g]
    L_d[i,g] = (1 - mu[i,g]) * beta_g  * R~[i,g] / w_i
    E_d[i,g] = (1 - mu[i,g]) * gamma_g * R~[i,g] / P[i,ENERGY]
    """
    r_tilde = revenue_last + eta[:, None] * subsidy
    budget = (1.0 - mu) * r_tilde
    L_d = budget * beta[None, :] / wage[:, None]
    E_d = budget * gamma[None, :] / p_energy[:, None]
    return L_d, E_d


def ration_to_limit(demand: np.ndarray, limit: np.ndarray) -> np.ndarray:
    """Scale each country's row down proportionally if its total exceeds the limit.

    If sum_g demand[i,g] > limit_i, every demand[i,.] is multiplied by limit_i / sum.
    """
    total = demand.sum(axis=1)
    scale = np.where(total > limit, limit / np.where(total > 0, total, 1.0), 1.0)
    return demand * scale[:, None]


def cobb_douglas(
    A_eff: np.ndarray, L: np.ndarray, E: np.ndarray, beta: np.ndarray, gamma: np.ndarray
) -> np.ndarray:
    """Q[i,g] = A_eff[i,g] * L[i,g]^beta_g * E[i,g]^gamma_g."""
    return A_eff * np.power(L, beta[None, :]) * np.power(E, gamma[None, :])


@dataclass(frozen=True)
class ProductionResult:
    labor_demand: np.ndarray  # L_d before rationing
    energy_demand: np.ndarray  # E_d before rationing
    labor: np.ndarray  # L used
    energy: np.ndarray  # E used
    output: np.ndarray  # Q
    stock_after_inputs: np.ndarray  # stock with energy inputs removed (ENERGY column)


def produce(
    *,
    A: np.ndarray,
    shock_mult: np.ndarray,
    nationalized: np.ndarray,
    delta_nat: float,
    revenue_last: np.ndarray,
    subsidy: np.ndarray,
    eta: np.ndarray,
    mu: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    wage: np.ndarray,
    price: np.ndarray,
    labor_force: np.ndarray,
    stock: np.ndarray,
) -> ProductionResult:
    """Step 1 of §6.1: factor demand, rationing, Cobb-Douglas, energy removed from stock."""
    L_d, E_d = factor_demand(revenue_last, subsidy, eta, mu, beta, gamma, wage, price[:, ENERGY])
    L = ration_to_limit(L_d, labor_force)
    E = ration_to_limit(E_d, stock[:, ENERGY])
    Q = cobb_douglas(effective_productivity(A, shock_mult, nationalized, delta_nat), L, E, beta, gamma)
    stock_after = stock.copy()
    stock_after[:, ENERGY] = np.maximum(stock[:, ENERGY] - E.sum(axis=1), 0.0)
    return ProductionResult(L_d, E_d, L, E, Q, stock_after)
