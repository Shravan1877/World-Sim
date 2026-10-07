"""Household, government and firm-energy demand, the food floor, household cash (§6.3, D28c/d, D30).

All functions are pure. Shapes: (6, 5) country x sector, (6,) per country.

Households spend their non-saved income plus a slice c_w of their cash stock H:
    Y_spend = Y_disp * (1 - s) + c_w * H
    H'      = H + s * Y_disp - c_w * H
Without the c_w term, savings leak out every turn and nominal income decays to zero
(see tests/engine/test_demand.py::test_wealth_term_keeps_income_alive).
"""

from __future__ import annotations

import numpy as np

FOOD = 0
ENERGY = 1
GOODS = 2


def spendable_income(y_disp: np.ndarray, saving_rate: np.ndarray, cash: np.ndarray, c_w: float) -> np.ndarray:
    """Y_spend = Y_disp * (1 - s) + c_w * H."""
    return y_disp * (1.0 - saving_rate) + c_w * cash


def update_household_cash(
    cash: np.ndarray, y_disp: np.ndarray, saving_rate: np.ndarray, c_w: float
) -> np.ndarray:
    """H' = H + s * Y_disp - c_w * H (the saved part goes in, the spent slice goes out)."""
    return cash + saving_rate * y_disp - c_w * cash


def household_demand(
    shares: np.ndarray,
    y_spend: np.ndarray,
    price: np.ndarray,
    f_min: float | np.ndarray,
    population: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Household quantity demand with the minimum food floor.

    D[i,g] = share[i,g] * Y_spend_i / P[i,g]
    D[i,FOOD] = max(D[i,FOOD], f_min * pop_i)
    If the floor binds, the other goods' spending is scaled down so total spending <= Y_spend.
    Edge case not covered by CLAUDE.md: if the floor alone costs more than Y_spend, food demand
    is capped at Y_spend / P_food (all money goes to food) so spending never exceeds Y_spend.

    Returns (demand (6, 5), floor_binds (6,) bool).
    """
    spend = shares * y_spend[:, None]
    food_floor_qty = f_min * population
    floor_binds = spend[:, FOOD] / price[:, FOOD] < food_floor_qty

    food_spend = np.where(floor_binds, food_floor_qty * price[:, FOOD], spend[:, FOOD])
    food_spend = np.minimum(food_spend, y_spend)
    other_spend = spend.sum(axis=1) - spend[:, FOOD]
    remaining = np.maximum(y_spend - food_spend, 0.0)
    safe_other = np.where(other_spend > 0, other_spend, 1.0)
    scale = np.where(floor_binds & (other_spend > 0), remaining / safe_other, 1.0)

    new_spend = spend * scale[:, None]
    new_spend[:, FOOD] = food_spend
    return new_spend / price, floor_binds


def government_goods_demand(military_spend: np.ndarray, p_goods: np.ndarray) -> np.ndarray:
    """D_gov[i,GOODS] = military_i / P[i,GOODS] (military spending buys domestic GOODS)."""
    return military_spend / p_goods


def total_demand(
    household: np.ndarray, gov_goods: np.ndarray, firm_energy: np.ndarray | None = None
) -> np.ndarray:
    """Total demand used in trade, prices and consumption.

    D[i,GOODS]  = D_house + D_gov                       (D28d)
    D[i,ENERGY] = D_house + sum_g E_d[i,g]              (D30: firms refill their energy stock)
    firm_energy is this turn's planned firm input demand (before rationing), the expectation for
    next turn. That part is stock-building: it is delivered into stock, never consumed.
    """
    total = household.copy()
    total[:, GOODS] = total[:, GOODS] + gov_goods
    if firm_energy is not None:
        total[:, ENERGY] = total[:, ENERGY] + firm_energy
    return total
