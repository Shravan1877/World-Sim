"""Employment, unemployment and wage updates (CLAUDE.md §6.6). Pure functions over (6,) arrays.

  employed_i = min(LF_i, sum_g L_d[i,g])
  u_i        = 1 - employed_i / LF_i
  w'_i       = w_i * (1 + clip(psi * (sum_g L_d[i,g] - LF_i) / LF_i, -cap, +cap))
L_d is labor demand BEFORE rationing (§6.2), so excess demand pushes wages up.
"""

from __future__ import annotations

import numpy as np


def employment(labor_demand: np.ndarray, labor_force: np.ndarray) -> np.ndarray:
    return np.minimum(labor_force, labor_demand.sum(axis=1))


def unemployment_rate(labor_demand: np.ndarray, labor_force: np.ndarray) -> np.ndarray:
    return 1.0 - employment(labor_demand, labor_force) / labor_force


def update_wage(
    wage: np.ndarray, labor_demand: np.ndarray, labor_force: np.ndarray, psi: float, cap: float
) -> np.ndarray:
    excess = (labor_demand.sum(axis=1) - labor_force) / labor_force
    return wage * (1.0 + np.clip(psi * excess, -cap, cap))
