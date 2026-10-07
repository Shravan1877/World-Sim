"""Taylor rule, AURELIA's policy-rate override, saving response, savings interest (CLAUDE.md §6.8).

  r_i = max(r_n + pi_t + a (pi_i^annual - pi_t) + b (u_n - u_i), floor)
  s_i = clip(s0 + k_r (r_i - r_n), s_min, s_max)
The rate acts on demand only through the saving rate. There is no separate savings-interest payment
(D42: it created money and drove an inflation loop); households receive interest through the D37
sweep of the bond market's surplus (world/engine/recycle.py).
"""

from __future__ import annotations

import numpy as np


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
