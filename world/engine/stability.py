"""Stability update, unrest and leader-fall checks (CLAUDE.md §6.11).

  dStab = - k_u  max(u - u_n, 0) * 100
          - k_pi |pi_annual - pi_t| * 100
          - k_f  food_shortage_frac * 100
          - k_e  energy_shortage_frac * 100
          + k_w  (welfare/GDP - w_ref) * 100
          + k_m  (70 - Stab) / 10
          - sanction self-costs - shock effects
  Stab' = clip(Stab + dStab, 0, 100)
Below 30: unrest with probability (30 - Stab)/60 -> output -5% next turn, Stab -5.
Below 15: the leader falls with probability 0.5 (CERES 0.8). For now this only sets a flag.
Draws use the SHOCK stream with keys (seed, turn, SHOCK, "unrest"|"leader_fall", country).
Order inside the turn: stability update -> unrest check -> leader-fall check (on the post-unrest value).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world.config import StabilityCfg
from world.rng import RngBundle, Stream


def stability_change(
    *,
    stability: np.ndarray,
    u: np.ndarray,
    pi_annual: np.ndarray,
    food_shortage_frac: np.ndarray,
    energy_shortage_frac: np.ndarray,
    welfare_to_gdp: np.ndarray,
    sanction_cost: np.ndarray,
    shock_effects: np.ndarray,
    u_n: float,
    pi_target: float,
    p: StabilityCfg,
) -> np.ndarray:
    return (
        -p.k_u * np.maximum(u - u_n, 0.0) * 100
        - p.k_pi * np.abs(pi_annual - pi_target) * 100
        - p.k_f * food_shortage_frac * 100
        - p.k_e * energy_shortage_frac * 100
        + p.k_w * (welfare_to_gdp - p.w_ref) * 100
        + p.k_m * (p.normal_level - stability) / p.mean_reversion_scale
        - sanction_cost
        - shock_effects
    )


def update_stability(stability: np.ndarray, delta: np.ndarray, p: StabilityCfg) -> np.ndarray:
    return np.clip(stability + delta, p.min_, p.max_)


@dataclass(frozen=True)
class UnrestResult:
    stability: np.ndarray
    unrest: np.ndarray  # bool: output -5% next turn
    leader_falls: np.ndarray  # bool: flag only (Phase 2)


def unrest_and_leader_fall(
    stability: np.ndarray,
    leader_fall_prob: np.ndarray,
    p: StabilityCfg,
    rng: RngBundle,
    turn: int,
) -> UnrestResult:
    n = stability.shape[0]
    stab = stability.copy()
    unrest = np.zeros(n, dtype=bool)
    falls = np.zeros(n, dtype=bool)
    for i in range(n):
        if stab[i] < p.unrest_threshold:
            prob = (p.unrest_threshold - stab[i]) / p.unrest_prob_divisor
            if rng.gen(turn, Stream.SHOCK, "unrest", i).random() < prob:
                unrest[i] = True
                stab[i] = max(stab[i] - p.unrest_stability_hit, p.min_)
        if (
            stab[i] < p.leader_fall_threshold
            and rng.gen(turn, Stream.SHOCK, "leader_fall", i).random() < leader_fall_prob[i]
        ):
            falls[i] = True
    return UnrestResult(stab, unrest, falls)


@dataclass(frozen=True)
class LeaderChange:
    stability: np.ndarray
    leader_changes: np.ndarray
    leader_changed_turn: np.ndarray


def leader_change(
    stability: np.ndarray,
    leader_changes: np.ndarray,
    leader_changed_turn: np.ndarray,
    country: int,
    turn: int,
    p: StabilityCfg,
) -> LeaderChange:
    """§6.11: a new leader takes over. Stability +15 (clipped), the change is counted, and
    leader_changed_turn[i] = turn is the flag the agent layer uses to wipe the leader's memory.
    The same model keeps playing (the model is the experimental unit)."""
    stab = stability.copy()
    count = leader_changes.copy()
    when = leader_changed_turn.copy()
    stab[country] = min(stab[country] + p.leader_change_stability_bonus, p.max_)
    count[country] += 1
    when[country] = turn
    return LeaderChange(stab, count, when)
