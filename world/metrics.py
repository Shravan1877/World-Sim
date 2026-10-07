"""Behavior and economy metrics (CLAUDE.md §12). Phase 4 has only the Power index (§12.2), which the
no-dominance check (§6.15) needs; the rest comes in Phase 7.
"""

from __future__ import annotations

import numpy as np

from world.config import Config
from world.engine.state import WorldState


def net_reserves_share(treasury: np.ndarray, debt: np.ndarray) -> np.ndarray:
    """NetReserves = treasury - debt, shifted so the lowest country is at 0 (our reading of "shifted
    to be positive", §12.2), then divided by the sum. Equal shares if every country is the same."""
    nr = treasury - debt
    shifted = nr - nr.min()
    total = shifted.sum()
    return shifted / total if total > 0 else np.full_like(nr, 1.0 / len(nr))


def power(state: WorldState, cfg: Config) -> np.ndarray:
    """§12.2: Power_i = w_g GDP_i/sum GDP + w_m Mil_i/sum Mil + w_r NetReserves_i+/sum NetReserves+."""
    w = cfg.world.metrics.power_weights
    gdp = np.maximum(state.gdp, 0.0)
    return (
        w.gdp * gdp / gdp.sum()
        + w.military * state.military / state.military.sum()
        + w.reserves * net_reserves_share(state.treasury, state.debt)
    )


def power_gain(start: WorldState, end: WorldState, cfg: Config) -> np.ndarray:
    """PG_i = Power_i(T) - Power_i(0)."""
    return power(end, cfg) - power(start, cfg)
