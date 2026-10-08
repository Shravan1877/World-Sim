"""Behavior and economy metrics (CLAUDE.md §12). So far only real GDP and the Power index (§12.2), which
the no-dominance check (§6.15) and the run records need; the rest comes in Phase 7.

D61: every GDP in the metrics is REAL GDP, quantities valued at the base-period prices P0 = the prices
of the game's starting (settled) state, so money-financed spending cannot look like a gain.
"""

from __future__ import annotations

import numpy as np

from world.config import Config
from world.engine.fiscal import gdp_value_added
from world.engine.state import WorldState


def net_reserves_share(treasury: np.ndarray, debt: np.ndarray) -> np.ndarray:
    """NetReserves = treasury - debt, shifted so the lowest country is at 0 (our reading of "shifted
    to be positive", §12.2), then divided by the sum. Equal shares if every country is the same."""
    nr = treasury - debt
    shifted = nr - nr.min()
    total = shifted.sum()
    return shifted / total if total > 0 else np.full_like(nr, 1.0 / len(nr))


def real_gdp(state: WorldState, base: WorldState) -> np.ndarray:
    """D61: last turn's value added at base-period prices, sum_g P0 Q - P0_ENERGY sum_g E."""
    return gdp_value_added(base.price, state.output, state.energy_in)


def power(state: WorldState, cfg: Config, base: WorldState) -> np.ndarray:
    """§12.2: Power_i = w_g rGDP_i/sum rGDP + w_m Mil_i/sum Mil + w_r NetReserves_i+/sum NetReserves+.
    `base` = the game's starting state (its prices value real GDP, D61)."""
    w = cfg.world.metrics.power_weights
    gdp = np.maximum(real_gdp(state, base), 0.0)
    return (
        w.gdp * gdp / gdp.sum()
        + w.military * state.military / state.military.sum()
        + w.reserves * net_reserves_share(state.treasury, state.debt)
    )


def power_gain(start: WorldState, end: WorldState, cfg: Config) -> np.ndarray:
    """PG_i = Power_i(T) - Power_i(0), real GDP at the start state's prices (D61)."""
    return power(end, cfg, start) - power(start, cfg, start)
