"""Cournot markups, antitrust, entry/exit, HHI, nationalization (CLAUDE.md §6.9).

Phase 1 holds only the base markup rule (needed to build the initial state). Antitrust,
entry/exit, breakups and HHI come in a later phase.
"""

from __future__ import annotations

import numpy as np


def base_markup(n_firms: np.ndarray, mu_dominant: float) -> np.ndarray:
    """Markup share mu from the number of firms: 1/n for n >= 2, mu_dominant for n = 1 (§6.9)."""
    n = np.asarray(n_firms)
    if np.any(n < 1):
        raise ValueError("n_firms must be >= 1")
    return np.where(n >= 2, 1.0 / np.maximum(n, 1), mu_dominant).astype(float)
