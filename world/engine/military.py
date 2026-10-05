"""Military stock (CLAUDE.md §6.10). Pure function over (6,) arrays. There is no war in v1.

  Mil'_i = Mil_i (1 - depreciation) + eff_i * military_goods_i
military_goods_i = military_spend_i / P[i,GOODS] is the GOODS the government actually bought this
turn (less than planned if GOODS were rationed). eff = 1, FALKEN 1.5.
"""

from __future__ import annotations

import numpy as np


def update_military(
    military: np.ndarray, military_goods: np.ndarray, efficiency: np.ndarray, depreciation: float
) -> np.ndarray:
    return military * (1.0 - depreciation) + efficiency * military_goods
