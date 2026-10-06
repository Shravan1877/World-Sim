"""Circular-flow check (owner's done-when, Phase 2 completion). Do not loosen these bands.

After burn-in (D38), 60 turns, no shocks, status quo, 20 seeds:
- every country's GDP stays between 50% and 200% of its start value
- all prices between 0.5 and 2.0
- unemployment below 25%, stability between 40 and 90
- bond_market and treasury balances stay bounded (our bound: |bond_market| <= 2 x starting world
  quarterly GDP; each treasury <= 2 x its country's starting quarterly GDP)
- all invariants hold (step() raises otherwise), money conserved to 1e-9
"""

import numpy as np
import pytest

from tests.runs import run
from world.engine.invariants import check_money
from world.ledger import BOND_MARKET

TURNS = 60


@pytest.mark.parametrize("seed", range(1, 21))
def test_status_quo_economy_stays_sane(seed: int) -> None:
    _, states, _ = run(seed, TURNS)
    start = states[0]
    bad = []
    for s in states[1:]:
        t = f"t{s.turn}"
        g = s.gdp / start.gdp
        if np.any(g < 0.5) or np.any(g > 2.0):
            bad.append(f"{t}: GDP/start {np.round(g, 2).tolist()}")
        if s.price.min() < 0.5 or s.price.max() > 2.0:
            bad.append(f"{t}: prices in [{s.price.min():.3f}, {s.price.max():.3f}]")
        if np.any(s.unemployment >= 0.25):
            bad.append(f"{t}: unemployment {np.round(s.unemployment, 3).tolist()}")
        if np.any(s.stability < 40) or np.any(s.stability > 90):
            bad.append(f"{t}: stability {np.round(s.stability, 1).tolist()}")
        if abs(s.ledger.balance(BOND_MARKET)) > 2 * start.gdp.sum():
            bad.append(f"{t}: bond_market {s.ledger.balance(BOND_MARKET):.2f}")
        if np.any(s.treasury > 2 * start.gdp):
            bad.append(f"{t}: treasury/start GDP {np.round(s.treasury / start.gdp, 2).tolist()}")
        assert check_money(s.ledger, 1e-9) == []
    assert not bad, "\n".join(bad[:10]) + (f"\n... {len(bad)} problems in total" if len(bad) > 10 else "")
