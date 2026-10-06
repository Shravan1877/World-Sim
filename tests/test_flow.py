"""Circular-flow check (D28, D30-D33): 60 turns, fixed policies, no shocks.

The economy must neither die nor blow up. Bands (our choice, relative to the starting state):
  world nominal GDP      within [0.5, 2.0] x its start, every turn
  each country's GDP     at least 0.25 x its start, every turn ("no country dies")
  ENERGY stock           each country in [0, 10 x start]; world at least 0.25 x start
"""

import numpy as np

from tests.runs import run

TURNS = 60


def test_nominal_gdp_and_energy_stay_in_a_sane_band() -> None:
    _, states, _ = run(1, TURNS)
    start = states[0]
    world0 = start.gdp.sum()
    e0 = start.stock[:, 1]
    bad = []
    for s in states[1:]:
        ratio = s.gdp.sum() / world0
        if not 0.5 <= ratio <= 2.0:
            bad.append(f"t{s.turn}: world GDP x{ratio:.2f}")
        dead = np.nonzero(s.gdp < 0.25 * start.gdp)[0]
        if dead.size:
            bad.append(f"t{s.turn}: GDP below 25% of start in countries {dead.tolist()}")
        e = s.stock[:, 1]
        if np.any(e < 0) or np.any(e > 10 * e0) or e.sum() < 0.25 * e0.sum():
            bad.append(f"t{s.turn}: ENERGY stock out of band {np.round(e / e0, 2).tolist()}")
    assert not bad, "\n".join(bad[:12]) + (f"\n... {len(bad)} problems in total" if len(bad) > 12 else "")
