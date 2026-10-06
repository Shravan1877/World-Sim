"""D30 energy refill: 14 turns, fixed (status-quo) policies, no shocks.

- No country's firms are starved of energy: firms get at least 90% of their planned energy inputs
  (5% ENERGY spoilage means a refilled stock covers about 95%).
- DORNE's exports never cut into its own firms' needs: in every turn it exports, its firm energy
  demand is filled in full at home, and exports never exceed its surplus over home + firm demand.
"""

import numpy as np

from tests.runs import CFG, run

DORNE, ENERGY = 0, 1
MIN_FILL = 0.9


def test_no_country_starved_of_energy() -> None:
    _, _, logs = run(1, 14)
    worst = []
    for log in logs:
        fill = log.extra["energy_used"].sum(axis=1) / log.extra["energy_demand"].sum(axis=1)
        if np.any(fill < MIN_FILL):
            worst.append(f"t{log.turn}: firm energy fill {np.round(fill, 3).tolist()}")
    assert not worst, "\n".join(worst)


def test_dorne_exports_never_cut_its_own_firms() -> None:
    """Whenever DORNE exports ENERGY, its own firms' planned energy is filled in full at home, and
    exports never exceed the surplus over home demand + firm demand. (If DORNE's own output falls
    short it exports nothing; that case is covered by the starvation test above.)"""
    _, states, logs = run(1, 14)
    for s, log in zip(states[1:], logs, strict=True):
        exports = log.exports[DORNE, ENERGY]
        assert exports <= log.extra["surplus"][DORNE, ENERGY] * (1 + 1e-12)
        if exports > 0:
            need = log.extra["energy_demand"][DORNE].sum() / (1 - CFG.world.spoilage["ENERGY"])  # D39
            assert np.isclose(log.extra["firm_energy_filled"][DORNE], need, rtol=1e-12), f"t{s.turn}"
