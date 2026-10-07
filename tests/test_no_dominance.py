"""No country dominates the CooperativeBot baseline by construction (§4.1 design rule, §6.15).

CooperativeBot self-play, random shocks on, 14 turns, seeds 1-10. Power = §12.2 index (world/metrics.py).
"Dominates" (our operational definition, docs/calibration.md):
  1. any country's power share above 2x the equal share (1/3) at any turn, or
  2. the same country is the top power gainer in every seed, or
  3. any country's mean power gain over the seeds above 0.02 (12% of an equal share) in absolute value.
None of these may happen.
"""

from collections import Counter

import numpy as np
import pytest

from tests.runs import CFG, run_bots
from world.config import COUNTRIES
from world.metrics import power, power_gain

SEEDS = range(1, 11)
EQUAL = 1.0 / len(COUNTRIES)


@pytest.fixture(scope="module")
def games():
    out = []
    for seed in SEEDS:
        r = run_bots("cooperative", seed, 14, shocks_on=True)
        assert r.status == "ok", r.error
        out.append(r)
    return out


def test_power_shares_sum_to_one(games) -> None:
    for r in games:
        for s in r.states:
            assert power(s, CFG).sum() == pytest.approx(1.0)


def test_no_country_above_twice_equal_share(games) -> None:
    peak = max(float(power(s, CFG).max()) for r in games for s in r.states)
    print(f"largest power share at any turn: {peak:.3f} (limit {2 * EQUAL:.3f})")
    assert peak <= 2 * EQUAL


def test_no_country_always_gains_most(games) -> None:
    tops = Counter(COUNTRIES[int(np.argmax(power_gain(r.states[0], r.final, CFG)))] for r in games)
    print("top power gainer per seed:", dict(tops))
    assert max(tops.values()) < len(SEEDS)


def test_mean_power_gain_is_small(games) -> None:
    pg = np.mean([power_gain(r.states[0], r.final, CFG) for r in games], axis=0)
    print("mean power gain:", dict(zip(COUNTRIES, np.round(pg, 4), strict=True)))
    assert np.all(np.abs(pg) <= 0.02)
