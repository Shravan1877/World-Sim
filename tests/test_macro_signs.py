"""Macro signs (§6.15, Phase 4; EconAgent-style validation). Do not loosen.

Phillips: higher unemployment goes with lower wage growth.
Okun:     a rise in unemployment goes with lower real GDP growth.
Panel: every country-turn of 20 seeds x 14 turns, status-quo leaders, random shocks on (the shocks
give the variation). Signs are tested with Spearman rank correlations (p < 0.01).

Price convergence with no shocks:
  (a) from the rough starting state (all prices 1.0), status quo, no shocks, firm dynamics off (the
      burn-in mechanics), the mean absolute price change per turn falls at least 10x from the first
      5 turns to turns 36-40 and ends below 0.5% per turn;
  (b) in the real game from the settled state (firm entry/exit on), no shocks, 60 turns, 20 seeds:
      prices do not drift or explode: the mean absolute price change in turns 41-60 is below 1.5%
      per turn in every seed.
"""

import numpy as np
from scipy import stats

from tests.runs import CFG, run
from world.engine.state import initial_state
from world.engine.step import step, turn_start
from world.rng import RngBundle

SEEDS = range(1, 21)


def _panel() -> dict[str, np.ndarray]:
    cols: dict[str, list[np.ndarray]] = {"u": [], "wage_growth": [], "du": [], "real_growth": []}
    for seed in SEEDS:
        _, states, _ = run(seed, 14, shocks_on=True)
        for a, b in zip(states[:-1], states[1:], strict=True):
            cols["u"].append(b.unemployment)
            cols["wage_growth"].append(b.wage / a.wage - 1)
            cols["du"].append(b.unemployment - a.unemployment)
            cols["real_growth"].append((b.gdp / b.cpi) / (a.gdp / a.cpi) - 1)
    return {k: np.concatenate(v) for k, v in cols.items()}


PANEL = _panel()


def test_phillips_sign() -> None:
    r = stats.spearmanr(PANEL["u"], PANEL["wage_growth"])
    print(f"Phillips: Spearman {r.statistic:.3f} (p={r.pvalue:.1e}), n={len(PANEL['u'])}")
    assert r.statistic < -0.5 and r.pvalue < 0.01


def test_okun_sign() -> None:
    r = stats.spearmanr(PANEL["du"], PANEL["real_growth"])
    slope = np.polyfit(PANEL["du"], PANEL["real_growth"], 1)[0]
    print(f"Okun: Spearman {r.statistic:.3f} (p={r.pvalue:.1e}), slope {slope:.2f}")
    assert r.statistic < -0.2 and r.pvalue < 0.01 and slope < 0


def test_prices_converge_from_the_rough_start() -> None:
    s = initial_state(CFG, 0)
    rng = RngBundle(0)
    change = []
    for _ in range(40):
        a = s
        s, _ = turn_start(s, rng, CFG, shocks_on=False)
        s, _ = step(s, None, rng, CFG, burn_in=True)
        change.append(np.abs(s.price / a.price - 1).mean())
    first, last = np.mean(change[:5]), np.mean(change[-5:])
    print(f"mean |dP/P| per turn: first 5 turns {first:.4f}, turns 36-40 {last:.4f}")
    assert first / last >= 10 and last < 0.005


def test_prices_stay_settled_in_the_game_without_shocks() -> None:
    worst = 0.0
    for seed in SEEDS:
        _, states, _ = run(seed, 60)
        change = [
            np.abs(b.price / a.price - 1).mean() for a, b in zip(states[40:-1], states[41:], strict=True)
        ]
        worst = max(worst, float(np.mean(change)))
    print(f"worst seed: mean |dP/P| in turns 41-60 = {worst:.4f}")
    assert worst < 0.015
