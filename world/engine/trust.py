"""Trust matrix updates (CLAUDE.md §6.13). trust[i, j] = how much i trusts j, in [0, 1].

Events come in from outside (the policy and treaty layers, and contract deliveries in step):
  j violates a treaty with i   -> trust[i, j] += violation_victim (-0.20)
                                  trust[k, j] += violation_third_party (-0.10) for every other k
  j sanctions i                -> trust[i, j] += sanctioned (-0.10)
  a treaty between i and j is honored this turn -> trust[i, j] and trust[j, i] += 0.02
  FALKEN renounces a treaty    -> no trust change (it costs FALKEN stability instead, see stability)
Then every off-diagonal entry drifts toward 0.7:  trust += drift_rate * (0.7 - trust).
The diagonal (self-trust) is not used by the engine and is left unchanged.
Result is clipped to [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world.config import TrustCfg


@dataclass(frozen=True)
class TrustEvents:
    violations: tuple[tuple[int, int], ...] = ()  # (violator j, victim i)
    sanctions: tuple[tuple[int, int], ...] = ()  # (sanctioner j, target i): newly imposed this turn
    honored: tuple[tuple[int, int], ...] = ()  # (a, b): a treaty between a and b was honored
    renounced: tuple[tuple[int, int], ...] = ()  # (renouncer, counterpart): logged, no trust effect


def apply_events(trust: np.ndarray, ev: TrustEvents, p: TrustCfg) -> np.ndarray:
    t = trust.astype(float).copy()
    n = t.shape[0]
    for j, i in ev.violations:
        t[i, j] += p.violation_victim
        for k in range(n):
            if k not in (i, j):
                t[k, j] += p.violation_third_party
    for j, i in ev.sanctions:
        t[i, j] += p.sanctioned
    for a, b in ev.honored:
        t[a, b] += p.treaty_honored
        t[b, a] += p.treaty_honored
    return t


def drift(trust: np.ndarray, rate: float, target: float) -> np.ndarray:
    off = ~np.eye(trust.shape[0], dtype=bool)
    return np.where(off, trust + rate * (target - trust), trust)


def update_trust(trust: np.ndarray, ev: TrustEvents, p: TrustCfg) -> np.ndarray:
    """Events first, then drift, then clip to [0, 1]."""
    return np.clip(drift(apply_events(trust, ev, p), p.drift_rate, p.drift_target), 0.0, 1.0)
