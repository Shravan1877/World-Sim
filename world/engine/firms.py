"""Cournot markups, antitrust, breakups, startup entry, exit, HHI, nationalization (CLAUDE.md §6.9).

Each (country, sector) has a tuple of Firm(id, share, owner). n = number of firms.

Markup share mu (the part of revenue that is not spent on labor and energy):
    mu = 1/n                       if n >= 2      (Cournot with n equal firms, unit-elastic demand)
    mu = mu_dom (0.5)              if n = 1       (dominant-firm rule; Cournot gives 0 at n = 1)
    mu = mu * (1 - e_i)^k          after k antitrust actions THIS turn (the cut lasts one turn)
    mu = 0                         if the sector is nationalized
Breakup (Schumpeter): if the largest share > s_max for o turns in a row, P(split in two) = 1 - omega^o.
Startup entry: h = h0 (1 + k_mu mu)(1 + o / 8) entry_mult_i (x boost from industry_leader_death).
Exit: if n >= 3, one firm leaves with probability exit_prob per turn.
All draws use the FIRMS stream, one generator per (turn, country, sector), and the same 4 dice are
always drawn whether or not they are used, so every run with the same seed sees the same dice.

Our choices where CLAUDE.md is silent (flagged in the Phase 3 summary):
- an entrant gets share 1/(n+1); the others shrink in proportion;
- a breakup splits the largest firm into two equal halves;
- the exiting firm is picked uniformly (4th die); its share is spread over the rest in proportion;
- a nationalized sector is a state monopoly: no breakup, entry or exit.

This module decides who owns what (private firms, or a nationalized state sector) and how much
market power they have. The money (wages, energy inputs, profits and losses to the owners) moves in
the income step, world/engine/income.py.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from world.config import FirmsCfg
from world.engine.state import Event, Firm
from world.rng import RngBundle, Stream

N_DICE = 4  # breakup, entry, exit, exit pick


def base_markup(n_firms: np.ndarray, mu_dominant: float) -> np.ndarray:
    """Markup share mu from the number of firms: 1/n for n >= 2, mu_dominant for n = 1 (§6.9)."""
    n = np.asarray(n_firms)
    if np.any(n < 1):
        raise ValueError("n_firms must be >= 1")
    return np.where(n >= 2, 1.0 / np.maximum(n, 1), mu_dominant).astype(float)


def cournot(a: float, d: float, n: int) -> tuple[float, float, float]:
    """Cournot with n equal firms, unit cost 1/a and unit-elastic demand (spending d).

    Returns (output per firm q = a d (n-1)/n^2, price = d / (n q), markup share = 1/n).
    """
    if n < 2:
        raise ValueError("Cournot needs n >= 2 (n = 1 uses the dominant-firm rule)")
    q = a * d * (n - 1) / n**2
    return q, d / (n * q), 1.0 / n


def effective_markup(
    n_firms: np.ndarray,
    nationalized: np.ndarray,
    antitrust_count: np.ndarray,
    antitrust_strength: np.ndarray,
    mu_dominant: float,
) -> np.ndarray:
    """mu used in this turn's production: base rule, antitrust cut (1 - e_i)^k, 0 if nationalized."""
    cut = np.power(1.0 - antitrust_strength[:, None], antitrust_count)
    return np.where(nationalized, 0.0, base_markup(n_firms, mu_dominant) * cut)


def hhi(shares: np.ndarray | list[float] | tuple[float, ...]) -> float:
    """Herfindahl index: sum of squared market shares (4 equal -> 0.25, monopoly -> 1)."""
    return float(np.sum(np.square(np.asarray(shares, dtype=float))))


def hhi_matrix(firm_lists: tuple[tuple[Firm, ...], ...], n_sectors: int) -> np.ndarray:
    vals = [hhi([f.share for f in fl]) for fl in firm_lists]
    return np.array(vals).reshape(-1, n_sectors)


def breakup_probability(omega: float | np.ndarray, o: int | np.ndarray) -> np.ndarray:
    """P(split) = 1 - omega^o, o = consecutive turns of dominance."""
    return 1.0 - np.power(omega, o)


def entry_hazard(
    mu: np.ndarray | float,
    dominance_age: np.ndarray | int,
    entry_mult: np.ndarray | float,
    boost: np.ndarray | float,
    p: FirmsCfg,
) -> np.ndarray:
    """h = h0 (1 + k_mu mu)(1 + age_dom / 8) entry_mult (x boost), capped at 1."""
    h = p.entry_h0 * (1.0 + p.entry_k_mu * mu) * (1.0 + dominance_age / p.entry_age_scale) * entry_mult
    return np.minimum(h * boost, 1.0)


# ------------------------------------------------------------------------ firm list operations


def _renormalize(firm_list: list[Firm]) -> tuple[Firm, ...]:
    total = sum(f.share for f in firm_list)
    return tuple(replace(f, share=f.share / total) for f in firm_list)


def largest_index(firm_list: tuple[Firm, ...]) -> int:
    """Index of the largest firm (first one on ties)."""
    return int(np.argmax([f.share for f in firm_list]))


def split_largest(firm_list: tuple[Firm, ...], new_id: str) -> tuple[Firm, ...]:
    k = largest_index(firm_list)
    big = firm_list[k]
    half = replace(big, share=big.share / 2)
    out = list(firm_list)
    out[k] = half
    out.append(Firm(new_id, big.share / 2, big.owner))
    return tuple(out)


def add_entrant(firm_list: tuple[Firm, ...], new_id: str) -> tuple[Firm, ...]:
    n = len(firm_list)
    out = [replace(f, share=f.share * n / (n + 1)) for f in firm_list]
    out.append(Firm(new_id, 1.0 / (n + 1)))
    return tuple(out)


def remove_firm(firm_list: tuple[Firm, ...], index: int) -> tuple[Firm, ...]:
    if len(firm_list) <= 1:
        raise ValueError("cannot remove the last firm of a sector")
    return _renormalize([f for k, f in enumerate(firm_list) if k != index])


def remove_smallest(firm_list: tuple[Firm, ...]) -> tuple[Firm, ...]:
    """Used by industry_collapse (n - 1, min 1). Ties: the last smallest firm leaves."""
    if len(firm_list) <= 1:
        return firm_list
    shares = [f.share for f in firm_list]
    k = len(shares) - 1 - int(np.argmin(shares[::-1]))
    return remove_firm(firm_list, k)


def nationalize_firms(firm_list: tuple[Firm, ...]) -> tuple[Firm, ...]:
    return tuple(replace(f, owner="state") for f in firm_list)


# ------------------------------------------------------------------------- the step-11 update


@dataclass(frozen=True)
class FirmsUpdate:
    firms: tuple[tuple[Firm, ...], ...]
    n_firms: np.ndarray
    dominance_age: np.ndarray
    markup: np.ndarray  # base markup for next turn (no antitrust; 0 if nationalized)
    hhi: np.ndarray
    events: tuple[Event, ...]
    dice: np.ndarray  # (6, 5, 4): every FIRMS die drawn this turn (for common-random-number checks)


def firm_dice(rng: RngBundle, turn: int, i: int, g: int) -> np.ndarray:
    return rng.gen(turn, Stream.FIRMS, i, g).random(N_DICE)


def update_firms(
    *,
    firm_lists: tuple[tuple[Firm, ...], ...],
    nationalized: np.ndarray,
    dominance_age: np.ndarray,
    mu: np.ndarray,
    entry_boost: np.ndarray,
    breakup_omega: np.ndarray,
    entry_mult: np.ndarray,
    p: FirmsCfg,
    rng: RngBundle,
    turn: int,
    country_names: tuple[str, ...],
    sector_names: tuple[str, ...],
) -> FirmsUpdate:
    """Dominance age -> breakup -> startup entry -> exit, per (country, sector), FIRMS stream."""
    n_c, n_s = nationalized.shape
    lists = list(firm_lists)
    age = dominance_age.copy()
    events: list[Event] = []
    dice = np.zeros((n_c, n_s, N_DICE))

    for i in range(n_c):
        for g in range(n_s):
            u = firm_dice(rng, turn, i, g)
            dice[i, g] = u
            k = i * n_s + g
            fl = lists[k]
            prefix = f"{country_names[i]}-{sector_names[g]}-t{turn}"
            dominant = max(f.share for f in fl) > p.dominance_share_threshold
            age[i, g] = age[i, g] + 1 if dominant else 0
            if nationalized[i, g]:
                continue  # state monopoly: no breakup, entry or exit

            if age[i, g] > 0 and u[0] < breakup_probability(breakup_omega[i], age[i, g]):
                n0 = len(fl)
                fl = split_largest(fl, f"{prefix}-split")
                events.append(Event("antitrust_breakup", i, g, f"n {n0}->{len(fl)}", float(u[0])))
                age[i, g] = 0

            h = entry_hazard(mu[i, g], age[i, g], entry_mult[i], entry_boost[i, g], p)
            if u[1] < h:
                n0 = len(fl)
                fl = add_entrant(fl, f"{prefix}-startup")
                kind = "startup_kills_monopoly" if n0 == 1 else "startup_disruption"
                events.append(Event(kind, i, g, f"n {n0}->{len(fl)}", float(u[1])))

            if len(fl) >= p.exit_min_firms and u[2] < p.exit_prob:
                n0 = len(fl)
                fl = remove_firm(fl, min(int(u[3] * n0), n0 - 1))
                events.append(Event("firm_exit", i, g, f"n {n0}->{len(fl)}", float(u[2])))
            lists[k] = fl

    firms_t = tuple(lists)
    n_firms = np.array([len(fl) for fl in firms_t], dtype=np.int64).reshape(n_c, n_s)
    markup = np.where(nationalized, 0.0, base_markup(n_firms, p.mu_dominant))
    return FirmsUpdate(firms_t, n_firms, age, markup, hhi_matrix(firms_t, n_s), tuple(events), dice)
