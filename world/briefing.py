"""The per-turn briefing a leader sees (CLAUDE.md §11.2), as data.

build_briefing() reads ONLY engine state and the game history, never computes anything new about the
economy, and rounds every number to `briefing_sig_figs` (3) significant figures. The same numbers are
the "true values" for the Layer-2 fact check (Briefing.facts()). Text rendering for the LLM prompt
(briefing.md.j2) comes in Phase 5; bots read the data directly.

Sections (§11.2): 1 quarter; 2 your numbers; 3 other countries (public numbers, trust both ways);
4 this quarter's shocks and last quarter's events; 5 this quarter so far (earlier movers' accepted
actions and public statements); 6 still to move; 7 who hurt you (last 2 turns); 8 treaties;
9 your last 2 turns and the actions rejected last turn with reasons.
Never shown: other countries' private plans, predictions, forecasts or stances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from world.actions import Action
from world.config import COUNTRIES, SECTORS, Config
from world.engine import treaties as treaties_mod
from world.engine.state import WorldState
from world.history import SeatRecord, TurnRecord

PUBLIC_METRICS = (
    "gdp",
    "gdp_growth_pct",
    "inflation_pct",
    "unemployment_pct",
    "stability",
    "debt_to_gdp",
    "military",
    "treasury",
    "interest_rate_pct",
    "wage",
)


def sig(x: float, n: int = 3) -> float:
    """Round to n significant figures (0 stays 0)."""
    x = float(x)
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, n - 1 - int(math.floor(math.log10(abs(x)))))


def country_numbers(s: WorldState, i: int, cfg: Config) -> dict[str, float]:
    """Public numbers of country i (unrounded), every metric of the facts_used enum."""
    f = cfg.world.fiscal
    gdp_ref = max(float(s.gdp_hist[i].mean()), f.gdp_floor_share_of_start * float(s.gdp_start[i]), 1e-12)
    prev = float(s.gdp_prev[i])
    return {
        "gdp": float(s.gdp[i]),
        "gdp_growth_pct": 100.0 * (float(s.gdp[i]) / prev - 1.0) if prev > 0 else 0.0,
        "inflation_pct": 100.0 * ((1.0 + float(s.inflation_q[i])) ** cfg.world.periods_per_year - 1.0),
        "unemployment_pct": 100.0 * float(s.unemployment[i]),
        "stability": float(s.stability[i]),
        "debt_to_gdp": float(s.debt[i]) / (cfg.world.periods_per_year * gdp_ref),
        "military": float(s.military[i]),
        "treasury": float(s.treasury[i]),
        "interest_rate_pct": 100.0 * float(s.policy_rate[i]),
        "wage": float(s.wage[i]),
    }


@dataclass(frozen=True)
class TreatyView:
    id: str
    kind: str
    proposer: str
    addressee: str
    terms: dict[str, float | int | str]
    duration: int
    status: str
    proposed_turn: int
    start_turn: int


@dataclass(frozen=True)
class SeatView:
    """What others may see of a seat: accepted actions and the public statement."""

    country: str
    actions: tuple[Action, ...]
    public_statement: str


@dataclass(frozen=True)
class Briefing:
    turn: int
    country: str
    index: int
    seed: int
    order: tuple[str, ...]
    still_to_move: tuple[str, ...]
    numbers: dict[str, dict[str, float]]  # every country -> PUBLIC_METRICS (rounded)
    own: dict[str, object]  # own-only details: policies, stocks, shortages, output, firms
    trust_toward_me: dict[str, float]
    my_trust: dict[str, float]
    shocks: tuple[str, ...]
    last_events: tuple[str, ...]
    so_far: tuple[SeatView, ...]
    who_hurt_you: tuple[tuple[int, str, str], ...]  # (turn, country, kind)
    treaties_active: tuple[TreatyView, ...]
    proposals_to_me: tuple[TreatyView, ...]
    my_proposals: tuple[TreatyView, ...]
    my_last_turns: tuple[SeatRecord, ...]
    rejected_last_turn: tuple[tuple[str, str], ...]  # (action type, reason)
    leader_removed: bool
    extra: dict[str, object] = field(default_factory=dict)

    def facts(self) -> dict[tuple[str, str], float]:
        """(country, metric) -> the value shown in the briefing (Layer-2 truth)."""
        return {(c, m): v for c, d in self.numbers.items() for m, v in d.items()}


def _treaty_view(t, n: int) -> TreatyView:
    terms = {}
    for k, v in t.terms:
        if k in ("seller", "buyer", "lender", "borrower"):
            terms[k] = COUNTRIES[int(v)]
        elif k == "good":
            terms[k] = SECTORS[int(v)]
        else:
            terms[k] = sig(v, n)
    return TreatyView(
        t.id, t.kind, COUNTRIES[t.proposer], COUNTRIES[t.addressee], terms, t.duration, t.status,
        t.proposed_turn, t.start_turn,
    )  # fmt: skip


def build_briefing(
    state: WorldState,
    country: str,
    order: tuple[str, ...],
    shocks: tuple[str, ...],
    so_far: tuple[SeatRecord, ...],
    history: tuple[TurnRecord, ...],
    cfg: Config,
) -> Briefing:
    n = cfg.world.agents.briefing_sig_figs
    i = COUNTRIES.index(country)
    s = state
    numbers = {
        c: {m: sig(v, n) for m, v in country_numbers(s, j, cfg).items()} for j, c in enumerate(COUNTRIES)
    }
    seat = order.index(country)
    own = {
        "tax_rate": sig(s.tax_rate[i], n),
        "welfare_share": sig(s.welfare_share[i], n),
        "military_share": sig(s.military_share[i], n),
        "subsidy_share": sig(s.subsidy_share[i], n),
        "subsidy_target": SECTORS[s.subsidy_target[i]] if s.subsidy_target[i] >= 0 else None,
        "industry_subsidy": {SECTORS[g]: sig(s.industry_subsidy[i, g], n) for g in range(len(SECTORS))},
        "stocks": {SECTORS[g]: sig(s.stock[i, g], n) for g in range(len(SECTORS))},
        "shortages": {SECTORS[g]: sig(s.shortage[i, g], n) for g in range(len(SECTORS))},
        "output": {SECTORS[g]: sig(s.output[i, g], n) for g in range(len(SECTORS))},
        "n_firms": {SECTORS[g]: int(s.n_firms[i, g]) for g in range(len(SECTORS))},
        "nationalized": [SECTORS[g] for g in range(len(SECTORS)) if s.nationalized[i, g]],
        "my_tariffs": {COUNTRIES[j]: sig(s.tariff[i, j].max(), n) for j in range(len(COUNTRIES)) if j != i},
        "tariffs_on_me": {
            COUNTRIES[j]: sig(s.tariff[j, i].max(), n) for j in range(len(COUNTRIES)) if j != i
        },
        "my_sanctions": [COUNTRIES[j] for j in range(len(COUNTRIES)) if s.sanction[i, j]],
        "sanctions_on_me": [COUNTRIES[j] for j in range(len(COUNTRIES)) if s.sanction[j, i]],
        "energy_levy": sig(s.levy[i, 1], n),
        "policy_rate_override": None
        if np.isnan(s.policy_rate_override[i])
        else sig(s.policy_rate_override[i], n),
        "in_default": bool(s.default_turns_left[i] > 0),
        "gov_revenue": sig(s.gov_revenue[i], n),
    }
    hurt: list[tuple[int, str, str]] = []
    for rec in history[-2:]:
        for r in rec.seats:
            hurt += [(rec.turn, r.country, kind) for kind, victim in r.hostile if victim == country]
        hurt += [(rec.turn, v, "treaty_violation") for v, victim, _ in rec.violations if victim == country]
    for r in so_far:
        hurt += [(state.turn, r.country, kind) for kind, victim in r.hostile if victim == country]
    expiry = cfg.world.treaties.proposal_expiry_turns
    mine = [t for t in s.treaties if i in t.parties()]
    last = history[-1].seat(country) if history else None
    return Briefing(
        turn=s.turn,
        country=country,
        index=i,
        seed=s.seed,
        order=order,
        still_to_move=order[seat + 1 :],
        numbers=numbers,
        own=own,
        trust_toward_me={c: sig(s.trust[j, i], n) for j, c in enumerate(COUNTRIES) if j != i},
        my_trust={c: sig(s.trust[i, j], n) for j, c in enumerate(COUNTRIES) if j != i},
        shocks=shocks,
        last_events=history[-1].events if history else (),
        so_far=tuple(SeatView(r.country, r.accepted, r.public_statement) for r in so_far),
        who_hurt_you=tuple(hurt),
        treaties_active=tuple(_treaty_view(t, n) for t in mine if t.status == "active"),
        proposals_to_me=tuple(
            _treaty_view(t, n)
            for t in mine
            if t.addressee == i and treaties_mod.proposal_open(t, s.turn, expiry)
        ),
        my_proposals=tuple(
            _treaty_view(t, n)
            for t in mine
            if t.proposer == i and treaties_mod.proposal_open(t, s.turn, expiry)
        ),
        my_last_turns=tuple(r for rec in history[-2:] for r in rec.seats if r.country == country),
        rejected_last_turn=tuple((a.type, why) for a, why in last.rejected) if last else (),
        leader_removed=bool(s.leader_changed_turn[i] == s.turn - 1 and s.turn > 1),
    )
