"""Apply accepted actions to the state's POLICY fields, immediately (CLAUDE.md §9 last line).

A seat's accepted actions change tariffs, sanctions, quotas, spending plans, treaties, ... at once, so
later movers' briefings see them. Nothing economic happens here: production, trade and money only
move in step(). Things step() needs beyond the policy fields (antitrust actions this turn, newly
imposed sanctions for trust, renounced treaties for stability) are collected in PolicyEffects.

Special powers (§4.1, §9) are numbers in countries.yaml (subsidy efficiency, antitrust strength,
cheap sanctions, ...) or actions that only some countries may take; the validator checks who may take
which action, this module only applies them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from world.actions import (
    AcceptTreaty,
    Action,
    Antitrust,
    Nationalize,
    ProposeTreaty,
    RejectTreaty,
    RenounceTreaty,
    SetEnergyExportLevy,
    SetEnergyExportQuota,
    SetFoodExportBan,
    SetPolicyRate,
    SetSanction,
    SetSpending,
    SetSubsidyTarget,
    SetTariff,
    SetTax,
    SubsidizeIndustry,
    Wait,
)
from world.config import COUNTRIES, SECTORS, Config
from world.engine import treaties as treaties_mod
from world.engine.firms import nationalize_firms
from world.engine.state import ENERGY, FOOD, N_SECTORS, N_TRADED, WorldState
from world.engine.step import TurnInputs


@dataclass
class PolicyEffects:
    """What step() needs from this turn's accepted actions (accumulated over all seats)."""

    antitrust: np.ndarray = field(
        default_factory=lambda: np.zeros((len(COUNTRIES), N_SECTORS), dtype=np.int64)
    )
    sanctions_imposed: list[tuple[int, int]] = field(default_factory=list)
    renounced: list[tuple[int, int]] = field(default_factory=list)
    proposed: list[str] = field(default_factory=list)  # treaty ids created this turn

    def turn_inputs(self) -> TurnInputs:
        return TurnInputs(
            antitrust=self.antitrust.copy(),
            sanctions_imposed=tuple(self.sanctions_imposed),
            renounced=tuple(self.renounced),
        )


def targets(target: str, actor: int) -> list[int]:
    """'ALL' -> every other country; a name -> [its index]."""
    if target == "ALL":
        return [j for j in range(len(COUNTRIES)) if j != actor]
    return [COUNTRIES.index(target)]


def goods(good: str) -> list[int]:
    return list(range(N_TRADED)) if good == "ALL" else [SECTORS.index(good)]


def next_treaty_id(state: WorldState, proposer: int) -> str:
    prefix = f"T{state.turn}-{COUNTRIES[proposer]}-"
    n = sum(t.id.startswith(prefix) for t in state.treaties)
    return f"{prefix}{n + 1}"


def apply_actions(
    state: WorldState, country: int, actions: list[Action], cfg: Config, effects: PolicyEffects
) -> WorldState:
    """Return a copy of `state` with `country`'s accepted actions applied (in order)."""
    s = state.copy()
    for a in actions:
        _apply(s, country, a, cfg, effects)
    return s


def _apply(s: WorldState, i: int, a: Action, cfg: Config, fx: PolicyEffects) -> None:  # noqa: C901
    if isinstance(a, SetTax):
        s.tax_rate[i] = a.rate
    elif isinstance(a, SetSpending):
        s.welfare_share[i], s.military_share[i], s.subsidy_share[i] = a.welfare, a.military, a.subsidy
    elif isinstance(a, SetSubsidyTarget):
        s.subsidy_target[i] = SECTORS.index(a.sector)
    elif isinstance(a, SubsidizeIndustry):
        s.industry_subsidy[i, SECTORS.index(a.sector)] = a.amount
    elif isinstance(a, SetTariff):
        j = COUNTRIES.index(a.target)
        for g in goods(a.good):
            s.tariff[i, j, g] = a.rate
    elif isinstance(a, SetSanction):
        j = COUNTRIES.index(a.target)
        if a.on and not s.sanction[i, j]:
            fx.sanctions_imposed.append((i, j))
        s.sanction[i, j] = a.on
    elif isinstance(a, ProposeTreaty):
        tid = next_treaty_id(s, i)
        j = COUNTRIES.index(a.target)
        s.treaties = s.treaties + (treaties_mod.make_treaty(tid, a.kind, i, j, a.terms, a.duration, s.turn),)
        fx.proposed.append(tid)
    elif isinstance(a, AcceptTreaty | RejectTreaty | RenounceTreaty):
        _treaty_answer(s, i, a, fx)
    elif isinstance(a, Antitrust):
        fx.antitrust[i, SECTORS.index(a.sector)] += 1
    elif isinstance(a, SetEnergyExportQuota):
        for j in targets(a.target, i):
            s.export_cap[j, i, ENERGY] = a.rate
    elif isinstance(a, SetEnergyExportLevy):
        s.levy[i, ENERGY] = a.rate
    elif isinstance(a, SetFoodExportBan):
        for j in targets(a.target, i):
            s.export_cap[j, i, FOOD] = 0.0 if a.on else cfg.world.trade.export_cap_default
    elif isinstance(a, Nationalize):
        g = SECTORS.index(a.sector)
        s.nationalized[i, g] = True
        k = i * N_SECTORS + g
        s.firms = s.firms[:k] + (nationalize_firms(s.firms[k]),) + s.firms[k + 1 :]
    elif isinstance(a, SetPolicyRate):
        s.policy_rate_override[i] = np.nan if a.rate is None else a.rate
    elif isinstance(a, Wait):
        pass
    else:  # pragma: no cover - the Action union is closed
        raise TypeError(f"unknown action {a!r}")


def _treaty_answer(
    s: WorldState, i: int, a: AcceptTreaty | RejectTreaty | RenounceTreaty, fx: PolicyEffects
) -> None:
    new = []
    for t in s.treaties:
        if t.id != a.treaty_id:
            new.append(t)
        elif isinstance(a, AcceptTreaty):
            new.append(replace(t, status="active", start_turn=s.turn))
        elif isinstance(a, RejectTreaty):
            new.append(replace(t, status="rejected", end_turn=s.turn, note="rejected"))
        else:
            other = t.addressee if t.proposer == i else t.proposer
            fx.renounced.append((i, other))
            new.append(replace(t, status="ended", end_turn=s.turn, note="renounced"))
    s.treaties = tuple(new)


def hostile_acts(before: WorldState, after: WorldState, actor: int) -> list[tuple[str, int]]:
    """Hostile moves by `actor` between two states (§11.2 "who hurt you"): (kind, victim).

    raise_tariff: any tariff on the victim's goods went up; sanction: a new sanction;
    cut_exports: the energy quota or food ban on the victim tightened.
    """
    out: list[tuple[str, int]] = []
    for j in range(len(COUNTRIES)):
        if j == actor:
            continue
        if np.any(after.tariff[actor, j] > before.tariff[actor, j] + 1e-12):
            out.append(("raise_tariff", j))
        if after.sanction[actor, j] and not before.sanction[actor, j]:
            out.append(("sanction", j))
        if np.any(after.export_cap[j, actor] < before.export_cap[j, actor] - 1e-12):
            out.append(("cut_exports", j))
    return out
