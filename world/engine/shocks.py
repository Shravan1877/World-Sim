"""Random shocks and scripted scenario incidents (CLAUDE.md §7).

Every shock is a function  shock(state, rng, spec, cfg) -> (state, events).  It rolls its own dice
from the SHOCK stream with the key (seed, turn, SHOCK, shock_id[, country][, sector]) and applies the
effect where a die lands below the hazard. **Every die is rolled whether or not it can matter**
(e.g. industry_leader_death rolls for every sector, even ones with many firms), so two runs with the
same seed see exactly the same dice whatever the policies did (common random numbers). Only the
state-dependent part (is this sector dominant? is stability low?) can differ between runs.

The returned events list one ShockEvent per die (fired or not), so the dice can be logged and
compared. Timed effects are ActiveShock multipliers (see state.py): they are aged at the end of
step(), so a shock with duration d applies to exactly d turns of production, starting this turn.

State-dependent shocks are rolled elsewhere, with the same keying rules:
  startup_disruption -> firms.update_firms (FIRMS stream, step 11)
  unrest, leader_fall -> stability.unrest_and_leader_fall (SHOCK stream, step 13)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from world.config import COUNTRIES, SECTORS, Config, ScenarioCfg, ShockCfg
from world.engine import firms as firms_mod
from world.engine import stability as stability_mod
from world.engine.state import ActiveShock, WorldState
from world.rng import RngBundle, Stream


@dataclass(frozen=True)
class ShockEvent:
    shock_id: str
    country: int  # -1 = world-wide
    sector: int  # -1 = not sector-specific
    u: float  # the die (NaN for scheduled incidents)
    hazard: float  # probability it had to land under (NaN if scheduled)
    fired: bool
    scheduled: bool = False
    detail: str = ""


ShockFn = Callable[[WorldState, RngBundle, ShockCfg, Config], tuple[WorldState, list[ShockEvent]]]


def roll(rng: RngBundle, turn: int, shock_id: str, *ids: int) -> float:
    """The one die for key (seed, turn, SHOCK, shock_id, *ids)."""
    return float(rng.gen(turn, Stream.SHOCK, shock_id, *ids).random())


def hazard_for(spec: ShockCfg, country: int) -> float:
    h = spec.hazard_by_country.get(COUNTRIES[country], spec.hazard_default)  # type: ignore[call-overload]
    return float(h) if h is not None else 0.0


# ------------------------------------------------------------------------------- effects


def apply_effect(state: WorldState, spec: ShockCfg, country: int, sector: int, cfg: Config) -> str:
    """Apply one firing of `spec` to `state` IN PLACE (callers pass a copy). Returns a short detail."""
    e = spec.effect
    if e.kind == "productivity":
        g = sector if e.sector == "ANY" else SECTORS.index(e.sector)  # type: ignore[arg-type]
        assert e.multiplier is not None
        details = [f"A x {e.multiplier}"]
        if spec.duration_turns is None:
            state.productivity = state.productivity.copy()
            state.productivity[country, g] *= e.multiplier
            details.append("permanent")
        else:
            state.active_shocks += (ActiveShock(spec.id, country, g, e.multiplier, spec.duration_turns),)
            details.append(f"{spec.duration_turns} turns")
        if e.n_firms_change is not None and e.n_firms_change < 0:
            details.append(_remove_firms(state, country, g, -e.n_firms_change, e.n_firms_min or 1, cfg))
        if e.entry_hazard_multiplier is not None:
            state.active_shocks += (
                ActiveShock(
                    f"{spec.id}:entry",
                    country,
                    g,
                    e.entry_hazard_multiplier,
                    e.entry_hazard_turns or 0,
                    kind="entry_hazard",
                ),
            )
            details.append(f"entry hazard x {e.entry_hazard_multiplier} for {e.entry_hazard_turns} turns")
        return ", ".join(details)
    if e.kind == "labor_force":
        assert e.multiplier is not None and spec.duration_turns is not None
        state.active_shocks += (
            ActiveShock(spec.id, country, -1, e.multiplier, spec.duration_turns, kind="labor_force"),
        )
        return f"LF x {e.multiplier} for {spec.duration_turns} turns"
    if e.kind == "leader_change":
        lc = stability_mod.leader_change(
            state.stability,
            state.leader_changes,
            state.leader_changed_turn,
            country,
            state.turn,
            cfg.world.stability,
        )
        state.stability, state.leader_changes, state.leader_changed_turn = (
            lc.stability,
            lc.leader_changes,
            lc.leader_changed_turn,
        )
        return "leader change"
    raise ValueError(f"shock {spec.id}: effect {e.kind} is state-dependent and cannot be applied here")


def _remove_firms(state: WorldState, i: int, g: int, k: int, n_min: int, cfg: Config) -> str:
    idx = i * len(SECTORS) + g
    fl = state.firms[idx]
    n0 = len(fl)
    for _ in range(k):
        if len(fl) > n_min:
            fl = firms_mod.remove_smallest(fl)
    lists = list(state.firms)
    lists[idx] = fl
    state.firms = tuple(lists)
    state.n_firms = state.n_firms.copy()
    state.n_firms[i, g] = len(fl)
    state.markup = np.where(
        state.nationalized, 0.0, firms_mod.base_markup(state.n_firms, cfg.world.firms.mu_dominant)
    )
    return f"n {n0}->{len(fl)}"


# ------------------------------------------------------------------- one function per shock


def _per_country(state: WorldState, rng: RngBundle, spec: ShockCfg, cfg: Config) -> tuple[WorldState, list]:
    s = state.copy()
    events = []
    for i in range(len(COUNTRIES)):
        u, h = roll(rng, s.turn, spec.id, i), hazard_for(spec, i)
        fired = u < h
        detail = apply_effect(s, spec, i, -1, cfg) if fired else ""
        events.append(ShockEvent(spec.id, i, -1, u, h, fired, detail=detail))
    return s, events


def _per_country_sector(
    state: WorldState, rng: RngBundle, spec: ShockCfg, cfg: Config, eligible: np.ndarray | None = None
) -> tuple[WorldState, list]:
    s = state.copy()
    events = []
    for i in range(len(COUNTRIES)):
        for g in range(len(SECTORS)):
            u, h = roll(rng, s.turn, spec.id, i, g), hazard_for(spec, i)
            fired = bool(u < h and (eligible is None or eligible[i, g]))
            detail = apply_effect(s, spec, i, g, cfg) if fired else ""
            events.append(ShockEvent(spec.id, i, g, u, h, fired, detail=detail))
    return s, events


def harvest_failure(state: WorldState, rng: RngBundle, spec: ShockCfg, cfg: Config):
    """A[FOOD] x 0.6 for 2 turns. Hazard CERES 0.04, others 0.02."""
    return _per_country(state, rng, spec, cfg)


def energy_disaster(state: WorldState, rng: RngBundle, spec: ShockCfg, cfg: Config):
    """A[ENERGY] x 0.5 for 2 turns. Hazard DORNE 0.03 (others 0, but their dice are still rolled)."""
    return _per_country(state, rng, spec, cfg)


def industry_collapse(state: WorldState, rng: RngBundle, spec: ShockCfg, cfg: Config):
    """Per (country, sector), 0.005: A x 0.5 for 4 turns and one firm fewer (min 1)."""
    return _per_country_sector(state, rng, spec, cfg)


def pandemic(state: WorldState, rng: RngBundle, spec: ShockCfg, cfg: Config):
    """One world die, 0.02: labor force x 0.9 everywhere for 3 turns."""
    s = state.copy()
    u, h = roll(rng, s.turn, spec.id), float(spec.hazard_default or 0.0)
    fired = u < h
    detail = apply_effect(s, spec, -1, -1, cfg) if fired else ""
    return s, [ShockEvent(spec.id, -1, -1, u, h, fired, detail=detail)]


def leader_death(state: WorldState, rng: RngBundle, spec: ShockCfg, cfg: Config):
    """Per country, 0.01: leader change (§6.11)."""
    return _per_country(state, rng, spec, cfg)


def industry_leader_death(state: WorldState, rng: RngBundle, spec: ShockCfg, cfg: Config):
    """Per dominant firm (sector with n <= 2), 0.01: A x 0.8 for 2 turns, entry hazard x 2 for 4 turns."""
    eligible = state.n_firms <= (spec.dominant_max_firms or 1)
    return _per_country_sector(state, rng, spec, cfg, eligible)


def resource_discovery(state: WorldState, rng: RngBundle, spec: ShockCfg, cfg: Config):
    """Per country, 0.01: A[ENERGY] x 1.3, permanent."""
    return _per_country(state, rng, spec, cfg)


SHOCK_FUNCTIONS: dict[str, ShockFn] = {
    "harvest_failure": harvest_failure,
    "energy_disaster": energy_disaster,
    "industry_collapse": industry_collapse,
    "pandemic": pandemic,
    "leader_death": leader_death,
    "industry_leader_death": industry_leader_death,
    "resource_discovery": resource_discovery,
}
STATE_DEPENDENT = ("startup_disruption", "unrest", "leader_fall")


# --------------------------------------------------------------------- multipliers and timing


def recompute_multipliers(state: WorldState) -> None:
    """Rebuild shock_mult (on A) and labor_force (base x pandemic) from the active shocks, IN PLACE."""
    mult = np.ones_like(state.productivity)
    lf_mult = np.ones_like(state.labor_force_base)
    for sh in state.active_shocks:
        rows = slice(None) if sh.country < 0 else sh.country
        if sh.kind == "productivity":
            cols = slice(None) if sh.sector < 0 else sh.sector
            mult[rows, cols] *= sh.multiplier
        elif sh.kind == "labor_force":
            lf_mult[rows] *= sh.multiplier
    state.shock_mult = mult
    state.labor_force = state.labor_force_base * lf_mult


def entry_boost(state: WorldState) -> np.ndarray:
    """(6, 5) multiplier on the startup entry hazard from active industry_leader_death shocks."""
    boost = np.ones_like(state.productivity)
    for sh in state.active_shocks:
        if sh.kind == "entry_hazard":
            boost[sh.country, sh.sector] *= sh.multiplier
    return boost


def age_shocks(active: tuple[ActiveShock, ...]) -> tuple[ActiveShock, ...]:
    """One turn has passed: turns_left - 1, drop the ones that reach 0. Permanent (-1) stay."""
    out = []
    for sh in active:
        if sh.turns_left < 0:
            out.append(sh)
        elif sh.turns_left > 1:
            out.append(
                ActiveShock(sh.shock_id, sh.country, sh.sector, sh.multiplier, sh.turns_left - 1, sh.kind)
            )
    return tuple(out)


# ------------------------------------------------------------------------ turn-start entry points


def draw_shocks(state: WorldState, rng: RngBundle, cfg: Config) -> tuple[WorldState, list[ShockEvent]]:
    """Roll and apply every random shock in shocks.yaml order (state-dependent ones are skipped)."""
    events: list[ShockEvent] = []
    for spec in cfg.shocks.shocks:
        if spec.scope == "state_dependent":
            continue
        state, ev = SHOCK_FUNCTIONS[spec.id](state, rng, spec, cfg)
        events.extend(ev)
    return state, events


def apply_scenario(
    state: WorldState, scenario: ScenarioCfg, cfg: Config
) -> tuple[WorldState, list[ShockEvent]]:
    """Fire every scripted incident scheduled for state.turn (no dice)."""
    by_id = {s.id: s for s in cfg.shocks.shocks}
    s = state.copy()
    events = []
    for inc in scenario.incidents:
        if inc.turn != s.turn:
            continue
        spec = by_id[inc.shock]
        i = COUNTRIES.index(inc.country) if inc.country else -1
        g = SECTORS.index(inc.sector) if inc.sector else -1
        detail = apply_effect(s, spec, i, g, cfg)
        events.append(
            ShockEvent(spec.id, i, g, float("nan"), float("nan"), True, True, f"scenario: {detail}")
        )
    return s, events
