"""Per-country engine parameters as (6,) arrays: world.yaml defaults with countries.yaml overrides."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world.config import Config


def _param(cfg: Config, name: str, default: float) -> np.ndarray:
    vals = []
    for c in cfg.countries.countries:
        v = getattr(c.params, name)
        vals.append(default if v is None else v)
    return np.array(vals, dtype=float)


@dataclass(frozen=True)
class CountryParams:
    subsidy_efficiency: np.ndarray  # eta
    military_efficiency: np.ndarray
    antitrust_strength: np.ndarray
    breakup_omega: np.ndarray
    entry_mult: np.ndarray
    sanction_self_cost: np.ndarray
    leader_fall_prob: np.ndarray
    renounce_stability_cost: np.ndarray


def country_params(cfg: Config) -> CountryParams:
    w = cfg.world
    return CountryParams(
        subsidy_efficiency=_param(cfg, "subsidy_efficiency", w.production.subsidy_efficiency_default),
        military_efficiency=_param(cfg, "military_efficiency", w.military.efficiency_default),
        antitrust_strength=_param(cfg, "antitrust_strength", w.firms.antitrust_strength_default),
        breakup_omega=_param(cfg, "breakup_omega", w.firms.breakup_omega_default),
        entry_mult=_param(cfg, "entry_mult", w.firms.entry_mult_default),
        sanction_self_cost=_param(cfg, "sanction_self_cost", w.trade.sanction_self_cost_default),
        leader_fall_prob=_param(cfg, "leader_fall_prob", w.stability.leader_fall_prob_default),
        renounce_stability_cost=_param(
            cfg, "renounce_stability_cost", w.stability.renounce_stability_cost_default
        ),
    )
