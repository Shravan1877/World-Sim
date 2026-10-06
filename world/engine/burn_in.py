"""Burn-in (CLAUDE.md §6.15, D38): settle the rough starting state before turn 1.

Run `burn_in.turns` (8) turns with the status-quo policy (nobody acts), no shocks, firm dynamics off,
and stability held at its starting value with no effects (step(..., burn_in=True)). Then:
  - reset the turn counter to 0 and stability to its configured starting value (§4.3);
  - discard the burn-in history: a fresh ledger that opens at the settled balances, GDP window and
    starting GDP (D32) = the settled GDP, no leader changes, no active shocks.
The burn-in draws no random numbers, so the settled state is the same for every seed (only the
`seed` field differs). tests/fixtures/settled_state.pkl holds it for the tests; regenerate with
    uv run python -m world.engine.burn_in
"""

from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np

from world.config import Config, load_config
from world.engine.state import WorldState, initial_state
from world.engine.step import step, turn_start
from world.ledger import Ledger, all_accounts
from world.rng import RngBundle

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "settled_state.pkl"


def run_burn_in(cfg: Config, seed: int = 0) -> tuple[WorldState, list[WorldState]]:
    """Returns (settled state, the states after each burn-in turn, for calibration reports)."""
    s = initial_state(cfg, seed)
    rng = RngBundle(seed)
    history = [s]
    for _ in range(cfg.world.burn_in.turns):
        s, _ = turn_start(s, rng, cfg, shocks_on=cfg.world.burn_in.shocks)
        s, _ = step(s, None, rng, cfg, burn_in=True)
        history.append(s)
    return settle(s, cfg), history


def settle(s: WorldState, cfg: Config) -> WorldState:
    """Turn the last burn-in state into the turn-0 starting state."""
    start = initial_state(cfg, s.seed)
    n_hist = s.gdp_hist.shape[1]
    return replace(
        s.copy(),
        turn=0,
        stability=start.stability.copy(),
        ledger=Ledger({acc: s.ledger.balance(acc) for acc in all_accounts()}),
        gdp_prev=s.gdp.copy(),
        gdp_hist=np.tile(s.gdp[:, None], (1, n_hist)),
        gdp_start=s.gdp.copy(),
        cpi_prev=s.cpi.copy(),
        inflation_q=np.zeros_like(s.inflation_q),
        leader_changes=np.zeros_like(s.leader_changes),
        leader_changed_turn=np.full_like(s.leader_changed_turn, -1),
        default_premium=np.zeros_like(s.default_premium),
        default_turns_left=np.zeros_like(s.default_turns_left),
        active_shocks=(),
    )


def settled_state(cfg: Config, seed: int) -> WorldState:
    """The starting state the engine uses (computed; seed only labels the state)."""
    return run_burn_in(cfg, seed)[0]


def load_fixture(seed: int) -> WorldState:
    """The saved settled state, relabelled with `seed`."""
    with FIXTURE.open("rb") as f:
        s: WorldState = pickle.load(f)
    return replace(s.copy(), seed=seed)


def write_fixture(cfg: Config | None = None) -> WorldState:
    s = settled_state(cfg or load_config(), 0)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE.open("wb") as f:
        pickle.dump(s, f, protocol=pickle.HIGHEST_PROTOCOL)
    return s


if __name__ == "__main__":
    print("settled state hash:", write_fixture().state_hash())
