"""Command line: run one game, resume a paused or crashed run, or fork a finished one (§11.6, §15).

    uv run python -m world.runner run --seed 3 --policies all=status_quo [--scenario energy_crunch]
                                      [--run-id R] [--turns 12] [--stop-after 5] [--no-shocks]
    uv run python -m world.runner resume R
    uv run python -m world.runner fork R --turn 6 --policies DORNE=aggressor [--run-id R2]

Checkpoints go to data/checkpoints.db (thread_id = run_id), results to data/experiments.db.
`--policies`: 'all=NAME', or 'COUNTRY=NAME,...' (others status_quo). Phase 5 knows the bots only:
status_quo, tit_for_tat, greedy, cooperative, random, aggressor.
`--stop-after N` pauses the run after turn N is recorded (status 'paused'); `resume` continues it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from world.config import COUNTRIES, Config, load_config
from world.engine.burn_in import settled_state
from world.graph import (
    CHECKPOINTS_DB,
    EXPERIMENTS_DB,
    build_graph,
    drive,
    fork_run,
    initial_state,
    make_policy,
    open_checkpointer,
    parse_policies,
    thread_config,
)
from world.storage import Storage


def _app(policies: dict[str, str], cfg: Config, checkpoints: Path, storage: Storage, scenario: str | None):
    seats = {c: make_policy(policies[c], cfg) for c in COUNTRIES}
    return build_graph(
        seats, cfg=cfg, checkpointer=open_checkpointer(checkpoints), storage=storage, scenario_name=scenario
    )


def _finish(storage: Storage, run_id: str, values: dict, stop_after: int | None) -> int:
    status = values["status"]
    if stop_after is not None and values["turn"] < values["horizon"] and status not in ("invariant_failed",):
        status = "paused"
        storage.set_status(run_id, status, degraded_seats=values["degraded_seats"])
    print(f"run {run_id}: status {status}, turn {values['turn']}")
    for row in values["logs"][-1:]:
        print(f"  last state_hash {row['state_hash'][:16]}")
    return 1 if status == "invariant_failed" else 0


def cmd_run(a: argparse.Namespace, cfg: Config) -> int:
    policies = parse_policies(a.policies)
    run_id = a.run_id or f"s{a.seed}-{abs(hash(tuple(policies.values()))) % 10**6:06d}"
    storage = Storage(a.db)
    start = settled_state(cfg, a.seed)
    init = initial_state(
        run_id, a.seed, policies, start, cfg=cfg, turns=a.turns, scenario=a.scenario,
        shocks_on=not a.no_shocks, experiment=a.experiment,
    )  # fmt: skip
    storage.write_run(
        run_id, seed=a.seed, policies=policies, horizon=init["horizon"], experiment=a.experiment,
        scenario=a.scenario, shocks_on=not a.no_shocks,
    )  # fmt: skip
    app = _app(policies, cfg, a.checkpoints, storage, a.scenario)
    values = drive(app, init, run_id, stop_after_turn=a.stop_after)
    return _finish(storage, run_id, values, a.stop_after)


def _meta(app_checkpoints: Path, run_id: str, cfg: Config) -> dict:
    """run_meta of a thread, read with a throwaway status-quo graph (meta is in the state)."""
    probe = build_graph({c: make_policy("status_quo", cfg) for c in COUNTRIES}, cfg=cfg,
                        checkpointer=open_checkpointer(app_checkpoints))  # fmt: skip
    snap = probe.get_state(thread_config(run_id))
    if not snap.values:
        raise SystemExit(f"no checkpoints for run {run_id} in {app_checkpoints}")
    return snap.values["run_meta"]


def cmd_resume(a: argparse.Namespace, cfg: Config) -> int:
    meta = _meta(a.checkpoints, a.run_id, cfg)
    storage = Storage(a.db)
    app = _app(meta["policies"], cfg, a.checkpoints, storage, meta["scenario"])
    if not app.get_state(thread_config(a.run_id)).next:
        print(f"run {a.run_id} is already finished")
        return 0
    storage.set_status(a.run_id, "running")
    values = drive(app, None, a.run_id, stop_after_turn=a.stop_after)
    return _finish(storage, a.run_id, values, a.stop_after)


def cmd_fork(a: argparse.Namespace, cfg: Config) -> int:
    meta = _meta(a.checkpoints, a.run_id, cfg)
    policies = {**meta["policies"], **_named(a.policies)}
    new_id = a.new_run_id or f"{a.run_id}-fork{a.turn}"
    storage = Storage(a.db)
    app = _app(policies, cfg, a.checkpoints, storage, meta["scenario"])
    horizon = app.get_state(thread_config(a.run_id)).values["horizon"]
    storage.write_run(
        new_id, seed=meta["seed"], policies=policies, horizon=horizon, experiment=meta["experiment"],
        scenario=meta["scenario"], shocks_on=meta["shocks_on"], parent_run_id=a.run_id, fork_turn=a.turn,
    )  # fmt: skip
    storage.copy_turns(a.run_id, new_id, a.turn)
    fork_run(app, a.run_id, a.turn, new_id, policies)
    values = drive(app, None, new_id, stop_after_turn=a.stop_after)
    return _finish(storage, new_id, values, a.stop_after)


def _named(spec: str | None) -> dict[str, str]:
    """Only the seats named in a fork's --policies change; 'all=X' changes every seat."""
    if not spec:
        return {}
    full = parse_policies(spec)
    if any(p.split("=")[0].strip().lower() == "all" or "=" not in p for p in spec.split(",")):
        return full
    named = {p.split("=")[0].strip().upper() for p in spec.split(",") if p.strip()}
    return {c: full[c] for c in named}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="world.runner", description=__doc__.split("\n\n")[0])
    p.add_argument("--db", type=Path, default=EXPERIMENTS_DB, help="experiment database")
    p.add_argument("--checkpoints", type=Path, default=CHECKPOINTS_DB, help="LangGraph checkpoint database")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one game")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--policies", default="all=status_quo")
    r.add_argument("--scenario", default=None)
    r.add_argument("--run-id", default=None)
    r.add_argument("--turns", type=int, default=None, help="fixed length (default: hidden horizon, §8)")
    r.add_argument("--stop-after", type=int, default=None, help="pause after this turn")
    r.add_argument("--no-shocks", action="store_true")
    r.add_argument("--experiment", default="")

    s = sub.add_parser("resume", help="continue a paused or crashed run from its last checkpoint")
    s.add_argument("run_id")
    s.add_argument("--stop-after", type=int, default=None)

    f = sub.add_parser("fork", help="branch a run at the start of a turn, with new policies for some seats")
    f.add_argument("run_id")
    f.add_argument("--turn", type=int, required=True)
    f.add_argument("--policies", default=None, help="seats that change, e.g. DORNE=aggressor")
    f.add_argument("--run-id", dest="new_run_id", default=None)
    f.add_argument("--stop-after", type=int, default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    a = parser().parse_args(argv)
    cfg = load_config()
    return {"run": cmd_run, "resume": cmd_resume, "fork": cmd_fork}[a.cmd](a, cfg)


if __name__ == "__main__":
    sys.exit(main())
