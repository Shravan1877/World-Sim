"""Command line: run one game, resume a paused or crashed run, fork a run, show quota (§11.6, §11.7, §15).

    uv run python -m world.runner run --seed 3 [--policies all=status_quo] [--models all=m8b]
                                      [--scenario energy_crunch] [--run-id R] [--turns 12]
                                      [--stop-after 5] [--no-shocks] [--yes] [--allow-unverified]
    uv run python -m world.runner resume R
    uv run python -m world.runner fork R --turn 6 --policies DORNE=aggressor [--run-id R2]
    uv run python -m world.runner quota

Checkpoints: data/checkpoints.db (thread_id = run_id); results: data/experiments.db; quota ledger:
data/quota.db; LLM cache: data/cache.db.
`--policies`: bots, 'all=NAME' or 'COUNTRY=NAME,...' (others status_quo): status_quo, tit_for_tat,
greedy, cooperative, random, aggressor. `--models`: LLM leaders, 'all=KEY' or 'COUNTRY=KEY,...' with
model keys from config/models.yaml; these seats override --policies. Before a run with LLM seats the
runner prints the estimated calls and tokens per model and today's remaining quota, and asks for
confirmation (--yes skips the question). Only models with status 'verified' run unless
--allow-unverified (smoke tests only).
A run pauses (status 'paused', resume_after set) when a quota or rate limit blocks it; it stops with
status blocked_model / auth_error on a zero-allowance 429 or a 401/403. `resume` continues from the
last seat checkpoint. `--stop-after N` pauses after turn N.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
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
from world.llm.cache import CACHE_DB, LLMCache
from world.llm.limiter import QuotaPause
from world.llm.quota import QUOTA_DB, QuotaGuard
from world.llm.router import Router, RunStop
from world.storage import Storage

RETRY_ALLOWANCE = 1.10  # +10% calls for retries (§13.2)
DEFAULT_TOKENS_PER_CALL = 4000  # planning number until measured (D21)


def seat_spec(policies: str, models: str | None) -> dict[str, str]:
    """Bots from --policies, LLM seats ('lite:KEY') from --models on top."""
    seats = parse_policies(policies)
    if models:
        named = parse_policies(models) if "all" in models.lower() or "=" not in models else None
        if named is None:
            named = {}
            for part in filter(None, (p.strip() for p in models.split(","))):
                c, _, k = part.partition("=")
                named[c.strip().upper()] = k.strip()
            unknown = set(named) - set(COUNTRIES)
            if unknown:
                raise SystemExit(f"unknown countries in --models: {sorted(unknown)}")
        for c, k in named.items():
            seats[c] = f"lite:{k}"
    return seats


def _router(a: argparse.Namespace, cfg: Config) -> Router:
    return Router(
        cfg.models,
        quota=QuotaGuard(cfg.models, a.quota_db),
        cache=LLMCache(a.cache_db),
        allow_unverified=getattr(a, "allow_unverified", False),
    )


def _app(
    policies: dict[str, str], cfg: Config, a: argparse.Namespace, storage: Storage, scenario: str | None
):
    router = _router(a, cfg) if any(p.startswith("lite:") for p in policies.values()) else None
    seats = {c: make_policy(policies[c], cfg, router, c) for c in COUNTRIES}
    return build_graph(
        seats, cfg=cfg, checkpointer=open_checkpointer(a.checkpoints), storage=storage, scenario_name=scenario
    )


def estimate(policies: dict[str, str], turns: int, cfg: Config, quota: QuotaGuard) -> list[dict]:
    """Estimated calls and tokens per model key for one run, with what today's quota allows."""
    seats: dict[str, int] = {}
    for p in policies.values():
        if p.startswith("lite:"):
            k = p.split(":", 1)[1]
            seats[k] = seats.get(k, 0) + 1
    out = []
    for k, n in seats.items():
        m = cfg.models.models[k]
        per_call = m.tokens_per_call_measured or DEFAULT_TOKENS_PER_CALL
        calls = math.ceil(n * turns * RETRY_ALLOWANCE)
        adm = quota.admit(k, calls, per_call)
        out.append({
            "model": k, "seats": n, "calls": calls, "tokens": calls * per_call, "tokens_per_call": per_call,
            "fits_today": adm.fits, "can_start": adm.can_start, "remaining": quota.remaining(k),
            "next_reset_utc": adm.next_reset.isoformat(timespec="minutes"),
        })  # fmt: skip
    return out


def _confirm(rows: list[dict], yes: bool) -> bool:
    for r in rows:
        print(
            f"  {r['model']}: {r['seats']} seats, ~{r['calls']} calls, ~{r['tokens']:,} tokens "
            f"({r['tokens_per_call']}/call); remaining {r['remaining']}; fits today: {r['fits_today']}; "
            f"next reset {r['next_reset_utc']} UTC"
        )
        if not r["can_start"]:
            print(f"  {r['model']}: no quota left in this window; start after the reset")
            return False
    if yes or not rows:
        return True
    return input("Start this run? [y/N] ").strip().lower() == "y"


def _drive(app, storage: Storage, run_id: str, inputs, stop_after: int | None) -> int:
    """Run or continue a thread; turn a quota pause or a hard stop into the run's status."""
    try:
        values = drive(app, inputs, run_id, stop_after_turn=stop_after)
    except QuotaPause as e:
        when = e.resume_at
        if when is None and e.resume_after_s is not None:
            when = (dt.datetime.now(dt.UTC) + dt.timedelta(seconds=e.resume_after_s)).isoformat(
                timespec="seconds"
            )
        storage.set_status(run_id, "paused", error=e.reason, resume_after=when)
        print(f"run {run_id}: PAUSED at a seat boundary ({e.reason}); resume after {when} UTC")
        return 3
    except RunStop as e:
        storage.set_status(run_id, e.status, error=str(e), ended=True)
        print(f"run {run_id}: STOPPED, status {e.status}: {e}")
        return 2
    status = values["status"]
    if stop_after is not None and values["turn"] < values["horizon"] and status != "invariant_failed":
        status = "paused"
        storage.set_status(run_id, status)
    print(f"run {run_id}: status {status}, turn {values['turn']}")
    for row in values["logs"][-1:]:
        print(f"  last state_hash {row['state_hash'][:16]}")
    return 1 if status == "invariant_failed" else 0


def cmd_run(a: argparse.Namespace, cfg: Config) -> int:
    policies = seat_spec(a.policies, a.models)
    run_id = a.run_id or f"s{a.seed}-{dt.datetime.now(dt.UTC):%Y%m%d-%H%M%S}"
    start = settled_state(cfg, a.seed)
    init = initial_state(
        run_id, a.seed, policies, start, cfg=cfg, turns=a.turns, scenario=a.scenario,
        shocks_on=not a.no_shocks, experiment=a.experiment,
    )  # fmt: skip
    if any(p.startswith("lite:") for p in policies.values()):
        horizon = a.turns or cfg.world.horizon.max_  # never print the drawn horizon (§8)
        print(f"run {run_id}: estimate for up to {horizon} turns")
        if not _confirm(estimate(policies, horizon, cfg, QuotaGuard(cfg.models, a.quota_db)), a.yes):
            print("not started")
            return 4
    storage = Storage(a.db)
    storage.write_run(
        run_id, seed=a.seed, policies=policies, horizon=init["horizon"], experiment=a.experiment,
        scenario=a.scenario, shocks_on=not a.no_shocks,
    )  # fmt: skip
    app = _app(policies, cfg, a, storage, a.scenario)
    return _drive(app, storage, run_id, init, a.stop_after)


def _meta(a: argparse.Namespace, run_id: str, cfg: Config) -> dict:
    """run_meta of a thread, read with a throwaway status-quo graph (meta is in the state)."""
    probe = build_graph({c: make_policy("status_quo", cfg) for c in COUNTRIES}, cfg=cfg,
                        checkpointer=open_checkpointer(a.checkpoints))  # fmt: skip
    snap = probe.get_state(thread_config(run_id))
    if not snap.values:
        raise SystemExit(f"no checkpoints for run {run_id} in {a.checkpoints}")
    return snap.values["run_meta"]


def cmd_resume(a: argparse.Namespace, cfg: Config) -> int:
    meta = _meta(a, a.run_id, cfg)
    storage = Storage(a.db)
    app = _app(meta["policies"], cfg, a, storage, meta["scenario"])
    if not app.get_state(thread_config(a.run_id)).next:
        print(f"run {a.run_id} is already finished")
        return 0
    storage.set_status(a.run_id, "running")
    return _drive(app, storage, a.run_id, None, a.stop_after)


def cmd_fork(a: argparse.Namespace, cfg: Config) -> int:
    meta = _meta(a, a.run_id, cfg)
    policies = {**meta["policies"], **_named(a.policies)}
    new_id = a.new_run_id or f"{a.run_id}-fork{a.turn}"
    storage = Storage(a.db)
    app = _app(policies, cfg, a, storage, meta["scenario"])
    horizon = app.get_state(thread_config(a.run_id)).values["horizon"]
    storage.write_run(
        new_id, seed=meta["seed"], policies=policies, horizon=horizon, experiment=meta["experiment"],
        scenario=meta["scenario"], shocks_on=meta["shocks_on"], parent_run_id=a.run_id, fork_turn=a.turn,
    )  # fmt: skip
    storage.copy_turns(a.run_id, new_id, a.turn)
    fork_run(app, a.run_id, a.turn, new_id, policies)
    return _drive(app, storage, new_id, None, a.stop_after)


def cmd_quota(a: argparse.Namespace, cfg: Config) -> int:
    for row in QuotaGuard(cfg.models, a.quota_db).usage_table():
        print(json.dumps(row))
    return 0


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
    p.add_argument("--quota-db", type=Path, default=QUOTA_DB, help="quota ledger")
    p.add_argument("--cache-db", type=Path, default=CACHE_DB, help="LLM answer cache")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one game")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--policies", default="all=status_quo")
    r.add_argument("--models", default=None, help="LLM seats: all=KEY or COUNTRY=KEY,...")
    r.add_argument("--scenario", default=None)
    r.add_argument("--run-id", default=None)
    r.add_argument("--turns", type=int, default=None, help="fixed length (default: hidden horizon, §8)")
    r.add_argument("--stop-after", type=int, default=None, help="pause after this turn")
    r.add_argument("--no-shocks", action="store_true")
    r.add_argument("--experiment", default="")
    r.add_argument("--yes", action="store_true", help="do not ask before a run with LLM seats")
    r.add_argument("--allow-unverified", action="store_true", help="smoke tests only")

    s = sub.add_parser("resume", help="continue a paused or crashed run from its last checkpoint")
    s.add_argument("run_id")
    s.add_argument("--stop-after", type=int, default=None)
    s.add_argument("--allow-unverified", action="store_true")

    f = sub.add_parser("fork", help="branch a run at the start of a turn, with new policies for some seats")
    f.add_argument("run_id")
    f.add_argument("--turn", type=int, required=True)
    f.add_argument("--policies", default=None, help="seats that change, e.g. DORNE=aggressor")
    f.add_argument("--run-id", dest="new_run_id", default=None)
    f.add_argument("--stop-after", type=int, default=None)
    f.add_argument("--allow-unverified", action="store_true")

    sub.add_parser("quota", help="today's usage and remaining budget per model")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(".env")  # API keys come from the environment only; never printed
    a = parser().parse_args(argv)
    cfg = load_config()
    cmds = {"run": cmd_run, "resume": cmd_resume, "fork": cmd_fork, "quota": cmd_quota}
    return cmds[a.cmd](a, cfg)


if __name__ == "__main__":
    sys.exit(main())
