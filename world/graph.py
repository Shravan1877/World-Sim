"""The LangGraph turn loop (CLAUDE.md §11.6). A thin orchestration layer over the engine.

    START -> turn_start (shocks, move order) -> leader (seat slot_index) -> validate_apply
          -> [slot_index < 6] -> leader                         (the 6 seats, in this turn's order)
          -> [slot_index = 6] -> resolve (step) -> record (database + metrics)
          -> [turn < horizon and the run is not paused/failed] -> turn_start, else -> END

The nodes call the same seat and turn helpers as the plain loop (world/game.py), so for the same seed
and policies both loops give identical states every turn (tests/test_graph_bots.py).

Graph state is plain data (world/serial.py encodes WorldState, records and decisions exactly), so the
SqliteSaver checkpointer (data/checkpoints.db, thread_id = run_id) can save it after every node. This
gives resume after a crash or a pause at a seat boundary (invoke(None, config)), replay, and forks
(fork_run: get_state_history -> update_state on a new thread -> invoke(None, fork_config)).

Policies are NOT in the graph state (they are objects, and LLM policies hold clients); the graph is
built around a {country: policy} map, and run_meta["policies"] keeps their names so a resumed or
forked run can rebuild them. Only the last 2 turn records are kept for briefings (all a briefing reads,
see briefing.py); the full history goes to the experiment database.

The engine never imports this module or LangGraph (§18 rule 7).
"""

from __future__ import annotations

import operator
import sqlite3
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from world.config import COUNTRIES, Config, load_config, load_scenario
from world.engine.invariants import InvariantError
from world.engine.policy import PolicyEffects
from world.engine.state import WorldState
from world.engine.step import step, turn_start
from world.game import (
    apply_seat,
    draw_horizon,
    leader_deaths,
    make_bot,
    policy_name,
    seat_briefing,
    shock_text,
    turn_record,
)
from world.history import TurnRecord
from world.policies.base import LeaderPolicy, notify_leader_change
from world.policies.bots import StatusQuoBot
from world.rng import RngBundle
from world.serial import decode, encode
from world.storage import Storage

CHECKPOINTS_DB = Path("data/checkpoints.db")
EXPERIMENTS_DB = Path("data/experiments.db")
RECURSION_LIMIT = 100_000  # one game is ~15 graph steps per turn; LangGraph's default limit is 25
HISTORY_KEPT = 2  # turn records a briefing reads (§11.2 sections 4, 7, 9)


class GraphState(TypedDict, total=False):
    world: dict  # encoded WorldState (current)
    start_world: dict  # encoded starting (settled) state: base prices for real GDP (D61)
    turn: int
    horizon: int  # hidden game length (§8); never passed to policies
    order: list[str]
    slot_index: int  # next seat to move (0..5); 6 = everyone has moved
    shocks: list[str]  # this turn's fired shocks, as text
    shock_events: list  # encoded ShockEvents (every die, for the shocks table)
    turn_seats: list  # encoded SeatRecords of this turn so far
    turn_actions: list  # [[country, action text], ...] accepted so far this turn (readable view)
    turn_statements: list  # [[country, public statement], ...] this turn so far
    effects: dict  # encoded PolicyEffects (what step() needs from this turn's actions)
    pending: dict | None  # encoded DecisionResult of the seat that just decided
    last_log: dict | None  # encoded TurnLog between resolve and record
    recent: list  # encoded TurnRecords, the last HISTORY_KEPT turns
    streak: dict  # country -> consecutive parse failures
    degraded_seats: list[str]
    status: str  # running | ok | degraded | invariant_failed
    error: str
    decisions: Annotated[list, operator.add]  # one compact row per seat (full rows in the database)
    logs: Annotated[list, operator.add]  # one row per resolved turn: turn, state_hash, events
    run_meta: dict  # run_id, seed, policies, scenario, shocks_on, experiment, parent_run_id, fork_turn


PolicyFactory = Callable[[str, Config], LeaderPolicy]


def make_policy(name: str, cfg: Config) -> LeaderPolicy:
    """Policy by name. Phase 5: the scripted bots only (LLM policies arrive in Phase 6)."""
    return make_bot(name, cfg)


def parse_policies(spec: str | Mapping[str, str]) -> dict[str, str]:
    """'all=greedy' | 'greedy' | 'DORNE=greedy,CERES=aggressor' | {country: name} -> {country: name}.
    Countries not named play status_quo."""
    if isinstance(spec, Mapping):
        named = dict(spec)
    else:
        named = {}
        for part in filter(None, (p.strip() for p in spec.split(","))):
            key, _, val = part.partition("=")
            if not val:
                key, val = "all", key
            named[key.strip().upper() if key.strip().lower() != "all" else "all"] = val.strip()
    unknown = set(named) - set(COUNTRIES) - {"all"}
    if unknown:
        raise ValueError(f"unknown countries in policy spec: {sorted(unknown)}")
    default = named.pop("all", "status_quo")
    return {c: named.get(c, default) for c in COUNTRIES}


def open_checkpointer(path: str | Path = CHECKPOINTS_DB) -> SqliteSaver:
    """SqliteSaver on its own SQLite file. Strict serializer: graph state is plain data only."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=None))


def thread_config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}, "recursion_limit": RECURSION_LIMIT}


# ---------------------------------------------------------------------------------- graph


def build_graph(
    policies: Mapping[str, LeaderPolicy],
    *,
    cfg: Config | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    storage: Storage | None = None,
    scenario_name: str | None = None,
):
    """Compile the turn loop around one {country: policy} map. `storage` = the experiment database
    (None: nothing is written, e.g. in tests)."""
    cfg = cfg or load_config()
    scenario = load_scenario(scenario_name) if scenario_name else None
    fallback = StatusQuoBot(cfg=cfg)

    def seat_policy(country: str, degraded: list[str]) -> LeaderPolicy:
        return fallback if country in degraded else policies[country]

    def history(st: GraphState) -> tuple[TurnRecord, ...]:
        return tuple(decode(r) for r in st["recent"])

    def n_turn_start(st: GraphState) -> dict[str, Any]:
        meta = st["run_meta"]
        s, ts = turn_start(
            decode(st["world"]), RngBundle(meta["seed"]), cfg, scenario=scenario, shocks_on=meta["shocks_on"]
        )
        shocks = [shock_text(e) for e in ts.fired]
        for c in leader_deaths(tuple(shocks)):
            notify_leader_change(seat_policy(c, st["degraded_seats"]), c)
        return {
            "world": encode(s),
            "turn": s.turn,
            "order": list(ts.order),
            "slot_index": 0,
            "shocks": shocks,
            "shock_events": [encode(e) for e in ts.shock_events],
            "turn_seats": [],
            "turn_actions": [],
            "turn_statements": [],
            "effects": encode(PolicyEffects()),
            "pending": None,
        }

    def briefing_for(st: GraphState):
        country = st["order"][st["slot_index"]]
        done = tuple(decode(r) for r in st["turn_seats"])
        b = seat_briefing(
            decode(st["world"]), country, tuple(st["order"]), tuple(st["shocks"]), done, history(st), cfg
        )
        return country, b

    def n_leader(st: GraphState) -> dict[str, Any]:
        country, b = briefing_for(st)
        out = seat_policy(country, st["degraded_seats"]).decide(b)
        return {"pending": encode(out)}

    def n_validate_apply(st: GraphState) -> dict[str, Any]:
        country, b = briefing_for(st)  # the same (pure) briefing the leader saw
        out = decode(st["pending"])
        effects: PolicyEffects = decode(st["effects"])
        degraded = list(st["degraded_seats"])
        so = apply_seat(
            decode(st["world"]), st["slot_index"] + 1, b, out,
            policy_name(seat_policy(country, degraded)), effects, st["streak"][country],
            country in degraded, cfg,
        )  # fmt: skip
        status = st["status"]
        if so.degrade:
            degraded.append(country)
            status = "degraded"
        r = so.record
        return {
            "world": encode(so.state),
            "effects": encode(effects),
            "turn_seats": [*st["turn_seats"], encode(r)],
            "turn_actions": [*st["turn_actions"], *([country, a.model_dump_json()] for a in r.accepted)],
            "turn_statements": [*st["turn_statements"], [country, r.public_statement]],
            "streak": {**st["streak"], country: so.streak},
            "degraded_seats": degraded,
            "status": status,
            "slot_index": st["slot_index"] + 1,
            "pending": None,
            "decisions": [
                {
                    "turn": r.turn, "country": country, "seat": r.seat, "policy": r.policy,
                    "parse_failure": r.parse_failure, "degraded": r.degraded,
                    "accepted": len(r.accepted), "rejected": len(r.rejected),
                }
            ],
        }  # fmt: skip

    def after_seat(st: GraphState) -> str:
        return "leader" if st["slot_index"] < len(st["order"]) else "resolve"

    def n_resolve(st: GraphState) -> dict[str, Any]:
        effects: PolicyEffects = decode(st["effects"])
        try:
            s, log = step(decode(st["world"]), effects.turn_inputs(), RngBundle(st["run_meta"]["seed"]), cfg)
        except InvariantError as e:
            return {"status": "invariant_failed", "error": str(e), "last_log": None}
        # Counterfactual replays for collateral damage (§12.2) are added here in Phase 7.
        return {"world": encode(s), "last_log": encode(log)}

    def n_record(st: GraphState) -> dict[str, Any]:
        run_id = st["run_meta"]["run_id"]
        if st["status"] == "invariant_failed":
            if storage is not None:
                storage.set_status(run_id, "invariant_failed", error=st["error"], ended=True)
            return {}
        s: WorldState = decode(st["world"])
        log = decode(st["last_log"])
        seats = tuple(decode(r) for r in st["turn_seats"])
        rec = turn_record(s.turn, tuple(st["order"]), tuple(st["shocks"]), seats, log)
        for c in rec.leader_changes:
            notify_leader_change(seat_policy(c, st["degraded_seats"]), c)
        finished = s.turn >= st["horizon"]
        status = st["status"]
        if finished:
            status = "degraded" if st["degraded_seats"] else "ok"
        if storage is not None:
            storage.write_turn(
                run_id, rec, s, log, decode(st["start_world"]), cfg,
                shock_events=(decode(e) for e in st["shock_events"]),
            )  # fmt: skip
            storage.set_status(
                run_id, status, degraded_seats=st["degraded_seats"], error=st["error"], ended=finished
            )
        return {
            "recent": [*st["recent"], encode(rec)][-HISTORY_KEPT:],
            "last_log": None,
            "status": status,
            "logs": [{"turn": s.turn, "state_hash": s.state_hash(), "events": list(rec.events)}],
        }

    def after_record(st: GraphState) -> str:
        if st["status"] in ("invariant_failed", "paused"):
            return END
        return "turn_start" if st["turn"] < st["horizon"] else END

    g = StateGraph(GraphState)
    g.add_node("turn_start", n_turn_start)
    g.add_node("leader", n_leader)
    g.add_node("validate_apply", n_validate_apply)
    g.add_node("resolve", n_resolve)
    g.add_node("record", n_record)
    g.add_edge(START, "turn_start")
    g.add_edge("turn_start", "leader")
    g.add_edge("leader", "validate_apply")
    g.add_conditional_edges("validate_apply", after_seat, ["leader", "resolve"])
    g.add_edge("resolve", "record")
    g.add_conditional_edges("record", after_record, ["turn_start", END])
    return g.compile(checkpointer=checkpointer)


# ------------------------------------------------------------------------------ run helpers


def initial_state(
    run_id: str,
    seed: int,
    policies: Mapping[str, str],
    start: WorldState,
    *,
    cfg: Config,
    turns: int | None = None,
    scenario: str | None = None,
    shocks_on: bool = True,
    experiment: str = "",
) -> GraphState:
    """The graph input for a new run. The horizon is drawn here from the HORIZON stream (§8) unless a
    fixed `turns` is given (tests, calibration)."""
    return GraphState(
        world=encode(start),
        start_world=encode(start),
        turn=start.turn,
        horizon=turns if turns is not None else draw_horizon(seed, cfg),
        order=[],
        slot_index=0,
        shocks=[],
        shock_events=[],
        turn_seats=[],
        turn_actions=[],
        turn_statements=[],
        effects=encode(PolicyEffects()),
        pending=None,
        last_log=None,
        recent=[],
        streak=dict.fromkeys(COUNTRIES, 0),
        degraded_seats=[],
        status="running",
        error="",
        decisions=[],
        logs=[],
        run_meta={
            "run_id": run_id,
            "seed": seed,
            "policies": dict(policies),
            "scenario": scenario,
            "shocks_on": shocks_on,
            "experiment": experiment,
            "parent_run_id": None,
            "fork_turn": None,
        },
    )


def drive(
    app, inputs: GraphState | None, run_id: str, *, stop_after_turn: int | None = None
) -> dict[str, Any]:
    """Run (inputs = initial state) or continue (inputs = None) a thread. With stop_after_turn the
    run pauses right after that turn is recorded (a static interrupt after `record`); a later
    drive(app, None, run_id) resumes it from the checkpoint."""
    config = thread_config(run_id)
    if stop_after_turn is None:
        app.invoke(inputs, config)
    else:
        while True:
            app.invoke(inputs, config, interrupt_after=["record"])
            inputs = None
            snap = app.get_state(config)
            if not snap.next or snap.values["turn"] >= stop_after_turn:
                break
    return app.get_state(config).values


def fork_snapshot(app, run_id: str, fork_turn: int):
    """The checkpoint of `run_id` just before turn `fork_turn` starts (after turn fork_turn - 1 was
    recorded; for fork_turn = 1, the run's input checkpoint)."""
    for snap in app.get_state_history(thread_config(run_id)):  # newest first
        if snap.next == ("turn_start",) and snap.values.get("turn") == fork_turn - 1:
            return snap
    raise ValueError(f"run {run_id}: no checkpoint before turn {fork_turn}")


def fork_run(
    app,
    run_id: str,
    fork_turn: int,
    new_run_id: str,
    new_policies: Mapping[str, str],
) -> dict[str, Any]:
    """Branch `run_id` at the start of turn `fork_turn` into a NEW thread `new_run_id` whose seats play
    `new_policies` (`app` must be built around those policies, with the same checkpointer). The
    original thread is not touched. Returns the fork config to continue with invoke(None, config)."""
    snap = fork_snapshot(app, run_id, fork_turn)
    values = dict(snap.values)
    values["run_meta"] = {
        **values["run_meta"],
        "run_id": new_run_id,
        "policies": dict(new_policies),
        "parent_run_id": run_id,
        "fork_turn": fork_turn,
    }
    values["status"] = "running"
    return app.update_state(thread_config(new_run_id), values, as_node="record")
