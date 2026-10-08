"""Phase 5: the LangGraph turn loop (world/graph.py) with bots, no LLM.

- the graph gives the same state_hash every turn as the plain loop (world/game.py), 3 seeds;
- a run interrupted after turn 5 and resumed (new graph, new connection) ends in the same state;
  so does a run that crashes in the middle of a turn and is resumed from the last seat checkpoint;
- a fork at turn 6 with another policy for one seat is a valid branch (equal to a plain-loop game
  whose seat switches policy at turn 6) and leaves the original thread unchanged;
- the leader-change memory-wipe hook fires for every leader change, the same way in both loops;
- 14 turns x 20 seeds with mixed bots still run through the graph with no invariant failure.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tests.runs import CFG
from tests.test_bot_game import mixed
from world.config import COUNTRIES
from world.engine.burn_in import load_fixture
from world.game import bot_seats, make_bot, run_game
from world.graph import (
    RECURSION_LIMIT,
    build_graph,
    drive,
    fork_run,
    initial_state,
    open_checkpointer,
    parse_policies,
    thread_config,
)
from world.history import leader_death_text
from world.serial import decode


def graph_game(spec, seed, *, turns=None, checkpointer=None, run_id="run", seats=None, **kw):
    pol = parse_policies(spec)
    app = build_graph(seats or bot_seats(pol, CFG), cfg=CFG, checkpointer=checkpointer)
    init = initial_state(run_id, seed, pol, load_fixture(seed), cfg=CFG, turns=turns, **kw)
    if checkpointer is None:
        return app.invoke(init, {"recursion_limit": RECURSION_LIMIT}), app
    return drive(app, init, run_id), app


def hashes(values) -> list[str]:
    return [row["state_hash"] for row in values["logs"]]


# --------------------------------------------------------------------------- graph == loop


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_graph_equals_plain_loop_every_turn(seed: int, tmp_path) -> None:
    plain = run_game(bot_seats(mixed(seed), CFG), seed=seed, cfg=CFG, start=load_fixture(seed))
    values, app = graph_game(mixed(seed), seed, checkpointer=open_checkpointer(tmp_path / "cp.db"))
    assert values["horizon"] == plain.horizon  # same hidden horizon from the HORIZON stream
    assert values["status"] == plain.status == "ok"
    assert hashes(values) == [s.state_hash() for s in plain.states[1:]]
    assert decode(values["world"]).state_hash() == plain.final.state_hash()
    assert tuple(decode(r) for r in values["recent"]) == tuple(plain.records[-2:])
    assert len(values["decisions"]) == 6 * plain.horizon
    assert app.get_state(thread_config("run")).next == ()  # finished


# ------------------------------------------------------------------------ interrupt / resume


def test_interrupt_after_turn_5_then_resume_gives_identical_final_state(tmp_path) -> None:
    seed, spec = 7, mixed(7)
    full, _ = graph_game(spec, seed, turns=12)
    db = tmp_path / "cp.db"
    pol = parse_policies(spec)
    app = build_graph(bot_seats(pol, CFG), cfg=CFG, checkpointer=open_checkpointer(db))
    init = initial_state("r", seed, pol, load_fixture(seed), cfg=CFG, turns=12)
    paused = drive(app, init, "r", stop_after_turn=5)
    assert paused["turn"] == 5 and len(paused["logs"]) == 5
    assert app.get_state(thread_config("r")).next == ("turn_start",)
    # a new process: new connection, new policy objects, new graph
    app2 = build_graph(bot_seats(pol, CFG), cfg=CFG, checkpointer=open_checkpointer(db))
    done = drive(app2, None, "r")
    assert done["turn"] == 12
    assert hashes(done) == hashes(full)
    assert done["world"] == full["world"]


@dataclass
class CrashAt:
    """Wraps a bot and raises once at (turn, country): a crash in the middle of a turn."""

    inner: object
    turn: int
    country: str
    armed: bool = True

    @property
    def name(self) -> str:
        return self.inner.name

    def decide(self, b):
        if self.armed and b.turn == self.turn and b.country == self.country:
            self.armed = False
            raise RuntimeError("simulated crash")
        return self.inner.decide(b)


def test_crash_mid_turn_then_resume_from_seat_checkpoint(tmp_path) -> None:
    seed, spec = 8, mixed(8)
    full, _ = graph_game(spec, seed, turns=10)
    db = tmp_path / "cp.db"
    pol = parse_policies(spec)
    seats = bot_seats(pol, CFG)
    seats["FALKEN"] = CrashAt(seats["FALKEN"], 6, "FALKEN")
    app = build_graph(seats, cfg=CFG, checkpointer=open_checkpointer(db))
    with pytest.raises(RuntimeError, match="simulated crash"):
        drive(app, initial_state("r", seed, pol, load_fixture(seed), cfg=CFG, turns=10), "r")
    snap = app.get_state(thread_config("r"))
    assert snap.next == ("leader",) and snap.values["turn"] == 6 and snap.values["slot_index"] > 0
    app2 = build_graph(bot_seats(pol, CFG), cfg=CFG, checkpointer=open_checkpointer(db))
    done = drive(app2, None, "r")
    assert hashes(done) == hashes(full)


# -------------------------------------------------------------------------------------- fork


@dataclass
class SwitchAt:
    """Plays `before` until turn `turn`, then `after` (the plain-loop twin of a fork)."""

    before: object
    after: object
    turn: int
    name: str = "switch"

    def decide(self, b):
        return (self.before if b.turn < self.turn else self.after).decide(b)


def test_fork_at_turn_6_is_a_valid_branch_and_original_is_unchanged(tmp_path) -> None:
    seed, spec, turns = 4, "all=tit_for_tat", 12
    cp = open_checkpointer(tmp_path / "cp.db")
    original, app = graph_game(spec, seed, turns=turns, checkpointer=cp, run_id="orig")
    n_checkpoints = len(list(app.get_state_history(thread_config("orig"))))

    new_pol = {**parse_policies(spec), "DORNE": "aggressor"}
    fork_app = build_graph(bot_seats(new_pol, CFG), cfg=CFG, checkpointer=cp)
    fork_config = fork_run(fork_app, "orig", 6, "fork", new_pol)
    fork_app.invoke(None, {**fork_config, "recursion_limit": RECURSION_LIMIT})
    fork = fork_app.get_state(thread_config("fork")).values

    # the original thread is untouched
    after = app.get_state(thread_config("orig"))
    assert after.values["logs"] == original["logs"] and after.values["world"] == original["world"]
    assert len(list(app.get_state_history(thread_config("orig")))) == n_checkpoints
    # the branch shares turns 1-5, then diverges, and runs to the same horizon
    assert hashes(fork)[:5] == hashes(original)[:5]
    assert hashes(fork)[5] != hashes(original)[5]
    assert fork["turn"] == turns and fork["status"] == "ok"
    assert fork["run_meta"]["parent_run_id"] == "orig" and fork["run_meta"]["fork_turn"] == 6
    assert fork["run_meta"]["policies"]["DORNE"] == "aggressor"
    # and it is exactly the game where DORNE switches to the aggressor at turn 6
    seats = bot_seats(parse_policies(spec), CFG)
    seats["DORNE"] = SwitchAt(make_bot("tit_for_tat", CFG), make_bot("aggressor", CFG), 6)
    twin = run_game(seats, seed=seed, cfg=CFG, turns=turns, start=load_fixture(seed))
    assert hashes(fork) == [s.state_hash() for s in twin.states[1:]]


# ------------------------------------------------------------------------ leader-change hook


@dataclass
class Recorder:
    inner: object
    events: list

    @property
    def name(self) -> str:
        return self.inner.name

    def decide(self, b):
        self.events.append(("decide", b.turn, b.country, b.leader_removed, len(b.my_last_turns)))
        return self.inner.decide(b)

    def on_leader_change(self, country: str) -> None:
        self.events.append(("wipe", country))


def test_leader_change_hook_and_memory_wipe_same_in_both_loops() -> None:
    seed, turns = 3, 14
    ev_plain: list = []
    plain = run_game({c: Recorder(make_bot("aggressor", CFG), ev_plain) for c in COUNTRIES}, seed=seed,
                     cfg=CFG, turns=turns, start=load_fixture(seed))  # fmt: skip
    changes = [(r.turn, c) for r in plain.records for c in r.leader_changes]
    deaths = [(r.turn, c) for r in plain.records for c in COUNTRIES if leader_death_text(c) in r.shocks]
    assert changes, "aggressor self-play should topple some leader (S5)"
    assert sum(e[0] == "wipe" for e in ev_plain) == len(changes) + len(deaths)
    for t, c in changes:  # the new leader's first briefing says so and shows no earlier turns
        first = (
            next(e for e in ev_plain if e[0] == "decide" and e[1] == t + 1 and e[2] == c)
            if t < turns
            else None
        )
        if first is not None:
            assert first[3] is True and first[4] == 0
    ev_graph: list = []
    seats = {c: Recorder(make_bot("aggressor", CFG), ev_graph) for c in COUNTRIES}
    values, _ = graph_game("all=aggressor", seed, turns=turns, seats=seats)
    assert ev_graph == ev_plain
    assert hashes(values) == [s.state_hash() for s in plain.states[1:]]


# ------------------------------------------------------------------------------ 14 x 20 seeds


@pytest.mark.parametrize("seed", range(1, 21))
def test_mixed_bot_game_14_turns_through_graph(seed: int) -> None:
    values, _ = graph_game(mixed(seed), seed, turns=14)
    assert values["status"] == "ok", values["error"]
    assert values["turn"] == 14 and len(values["logs"]) == 14
    plain = run_game(bot_seats(mixed(seed), CFG), seed=seed, cfg=CFG, turns=14, start=load_fixture(seed))
    assert hashes(values) == [s.state_hash() for s in plain.states[1:]]
