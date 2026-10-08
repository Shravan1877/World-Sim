"""Experiment database (§15): write a run through the graph, read it back; per-turn writes are
idempotent; the runner CLI writes runs, pauses, resumes and forks (data in tmp_path)."""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest
from pydantic import TypeAdapter

from tests.runs import CFG
from tests.test_bot_game import mixed
from world.actions import Action
from world.config import COUNTRIES, SECTORS
from world.engine.burn_in import load_fixture
from world.game import bot_seats
from world.graph import RECURSION_LIMIT, build_graph, initial_state, parse_policies
from world.metrics import real_gdp
from world.runner import main
from world.serial import decode
from world.storage import TURN_TABLES, Storage

SEED, TURNS = 6, 5


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    db = Storage(tmp_path_factory.mktemp("db") / "experiments.db")
    pol = parse_policies(mixed(SEED))
    start = load_fixture(SEED)
    init = initial_state("R1", SEED, pol, start, cfg=CFG, turns=TURNS, scenario="energy_crunch")
    db.write_run("R1", seed=SEED, policies=pol, horizon=TURNS, experiment="test", scenario="energy_crunch")
    app = build_graph(bot_seats(pol, CFG), cfg=CFG, storage=db, scenario_name="energy_crunch")
    values = app.invoke(init, {"recursion_limit": RECURSION_LIMIT})
    return db, values, start, pol


def test_run_row_round_trip(stored) -> None:
    db, values, _, pol = stored
    run = db.read_run("R1")
    assert run["seed"] == SEED and run["horizon"] == TURNS and run["experiment"] == "test"
    assert run["models_per_seat"] == pol and run["scenario"] == "energy_crunch"
    assert run["status"] == "ok" and run["ended_at"] and len(run["config_hash"]) == 16
    assert db.run_ids() == ["R1"] and db.read_run("nope") is None


def test_turn_rows_match_the_graph_state(stored) -> None:
    db, values, start, _ = stored
    turns = db.read("turns", "R1")
    assert list(turns.turn) == list(range(1, TURNS + 1))
    assert list(turns.state_hash) == [r["state_hash"] for r in values["logs"]]
    final = decode(values["world"])
    cs = db.read("country_state", "R1").query(f"turn == {TURNS}").set_index("country").loc[list(COUNTRIES)]
    assert np.array_equal(cs.gdp.to_numpy(), final.gdp)
    assert np.array_equal(cs.real_gdp.to_numpy(), real_gdp(final, start))
    assert np.array_equal(cs.stability.to_numpy(), final.stability)
    assert np.array_equal(np.array([json.loads(x) for x in cs.price]), final.price)
    assert cs.power.sum() == pytest.approx(1.0)
    assert len(db.read("bilateral", "R1")) == TURNS * 30
    assert len(db.read("firms", "R1")) == TURNS * 30 and set(db.read("firms", "R1").sector) == set(SECTORS)
    shocks = db.read("shocks", "R1")
    assert shocks.scheduled.sum() >= 1  # the energy_crunch incident


def test_decision_and_action_rows_match_the_records(stored) -> None:
    db, values, _, _ = stored
    dec = db.read("decisions", "R1")
    assert len(dec) == 6 * TURNS and dec.parse_failure.sum() == 0
    last = tuple(decode(r) for r in values["recent"])[-1]
    acts = db.read("actions", "R1").query(f"turn == {TURNS}")
    adapter = TypeAdapter(Action)
    for seat in last.seats:
        mine = acts[(acts.country == seat.country) & (acts.status == "accepted")]
        assert [adapter.validate_json(j) for j in mine.action_json] == list(seat.accepted)
        row = dec[(dec.turn == TURNS) & (dec.country == seat.country)].iloc[0]
        assert row.public_statement == seat.public_statement and row.seat == seat.seat
    assert len(db.read("facts_checks", "R1")) > 0 and len(db.read("predictions", "R1")) > 0


def test_write_turn_is_idempotent(stored, tmp_path) -> None:
    db, values, start, _ = stored
    other = Storage(tmp_path / "copy.db")
    rec = tuple(decode(r) for r in values["recent"])[-1]
    # a resumed run may record the same turn twice: rows are replaced, never duplicated
    log_like = _fake_log()
    for _ in range(2):
        other.write_turn("X", rec, decode(values["world"]), log_like, start, CFG)
    counts = {t: len(other.read(t, "X")) for t in TURN_TABLES}
    other.write_turn("X", rec, decode(values["world"]), log_like, start, CFG)
    assert counts == {t: len(other.read(t, "X")) for t in TURN_TABLES}
    assert counts["turns"] == 1 and counts["country_state"] == 6


def _fake_log():
    from world.engine.step import TurnLog

    z65, z64 = np.zeros((6, 5)), np.zeros((6, 4))
    return TurnLog(
        turn=TURNS, events=(), consumption=z65, shortage=z65, imports=z64, exports=z64, taxes=np.zeros(6),
        profit=z65, interest=np.zeros(6), borrowing=np.zeros(6), hhi=z65, deliveries=(),
        firm_dice=np.zeros((6, 5, 4)),
    )  # fmt: skip


def test_runner_cli_run_pause_resume_fork(tmp_path) -> None:
    db, cp = tmp_path / "e.db", tmp_path / "c.db"
    base = ["--db", str(db), "--checkpoints", str(cp)]
    assert main([*base, "run", "--seed", "3", "--policies", "all=tit_for_tat", "--run-id", "A",
                 "--turns", "9", "--stop-after", "4"]) == 0  # fmt: skip
    st = Storage(db)
    assert st.read_run("A")["status"] == "paused" and len(st.read("turns", "A")) == 4
    assert main([*base, "resume", "A"]) == 0
    assert st.read_run("A")["status"] == "ok" and len(st.read("turns", "A")) == 9
    assert main([*base, "fork", "A", "--turn", "6", "--policies", "CERES=aggressor", "--run-id", "B"]) == 0
    b = st.read_run("B")
    assert b["parent_run_id"] == "A" and b["fork_turn"] == 6 and b["status"] == "ok"
    assert b["models_per_seat"]["CERES"] == "aggressor" and b["models_per_seat"]["DORNE"] == "tit_for_tat"
    ha, hb = (list(st.read("turns", r).state_hash) for r in ("A", "B"))
    assert len(hb) == 9 and hb[:5] == ha[:5] and hb[5] != ha[5]
    with sqlite3.connect(cp) as c:  # both threads live in the one checkpoint file
        threads = {r[0] for r in c.execute("SELECT DISTINCT thread_id FROM checkpoints")}
    assert threads == {"A", "B"}


def test_ledger_transfers_in_db_reconcile_every_account(stored) -> None:
    """D68: the graph state keeps balances only; the database holds every transfer, so each account's
    final balance = its starting balance + inflows - outflows recomputed from the database."""
    import math

    from world.ledger import all_accounts

    db, values, start, _ = stored
    tr = db.read("ledger_transfers", "R1")
    assert len(tr) > 0 and set(tr.turn) == set(range(1, TURNS + 1))
    final = decode(values["world"])
    for acc in all_accounts():
        name = str(acc)
        flow = math.fsum(tr.amount[tr.dst == name]) - math.fsum(tr.amount[tr.src == name])
        assert start.ledger.balance(acc) + flow == pytest.approx(
            final.ledger.balance(acc), rel=1e-9, abs=1e-9
        )
