"""LIVE smoke games (Phase 6). Skipped unless `pytest --live`. They use real API quota.

1 LLM + 5 bots for 6 turns, then 6 LLMs for 6 turns, with m8b and with g35. Results go to the real
data/ databases (run ids smoke-*). A budget guard reads the quota ledger and refuses to start a game
whose worst case (every seat, every turn, call + retry) would pass the session caps below.
"""

from __future__ import annotations

import datetime as dt
import os
import sqlite3

import pytest

from world.llm.quota import QUOTA_DB
from world.runner import main
from world.storage import Storage

SESSION_START = os.environ.get("WORLDSIM_SESSION_START", "2026-10-08T00:00:00")
SESSION_CAP = int(os.environ.get("WORLDSIM_SESSION_CAP", "250"))
GOOGLE_CAP = int(os.environ.get("WORLDSIM_GOOGLE_CAP", "150"))
TURNS = 6


def used_since_start() -> tuple[int, int]:
    with sqlite3.connect(QUOTA_DB) as c:
        rows = c.execute("SELECT model_key, SUM(requests) FROM calls WHERE at >= ? GROUP BY model_key",
                         (SESSION_START,)).fetchall()  # fmt: skip
    total = sum(n for _, n in rows)
    google = sum(n for k, n in rows if k.startswith("g"))
    return total, google


def guard(key: str, llm_seats: int) -> None:
    worst = llm_seats * TURNS * 2
    total, google = used_since_start()
    if total + worst > SESSION_CAP or (key.startswith("g") and google + worst > GOOGLE_CAP):
        pytest.skip(f"session budget: used {total} (google {google}), this game may need {worst}")


@pytest.mark.live
@pytest.mark.parametrize("key", ["m8b", "g35"])
@pytest.mark.parametrize("all_llm", [False, True], ids=["1llm_5bots", "6llm"])
def test_live_smoke_game(key: str, all_llm: bool) -> None:
    seats = 6 if all_llm else 1
    guard(key, seats)
    run_id = f"smoke-{key}-{'6llm' if all_llm else '1v5'}-{dt.datetime.now(dt.UTC):%H%M%S}"
    models = f"all={key}" if all_llm else f"DORNE={key}"
    args = ["run", "--seed", "1", "--policies", "all=tit_for_tat", "--models", models]
    code = main([*args, "--turns", str(TURNS), "--run-id", run_id, "--yes"])
    db = Storage("data/experiments.db")
    run = db.read_run(run_id)
    calls = db.read("llm_calls", run_id)
    print(run_id, run["status"], len(calls), "calls")
    assert code == 0 and run["status"] in ("ok", "degraded")
    assert len(db.read("turns", run_id)) == TURNS
    assert len(calls) >= seats * TURNS
    assert calls.model_id.nunique() == 1  # never switched model id
