"""Full bot games through the plain loop (Phase 4 done-when): 14 turns x 20 seeds, no invariant
failures, and the whole pipeline (briefing -> TurnDecision -> validator -> policy -> step) holds up."""

import numpy as np
import pytest

from tests.runs import CFG, run_bots
from world.config import COUNTRIES, load_scenario
from world.engine.burn_in import load_fixture
from world.game import BOTS, bot_seats, draw_horizon, run_game

TYPES = ("status_quo", "tit_for_tat", "greedy", "cooperative", "random", "aggressor")


def mixed(seed: int) -> dict[str, str]:
    """Every bot type in every game, rotated over the seats by seed."""
    return {c: TYPES[(k + seed) % len(TYPES)] for k, c in enumerate(COUNTRIES)}


@pytest.mark.parametrize("seed", range(1, 21))
def test_mixed_bot_game_14_turns_no_invariant_failure(seed: int) -> None:
    r = run_bots(mixed(seed), seed, 14, shocks_on=True)
    assert r.status == "ok", r.error
    assert len(r.records) == 14
    for rec in r.records:
        assert [c for c in rec.order if c != "EVERMERE"] == ["DORNE", "BRONTIA", "CERES", "FALKEN", "AURELIA"]
        assert rec.order.index("AURELIA") in (4, 5)
        for k, seat in enumerate(rec.seats):
            assert seat.decision is not None and not seat.parse_failure
            assert seat.country == rec.order[k] and seat.seat == k + 1
            assert all(p.country in rec.order[k + 1 :] for p in seat.predictions)
            assert not any(f.wrong for f in seat.fact_checks), "bots copy facts from the briefing"
            if seat.policy != "random":  # RandomBot tries out invalid actions on purpose
                bad = [why for _, why in seat.rejected if not why.startswith("in default")]
                assert not bad, (seat.country, seat.policy, bad)


@pytest.mark.parametrize("bot", sorted(BOTS))
def test_each_bot_in_self_play(bot: str) -> None:
    r = run_bots(bot, 5, 14, scenario=load_scenario("energy_crunch"))
    assert r.status == "ok", r.error


def test_bot_game_is_deterministic() -> None:
    a = run_bots(mixed(3), 3, 8)
    b = run_bots(mixed(3), 3, 8)
    assert [s.state_hash() for s in a.states] == [s.state_hash() for s in b.states]


def test_status_quo_bots_equal_no_policy_run() -> None:
    from tests.runs import run

    _, states, _ = run(4, 6, shocks_on=True)
    r = run_bots("status_quo", 4, 6)
    assert r.final.state_hash() == states[-1].state_hash()


def test_hidden_horizon_from_horizon_stream() -> None:
    horizons = {draw_horizon(s, CFG) for s in range(1, 200)}
    assert horizons == set(range(CFG.world.horizon.min_, CFG.world.horizon.max_ + 1))
    r = run_game(bot_seats("status_quo"), seed=2, start=load_fixture(2), shocks_on=False)
    assert len(r.records) == draw_horizon(2, CFG)


class _Broken:
    name = "broken"

    def decide(self, briefing):
        from world.policies.base import DecisionResult

        return DecisionResult(decision=None, parse_error="not json")


def test_parse_failures_fall_back_to_wait_then_status_quo_bot() -> None:
    seats = bot_seats("status_quo")
    seats["CERES"] = _Broken()
    r = run_game(seats, seed=1, turns=5, start=load_fixture(1), shocks_on=False)
    ceres = [rec.seat("CERES") for rec in r.records]
    assert [x.parse_failure for x in ceres] == [True, True, True, False, False]
    assert [x.policy for x in ceres] == ["broken", "broken", "status_quo", "status_quo", "status_quo"]
    assert r.status == "degraded" and r.degraded_seats == ["CERES"]


def test_cooperative_bots_sign_and_honor_treaties() -> None:
    r = run_bots("cooperative", 6, 10, shocks_on=False)
    statuses = [t.status for t in r.final.treaties]
    assert statuses.count("ended") + statuses.count("active") >= 6
    assert all(not rec.violations for rec in r.records)


def test_aggressor_record_shows_hostility_in_briefings() -> None:
    r = run_bots({"FALKEN": "aggressor", "CERES": "tit_for_tat"}, 2, 3, shocks_on=False)
    first = r.records[0].seat("FALKEN")
    assert {k for k, _ in first.hostile} == {"sanction"}
    # CERES (tit-for-tat) answers FALKEN's sanction with its own once it has seen it
    ceres_sanctions = [r.states[k].sanction[2, 3] for k in range(1, 4)]
    assert any(ceres_sanctions)
    assert np.all(r.final.sanction[3, [0, 1, 2, 4, 5]])
