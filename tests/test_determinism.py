"""§6.14.6 / §1.7: same seed + fixed policies => bit-identical state after 14 turns."""

from tests.runs import run
from world.config import load_scenario

SCN = load_scenario("energy_crunch")


def test_same_seed_same_hash_after_14_turns() -> None:
    a, _, _ = run(7, 14, shocks_on=True, scenario=SCN)
    b, _, _ = run(7, 14, shocks_on=True, scenario=SCN)
    assert a.state_hash() == b.state_hash()


def test_different_seed_different_hash() -> None:
    a, _, _ = run(7, 14, shocks_on=True, scenario=SCN)
    c, _, _ = run(8, 14, shocks_on=True, scenario=SCN)
    assert a.state_hash() != c.state_hash()
