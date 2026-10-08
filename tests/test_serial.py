"""world/serial.py: the graph-state encoding is exact and plain (msgpack-safe without custom types)."""

from __future__ import annotations

import numpy as np
import ormsgpack
import pytest

from tests.runs import run_bots
from tests.test_bot_game import mixed
from world.policies.base import DecisionResult
from world.serial import decode, encode


@pytest.fixture(scope="module")
def game():
    return run_bots(mixed(9), 9, 6)


def _plain(x) -> bool:
    if isinstance(x, dict):
        return all(isinstance(k, str) and _plain(v) for k, v in x.items())
    if isinstance(x, list):
        return all(_plain(v) for v in x)
    return x is None or isinstance(x, bool | int | float | str | bytes) and not isinstance(x, np.generic)


def test_world_state_round_trip_is_exact(game) -> None:
    for s in game.states:
        data = encode(s)
        assert _plain(data)
        back = decode(ormsgpack.unpackb(ormsgpack.packb(data)))
        assert back.state_hash() == s.state_hash()
        assert back.ledger.log == s.ledger.log and back.ledger.opening == s.ledger.opening
        assert back.treaties == s.treaties and back.firms == s.firms


def test_records_logs_and_decisions_round_trip(game) -> None:
    for obj in (
        game.records,
        game.logs[-1].events,
        DecisionResult(game.records[-1].seats[0].decision, "raw"),
    ):
        data = encode(obj)
        assert _plain(data)
        assert decode(ormsgpack.unpackb(ormsgpack.packb(data))) == obj


def test_numpy_scalars_and_tuples_keep_their_type() -> None:
    x = (np.float64(0.5), np.int64(3), 0.5, [1, (2, 3)], {("a", 1): np.bool_(True)})
    back = decode(encode(x))
    assert back == x
    assert type(back[0]) is np.float64 and type(back[1]) is np.int64 and type(back[3][1]) is tuple


def test_unknown_class_is_refused() -> None:
    with pytest.raises(ValueError, match="registry"):
        decode({"__dc__": "os.system", "f": {}})
    with pytest.raises(TypeError):
        encode(object())
