"""All randomness: seeded numpy generators per (seed, turn, stream, ids) (CLAUDE.md §8).

This is the ONLY module allowed to create random generators. Every draw in the project is
keyed by (seed, turn, stream, *ids), so the same seed always gives the same dice, and two
different uses (e.g. a shock draw and a firm-entry draw) never share a stream.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

Generator = np.random.Generator  # the type of every generator; import it from here, not numpy


class Stream(IntEnum):
    """Named random streams (§8). The numbers are part of the experiment design; never change them."""

    SHOCK = 1
    ORDER = 2
    HORIZON = 3
    FIRMS = 4
    BOT = 5


def stable_id(name: str) -> int:
    """Turn a string id (e.g. a shock id) into a stable non-negative int.

    Python's built-in hash() changes between processes, so we use sha256 instead.
    """
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")


def make_rng(seed: int, turn: int, stream: Stream, *ids: int | str) -> np.random.Generator:
    """Return the generator for key (seed, turn, stream, *ids).

    String ids are converted with stable_id(). All parts must be non-negative integers.
    """
    if not isinstance(stream, Stream):
        raise TypeError(f"stream must be a Stream, got {stream!r}")
    key = [seed, turn, int(stream)] + [stable_id(x) if isinstance(x, str) else x for x in ids]
    for part in key:
        if not isinstance(part, int | np.integer) or isinstance(part, bool) or part < 0:
            raise ValueError(f"rng key parts must be non-negative ints, got {key}")
    return np.random.default_rng([int(p) for p in key])


@dataclass(frozen=True)
class RngBundle:
    """A seed plus a way to get keyed generators. Passed into turn_start() and step()."""

    seed: int

    def gen(self, turn: int, stream: Stream, *ids: int | str) -> np.random.Generator:
        return make_rng(self.seed, turn, stream, *ids)
