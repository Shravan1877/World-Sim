"""Move order inside a turn (CLAUDE.md §4.2).

The five role-ordered countries keep their relative order: DORNE, BRONTIA, CERES, FALKEN, AURELIA.
EVERMERE is inserted at a random slot 1..6, drawn from the ORDER stream:
    slot = rng(seed, turn, ORDER).integers(1, 7)
So AURELIA is always 5th or 6th.
"""

from __future__ import annotations

from world.config import MOVE_ORDER, RANDOM_SLOT_COUNTRY
from world.rng import RngBundle, Stream


def evermere_slot(rng: RngBundle, turn: int) -> int:
    """EVERMERE's 1-based seat this turn."""
    return int(rng.gen(turn, Stream.ORDER).integers(1, len(MOVE_ORDER) + 2))


def move_order(rng: RngBundle, turn: int) -> tuple[str, ...]:
    order = list(MOVE_ORDER)
    order.insert(evermere_slot(rng, turn) - 1, RANDOM_SLOT_COUNTRY)
    return tuple(order)
