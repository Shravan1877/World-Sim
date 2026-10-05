"""Move order (§4.2): EVERMERE's slot comes from the ORDER stream; the other five keep their order."""

from collections import Counter

from world.order import evermere_slot, move_order
from world.rng import RngBundle, Stream, make_rng


def test_order_rule_over_many_turns() -> None:
    rng = RngBundle(11)
    slots = Counter()
    for turn in range(1, 601):
        order = move_order(rng, turn)
        assert sorted(order) == sorted(["DORNE", "BRONTIA", "CERES", "FALKEN", "AURELIA", "EVERMERE"])
        assert [c for c in order if c != "EVERMERE"] == ["DORNE", "BRONTIA", "CERES", "FALKEN", "AURELIA"]
        assert order.index("AURELIA") in (4, 5)
        slot = evermere_slot(rng, turn)
        assert order.index("EVERMERE") == slot - 1
        assert slot == int(make_rng(11, turn, Stream.ORDER).integers(1, 7))
        slots[slot] += 1
    assert set(slots) == {1, 2, 3, 4, 5, 6}
