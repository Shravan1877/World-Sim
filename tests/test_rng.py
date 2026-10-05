"""RNG (§8): keyed streams are reproducible and independent; no other module creates RNGs."""

import re
from pathlib import Path

import numpy as np
import pytest

from world.rng import RngBundle, Stream, make_rng, stable_id

WORLD = Path(__file__).resolve().parent.parent / "world"


def test_same_key_same_draws() -> None:
    a = make_rng(7, 3, Stream.SHOCK, "harvest_failure", 2).random(5)
    b = make_rng(7, 3, Stream.SHOCK, "harvest_failure", 2).random(5)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "other",
    [
        (8, 3, Stream.SHOCK, "harvest_failure", 2),
        (7, 4, Stream.SHOCK, "harvest_failure", 2),
        (7, 3, Stream.FIRMS, "harvest_failure", 2),
        (7, 3, Stream.SHOCK, "energy_disaster", 2),
        (7, 3, Stream.SHOCK, "harvest_failure", 3),
    ],
)
def test_any_key_change_gives_different_draws(other: tuple) -> None:
    base = make_rng(7, 3, Stream.SHOCK, "harvest_failure", 2).random(5)
    assert not np.array_equal(base, make_rng(*other).random(5))


def test_order_draw_matches_claude_md_form() -> None:
    slot = make_rng(1, 1, Stream.ORDER).integers(1, 7)
    assert 1 <= slot <= 6
    assert slot == np.random.default_rng([1, 1, 2]).integers(1, 7)


def test_bundle_and_stable_id() -> None:
    bundle = RngBundle(seed=5)
    expected = make_rng(5, 2, Stream.BOT, 1).random(3)
    np.testing.assert_array_equal(bundle.gen(2, Stream.BOT, 1).random(3), expected)
    assert stable_id("pandemic") == stable_id("pandemic") != stable_id("unrest")
    assert [s.value for s in Stream] == [1, 2, 3, 4, 5]


def test_bad_keys_rejected() -> None:
    with pytest.raises(ValueError):
        make_rng(-1, 0, Stream.SHOCK)
    with pytest.raises(TypeError):
        make_rng(1, 0, 1)  # type: ignore[arg-type]


FORBIDDEN = re.compile(
    r"default_rng|np\.random\.|numpy\.random|^\s*import random|^\s*from random|RandomState|SeedSequence",
    re.M,
)


def test_only_rng_module_creates_generators() -> None:
    offenders = [
        str(p.relative_to(WORLD))
        for p in WORLD.rglob("*.py")
        if p.name != "rng.py" and FORBIDDEN.search(p.read_text())
    ]
    assert offenders == []


def test_engine_imports_no_llm_or_orchestration() -> None:
    bad = re.compile(
        r"^\s*(from|import)\s+(world\.llm|world\.policies|world\.graph|langchain|langgraph|deepagents)",
        re.M,
    )
    offenders = [p.name for p in (WORLD / "engine").rglob("*.py") if bad.search(p.read_text())]
    assert offenders == []
