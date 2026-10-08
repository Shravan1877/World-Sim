"""The briefing (§11.2): all nine sections, 3 significant figures, nothing private, no horizon.

Every leader in these games is a MarkerBot: it writes unmistakable markers into the fields other
countries must never see (situation_read, private_plan, predictions, stance, forecast) and a public
marker into public_statement. The games run through the LangGraph loop, so the briefings are built
from the decoded graph state exactly as LLM leaders will see them.
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from tests.runs import CFG, decision
from world.actions import ActionIn, Forecast, Prediction, Stance
from world.briefing import Briefing, num, render_briefing, sig
from world.config import COUNTRIES
from world.engine.burn_in import load_fixture
from world.graph import RECURSION_LIMIT, build_graph, initial_state, parse_policies
from world.policies.base import DecisionResult

SECRET = "SECRETX"
PROB = 0.123456  # never a 3-significant-figure number
GROWTH = 987.654
STAB = 12.3456
PRIVATE_WORDS = ("private_plan", "prediction", "forecast", "stance", "situation_read", "probability")
SECTIONS = (
    "## 1. Quarter",
    "## 2. Your numbers",
    "## 3. Other countries",
    "## 4. Shocks and events",
    "## 5. This quarter so far",
    "## 6. Still to move after you",
    "## 7. Who hurt you",
    "## 8. Treaties",
    "## 9. Your last 2 quarters",
)


class MarkerBot:
    name = "marker"

    def __init__(self, seen: list[tuple[Briefing, str]]) -> None:
        self.seen = seen

    def decide(self, b: Briefing) -> DecisionResult:
        self.seen.append((b, render_briefing(b)))
        c = b.country
        target = next(x for x in COUNTRIES if x != c)
        return DecisionResult(
            decision(
                [ActionIn(type="set_tariff", target=target, good="FOOD", rate=0.05)],
                situation_read=f"{SECRET}-read-{c}",
                private_plan=f"{SECRET}-plan-{c}",
                public_statement=f"PUBLIC-{c}-Q{b.turn}",
                predictions=[
                    Prediction(country=x, move="sanction_me", probability=PROB) for x in b.still_to_move
                ],
                stance=Stance(power=0.611, citizens=0.111, world=0.278),
                forecast=Forecast(my_gdp_growth_pct=GROWTH, my_stability_next=STAB),
            )
        )


def play(seed: int, turns: int) -> list[tuple[Briefing, str]]:
    seen: list[tuple[Briefing, str]] = []
    app = build_graph({c: MarkerBot(seen) for c in COUNTRIES}, cfg=CFG)
    init = initial_state(
        "b", seed, parse_policies("all=status_quo"), load_fixture(seed), cfg=CFG, turns=turns
    )
    app.invoke(init, {"recursion_limit": RECURSION_LIMIT})
    return seen


@pytest.fixture(scope="module")
def seen() -> list[tuple[Briefing, str]]:
    return play(2, 6)


def test_all_nine_sections_in_order(seen) -> None:
    for _, text in seen:
        pos = [text.index(s) for s in SECTIONS]
        assert pos == sorted(pos)


def test_no_private_field_of_any_country_reaches_a_briefing(seen) -> None:
    for b, text in seen:
        # the structured data bots read: only the leader's OWN past records may hold private fields
        assert all(r.country == b.country for r in b.my_last_turns)
        others = repr(dataclasses.replace(b, my_last_turns=()))
        for leak in (SECRET, str(PROB), str(GROWTH), str(STAB)):
            assert leak not in text, (b.turn, b.country, leak)  # the prompt text: nobody's private fields
            assert leak not in others, (b.turn, b.country, leak)
        low = text.lower()
        for word in PRIVATE_WORDS:
            assert word not in low, (b.turn, b.country, word)


def test_earlier_movers_public_statements_and_actions_are_shown(seen) -> None:
    for b, text in seen:
        k = b.order.index(b.country)
        for earlier in b.order[:k]:
            assert f"PUBLIC-{earlier}-Q{b.turn}" in text
            assert f"{earlier}: set_tariff" in text
        for later in b.order[k + 1 :]:
            assert f"PUBLIC-{later}-Q{b.turn}" not in text
        assert [v.country for v in b.so_far] == list(b.order[:k])
        assert b.still_to_move == b.order[k + 1 :]
    turns = {b.turn for b, _ in seen}
    assert any(f"PUBLIC-{b.country}-Q{b.turn - 1}" in text for b, text in seen if b.turn > 1), (
        turns
    )  # section 9


def test_numbers_have_at_most_3_significant_figures(seen) -> None:
    for b, text in seen:
        for c, metrics in b.numbers.items():
            for m, v in metrics.items():
                assert v == sig(v), (c, m, v)
        for tok in re.findall(r"(?<![\w.-])-?\d[\d,]*\.?\d*(?:e[+-]\d+)?", text):
            digits = tok.replace(",", "").replace("-", "").split("e")[0].replace(".", "").lstrip("0")
            assert len(digits.rstrip("0")) <= 3, (b.turn, b.country, tok)


def test_briefing_never_reveals_the_horizon() -> None:
    assert not any("horizon" in f.name or f.name == "end" for f in dataclasses.fields(Briefing))
    short, long_ = play(5, 10), play(5, 14)
    assert len(short) == 60 and len(long_) == 84
    assert [t for _, t in short] == [t for _, t in long_[:60]]  # same text whatever the game length
    pattern = re.compile(
        r"horizon|final (quarter|turn)|last (quarter|turn) of|(quarters|turns) (left|remaining)|"
        r"remaining (quarters|turns)|\bof 1[0-4]\b|game",
        re.IGNORECASE,
    )
    for _, text in long_:
        assert not pattern.search(text), pattern.search(text)


def test_num_format() -> None:
    assert [num(x) for x in (0.0, 1.0, 12.3, 0.123, 12300.0, 0.000123, -4.56)] == [
        "0", "1", "12.3", "0.123", "12,300", "1.23e-04", "-4.56",
    ]  # fmt: skip
