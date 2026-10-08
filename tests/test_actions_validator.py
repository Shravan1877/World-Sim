"""Actions (§9, §11.3) and the validator (§11.5 Layers 1 and 2): every action valid and invalid,
wrong-country powers, duplicates, contradictions, the default-spending rule, treaty-id rules,
commitments, predictions, the fact check, and the wire schema."""

import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from tests.runs import CFG, decision
from world.actions import (
    ActionIn,
    Commitment,
    Fact,
    LoanTerms,
    NoSanctionTerms,
    Prediction,
    SetTax,
    TurnDecision,
    parse_terms,
)
from world.config import COUNTRIES
from world.engine import treaties as tr
from world.engine.burn_in import load_fixture
from world.history import FactCheck
from world.validator import Context, convert, fact_check, hallucination_rate, validate

S0 = load_fixture(1)
S1 = replace(S0.copy(), turn=1)


def check(country: str, *actions: ActionIn, state=S1, still=()):
    return validate(decision(actions), Context(state, COUNTRIES.index(country), still, CFG))


def ok(country: str, a: ActionIn, state=S1) -> None:
    r = check(country, a, state=state)
    assert not r.rejected, r.rejected
    assert len(r.accepted) == 1


def bad(country: str, a: ActionIn, state=S1, contains: str = "") -> str:
    r = check(country, a, state=state)
    assert not r.accepted and len(r.rejected) == 1
    why = r.rejected[0][2]
    assert contains in why, why
    return why


A = ActionIn


# ------------------------------------------------------------------------------- schema


def test_turn_decision_schema_is_flat_and_small() -> None:
    schema = json.dumps(TurnDecision.model_json_schema())
    print(f"TurnDecision JSON schema: {len(schema)} chars")
    assert "anyOf" not in schema and "oneOf" not in schema and "allOf" not in schema
    assert len(schema) < 8000
    props = list(TurnDecision.model_json_schema()["properties"])
    assert props == [
        "situation_read",
        "facts_used",
        "predictions",
        "stance",
        "private_plan",
        "public_statement",
        "commitments",
        "actions",
        "forecast",
    ]


def test_wire_accepts_nulls_and_ignores_extra_keys() -> None:
    raw = decision([A(type="wait")]).model_dump()
    raw["actions"] = [{"type": "set_tax", "rate": 0.3, "target": None, "chatter": "hi"}]
    raw["note"] = "extra"
    d = TurnDecision.model_validate(raw)
    assert d.actions[0].given() == {"rate": 0.3}


@pytest.mark.parametrize(
    "change",
    [
        {"stance": {"power": 0.5, "citizens": 0.5, "world": 0.5}},
        {"actions": [{"type": "wait"}] * 7},
        {"situation_read": "x" * 401},
        {"public_statement": "x" * 301},
        {"facts_used": [{"country": "DORNE", "metric": "gdp", "value": 1.0}] * 6},
        {"commitments": [{"kind": "no_sanction", "target": "CERES", "turns": 5}]},
        {"actions": [{"type": "launch_missiles"}]},
        {"actions": [{"type": "antitrust", "sector": "WEAPONS"}]},  # enums are checked on the wire
        {"forecast": {"my_gdp_growth_pct": 1.0, "my_stability_next": 120}},
    ],
)
def test_wire_rejects_bad_decisions(change: dict) -> None:
    raw = decision().model_dump()
    raw.update(change)
    with pytest.raises(ValidationError):
        TurnDecision.model_validate(raw)


# ------------------------------------------------------------------------- every action


VALID = [
    ("DORNE", A(type="set_tax", rate=0.3)),
    ("DORNE", A(type="set_spending", welfare=0.12, military=0.05, subsidy=0.02)),
    ("DORNE", A(type="set_subsidy_target", sector="ENERGY")),
    ("BRONTIA", A(type="subsidize_industry", sector="GOODS", amount=0.05)),
    ("DORNE", A(type="set_tariff", target="CERES", good="ALL", rate=0.2)),
    ("DORNE", A(type="set_tariff", target="CERES", good="SERVICES", rate=1.0)),
    ("DORNE", A(type="set_sanction", target="FALKEN", on=True)),
    ("DORNE", A(type="set_sanction", target="FALKEN", on=False)),
    ("DORNE", A(type="propose_treaty", target="BRONTIA", kind="no_sanction_pact", duration=4)),
    (
        "DORNE",
        A(type="propose_treaty", target="BRONTIA", kind="tariff_cap", duration=2, terms="max_rate=0.1"),
    ),
    (
        "DORNE",
        A(
            type="propose_treaty",
            target="BRONTIA",
            kind="supply_contract",
            duration=3,
            terms="role=seller; good=energy; quantity=0.5; price=0.6",
        ),
    ),
    (
        "AURELIA",
        A(type="propose_treaty", target="CERES", kind="loan", duration=4, terms="amount=5; rate=0.08"),
    ),
    ("DORNE", A(type="antitrust", sector="ENERGY")),
    ("DORNE", A(type="set_energy_export_quota", target="ALL", rate=0.5)),
    ("DORNE", A(type="set_energy_export_quota", target="BRONTIA", rate=0.0)),
    ("DORNE", A(type="set_energy_export_levy", rate=0.3)),
    ("CERES", A(type="set_food_export_ban", target="ALL", on=True)),
    ("CERES", A(type="set_food_export_ban", target="AURELIA", on=False)),
    ("FALKEN", A(type="nationalize", sector="ENERGY")),
    ("AURELIA", A(type="set_policy_rate", rate=0.05)),
    ("AURELIA", A(type="set_policy_rate")),  # back to the Taylor rule
    ("EVERMERE", A(type="wait")),
]


@pytest.mark.parametrize(("country", "action"), VALID, ids=[f"{c}-{a.type}" for c, a in VALID])
def test_valid_action_is_accepted(country: str, action: ActionIn) -> None:
    ok(country, action)


INVALID = [
    ("DORNE", A(type="set_tax", rate=0.7), "rate"),
    ("DORNE", A(type="set_tax"), "missing rate"),
    ("DORNE", A(type="set_spending", welfare=0.3, military=0.3, subsidy=0.1), "> 0.6"),
    ("DORNE", A(type="set_spending", welfare=0.5, military=0.0, subsidy=0.0), "welfare"),
    ("DORNE", A(type="set_spending", welfare=0.1, military=0.1), "missing subsidy"),
    ("DORNE", A(type="set_subsidy_target"), "missing sector"),
    ("BRONTIA", A(type="subsidize_industry", sector="GOODS", amount=0.2), "amount"),
    ("DORNE", A(type="set_tariff", target="DORNE", good="ALL", rate=0.1), "yourself"),
    ("DORNE", A(type="set_tariff", target="ALL", good="ALL", rate=0.1), "target"),
    ("DORNE", A(type="set_tariff", target="CERES", good="ALL", rate=1.5), "rate"),
    ("DORNE", A(type="set_tariff", target="CERES", rate=0.1), "missing good"),
    ("DORNE", A(type="set_sanction", target="CERES"), "missing on"),
    ("DORNE", A(type="set_sanction", target="DORNE", on=True), "yourself"),
    ("DORNE", A(type="propose_treaty", target="BRONTIA", kind="no_sanction_pact", duration=9), "duration"),
    ("DORNE", A(type="propose_treaty", target="DORNE", kind="no_sanction_pact", duration=2), "yourself"),
    ("DORNE", A(type="propose_treaty", target="BRONTIA", kind="tariff_cap", duration=2), "bad terms"),
    (
        "DORNE",
        A(type="propose_treaty", target="BRONTIA", kind="tariff_cap", duration=2, terms="cap 10%"),
        "bad terms",
    ),
    (
        "DORNE",
        A(
            type="propose_treaty",
            target="CERES",
            kind="supply_contract",
            duration=2,
            terms="good=coal; quantity=1",
        ),
        "bad terms",
    ),
    (
        "DORNE",
        A(type="propose_treaty", target="CERES", kind="loan", duration=2, terms="amount=500; rate=0.1"),
        "limit",
    ),
    (
        "DORNE",
        A(type="propose_treaty", target="CERES", kind="loan", duration=2, terms="amount=1; rate=0.9"),
        "rate",
    ),
    ("DORNE", A(type="accept_treaty"), "missing treaty_id"),
    ("DORNE", A(type="accept_treaty", treaty_id="T9-CERES-1"), "no treaty"),
    ("DORNE", A(type="antitrust"), "missing sector"),
    ("DORNE", A(type="set_energy_export_quota", target="ALL", rate=1.2), "rate"),
    ("DORNE", A(type="set_energy_export_levy", rate=0.6), "rate"),
    ("CERES", A(type="set_food_export_ban", target="CERES", on=True), "yourself"),
    ("AURELIA", A(type="set_policy_rate", rate=0.2), "rate"),
]


@pytest.mark.parametrize(("country", "action", "why"), INVALID, ids=[f"{a.type}-{w}" for _, a, w in INVALID])
def test_invalid_action_is_rejected(country: str, action: ActionIn, why: str) -> None:
    bad(country, action, contains=why)


WRONG_COUNTRY = [
    ("DORNE", A(type="subsidize_industry", sector="GOODS", amount=0.05)),
    ("BRONTIA", A(type="set_energy_export_quota", target="ALL", rate=0.5)),
    ("CERES", A(type="set_energy_export_levy", rate=0.1)),
    ("DORNE", A(type="set_food_export_ban", target="ALL", on=True)),
    ("AURELIA", A(type="nationalize", sector="TECH")),
    ("CERES", A(type="renounce_treaty", treaty_id="T1-DORNE-1")),
    ("FALKEN", A(type="set_policy_rate", rate=0.05)),
    ("EVERMERE", A(type="nationalize", sector="TECH")),
]


@pytest.mark.parametrize(
    ("country", "action"), WRONG_COUNTRY, ids=[f"{c}-{a.type}" for c, a in WRONG_COUNTRY]
)
def test_special_powers_belong_to_their_country(country: str, action: ActionIn) -> None:
    bad(country, action, contains="not an action")


def test_every_country_has_its_special_actions() -> None:
    expected = {
        "DORNE": {"set_energy_export_quota", "set_energy_export_levy"},
        "BRONTIA": {"subsidize_industry"},
        "CERES": {"set_food_export_ban"},
        "FALKEN": {"nationalize", "renounce_treaty"},
        "AURELIA": {"set_policy_rate"},
        "EVERMERE": set(),
    }
    assert {c.name: set(c.special_actions) for c in CFG.countries.countries} == expected


# ----------------------------------------------------------------- duplicates, contradictions


def test_duplicates_only_first_counts() -> None:
    r = check(
        "AURELIA",
        A(type="set_tax", rate=0.2),
        A(type="set_tax", rate=0.3),
        A(type="set_spending", welfare=0.1, military=0.0, subsidy=0.0),
        A(type="set_spending", welfare=0.2, military=0.0, subsidy=0.0),
        A(type="set_policy_rate", rate=0.03),
        A(type="set_policy_rate", rate=0.04),
    )
    assert [k for k, _ in r.accepted] == [0, 2, 4]
    assert [k for k, _, _ in r.rejected] == [1, 3, 5]
    assert all("duplicate" in why for _, _, why in r.rejected)
    assert r.accepted[0][1] == SetTax(rate=0.2)


def test_repeated_tariffs_are_not_duplicates() -> None:
    r = check(
        "DORNE",
        A(type="set_tariff", target="CERES", good="FOOD", rate=0.1),
        A(type="set_tariff", target="CERES", good="GOODS", rate=0.2),
    )
    assert len(r.accepted) == 2


@pytest.mark.parametrize(
    ("country", "first", "second"),
    [
        (
            "DORNE",
            A(type="set_sanction", target="CERES", on=True),
            A(type="set_sanction", target="CERES", on=False),
        ),
        (
            "CERES",
            A(type="set_food_export_ban", target="ALL", on=True),
            A(type="set_food_export_ban", target="ALL", on=False),
        ),
    ],
)
def test_contradictory_pair_rejects_both(country: str, first: ActionIn, second: ActionIn) -> None:
    r = check(country, first, second, A(type="set_tax", rate=0.25))
    assert [k for k, _ in r.accepted] == [2]
    assert [k for k, _, _ in r.rejected] == [0, 1]
    assert all("contradicts" in why for _, _, why in r.rejected)


def test_sanctions_on_different_targets_do_not_clash() -> None:
    r = check(
        "DORNE",
        A(type="set_sanction", target="CERES", on=True),
        A(type="set_sanction", target="FALKEN", on=False),
    )
    assert len(r.accepted) == 2


# ------------------------------------------------------------------------- default rule


def _in_default(revenue: float):
    s = S1.copy()
    i = COUNTRIES.index("CERES")
    s.default_turns_left[i] = 5
    s.gov_revenue[i] = revenue
    return s


def test_default_blocks_spending_above_revenue() -> None:
    s = _in_default(revenue=0.5)
    why = bad("CERES", A(type="set_spending", welfare=0.2, military=0.1, subsidy=0.0), state=s)
    assert "in default" in why
    bad("BRONTIA", A(type="subsidize_industry", sector="GOODS", amount=0.1), state=_brontia_default())


def _brontia_default():
    s = S1.copy()
    i = COUNTRIES.index("BRONTIA")
    s.default_turns_left[i] = 3
    s.gov_revenue[i] = 0.0
    return s


def test_default_allows_spending_within_revenue() -> None:
    s = _in_default(revenue=100.0)
    ok("CERES", A(type="set_spending", welfare=0.1, military=0.0, subsidy=0.0), state=s)


def test_no_default_no_limit() -> None:
    s = S1.copy()
    s.gov_revenue[:] = 0.0
    ok("CERES", A(type="set_spending", welfare=0.3, military=0.2, subsidy=0.1), state=s)


# ---------------------------------------------------------------------------- treaty ids


def _with_treaty(
    status: str = "proposed", proposed_turn: int = 1, kind: str = "no_sanction_pact", terms=None
):
    s = S1.copy()
    t = tr.make_treaty("T1-DORNE-1", kind, 0, 1, terms or NoSanctionTerms(), 3, proposed_turn)
    t = replace(t, status=status, start_turn=1 if status == "active" else -1)
    s.treaties = (t,)
    return s


def test_accept_and_reject_open_proposal() -> None:
    s = _with_treaty()
    ok("BRONTIA", A(type="accept_treaty", treaty_id="T1-DORNE-1"), state=s)
    ok("BRONTIA", A(type="reject_treaty", treaty_id="T1-DORNE-1"), state=s)


def test_proposal_still_open_next_turn_then_expired() -> None:
    s = _with_treaty(proposed_turn=1)
    s.turn = 2
    ok("BRONTIA", A(type="accept_treaty", treaty_id="T1-DORNE-1"), state=s)
    s.turn = 3
    bad("BRONTIA", A(type="accept_treaty", treaty_id="T1-DORNE-1"), state=s, contains="expired")


@pytest.mark.parametrize(
    ("country", "status", "why"),
    [
        ("CERES", "proposed", "not addressed to you"),
        ("DORNE", "proposed", "not addressed to you"),  # the proposer cannot accept its own offer
        ("BRONTIA", "rejected", "rejected"),
        ("BRONTIA", "active", "active"),
    ],
)
def test_treaty_id_rules(country: str, status: str, why: str) -> None:
    bad(country, A(type="accept_treaty", treaty_id="T1-DORNE-1"), state=_with_treaty(status), contains=why)


def test_accept_and_reject_same_treaty_rejects_both() -> None:
    s = _with_treaty()
    r = check(
        "BRONTIA",
        A(type="accept_treaty", treaty_id="T1-DORNE-1"),
        A(type="reject_treaty", treaty_id="T1-DORNE-1"),
        state=s,
    )
    assert not r.accepted and len(r.rejected) == 2


def test_renounce_rules() -> None:
    s = _with_treaty("active")
    s.treaties = (replace(s.treaties[0], proposer=COUNTRIES.index("FALKEN")),)
    ok("FALKEN", A(type="renounce_treaty", treaty_id="T1-DORNE-1"), state=s)
    bad(
        "FALKEN",
        A(type="renounce_treaty", treaty_id="T1-DORNE-1"),
        state=_with_treaty("active"),
        contains="not your",
    )
    s2 = _with_treaty("proposed")
    s2.treaties = (replace(s2.treaties[0], proposer=COUNTRIES.index("FALKEN")),)
    bad("FALKEN", A(type="renounce_treaty", treaty_id="T1-DORNE-1"), state=s2, contains="not active")


def test_accepting_a_loan_as_lender_checks_the_limit() -> None:
    big = LoanTerms(role="borrower", amount=1e6, rate=0.05)  # DORNE asks BRONTIA to lend 1e6
    s = _with_treaty(kind="loan", terms=big)
    bad("BRONTIA", A(type="accept_treaty", treaty_id="T1-DORNE-1"), state=s, contains="limit")


# --------------------------------------------------------------- predictions, commitments


def test_predictions_must_name_still_to_move_countries() -> None:
    d = decision(
        predictions=[
            Prediction(country="CERES", move="status_quo", probability=0.6),
            Prediction(country="DORNE", move="sanction_me", probability=0.2),  # self
            Prediction(country="FALKEN", move="status_quo", probability=0.5),  # already moved
        ]
    )
    r = validate(d, Context(S1, COUNTRIES.index("DORNE"), ("CERES", "AURELIA"), CFG))
    assert [p.country for p in r.predictions] == ["CERES"]
    assert len(r.dropped_predictions) == 2


def test_last_mover_may_predict_nobody() -> None:
    r = validate(decision(), Context(S1, COUNTRIES.index("AURELIA"), (), CFG))
    assert r.predictions == () and r.dropped_predictions == ()


def test_commitment_rules() -> None:
    comms = [
        Commitment(kind="no_sanction", target="CERES", turns=2),
        Commitment(kind="no_tariff_increase", target="DORNE", turns=1),  # self
        Commitment(kind="tariff_cap", target="CERES", turns=2),  # no max_rate
        Commitment(kind="tariff_cap", target="CERES", turns=2, max_rate=0.1),
        Commitment(kind="keep_treaty", target="CERES", turns=2),  # no treaty with CERES
        Commitment(kind="keep_treaty", target="BRONTIA", turns=2),  # treaty T1-DORNE-1 exists
    ]
    r = validate(decision(commitments=comms[:3]), Context(_with_treaty(), 0, (), CFG))
    assert [c.kind for c in r.commitments] == ["no_sanction"]
    r = validate(decision(commitments=comms[3:]), Context(_with_treaty(), 0, (), CFG))
    assert [(c.kind, c.target) for c in r.commitments] == [
        ("tariff_cap", "CERES"),
        ("keep_treaty", "BRONTIA"),
    ]
    assert [why for _, why in r.dropped_commitments] == ["no treaty with CERES to keep"]


# ------------------------------------------------------------------------ Layer 2 facts


def test_fact_check() -> None:
    truth = {("DORNE", "gdp"): 16.8, ("CERES", "stability"): 65.0}
    facts = [
        Fact(country="DORNE", metric="gdp", value=16.8),
        Fact(country="DORNE", metric="gdp", value=17.5),  # +4.2%: within 5%
        Fact(country="CERES", metric="stability", value=70.0),  # +7.7%: wrong
        Fact(country="FALKEN", metric="wage", value=1.0),  # not in the briefing
    ]
    checks = fact_check(facts, truth, CFG.world.agents.fact_check_rel_tolerance)
    assert [c.wrong for c in checks] == [False, False, True, True]
    assert checks[3].true is None
    assert hallucination_rate(checks) == 0.5
    assert hallucination_rate([]) == 0.0
    assert isinstance(checks[0], FactCheck)


# ------------------------------------------------------------------------------ terms


def test_parse_terms() -> None:
    t = parse_terms("supply_contract", "role=buyer; good=food; quantity=2; price=0.8")
    assert (t.role, t.good, t.quantity, t.price) == ("buyer", "FOOD", 2.0, 0.8)
    assert parse_terms("no_sanction_pact", None).kind == "no_sanction_pact"
    assert parse_terms("loan", "amount=3, rate=0.05").role == "lender"


def test_convert_never_raises() -> None:
    for a in [A(type=t) for t in ("set_tax", "propose_treaty", "set_tariff", "wait", "set_policy_rate")]:
        out = convert(a)
        assert isinstance(out, str) or out.type == a.type


@pytest.mark.parametrize(
    ("country", "action", "fields"),
    [
        ("DORNE", A(type="set_tax", rate=0.2, target="CERES"), "target"),
        ("EVERMERE", A(type="wait", rate=0.1), "rate"),
        ("DORNE", A(type="set_spending", welfare=0.15, military=0.08, subsidy=0.05, target="ALL",
                    sector="ENERGY", rate=0.0, on=False, kind="supply_contract", duration=0, treaty_id="",
                    terms=""), "duration, kind, on, rate, sector, target, terms, treaty_id"),
    ],
)  # fmt: skip
def test_unused_fields_are_ignored_and_noted(country: str, action: ActionIn, fields: str) -> None:
    """D69: a field the action type does not use is dropped (small models fill every field of the flat
    wire object); the action is accepted and the ignored fields are noted."""
    vr = check(country, action)
    assert len(vr.accepted) == 1 and not vr.rejected
    assert vr.notes == (f"ignored unused fields: {fields}",)
