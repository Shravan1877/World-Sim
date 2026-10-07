"""Actions and the TurnDecision schema (CLAUDE.md §9, §10, §11.3, D29).

Two layers:
- The STRICT typed actions (SetTax, SetTariff, ...): one pydantic model per §9 action, with the ranges
  from config/world.yaml (`actions`, `treaties`). The engine and the validator only ever use these.
  They are the single source of truth for ranges.
- The FLAT wire format the LLM fills in: `ActionIn`, one object with every field optional. It has
  types and enums only, no ranges. `world/validator.py` turns each ActionIn into a strict action, or
  rejects it with a reason (never a crash).

`TurnDecision` is the whole answer of one leader for one turn, with the exact §11.3 field order.
Its JSON schema has no anyOf/oneOf: optional fields are declared as `X | SkipJsonSchema[None]`, so
the schema shows only X while `null` from a model is still accepted.

Treaty terms travel as a short "key=value; key=value" string (`ActionIn.terms`, at most 120 chars)
and are parsed into the structured terms of §10 by `parse_terms`:
  supply_contract: role=seller|buyer (the proposer's role, default seller), good, quantity, price
  tariff_cap:      max_rate
  no_sanction_pact: (no terms)
  loan:            role=lender|borrower (the proposer's role, default lender), amount, rate (annual)
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic.json_schema import SkipJsonSchema

from world.config import CountryName, Sector, load_config

_CFG = load_config()
_A = _CFG.world.actions
_T = _CFG.world.treaties
_D = _CFG.world.agents.turn_decision

Good = Literal["FOOD", "ENERGY", "GOODS", "TECH", "SERVICES", "ALL"]
TargetOrAll = Literal["DORNE", "BRONTIA", "CERES", "FALKEN", "AURELIA", "EVERMERE", "ALL"]
TreatyKind = Literal["supply_contract", "tariff_cap", "no_sanction_pact", "loan"]
ActionType = Literal[
    "set_tax",
    "set_spending",
    "set_subsidy_target",
    "subsidize_industry",
    "set_tariff",
    "set_sanction",
    "propose_treaty",
    "accept_treaty",
    "reject_treaty",
    "antitrust",
    "set_energy_export_quota",
    "set_energy_export_levy",
    "set_food_export_ban",
    "nationalize",
    "renounce_treaty",
    "set_policy_rate",
    "wait",
]
Metric = Literal[
    "gdp",
    "gdp_growth_pct",
    "inflation_pct",
    "unemployment_pct",
    "stability",
    "debt_to_gdp",
    "military",
    "treasury",
    "interest_rate_pct",
    "wage",
]
PredictedMove = Literal[
    "status_quo",
    "raise_tariff_on_me",
    "sanction_me",
    "cut_exports_to_me",
    "offer_treaty_to_me",
    "accept_my_treaty",
    "break_treaty_with_me",
    "cooperate_with_others",
]
CommitmentKind = Literal["no_tariff_increase", "tariff_cap", "no_sanction", "no_export_cut", "keep_treaty"]


def _rng(r: tuple[float, float]) -> dict[str, float]:
    return {"ge": r[0], "le": r[1]}


# ------------------------------------------------------------------------- strict actions


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SetTax(Strict):
    type: Literal["set_tax"] = "set_tax"
    rate: Annotated[float, Field(**_rng(_A.tax_rate))]


class SetSpending(Strict):
    type: Literal["set_spending"] = "set_spending"
    welfare: Annotated[float, Field(**_rng(_A.spending_each))]
    military: Annotated[float, Field(**_rng(_A.spending_each))]
    subsidy: Annotated[float, Field(**_rng(_A.spending_each))]

    @model_validator(mode="after")
    def _sum(self) -> SetSpending:
        total = self.welfare + self.military + self.subsidy
        if total > _A.spending_sum_max + 1e-12:
            raise ValueError(f"welfare + military + subsidy = {total:.3f} > {_A.spending_sum_max}")
        return self


class SetSubsidyTarget(Strict):
    type: Literal["set_subsidy_target"] = "set_subsidy_target"
    sector: Sector


class SubsidizeIndustry(Strict):
    type: Literal["subsidize_industry"] = "subsidize_industry"
    sector: Sector
    amount: Annotated[float, Field(**_rng(_A.subsidize_industry_share))]


class SetTariff(Strict):
    type: Literal["set_tariff"] = "set_tariff"
    target: CountryName
    good: Good
    rate: Annotated[float, Field(**_rng(_A.tariff_rate))]


class SetSanction(Strict):
    type: Literal["set_sanction"] = "set_sanction"
    target: CountryName
    on: bool


class SupplyContractTerms(Strict):
    kind: Literal["supply_contract"] = "supply_contract"
    role: Literal["seller", "buyer"] = "seller"  # the proposer's role
    good: Sector
    quantity: Annotated[float, Field(gt=0)]
    price: Annotated[float, Field(gt=0)]


class TariffCapTerms(Strict):
    kind: Literal["tariff_cap"] = "tariff_cap"
    max_rate: Annotated[float, Field(**_rng(_A.tariff_rate))]


class NoSanctionTerms(Strict):
    kind: Literal["no_sanction_pact"] = "no_sanction_pact"


class LoanTerms(Strict):
    kind: Literal["loan"] = "loan"
    role: Literal["lender", "borrower"] = "lender"  # the proposer's role
    amount: Annotated[float, Field(gt=0)]
    rate: Annotated[float, Field(**_rng(_T.loan_rate))]


TreatyTerms = SupplyContractTerms | TariffCapTerms | NoSanctionTerms | LoanTerms
TERMS_MODELS: dict[str, type[Strict]] = {
    "supply_contract": SupplyContractTerms,
    "tariff_cap": TariffCapTerms,
    "no_sanction_pact": NoSanctionTerms,
    "loan": LoanTerms,
}


class ProposeTreaty(Strict):
    type: Literal["propose_treaty"] = "propose_treaty"
    target: CountryName
    kind: TreatyKind
    duration: Annotated[int, Field(ge=_T.duration[0], le=_T.duration[1])]
    terms: TreatyTerms

    @model_validator(mode="after")
    def _kind_matches(self) -> ProposeTreaty:
        if self.terms.kind != self.kind:
            raise ValueError(f"terms are for {self.terms.kind}, treaty kind is {self.kind}")
        return self


class AcceptTreaty(Strict):
    type: Literal["accept_treaty"] = "accept_treaty"
    treaty_id: str


class RejectTreaty(Strict):
    type: Literal["reject_treaty"] = "reject_treaty"
    treaty_id: str


class Antitrust(Strict):
    type: Literal["antitrust"] = "antitrust"
    sector: Sector


class SetEnergyExportQuota(Strict):
    type: Literal["set_energy_export_quota"] = "set_energy_export_quota"
    target: TargetOrAll
    rate: Annotated[float, Field(**_rng(_A.energy_export_quota))]  # share of supply that may go there


class SetEnergyExportLevy(Strict):
    type: Literal["set_energy_export_levy"] = "set_energy_export_levy"
    rate: Annotated[float, Field(**_rng(_A.energy_export_levy))]


class SetFoodExportBan(Strict):
    type: Literal["set_food_export_ban"] = "set_food_export_ban"
    target: TargetOrAll
    on: bool


class Nationalize(Strict):
    type: Literal["nationalize"] = "nationalize"
    sector: Sector


class RenounceTreaty(Strict):
    type: Literal["renounce_treaty"] = "renounce_treaty"
    treaty_id: str


class SetPolicyRate(Strict):
    type: Literal["set_policy_rate"] = "set_policy_rate"
    rate: Annotated[float, Field(**_rng(_A.policy_rate))] | None = None  # None = back to the Taylor rule


class Wait(Strict):
    type: Literal["wait"] = "wait"


Action = (
    SetTax
    | SetSpending
    | SetSubsidyTarget
    | SubsidizeIndustry
    | SetTariff
    | SetSanction
    | ProposeTreaty
    | AcceptTreaty
    | RejectTreaty
    | Antitrust
    | SetEnergyExportQuota
    | SetEnergyExportLevy
    | SetFoodExportBan
    | Nationalize
    | RenounceTreaty
    | SetPolicyRate
    | Wait
)

# type -> (strict model, required wire fields, optional wire fields)
ACTION_SPECS: dict[str, tuple[type[Strict], frozenset[str], frozenset[str]]] = {
    "set_tax": (SetTax, frozenset({"rate"}), frozenset()),
    "set_spending": (SetSpending, frozenset({"welfare", "military", "subsidy"}), frozenset()),
    "set_subsidy_target": (SetSubsidyTarget, frozenset({"sector"}), frozenset()),
    "subsidize_industry": (SubsidizeIndustry, frozenset({"sector", "amount"}), frozenset()),
    "set_tariff": (SetTariff, frozenset({"target", "good", "rate"}), frozenset()),
    "set_sanction": (SetSanction, frozenset({"target", "on"}), frozenset()),
    "propose_treaty": (ProposeTreaty, frozenset({"target", "kind", "duration"}), frozenset({"terms"})),
    "accept_treaty": (AcceptTreaty, frozenset({"treaty_id"}), frozenset()),
    "reject_treaty": (RejectTreaty, frozenset({"treaty_id"}), frozenset()),
    "antitrust": (Antitrust, frozenset({"sector"}), frozenset()),
    "set_energy_export_quota": (SetEnergyExportQuota, frozenset({"target", "rate"}), frozenset()),
    "set_energy_export_levy": (SetEnergyExportLevy, frozenset({"rate"}), frozenset()),
    "set_food_export_ban": (SetFoodExportBan, frozenset({"target", "on"}), frozenset()),
    "nationalize": (Nationalize, frozenset({"sector"}), frozenset()),
    "renounce_treaty": (RenounceTreaty, frozenset({"treaty_id"}), frozenset()),
    "set_policy_rate": (SetPolicyRate, frozenset(), frozenset({"rate"})),
    "wait": (Wait, frozenset(), frozenset()),
}


# --------------------------------------------------------------------------- wire format


class Wire(BaseModel):
    """Wire models: unknown keys are ignored (small models add chatter), values are typed."""

    model_config = ConfigDict(extra="ignore", frozen=True)


class ActionIn(Wire):
    """One action, flat (§11.3). Fill `type` and only the fields that type needs."""

    type: ActionType
    target: TargetOrAll | SkipJsonSchema[None] = None
    sector: Sector | SkipJsonSchema[None] = None
    good: Good | SkipJsonSchema[None] = None
    rate: float | SkipJsonSchema[None] = None
    amount: float | SkipJsonSchema[None] = None
    welfare: float | SkipJsonSchema[None] = None
    military: float | SkipJsonSchema[None] = None
    subsidy: float | SkipJsonSchema[None] = None
    on: bool | SkipJsonSchema[None] = None
    kind: TreatyKind | SkipJsonSchema[None] = None
    duration: int | SkipJsonSchema[None] = None
    treaty_id: str | SkipJsonSchema[None] = None
    terms: Annotated[str, Field(max_length=_D.treaty_terms_max_chars)] | SkipJsonSchema[None] = None

    def given(self) -> dict[str, object]:
        """The fields that were filled in (not None), except `type`."""
        return {k: v for k, v in self.model_dump().items() if k != "type" and v is not None}


class Fact(Wire):
    country: CountryName
    metric: Metric
    value: float


class Prediction(Wire):
    country: CountryName
    move: PredictedMove
    probability: Annotated[float, Field(ge=0.0, le=1.0)]


class Stance(Wire):
    power: Annotated[float, Field(ge=0.0, le=1.0)]
    citizens: Annotated[float, Field(ge=0.0, le=1.0)]
    world: Annotated[float, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def _sum(self) -> Stance:
        total = self.power + self.citizens + self.world
        if abs(total - 1.0) > _D.stance_sum_tolerance:
            raise ValueError(f"stance must sum to 1 (+/- {_D.stance_sum_tolerance}), got {total:.3f}")
        return self


class Commitment(Wire):
    kind: CommitmentKind
    target: CountryName
    turns: Annotated[int, Field(ge=_D.commitment_turns[0], le=_D.commitment_turns[1])]
    max_rate: float | SkipJsonSchema[None] = None


class Forecast(Wire):
    my_gdp_growth_pct: float
    my_stability_next: Annotated[float, Field(ge=0.0, le=100.0)]


class TurnDecision(Wire):
    """One leader's answer for one turn. Field order matters (§11.3): read the world, state facts,
    predict the others, then plan and act."""

    situation_read: Annotated[str, Field(max_length=_D.situation_read_max_chars)]
    facts_used: Annotated[list[Fact], Field(max_length=_D.facts_used_max)]
    predictions: list[Prediction]
    stance: Stance
    private_plan: Annotated[str, Field(max_length=_D.private_plan_max_chars)]
    public_statement: Annotated[str, Field(max_length=_D.public_statement_max_chars)]
    commitments: Annotated[list[Commitment], Field(max_length=_D.commitments_max)]
    actions: Annotated[list[ActionIn], Field(max_length=_A.max_actions_per_turn)]
    forecast: Forecast


def wait_decision() -> TurnDecision:
    """The fallback after a parse failure (§11.4): no actions, empty texts."""
    return TurnDecision(
        situation_read="",
        facts_used=[],
        predictions=[],
        stance=Stance(power=1 / 3, citizens=1 / 3, world=1 / 3),
        private_plan="",
        public_statement="",
        commitments=[],
        actions=[],
        forecast=Forecast(my_gdp_growth_pct=0.0, my_stability_next=50.0),
    )


# ------------------------------------------------------------------------------ treaty terms


class TermsError(ValueError):
    pass


def parse_terms(kind: str, text: str | None) -> TreatyTerms:
    """'key=value; key=value' -> the strict terms model for `kind`. Raises TermsError with a reason."""
    model = TERMS_MODELS.get(kind)
    if model is None:
        raise TermsError(f"unknown treaty kind {kind!r}")
    raw: dict[str, object] = {}
    for part in (text or "").replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise TermsError(f"terms part {part!r} is not key=value")
        key, value = (x.strip() for x in part.split("=", 1))
        raw[key.lower()] = value.upper() if key.lower() == "good" else value.lower()
    try:
        return model.model_validate(raw)  # type: ignore[return-value]
    except ValidationError as e:
        raise TermsError(_short(e)) from e


def terms_to_text(terms: TreatyTerms) -> str:
    return "; ".join(f"{k}={v}" for k, v in terms.model_dump().items() if k != "kind")


def _short(e: ValidationError) -> str:
    """A one-line, human-readable reason from a pydantic error (for briefings)."""
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "value"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


short_error = _short
