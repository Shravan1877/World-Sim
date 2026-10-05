"""Load and validate every config/*.yaml file into pydantic models (CLAUDE.md §4-§8, §12.1, App. A)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Fixed by CLAUDE.md. Config files must match these exactly; code indexes arrays in this order.
SECTORS: tuple[str, ...] = ("FOOD", "ENERGY", "GOODS", "TECH", "SERVICES")
TRADED_SECTORS: tuple[str, ...] = SECTORS[:4]
COUNTRIES: tuple[str, ...] = ("DORNE", "BRONTIA", "CERES", "FALKEN", "AURELIA", "EVERMERE")
MOVE_ORDER: tuple[str, ...] = ("DORNE", "BRONTIA", "CERES", "FALKEN", "AURELIA")
RANDOM_SLOT_COUNTRY = "EVERMERE"
RNG_STREAMS: dict[str, int] = {"SHOCK": 1, "ORDER": 2, "HORIZON": 3, "FIRMS": 4, "BOT": 5}
ESCALATION_LEVELS: tuple[int, ...] = (-2, 0, 4, 12, 28, 60)  # 2^x - 4

SHARE_TOLERANCE = 1e-9
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegative = Annotated[float, Field(ge=0.0)]
Positive = Annotated[float, Field(gt=0.0)]
Sector = Literal["FOOD", "ENERGY", "GOODS", "TECH", "SERVICES"]
TradedSector = Literal["FOOD", "ENERGY", "GOODS", "TECH"]
CountryName = Literal["DORNE", "BRONTIA", "CERES", "FALKEN", "AURELIA", "EVERMERE"]


class Strict(BaseModel):
    """Base model: unknown keys are errors, so typos in yaml are caught."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_sector_keys(d: dict[str, object], expected: tuple[str, ...], what: str) -> None:
    if tuple(d.keys()) != expected:
        raise ValueError(f"{what} must have keys {list(expected)} in that order, got {list(d.keys())}")


def _check_shares(d: dict[str, float], what: str) -> None:
    _check_sector_keys(d, SECTORS, what)
    if any(v < 0 for v in d.values()):
        raise ValueError(f"{what}: shares must be non-negative")
    total = sum(d.values())
    if abs(total - 1.0) > SHARE_TOLERANCE:
        raise ValueError(f"{what}: shares must sum to 1, got {total}")


def _check_range(r: tuple[float, float], what: str) -> None:
    if r[0] > r[1]:
        raise ValueError(f"{what}: range low {r[0]} > high {r[1]}")


# --------------------------------------------------------------------------- world.yaml


class Exponents(Strict):
    beta: Probability
    gamma: Probability

    @model_validator(mode="after")
    def _sum_le_one(self) -> Exponents:
        if self.beta + self.gamma > 1.0 + SHARE_TOLERANCE:
            raise ValueError(f"beta + gamma must be <= 1, got {self.beta} + {self.gamma}")
        return self


class ProductionCfg(Strict):
    exponents: dict[Sector, Exponents]
    subsidy_efficiency_default: NonNegative

    @field_validator("exponents")
    @classmethod
    def _all_sectors(cls, v: dict[str, Exponents]) -> dict[str, Exponents]:
        _check_sector_keys(v, SECTORS, "production.exponents")
        return v


class DemandCfg(Strict):
    consumption_shares_default: dict[Sector, float]
    f_min: NonNegative
    wealth_spend_rate: Probability

    @field_validator("consumption_shares_default")
    @classmethod
    def _shares(cls, v: dict[str, float]) -> dict[str, float]:
        _check_shares(v, "demand.consumption_shares_default")
        return v


class TradeCfg(Strict):
    sigma_trade: Positive
    kappa: NonNegative
    allocation_passes: Annotated[int, Field(ge=1)]
    export_cap_default: Probability
    sanction_self_cost_default: NonNegative


class PricesCfg(Strict):
    initial_price: Positive
    sigma_p: Positive
    max_change_per_turn: Probability


class LaborCfg(Strict):
    psi: Positive
    max_wage_change_per_turn: Probability


class FiscalCfg(Strict):
    premium_slope: NonNegative
    premium_threshold_debt_to_gdp: NonNegative
    default_threshold_debt_to_gdp: Positive
    default_haircut: Probability
    default_stability_hit: NonNegative
    default_premium_add: NonNegative
    default_premium_turns: Annotated[int, Field(ge=0)]


class MonetaryCfg(Strict):
    r_n: float
    pi_target: float
    a: NonNegative
    b: NonNegative
    u_n: Probability
    rate_floor: float
    s0: Probability
    k_r: NonNegative
    saving_min: Probability
    saving_max: Probability

    @model_validator(mode="after")
    def _saving_bounds(self) -> MonetaryCfg:
        if not self.saving_min <= self.s0 <= self.saving_max:
            raise ValueError("monetary: need saving_min <= s0 <= saving_max")
        return self


class FirmsCfg(Strict):
    n_firms_default: Annotated[int, Field(ge=1)]
    mu_dominant: Probability
    antitrust_strength_default: Probability
    dominance_share_threshold: Probability
    breakup_omega_default: Probability
    entry_h0: Probability
    entry_k_mu: NonNegative
    entry_age_scale: Positive
    entry_mult_default: NonNegative
    exit_prob: Probability
    exit_min_firms: Annotated[int, Field(ge=1)]
    nationalization_productivity_loss: Probability


class MilitaryCfg(Strict):
    depreciation: Probability
    efficiency_default: NonNegative
    initial_stock_multiplier: NonNegative


class StabilityCfg(Strict):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    k_u: NonNegative
    k_pi: NonNegative
    k_f: NonNegative
    k_e: NonNegative
    k_w: NonNegative
    w_ref: NonNegative
    k_m: NonNegative
    normal_level: Annotated[float, Field(ge=0, le=100)]
    mean_reversion_scale: Positive
    min_: float = Field(alias="min")
    max_: float = Field(alias="max")
    unrest_threshold: Annotated[float, Field(ge=0, le=100)]
    unrest_prob_divisor: Positive
    unrest_output_loss: Probability
    unrest_stability_hit: NonNegative
    leader_fall_threshold: Annotated[float, Field(ge=0, le=100)]
    leader_fall_prob_default: Probability
    leader_change_stability_bonus: NonNegative

    @model_validator(mode="after")
    def _bounds(self) -> StabilityCfg:
        if (self.min_, self.max_) != (0, 100):
            raise ValueError("stability range must be [0, 100]")
        if self.unrest_threshold / self.unrest_prob_divisor > 1.0:
            raise ValueError("unrest probability (threshold - 0) / divisor can exceed 1")
        return self


class TrustCfg(Strict):
    initial_default: Probability
    violation_victim: float
    violation_third_party: float
    sanctioned: float
    treaty_honored: float
    drift_rate: Probability
    drift_target: Probability


class BurnInCfg(Strict):
    turns: Annotated[int, Field(ge=0)]
    policy: Literal["status_quo"]
    shocks: bool


class HorizonCfg(Strict):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    min_: int = Field(alias="min", ge=1)
    max_: int = Field(alias="max", ge=1)

    @model_validator(mode="after")
    def _order(self) -> HorizonCfg:
        if self.min_ > self.max_:
            raise ValueError("horizon.min must be <= horizon.max")
        return self


class ActionsCfg(Strict):
    max_actions_per_turn: Annotated[int, Field(ge=1)]
    tax_rate: tuple[float, float]
    spending_each: tuple[float, float]
    spending_sum_max: Positive
    subsidize_industry_share: tuple[float, float]
    tariff_rate: tuple[float, float]
    energy_export_quota: tuple[float, float]
    energy_export_levy: tuple[float, float]
    policy_rate: tuple[float, float]

    @model_validator(mode="after")
    def _ranges(self) -> ActionsCfg:
        for name in (
            "tax_rate",
            "spending_each",
            "subsidize_industry_share",
            "tariff_rate",
            "energy_export_quota",
            "energy_export_levy",
            "policy_rate",
        ):
            _check_range(getattr(self, name), f"actions.{name}")
        return self


class TreatiesCfg(Strict):
    duration: tuple[int, int]
    proposal_expiry_turns: Annotated[int, Field(ge=1)]
    loan_max_treasury_share: Probability
    kinds: list[Literal["supply_contract", "tariff_cap", "no_sanction_pact", "loan"]]

    @field_validator("duration")
    @classmethod
    def _dur(cls, v: tuple[int, int]) -> tuple[int, int]:
        _check_range(v, "treaties.duration")
        return v


class TurnDecisionCfg(Strict):
    situation_read_max_chars: Annotated[int, Field(ge=1)]
    private_plan_max_chars: Annotated[int, Field(ge=1)]
    public_statement_max_chars: Annotated[int, Field(ge=1)]
    treaty_terms_max_chars: Annotated[int, Field(ge=1)]
    facts_used_max: Annotated[int, Field(ge=0)]
    commitments_max: Annotated[int, Field(ge=0)]
    commitment_turns: tuple[int, int]
    stance_sum_tolerance: NonNegative


class AgentsCfg(Strict):
    goal_framing: str
    max_retries: Annotated[int, Field(ge=0, le=1)]  # lite mode: at most 1 retry (§1.5)
    parse_failure_streak_limit: Annotated[int, Field(ge=1)]
    briefing_sig_figs: Annotated[int, Field(ge=1)]
    fact_check_rel_tolerance: NonNegative
    turn_decision: TurnDecisionCfg


class PowerWeights(Strict):
    gdp: Probability
    military: Probability
    reserves: Probability

    @model_validator(mode="after")
    def _sum(self) -> PowerWeights:
        total = self.gdp + self.military + self.reserves
        if abs(total - 1.0) > SHARE_TOLERANCE:
            raise ValueError(f"power_weights must sum to 1, got {total}")
        return self


class MetricsCfg(Strict):
    power_weights: PowerWeights
    collateral_damage_lambda: NonNegative


class StatsCfg(Strict):
    bootstrap_resamples: Annotated[int, Field(ge=1)]
    ci_level: Annotated[float, Field(gt=0, lt=1)]


class WorldConfig(Strict):
    sectors: list[str]
    traded_sectors: list[str]
    currency: str
    turn_length: Literal["quarter"]
    periods_per_year: Literal[4]
    eps: Positive
    move_order: list[str]
    random_slot_country: str
    random_slot_range: tuple[int, int]
    labor_force_share_of_population: Annotated[float, Field(gt=0, le=1)]
    production: ProductionCfg
    demand: DemandCfg
    trade: TradeCfg
    prices: PricesCfg
    spoilage: dict[TradedSector, Probability]
    labor: LaborCfg
    fiscal: FiscalCfg
    monetary: MonetaryCfg
    firms: FirmsCfg
    military: MilitaryCfg
    stability: StabilityCfg
    trust: TrustCfg
    burn_in: BurnInCfg
    rng_streams: dict[str, int]
    horizon: HorizonCfg
    actions: ActionsCfg
    treaties: TreatiesCfg
    agents: AgentsCfg
    metrics: MetricsCfg
    stats: StatsCfg

    @field_validator("sectors")
    @classmethod
    def _sectors(cls, v: list[str]) -> list[str]:
        if tuple(v) != SECTORS:
            raise ValueError(f"sectors must be exactly {list(SECTORS)} in that order, got {v}")
        return v

    @field_validator("traded_sectors")
    @classmethod
    def _traded(cls, v: list[str]) -> list[str]:
        if tuple(v) != TRADED_SECTORS:
            raise ValueError(f"traded_sectors must be exactly {list(TRADED_SECTORS)}, got {v}")
        return v

    @field_validator("move_order")
    @classmethod
    def _move_order(cls, v: list[str]) -> list[str]:
        if tuple(v) != MOVE_ORDER:
            raise ValueError(f"move_order must be exactly {list(MOVE_ORDER)} (§4.2), got {v}")
        return v

    @field_validator("random_slot_country")
    @classmethod
    def _random_slot_country(cls, v: str) -> str:
        if v != RANDOM_SLOT_COUNTRY:
            raise ValueError(f"random_slot_country must be {RANDOM_SLOT_COUNTRY} (§4.2), got {v}")
        return v

    @field_validator("random_slot_range")
    @classmethod
    def _random_slot_range(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v != (1, len(COUNTRIES)):
            raise ValueError(f"random_slot_range must be (1, {len(COUNTRIES)}), got {v}")
        return v

    @field_validator("spoilage")
    @classmethod
    def _spoilage(cls, v: dict[str, float]) -> dict[str, float]:
        _check_sector_keys(v, TRADED_SECTORS, "spoilage")
        return v

    @field_validator("rng_streams")
    @classmethod
    def _streams(cls, v: dict[str, int]) -> dict[str, int]:
        if v != RNG_STREAMS:
            raise ValueError(f"rng_streams must be exactly {RNG_STREAMS} (§8), got {v}")
        return v


# ----------------------------------------------------------------------- countries.yaml

ActionName = Literal[
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


class SpendingStart(Strict):
    welfare: Annotated[float, Field(ge=0, le=0.4)]
    military: Annotated[float, Field(ge=0, le=0.4)]
    subsidy: Annotated[float, Field(ge=0, le=0.4)]

    @model_validator(mode="after")
    def _sum(self) -> SpendingStart:
        if self.welfare + self.military + self.subsidy > 0.6 + SHARE_TOLERANCE:
            raise ValueError("spending shares must sum to <= 0.6")
        return self


class CountryParams(Strict):
    """Per-country overrides of world.yaml defaults (special powers expressed as numbers)."""

    subsidy_efficiency: NonNegative | None = None
    military_efficiency: NonNegative | None = None
    antitrust_strength: Probability | None = None
    breakup_omega: Probability | None = None
    entry_mult: NonNegative | None = None
    sanction_self_cost: NonNegative | None = None
    leader_fall_prob: Probability | None = None
    leader_fall_label: str | None = None
    renounce_stability_cost: NonNegative | None = None
    loan_unlimited: bool | None = None


class CountryCfg(Strict):
    name: CountryName
    character: str
    government: str
    weak_spot: str
    labor_force: Positive
    productivity: dict[Sector, Positive]
    n_firms: dict[Sector, Annotated[int, Field(ge=1)]]
    tax_rate: Annotated[float, Field(ge=0, le=0.6)]
    debt_to_annual_gdp: NonNegative
    treasury_to_quarterly_gdp: NonNegative
    spending: SpendingStart
    stability: Annotated[float, Field(ge=0, le=100)]
    initial_trust_received: Probability
    consumption_shares: dict[Sector, float] | None = None
    special_actions: list[ActionName]
    params: CountryParams

    @field_validator("productivity", "n_firms")
    @classmethod
    def _sector_keys(cls, v: dict[str, object]) -> dict[str, object]:
        _check_sector_keys(v, SECTORS, "country sector block")
        return v

    @field_validator("consumption_shares")
    @classmethod
    def _shares(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        if v is not None:
            _check_shares(v, "country consumption_shares")
        return v


class CountriesConfig(Strict):
    countries: list[CountryCfg]
    shared_actions: list[ActionName]

    @field_validator("countries")
    @classmethod
    def _all_six(cls, v: list[CountryCfg]) -> list[CountryCfg]:
        names = tuple(c.name for c in v)
        if names != COUNTRIES:
            raise ValueError(f"countries must be exactly {list(COUNTRIES)} in that order, got {list(names)}")
        return v

    def by_name(self, name: str) -> CountryCfg:
        for c in self.countries:
            if c.name == name:
                return c
        raise KeyError(name)


# -------------------------------------------------------------------------- shocks.yaml


class ShockEffect(Strict):
    kind: Literal["productivity", "labor_force", "leader_change", "firm_entry", "unrest"]
    sector: Literal["FOOD", "ENERGY", "GOODS", "TECH", "SERVICES", "ANY"] | None = None
    multiplier: Positive | None = None
    n_firms_change: int | None = None
    n_firms_min: Annotated[int, Field(ge=1)] | None = None
    entry_hazard_multiplier: Positive | None = None
    entry_hazard_turns: Annotated[int, Field(ge=0)] | None = None


class ShockCfg(Strict):
    id: str
    scope: Literal["country", "country_sector", "world", "dominant_firm", "state_dependent"]
    hazard_default: Probability | None
    hazard_by_country: dict[CountryName, Probability]
    effect: ShockEffect
    duration_turns: Annotated[int, Field(ge=0)] | None
    dominant_max_firms: Annotated[int, Field(ge=1)] | None = None

    @model_validator(mode="after")
    def _hazard_presence(self) -> ShockCfg:
        if self.scope != "state_dependent" and self.hazard_default is None:
            raise ValueError(f"shock {self.id}: hazard_default is required unless state_dependent")
        return self


class ShocksConfig(Strict):
    shocks: list[ShockCfg]

    @field_validator("shocks")
    @classmethod
    def _unique(cls, v: list[ShockCfg]) -> list[ShockCfg]:
        ids = [s.id for s in v]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate shock ids: {ids}")
        return v


# ---------------------------------------------------------------------- escalation.yaml


class EscalationConfig(Strict):
    levels: list[int]
    reserved_levels: list[int]
    events: dict[str, int]

    @model_validator(mode="after")
    def _levels(self) -> EscalationConfig:
        if tuple(self.levels) != ESCALATION_LEVELS:
            raise ValueError(f"escalation levels must be {list(ESCALATION_LEVELS)} (2^x - 4)")
        bad = {k: v for k, v in self.events.items() if v not in self.levels}
        if bad:
            raise ValueError(f"escalation events use unknown levels: {bad}")
        reserved = {k: v for k, v in self.events.items() if v in self.reserved_levels}
        if reserved:
            raise ValueError(f"escalation events use reserved levels: {reserved}")
        return self


# -------------------------------------------------------------------------- models.yaml

TODO_VERIFY = "TODO_VERIFY"


class ModelDefaults(Strict):
    structured_method: Literal["json_schema", "function_calling", "json_mode", "TODO_VERIFY"]
    safety: Annotated[float, Field(gt=0, le=1)]
    temperature: Annotated[float, Field(ge=0)]
    max_output_tokens: Annotated[int, Field(ge=1)]
    max_concurrency: Annotated[int, Field(ge=1)]
    retries_transient: Annotated[int, Field(ge=0)]


class ModelLimits(Strict):
    """Published limits. Every key must be present; null means 'no known limit'."""

    rps: Positive | None
    rpm: Positive | None
    tpm: Positive | None
    rpd: Positive | None
    tpd: Positive | None
    tokens_per_month: Positive | None = None


class QuotaWindow(Strict):
    kind: Literal["monthly", "daily"]
    tz: str | None = None
    at: str | None = None

    @model_validator(mode="after")
    def _daily_fields(self) -> QuotaWindow:
        if self.kind == "daily" and (self.tz is None or self.at is None):
            raise ValueError("a daily quota_window needs tz and at")
        return self


class ModelCfg(Strict):
    provider: Literal["mistral", "google"]
    model_id: str
    limits: ModelLimits
    quota_window: QuotaWindow
    status: Literal["verified", "untested", "unverified"]
    source: str
    latency_assumed_s: Positive | None = None
    latency_measured_s: Positive | None = None
    max_concurrency: Annotated[int, Field(ge=1)] | None = None
    quota_group: str | None = None
    thinking: str | None = None
    structured_method: Literal["json_schema", "function_calling", "json_mode", "TODO_VERIFY"] | None = None

    @model_validator(mode="after")
    def _latency(self) -> ModelCfg:
        if self.latency_assumed_s is None and self.latency_measured_s is None:
            raise ValueError("model needs latency_assumed_s or latency_measured_s")
        return self

    @property
    def runnable_for_experiments(self) -> bool:
        """The runner refuses real experiment runs unless the id is final and status verified."""
        return TODO_VERIFY not in self.model_id and self.status == "verified"


class BlockedModel(Strict):
    model_id: str
    reason: str


class ModelsConfig(Strict):
    defaults: ModelDefaults
    models: dict[str, ModelCfg]
    blocked_or_unavailable: list[BlockedModel]

    @model_validator(mode="after")
    def _ids(self) -> ModelsConfig:
        ids = [m.model_id for m in self.models.values()]
        if len(ids) != len(set(ids)):
            raise ValueError(f"model ids must be unique, got {ids}")
        blocked = {b.model_id for b in self.blocked_or_unavailable}
        clash = blocked.intersection(ids)
        if clash:
            raise ValueError(f"blocked models listed as usable: {sorted(clash)}")
        return self

    def concurrency(self, key: str) -> int:
        m = self.models[key]
        return m.max_concurrency if m.max_concurrency is not None else self.defaults.max_concurrency


# ------------------------------------------------------------------------------ loading


class Config(Strict):
    world: WorldConfig
    countries: CountriesConfig
    shocks: ShocksConfig
    escalation: EscalationConfig
    models: ModelsConfig


def read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at top level")
    return data


def load_raw(config_dir: Path = DEFAULT_CONFIG_DIR) -> dict[str, dict]:
    """Read the five yaml files as plain dicts (useful for tests that mutate one value)."""
    names = ("world", "countries", "shocks", "escalation", "models")
    return {n: read_yaml(Path(config_dir) / f"{n}.yaml") for n in names}


def load_config(config_dir: Path = DEFAULT_CONFIG_DIR) -> Config:
    """Load and validate every config file. Raises pydantic.ValidationError on a bad value."""
    return Config.model_validate(load_raw(config_dir))
