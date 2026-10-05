"""Config loading and validation: the real files load, and each rule rejects a bad value."""

import copy
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from world.config import (
    COUNTRIES,
    MOVE_ORDER,
    SECTORS,
    Config,
    load_config,
    load_raw,
)

Raw = dict[str, dict]


@pytest.fixture(scope="module")
def raw() -> Raw:
    return load_raw()


def _validate_mutated(raw: Raw, mutate: Callable[[Raw], None]) -> None:
    data = copy.deepcopy(raw)
    mutate(data)
    Config.model_validate(data)


def _country(data: Raw, name: str) -> dict:
    return next(c for c in data["countries"]["countries"] if c["name"] == name)


# ----------------------------------------------------------------------------- happy path


def test_config_loads() -> None:
    cfg = load_config()
    assert tuple(cfg.world.sectors) == SECTORS
    assert tuple(c.name for c in cfg.countries.countries) == COUNTRIES
    assert tuple(cfg.world.move_order) == MOVE_ORDER
    assert cfg.world.random_slot_country == "EVERMERE"
    assert set(cfg.models.models) == {"m3b", "m8b", "m14b", "g31", "g35"}


def test_spot_values_match_claude_md() -> None:
    cfg = load_config()
    assert cfg.countries.by_name("DORNE").productivity["ENERGY"] == 2.5
    assert cfg.countries.by_name("EVERMERE").n_firms["TECH"] == 1
    assert cfg.countries.by_name("FALKEN").initial_trust_received == 0.5
    assert cfg.countries.by_name("AURELIA").consumption_shares["SERVICES"] == 0.36
    assert cfg.world.production.exponents["GOODS"].gamma == 0.35
    assert cfg.world.trade.sigma_trade == 3.0
    assert cfg.world.burn_in.turns == 8
    assert (cfg.world.horizon.min_, cfg.world.horizon.max_) == (10, 14)
    assert cfg.escalation.events["impose_sanction"] == 12


def test_model_runnable_flags() -> None:
    models = load_config().models
    assert models.models["m8b"].runnable_for_experiments
    assert models.models["m14b"].runnable_for_experiments
    assert not models.models["m3b"].runnable_for_experiments  # untested
    assert not models.models["g31"].runnable_for_experiments  # unverified
    assert models.concurrency("m14b") == models.defaults.max_concurrency == 1
    assert models.concurrency("m3b") == 2


# ------------------------------------------------------------------------ rule violations

BAD_CASES: dict[str, Callable[[Raw], None]] = {
    # beta + gamma <= 1
    "beta_gamma_sum": lambda d: d["world"]["production"]["exponents"]["GOODS"].update(beta=0.7),
    # shares sum to 1
    "default_shares_sum": lambda d: d["world"]["demand"]["consumption_shares_default"].update(FOOD=0.30),
    "country_shares_sum": lambda d: _country(d, "AURELIA")["consumption_shares"].update(SERVICES=0.5),
    "power_weights_sum": lambda d: d["world"]["metrics"]["power_weights"].update(gdp=0.6),
    # probabilities in [0, 1]
    "shock_hazard_gt_1": lambda d: d["shocks"]["shocks"][0].update(hazard_default=1.5),
    "shock_hazard_negative": lambda d: d["shocks"]["shocks"][0]["hazard_by_country"].update(CERES=-0.1),
    "exit_prob": lambda d: d["world"]["firms"].update(exit_prob=1.2),
    "breakup_omega": lambda d: _country(d, "EVERMERE")["params"].update(breakup_omega=1.1),
    "leader_fall_prob": lambda d: _country(d, "CERES")["params"].update(leader_fall_prob=-0.2),
    "spoilage": lambda d: d["world"]["spoilage"].update(FOOD=2.0),
    "trust_start": lambda d: _country(d, "FALKEN").update(initial_trust_received=1.5),
    # exact country names and order
    "country_renamed": lambda d: _country(d, "CERES").update(name="CERESX"),
    "country_missing": lambda d: d["countries"]["countries"].pop(),
    "country_order": lambda d: d["countries"]["countries"].reverse(),
    # exact sector order
    "sector_order": lambda d: d["world"].update(sectors=["ENERGY", "FOOD", "GOODS", "TECH", "SERVICES"]),
    "sector_missing": lambda d: d["world"].update(sectors=["FOOD", "ENERGY", "GOODS", "TECH"]),
    "productivity_keys_order": lambda d: _country(d, "DORNE").update(
        productivity={"ENERGY": 2.5, "FOOD": 0.6, "GOODS": 0.6, "TECH": 0.4, "SERVICES": 0.8}
    ),
    # move order (§4.2)
    "move_order_swapped": lambda d: d["world"].update(
        move_order=["BRONTIA", "DORNE", "CERES", "FALKEN", "AURELIA"]
    ),
    "move_order_with_evermere": lambda d: d["world"].update(move_order=list(COUNTRIES)),
    "random_slot_country": lambda d: d["world"].update(random_slot_country="AURELIA"),
    "random_slot_range": lambda d: d["world"].update(random_slot_range=[0, 6]),
    # other ranges
    "rng_streams": lambda d: d["world"]["rng_streams"].update(SHOCK=9),
    "tax_rate_too_high": lambda d: _country(d, "DORNE").update(tax_rate=0.7),
    "spending_sum": lambda d: _country(d, "DORNE").update(
        spending={"welfare": 0.3, "military": 0.3, "subsidy": 0.1}
    ),
    "horizon_reversed": lambda d: d["world"].update(horizon={"min": 14, "max": 10}),
    "unknown_key_typo": lambda d: d["world"]["trade"].update(sigma_trad=3.0),
    "lite_retries": lambda d: d["world"]["agents"].update(max_retries=2),
    # escalation
    "escalation_bad_level": lambda d: d["escalation"]["events"].update(raise_tariff=10),
    "escalation_reserved_level": lambda d: d["escalation"]["events"].update(raise_tariff=60),
    # models.yaml (Appendix A fields)
    "model_missing_limit_key": lambda d: d["models"]["models"]["m8b"]["limits"].pop("rpd"),
    "model_missing_provider": lambda d: d["models"]["models"]["m8b"].pop("provider"),
    "model_missing_status": lambda d: d["models"]["models"]["g31"].pop("status"),
    "model_bad_status": lambda d: d["models"]["models"]["g31"].update(status="ok"),
    "model_missing_source": lambda d: d["models"]["models"]["m14b"].pop("source"),
    "model_no_latency": lambda d: d["models"]["models"]["m14b"].pop("latency_measured_s"),
    "safety_zero": lambda d: d["models"]["defaults"].update(safety=0.0),
    "safety_above_one": lambda d: d["models"]["defaults"].update(safety=1.2),
    "max_concurrency_zero": lambda d: d["models"]["models"]["m3b"].update(max_concurrency=0),
    "g31_g35_same_id": lambda d: d["models"]["models"]["g35"].update(model_id="gemini-3.1-flash-lite"),
    "daily_window_without_tz": lambda d: d["models"]["models"]["g31"]["quota_window"].pop("tz"),
    "blocked_model_used": lambda d: d["models"]["models"]["m8b"].update(model_id="mistral-small-2603"),
}


@pytest.mark.parametrize("case", sorted(BAD_CASES))
def test_bad_value_is_rejected(raw: Raw, case: str) -> None:
    with pytest.raises(ValidationError):
        _validate_mutated(raw, BAD_CASES[case])


def test_todo_verify_id_is_not_runnable(raw: Raw) -> None:
    data = copy.deepcopy(raw)
    data["models"]["models"]["m8b"]["model_id"] = "TODO_VERIFY"
    cfg = Config.model_validate(data)
    assert not cfg.models.models["m8b"].runnable_for_experiments
