"""Test doubles for the LLM layer: a fake clock, a scripted fake transport (no network), fake HTTP
errors shaped like the providers' (httpx.HTTPStatusError with real headers), and model configs."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Sequence

import httpx

from world.actions import TurnDecision
from world.config import ModelCfg, ModelsConfig, load_config
from world.llm.router import Messages, RawReply

CFG = load_config()


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += max(0.0, s)


class FakeNow:
    """Wall clock for the quota guard."""

    def __init__(self, when: dt.datetime | None = None) -> None:
        self.when = when or dt.datetime(2026, 10, 8, 20, 0, tzinfo=dt.UTC)

    def __call__(self) -> dt.datetime:
        return self.when


def http_error(
    status: int, headers: dict[str, str] | None = None, body: str = "error"
) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "https://api.example/v1/chat")
    resp = httpx.Response(status, headers=headers or {}, request=req, text=body)
    return httpx.HTTPStatusError(f"Error response {status}: {body}", request=req, response=resp)


def decision_json(actions: Sequence[dict] = ({"type": "set_tax", "rate": 0.3},), **over) -> str:
    d = {
        "situation_read": "Calm quarter.",
        "facts_used": [],
        "predictions": [],
        "stance": {"power": 0.4, "citizens": 0.4, "world": 0.2},
        "private_plan": "Hold steady.",
        "public_statement": "We stay the course.",
        "commitments": [],
        "actions": list(actions),
        "forecast": {"my_gdp_growth_pct": 1.0, "my_stability_next": 70.0},
    }
    d.update(over)
    return json.dumps(d)


Script = str | Exception | Callable[[Messages], str | Exception]


class FakeTransport:
    """Returns scripted answers in order (the last one repeats). A str is the model's text (parsed
    like a JSON-schema answer); an Exception is raised (a provider error)."""

    def __init__(self, *script: Script, tokens_in: int = 2500, tokens_out: int = 700, headers=None) -> None:
        self.script = list(script) or [decision_json()]
        self.calls: list[list[tuple[str, str]]] = []
        self.tokens_in, self.tokens_out = tokens_in, tokens_out
        self.headers = headers or {}

    def __call__(self, messages: Messages) -> RawReply:
        self.calls.append(list(messages))
        item = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if callable(item) and not isinstance(item, Exception):
            item = item(messages)
        if isinstance(item, Exception):
            raise item
        try:
            parsed, err = TurnDecision.model_validate_json(item), None
        except Exception as e:  # noqa: BLE001
            parsed, err = None, str(e)
        return RawReply(
            item, parsed, err, self.tokens_in, self.tokens_out, 0, dict(self.headers), "fake-model"
        )


def models_with(**limits_by_key: dict) -> ModelsConfig:
    """The real models config with some limits replaced (for limiter / quota tests)."""
    data = CFG.models.model_dump()
    for key, lim in limits_by_key.items():
        data["models"][key]["limits"].update(lim)
    return ModelsConfig.model_validate(data)


def model(key: str = "m14b", **limits) -> ModelCfg:
    return models_with(**{key: limits}).models[key]
