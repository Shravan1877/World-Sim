"""Model clients built from models.yaml; every call goes through the model key's limiter (§11.4, §11.7).

ModelClient.call(messages) -> CallOutcome:
  cache lookup (a hit costs nothing) -> limiter.acquire (spacing, 60 s windows, quota guard) ->
  structured call (LangChain with_structured_output(TurnDecision, method, include_raw=True)) ->
  record real usage (+ server headers) -> cache the answer.
Errors follow the §11.7 decision table (classify_error):
  temporary 429 (retry-after, or limit-req-minute > 0) -> wait retry-after x 1.2 (>= 1 s), slow down,
      retry up to 5 times
  zero-allowance 429 (limit-req-minute 0, "limit: 0", "not available on this plan") -> BlockedModel,
      never retried, the run stops with status blocked_model
  daily quota used (our counter at the cap, or a per-day quota in the 429) -> QuotaPause
  5xx, timeout, connection error -> retry 3 times (2, 4, 8 s)
  401, 403 -> AuthFailure, the run stops
  404 -> ModelUnavailable (wrong id)
Each attempt, retried or not, counts against the limits. A client is bound to ONE model id for its
whole life: there is no fallback to another model (D13).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from world.actions import TurnDecision
from world.config import ModelsConfig, load_config
from world.llm.cache import CachedAnswer, LLMCache, cache_key
from world.llm.limiter import Clock, ModelLimiter, QuotaPause, RealClock
from world.llm.quota import QuotaGuard, parse_rate_headers

Messages = Sequence[tuple[str, str]]  # (role, content), role in system | user | assistant
TEMP_429_RETRIES = 5
TRANSIENT_BACKOFF_S = (2.0, 4.0, 8.0)


class RunStop(Exception):  # noqa: N818 - control signal
    """The run must stop now (no retry): status in `status`."""

    status = "failed"


class BlockedModel(RunStop):
    status = "blocked_model"


class AuthFailure(RunStop):
    status = "auth_error"


class ModelUnavailable(RunStop):
    status = "model_unavailable"


# ------------------------------------------------------------------------------ transport


@dataclass
class RawReply:
    """What one provider call returned."""

    raw_text: str
    parsed: TurnDecision | None
    parse_error: str | None
    tokens_in: int
    tokens_out: int
    tokens_reasoning: int
    headers: dict[str, str] = field(default_factory=dict)
    model_name: str = ""


Transport = Callable[[Messages], RawReply]


def _usage(msg: Any) -> tuple[int, int, int]:
    u = getattr(msg, "usage_metadata", None) or {}
    details = u.get("output_token_details") or {}
    return (
        int(u.get("input_tokens") or 0),
        int(u.get("output_tokens") or 0),
        int(details.get("reasoning") or 0),
    )


def _raw_text(msg: Any) -> str:
    calls = getattr(msg, "tool_calls", None)
    if calls:
        return json.dumps(calls[0].get("args", {}))
    content = getattr(msg, "content", "")
    if isinstance(content, list):  # content blocks
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def to_decision(parsed: Any) -> TurnDecision:
    if isinstance(parsed, TurnDecision):
        return parsed
    if isinstance(parsed, BaseModel):
        parsed = parsed.model_dump()
    if isinstance(parsed, str):
        return TurnDecision.model_validate_json(parsed)
    return TurnDecision.model_validate(parsed)


class LangChainTransport:
    """A LangChain chat model with structured output. Captures Mistral response headers through an
    httpx event hook (google-genai does not expose headers on success)."""

    def __init__(self, chat_model: Any, method: str) -> None:
        self.chat_model = chat_model
        self.method = method
        self.runnable = chat_model.with_structured_output(TurnDecision, method=method, include_raw=True)
        self.last_headers: dict[str, str] = {}
        client = getattr(chat_model, "client", None)
        if client is not None and hasattr(client, "event_hooks"):
            hooks = dict(client.event_hooks)
            hooks["response"] = [*hooks.get("response", []), self._hook]
            client.event_hooks = hooks

    def _hook(self, response: Any) -> None:
        self.last_headers = {k.lower(): v for k, v in response.headers.items()}

    def __call__(self, messages: Messages) -> RawReply:
        self.last_headers = {}
        out = self.runnable.invoke([(role, content) for role, content in messages])
        raw = out.get("raw")
        tin, tout, treason = _usage(raw)
        parsed, err = None, out.get("parsing_error")
        if err is None and out.get("parsed") is not None:
            try:
                parsed = to_decision(out["parsed"])
            except ValidationError as e:
                err = e
        elif err is None:
            err = "no parsed output"
        meta = getattr(raw, "response_metadata", {}) or {}
        return RawReply(
            raw_text=_raw_text(raw),
            parsed=parsed,
            parse_error=None if err is None else _short(str(err)),
            tokens_in=tin,
            tokens_out=tout,
            tokens_reasoning=treason,
            headers=dict(self.last_headers),
            model_name=str(meta.get("model_name") or meta.get("model") or ""),
        )


def _short(text: str, head: int = 300, tail: int = 600) -> str:
    """Errors are kept short for prompts and logs: the start and the end (where the reason is)."""
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= head + tail + 5 else text[:head] + " ... " + text[-tail:]


def build_chat_model(key: str, models: ModelsConfig, *, method: str | None = None) -> Any:
    """The LangChain chat model for a model key (provider package per §2). Keys come from the
    environment (.env) and are never logged."""
    from langchain.chat_models import init_chat_model

    m = models.models[key]
    d = models.defaults
    if m.provider == "mistral":
        model = init_chat_model(
            m.model_id, model_provider="mistralai", temperature=d.temperature, max_tokens=d.max_output_tokens,
            max_retries=1, timeout=120,
        )  # fmt: skip
    else:
        kwargs: dict[str, Any] = {}
        if m.thinking and m.thinking != "TODO_VERIFY":
            kwargs["thinking_level"] = m.thinking
        model = init_chat_model(
            m.model_id, model_provider="google_genai", temperature=d.temperature,
            max_output_tokens=d.max_output_tokens, max_retries=0, timeout=120, **kwargs,
        )  # fmt: skip
    return model


# ------------------------------------------------------------------------------ errors


@dataclass(frozen=True)
class ErrorInfo:
    kind: (
        str  # temporary_429 | zero_allowance | daily_cap | transient | auth | not_found | bad_request | other
    )
    status: int | None
    retry_after_s: float | None
    text: str


_ZERO = re.compile(r"limit:\s*0\b|not available on (this|your) plan|limit-req-minute.{0,5}\b0\b", re.I)
_DAILY = re.compile(r"per.?day|PerDay|daily", re.I)
_RETRY_DELAY = re.compile(r"retry(?:Delay|_delay)?[\"']?\s*[:=]\s*[\"']?(\d+(?:\.\d+)?)s", re.I)


def _status_headers_text(exc: BaseException) -> tuple[int | None, dict[str, str], str]:
    """Walk the exception chain for an HTTP status, response headers and a text."""
    status, headers, texts = None, {}, []
    seen = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        texts.append(str(e))
        resp = getattr(e, "response", None)
        code = getattr(e, "code", None) or getattr(e, "status_code", None)
        if resp is not None:
            code = code or getattr(resp, "status_code", None)
            h = getattr(resp, "headers", None)
            if h is not None:
                with contextlib.suppress(TypeError, ValueError):
                    headers.update({k.lower(): v for k, v in dict(h).items()})
        details = getattr(e, "details", None)
        if details:
            texts.append(json.dumps(details, default=str))
        if isinstance(code, int) and status is None:
            status = code
        e = e.__cause__ or e.__context__
    text = " | ".join(texts)
    if status is None:
        m = re.search(r"\b(4\d\d|5\d\d)\b", text)
        status = int(m.group(1)) if m else None
    return status, headers, text


def classify_error(exc: BaseException) -> ErrorInfo:
    status, headers, text = _status_headers_text(exc)
    info = parse_rate_headers(headers)
    retry = info.retry_after_s
    if retry is None and (m := _RETRY_DELAY.search(text)):
        retry = float(m.group(1))
    name = type(exc).__name__.lower()
    if status == 429:
        if info.limit_req_minute == 0 or _ZERO.search(text):
            return ErrorInfo("zero_allowance", status, retry, _short(text))
        if _DAILY.search(text):
            return ErrorInfo("daily_cap", status, retry, _short(text))
        return ErrorInfo("temporary_429", status, retry, _short(text))
    if status in (401, 403):
        return ErrorInfo("auth", status, None, _short(text))
    if status == 404:
        return ErrorInfo("not_found", status, None, _short(text))
    if status is not None and status >= 500:
        return ErrorInfo("transient", status, retry, _short(text))
    if "timeout" in name or "connect" in name or "timeout" in text.lower()[:200]:
        return ErrorInfo("transient", status, None, _short(text))
    if status is not None and 400 <= status < 500:
        return ErrorInfo("bad_request", status, None, _short(text))
    return ErrorInfo("other", status, None, _short(text))


# ------------------------------------------------------------------------------ client


@dataclass(frozen=True)
class CallOutcome:
    parsed: TurnDecision | None
    raw_text: str
    parse_error: str | None
    tokens_in: int
    tokens_out: int
    tokens_reasoning: int
    latency_s: float
    transport_retries: int
    cached: bool
    model_name: str = ""
    headers: dict[str, str] = field(default_factory=dict)


class ModelClient:
    """One model key: its fixed model id, structured method, limiter, cache and transport."""

    def __init__(
        self,
        key: str,
        models: ModelsConfig,
        *,
        limiter: ModelLimiter,
        transport: Transport | None = None,
        cache: LLMCache | None = None,
        method: str | None = None,
        clock: Clock | None = None,
        transient_retries: int = len(TRANSIENT_BACKOFF_S),
    ) -> None:
        m = models.models[key]
        self.transient_retries = transient_retries
        self.key, self.provider, self.model_id = key, m.provider, m.model_id
        self.temperature = models.defaults.temperature
        self.method = method or m.structured_method or models.defaults.structured_method
        if self.method == "TODO_VERIFY":
            self.method = "json_schema"
        self.limiter = limiter
        self.cache = cache
        self.clock = clock or limiter.clock
        self._transport = transport
        self.models = models

    @property
    def transport(self) -> Transport:
        if self._transport is None:
            self._transport = LangChainTransport(build_chat_model(self.key, self.models), self.method)
        return self._transport

    def call(self, messages: Messages, *, sample_index: int = 0) -> CallOutcome:
        key = cache_key(self.provider, self.model_id, self.temperature, messages, sample_index)
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            parsed = TurnDecision.model_validate_json(hit.parsed_json) if hit.parsed_json else None
            return CallOutcome(
                parsed, hit.raw_text, hit.parse_error, hit.tokens_in, hit.tokens_out, hit.tokens_reasoning,
                hit.latency_s, 0, True,
            )  # fmt: skip
        est = self.limiter.estimate(sum(len(c) for _, c in messages))
        n429 = ntrans = 0
        while True:
            self.limiter.acquire(est)
            t0 = time.monotonic()
            try:
                reply = self.transport(messages)
            except Exception as exc:  # noqa: BLE001 - every provider error goes through the §11.7 table
                self.limiter.record_failed_attempt()
                err = classify_error(exc)
                if err.kind == "zero_allowance":
                    raise BlockedModel(
                        f"{self.key} ({self.model_id}) has a zero allowance: {err.text}"
                    ) from exc
                if err.kind == "auth":
                    raise AuthFailure(f"{self.key}: HTTP {err.status} (check the API key in .env)") from exc
                if err.kind == "not_found":
                    raise ModelUnavailable(
                        f"{self.key}: model id {self.model_id} not found: {err.text}"
                    ) from exc
                if err.kind == "daily_cap":
                    raise QuotaPause(f"{self.key}: provider daily quota used: {err.text}") from exc
                if err.kind == "temporary_429" and n429 < TEMP_429_RETRIES:
                    wait = self.limiter.slow_down(err.retry_after_s)
                    if wait > self.limiter.max_block_s:
                        raise QuotaPause(
                            f"{self.key}: 429 asks to wait {wait:.0f} s", resume_after_s=wait
                        ) from exc
                    n429 += 1
                    continue
                if err.kind == "transient" and ntrans < self.transient_retries:
                    self.clock.sleep(TRANSIENT_BACKOFF_S[ntrans])
                    ntrans += 1
                    continue
                if err.kind == "temporary_429":
                    raise QuotaPause(f"{self.key}: still rate limited after {n429} retries") from exc
                if err.kind == "bad_request":  # e.g. a schema the provider rejects: an invalid answer
                    return CallOutcome(None, "", f"HTTP {err.status}: {err.text}", 0, 0, 0,
                                       time.monotonic() - t0, n429 + ntrans, False)  # fmt: skip
                raise
            latency = time.monotonic() - t0
            self.limiter.record(reply.tokens_in, reply.tokens_out, reply.headers)
            out = CallOutcome(
                reply.parsed, reply.raw_text, reply.parse_error, reply.tokens_in, reply.tokens_out,
                reply.tokens_reasoning, latency, n429 + ntrans, False, reply.model_name, reply.headers,
            )  # fmt: skip
            if self.cache is not None:
                self.cache.put(key, self.provider, self.model_id, CachedAnswer(
                    out.raw_text, out.parsed.model_dump_json() if out.parsed else None, out.parse_error,
                    out.tokens_in, out.tokens_out, out.tokens_reasoning, out.latency_s,
                ))  # fmt: skip
            return out


class Router:
    """One ModelClient (and so one limiter) per model key for the whole process (D23)."""

    def __init__(
        self,
        models: ModelsConfig | None = None,
        *,
        quota: QuotaGuard | None = None,
        cache: LLMCache | None = None,
        clock: Clock | None = None,
        transports: Mapping[str, Transport] | None = None,
        allow_unverified: bool = False,
    ) -> None:
        self.models = models or load_config().models
        self.quota = quota
        self.cache = cache
        self.clock = clock or RealClock()
        self.transports = dict(transports or {})
        self.allow_unverified = allow_unverified
        self.clients: dict[str, ModelClient] = {}

    def check_runnable(self, key: str) -> None:
        if key not in self.models.models:
            raise KeyError(f"unknown model key {key!r}")
        m = self.models.models[key]
        if "TODO_VERIFY" in m.model_id:
            raise RunStop(f"{key}: model id is TODO_VERIFY")
        if not m.runnable_for_experiments and not self.allow_unverified:
            raise RunStop(
                f"{key}: status {m.status}; only verified models run (smoke/probe: allow_unverified)"
            )

    def client(self, key: str) -> ModelClient:
        if key not in self.clients:
            self.check_runnable(key)
            d = self.models.defaults
            limiter = ModelLimiter(
                key, self.models.models[key], safety=d.safety, max_output_tokens=d.max_output_tokens,
                clock=self.clock, quota=self.quota,
            )  # fmt: skip
            self.clients[key] = ModelClient(
                key, self.models, limiter=limiter, transport=self.transports.get(key), cache=self.cache,
                clock=self.clock,
            )  # fmt: skip
        return self.clients[key]


def keys_present() -> dict[str, bool]:
    """Whether each provider key is set (never the values)."""
    from dotenv import load_dotenv

    load_dotenv(".env")
    return {k: bool(os.environ.get(k)) for k in ("MISTRAL_API_KEY", "GOOGLE_API_KEY")}
