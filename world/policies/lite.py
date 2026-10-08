"""Lite-mode LLM leader (CLAUDE.md §11): exactly one model call per country per turn, plus at most one
retry (§1 rule 5).

decide(briefing):
  1. messages = [system prompt (fixed for the game), the turn's briefing]
  2. one structured call through the model key's ModelClient (limiter, quota guard, cache)
  3. if the answer is not a valid TurnDecision: ONE retry that adds the answer and
     "Your answer was invalid because ... Reply again in the required format."
  4. if that fails too: decision None -> the game falls back to `wait` and counts a parse failure
     (3 in a row -> StatusQuoBot for the rest of the run, run flagged degraded; world/game.py)
Every call (cached or not) becomes a CallLog in the DecisionResult; the graph writes them to
llm_calls. QuotaPause and RunStop are NOT caught here: they pause or stop the run at this seat
boundary (world/graph.py, world/runner.py). The client is bound to one model id for the whole run:
there is no fallback model (D13).
Memory: a lite leader's memory is the briefing's "your last 2 quarters" section, which already hides
earlier leaders' turns (D65); on_leader_change clears the little state the policy keeps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from world.briefing import Briefing
from world.config import Config
from world.llm.prompting import render_system_prompt, render_turn_message
from world.llm.router import CallOutcome, ModelClient
from world.policies.base import CallLog, DecisionResult

RETRY_TEMPLATE = "Your answer was invalid because {error}. Reply again in the required format."


@dataclass
class LitePolicy:
    client: ModelClient
    country: str
    cfg: Config
    sample_index: int = 0
    name: str = ""
    leader_changes: int = 0
    last_raw: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        self.name = self.name or f"lite:{self.client.key}"
        self.system = render_system_prompt(self.country, self.cfg)
        self.max_retries = self.cfg.world.agents.max_retries

    def _log(self, out: CallOutcome, attempt: int) -> CallLog:
        c = self.client
        return CallLog(
            provider=c.provider, model_id=c.model_id, model_key=c.key, method=c.method, attempt=attempt,
            tokens_in=out.tokens_in, tokens_out=out.tokens_out, tokens_reasoning=out.tokens_reasoning,
            latency_s=out.latency_s, transport_retries=out.transport_retries, raw_text=out.raw_text,
            parsed_json=out.parsed.model_dump_json() if out.parsed else None, error=out.parse_error,
            cached=out.cached, sample_index=self.sample_index,
        )  # fmt: skip

    def decide(self, briefing: Briefing) -> DecisionResult:
        if briefing.country != self.country:
            raise ValueError(f"policy for {self.country} asked to decide for {briefing.country}")
        messages: list[tuple[str, str]] = [("system", self.system), ("user", render_turn_message(briefing))]
        logs: list[CallLog] = []
        out = None
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                assert out is not None
                messages = [
                    *messages,
                    ("assistant", out.raw_text or "(no answer)"),
                    ("user", RETRY_TEMPLATE.format(error=out.parse_error or "it was not a valid answer")),
                ]
            out = self.client.call(messages, sample_index=self.sample_index)
            logs.append(self._log(out, attempt))
            if out.parsed is not None:
                break
        assert out is not None
        self.last_raw = out.raw_text
        return DecisionResult(
            decision=out.parsed,
            raw_text=out.raw_text,
            parse_error=None if out.parsed is not None else out.parse_error,
            tokens_in=sum(c.tokens_in for c in logs),
            tokens_out=sum(c.tokens_out for c in logs),
            latency_s=sum(c.latency_s for c in logs),
            retries=len(logs) - 1,
            calls=tuple(logs),
        )

    def on_leader_change(self, country: str) -> None:
        """§6.11 memory wipe. The 2-turn memory lives in the briefing (hidden from the new leader by
        D65); the policy itself only drops its last raw answer and counts the change."""
        self.leader_changes += 1
        self.last_raw = ""


def lite_seats(router, models: dict[str, str], cfg: Config, sample_index: int = 0) -> dict[str, LitePolicy]:
    """{country: model key} -> {country: LitePolicy}; all seats with the same key share one client
    (one limiter per model key, D23)."""
    return {c: LitePolicy(router.client(k), c, cfg, sample_index) for c, k in models.items()}
