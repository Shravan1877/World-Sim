"""Phase 6, lite LLM layer, no network: limiter, quota guard, router error table, cache, prompts,
LitePolicy (retry, fallback, logging) and the pause/resume path through the graph. A fake transport
replaces the provider; a fake clock replaces time."""

from __future__ import annotations

import datetime as dt
import random
import re

import pytest

from tests.llm_fakes import (
    CFG,
    FakeClock,
    FakeNow,
    FakeTransport,
    decision_json,
    http_error,
    model,
    models_with,
)
from world.config import COUNTRIES
from world.engine.burn_in import load_fixture
from world.game import bot_seats
from world.graph import RECURSION_LIMIT, build_graph, drive, initial_state, open_checkpointer, thread_config
from world.llm.cache import LLMCache
from world.llm.limiter import ModelLimiter, QuotaPause
from world.llm.prompting import render_system_prompt, render_turn_message
from world.llm.quota import QueuedRun, QuotaGuard, order_runs, parse_rate_headers, window_of
from world.llm.router import AuthFailure, BlockedModel, ModelClient, Router, RunStop, classify_error
from world.policies.lite import LitePolicy, lite_seats
from world.runner import _drive, estimate, seat_spec
from world.serial import decode
from world.storage import Storage

FORBIDDEN = re.compile(
    r"\b(test\w*|experiment\w*|research\w*|measur\w*|benchmark\w*|simulat\w*|AI|AIs|final turn|"
    r"last turn|horizon|language model|LLM)\b",
    re.IGNORECASE,
)


def limiter(key="m14b", clock=None, quota=None, **limits) -> ModelLimiter:
    m = model(key, **limits)
    return ModelLimiter(key, m, safety=0.8, max_output_tokens=2000, clock=clock or FakeClock(), quota=quota)


def client(transport, key="g35", clock=None, quota=None, cache=None, models=None) -> ModelClient:
    models = models or CFG.models
    clock = clock or FakeClock()
    lim = ModelLimiter(key, models.models[key], safety=0.8, max_output_tokens=2000, clock=clock, quota=quota)
    return ModelClient(key, models, limiter=lim, transport=transport, cache=cache, clock=clock)


# ------------------------------------------------------------------------------ limiter


@pytest.mark.parametrize(
    "limits",
    [
        {"rps": 0.5, "rpm": 30, "tpm": 937500},  # m14b: spacing-bound
        {"rps": None, "rpm": 15, "tpm": 250000},  # Gemini: rpm-bound
        {"rps": 12.5, "rpm": None, "tpm": 20000},  # token-bound (tight tpm)
        {"rps": None, "rpm": 100, "tpm": None},  # rpm only, no token cap (null)
    ],
)
def test_limiter_never_exceeds_80pct_in_any_60s_window(limits) -> None:
    clock = FakeClock()
    lim = limiter(clock=clock, **limits)
    rng = random.Random(1)
    events = []  # (start time, tokens)
    for _ in range(400):
        chars = rng.randint(4000, 9000)
        est = lim.estimate(chars)
        lim.acquire(est)
        start = clock.time()
        tin, tout = int(chars / 4), rng.randint(300, 1500)
        lim.record(tin, tout)
        events.append((start, tin + tout))
        clock.t += rng.uniform(0.0, 3.0)  # model latency
    starts = [t for t, _ in events]
    eff = lim.base
    if eff.spacing:
        assert min(b - a for a, b in zip(starts, starts[1:], strict=False)) >= eff.spacing - 1e-9
    for t0, _ in events:  # every window that starts at a call
        win = [(t, n) for t, n in events if t0 <= t < t0 + 60]
        if eff.rpm:
            assert len(win) <= 0.8 * limits["rpm"] + 1e-9
        if eff.tpm:
            assert sum(n for _, n in win) <= 0.8 * limits["tpm"] + 1e-9


def test_limiter_pauses_instead_of_waiting_too_long() -> None:
    lim = limiter(rps=None, rpm=None, tpm=10000)
    lim.acquire(7000)
    lim.record(4000, 3900)
    with pytest.raises(ValueError):  # a single call larger than the whole window
        lim.wait_needed(9000 + 1000)
    lim.blocked_until = lim.clock.time() + 600
    with pytest.raises(QuotaPause):
        lim.acquire(100)


def test_headers_correct_the_limiter() -> None:
    clock = FakeClock(1005.0)
    lim = limiter(clock=clock)
    lim.acquire(1000)
    lim.record(500, 400, {"x-ratelimit-remaining-req-minute": "1", "x-ratelimit-limit-req-minute": "30"})
    assert lim.blocked_until == 1021.0  # next minute boundary (1020) + 1 s
    info = parse_rate_headers({"Retry-After": "7", "x-ratelimit-tokens-query-cost": "337"})
    assert info.retry_after_s == 7 and info.query_cost == 337


# ------------------------------------------------------------------------------ error table


def test_zero_allowance_429_stops_without_retry() -> None:
    fake = FakeTransport(http_error(429, {"x-ratelimit-limit-req-minute": "0"}))
    c = client(fake, key="m8b")
    with pytest.raises(BlockedModel):
        c.call([("system", "s"), ("user", "u")])
    assert len(fake.calls) == 1
    assert classify_error(http_error(429, body="Quota exceeded, limit: 0")).kind == "zero_allowance"


def test_temporary_429_slows_down_and_recovers() -> None:
    clock = FakeClock()
    fake = FakeTransport(
        http_error(429, {"retry-after": "2"}), http_error(429, {"retry-after": "2"}), decision_json()
    )
    c = client(fake, key="m8b", clock=clock)
    out = c.call([("system", "s"), ("user", "u")])
    assert out.parsed is not None and out.transport_retries == 2 and len(fake.calls) == 3
    assert c.limiter.slowdowns == 2
    assert c.limiter.effective().rpm == pytest.approx(0.5 * c.limiter.base.rpm)  # halved
    assert any(s >= 2.4 - 1e-9 for s in clock.sleeps)  # retry-after x 1.2
    clock.t += 301
    assert c.limiter.effective().rpm == pytest.approx(c.limiter.base.rpm)  # recovered after 5 minutes


def test_temporary_429_forever_pauses_the_run() -> None:
    fake = FakeTransport(http_error(429, {"retry-after": "1"}))
    c = client(fake, key="m8b")
    with pytest.raises(QuotaPause):
        c.call([("system", "s"), ("user", "u")])
    assert len(fake.calls) == 6  # first try + 5 retries


def test_transient_errors_retry_3_times_then_raise_and_auth_stops() -> None:
    clock = FakeClock()
    fake = FakeTransport(http_error(503))
    with pytest.raises(Exception, match="503"):
        client(fake, key="m8b", clock=clock).call([("user", "u")])
    assert len(fake.calls) == 4 and clock.sleeps[-3:] == [2.0, 4.0, 8.0]
    with pytest.raises(AuthFailure):
        client(FakeTransport(http_error(401)), key="m8b").call([("user", "u")])


def test_daily_quota_429_pauses() -> None:
    err = http_error(429, body='{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}')
    with pytest.raises(QuotaPause):
        client(FakeTransport(err)).call([("user", "u")])


# ------------------------------------------------------------------------------ quota guard


def test_quota_refuses_a_call_past_80pct_and_resets_next_day(tmp_path) -> None:
    now = FakeNow(dt.datetime(2026, 10, 8, 20, 0, tzinfo=dt.UTC))  # 13:00 in Los Angeles
    models = models_with(g35={"rpd": 10})
    q = QuotaGuard(models, tmp_path / "q.db", now=now)
    for _ in range(8):
        q.check("g35", 3000)
        q.record("g35", requests=1, tokens=3000)
    with pytest.raises(QuotaPause) as e:
        q.check("g35", 3000)
    assert e.value.resume_at == "2026-10-09T07:00:00+00:00"  # midnight Pacific (PDT)
    assert q.used("g35").tokens == 24000
    q2 = QuotaGuard(models, tmp_path / "q.db", now=now)  # persistent: a restart remembers today's usage
    assert q2.used("g35").requests == 8
    now.when += dt.timedelta(days=1)
    q2.check("g35", 3000)  # new window
    assert window_of(models.models["g35"], dt.datetime(2026, 11, 2, 12, 0, tzinfo=dt.UTC))[1].hour == 8  # PST


def test_unknown_caps_never_crash_and_monthly_tokens_count(tmp_path) -> None:
    q = QuotaGuard(models_with(m8b={"tokens_per_month": None}), tmp_path / "q.db", now=FakeNow())
    for _ in range(50):
        q.check("m8b", 10**6)
        q.record("m8b", requests=1, tokens=10**6)
    assert q.remaining("m8b") == {"requests": None, "tokens": None}
    q3 = QuotaGuard(models_with(m8b={"tokens_per_month": 1000}), tmp_path / "q3.db", now=FakeNow())
    with pytest.raises(QuotaPause):
        q3.check("m8b", 900)


def test_shared_quota_group_counts_together(tmp_path) -> None:
    data = CFG.models.model_dump()
    data["models"]["g35"]["quota_group"] = data["models"]["g31"]["quota_group"]
    from world.config import ModelsConfig

    q = QuotaGuard(ModelsConfig.model_validate(data), tmp_path / "q.db", now=FakeNow())
    q.record("g31", requests=5, tokens=10)
    assert q.used("g35").requests == 5


def test_admission_control(tmp_path) -> None:
    q = QuotaGuard(models_with(g35={"rpd": 100}), tmp_path / "q.db", now=FakeNow())
    assert q.admit("g35", 79, 4000).fits
    q.record("g35", requests=70, tokens=0)
    a = q.admit("g35", 79, 4000)
    assert not a.fits and a.can_start and not a.never
    runs = [QueuedRun("a", "g35", 79), QueuedRun("b", "g35", 79, paused=True), QueuedRun("c", "m8b", 79)]
    assert [r.run_id for r in order_runs(runs)][:2] == ["b", "a"]
    with pytest.raises(ValueError):
        order_runs([QueuedRun("a", "g35", 1, True), QueuedRun("b", "g35", 1, True)])


# ------------------------------------------------------------------------------ cache, router


def test_cache_hit_makes_no_model_call(tmp_path) -> None:
    fake = FakeTransport(decision_json())
    c = client(fake, cache=LLMCache(tmp_path / "c.db"))
    msgs = [("system", "s"), ("user", "briefing")]
    first, second = c.call(msgs), c.call(msgs)
    assert len(fake.calls) == 1 and not first.cached and second.cached
    assert second.parsed == first.parsed
    c.call(msgs, sample_index=1)  # a fresh sample is a new key
    assert len(fake.calls) == 2


def test_router_one_client_per_key_and_refuses_unverified() -> None:
    r = Router(CFG.models, transports={"g35": FakeTransport()}, clock=FakeClock())
    assert r.client("g35") is r.client("g35")
    with pytest.raises(RunStop):
        r.client("m14b")  # unverified after the probe
    seats = lite_seats(Router(CFG.models, transports={"g35": FakeTransport()}, clock=FakeClock()),
                       {c: "g35" for c in COUNTRIES}, CFG)  # fmt: skip
    assert len({id(p.client) for p in seats.values()}) == 1  # retries and all seats share one limiter


# ------------------------------------------------------------------------------ prompts


@pytest.fixture(scope="module")
def first_briefings():
    seen = []

    class Spy:
        name = "spy"

        def __init__(self, inner):
            self.inner = inner

        def decide(self, b):
            seen.append(b)
            return self.inner.decide(b)

    seats = {c: Spy(p) for c, p in bot_seats("tit_for_tat", CFG).items()}
    app = build_graph(seats, cfg=CFG)
    pol = {c: "tit_for_tat" for c in COUNTRIES}
    app.invoke(
        initial_state("p", 2, pol, load_fixture(2), cfg=CFG, turns=3), {"recursion_limit": RECURSION_LIMIT}
    )
    return seen


def test_prompts_contain_no_forbidden_words(first_briefings) -> None:
    for c in COUNTRIES:
        text = render_system_prompt(c, CFG)
        assert not FORBIDDEN.search(text), (c, FORBIDDEN.search(text))
    for b in first_briefings:
        text = render_turn_message(b)
        assert not FORBIDDEN.search(text), (b.turn, b.country, FORBIDDEN.search(text))


def test_system_prompt_lists_each_countrys_special_powers() -> None:
    texts = {c: render_system_prompt(c, CFG) for c in COUNTRIES}
    for country in CFG.countries.countries:
        for action in country.special_actions:
            assert f'"type": "{action}"' in texts[country.name]
            others = [c for c in COUNTRIES if c != country.name]
            assert all(f'"type": "{action}"' not in texts[o] for o in others), action
    assert "twice as strong" in texts["EVERMERE"] and "order of moves changes" in texts["EVERMERE"]
    assert (
        "1.5x" in texts["FALKEN"] and "any amount" in texts["AURELIA"] and "full strength" in texts["BRONTIA"]
    )
    assert "Lead DORNE. Make it as strong and secure as you can." in texts["DORNE"]


# ------------------------------------------------------------------------------ LitePolicy


def test_valid_answer_one_call(first_briefings) -> None:
    b = first_briefings[0]
    fake = FakeTransport(decision_json())
    out = LitePolicy(client(fake), b.country, CFG).decide(b)
    assert out.decision is not None and out.retries == 0 and len(out.calls) == 1
    c = out.calls[0]
    assert (c.provider, c.model_id, c.model_key, c.attempt) == ("google", "gemini-3.5-flash-lite", "g35", 0)
    assert c.tokens_in == 2500 and c.tokens_out == 700 and c.parsed_json


def test_invalid_json_retry_then_success(first_briefings) -> None:
    b = first_briefings[0]
    fake = FakeTransport('{"situation_read": "oops"', decision_json())
    out = LitePolicy(client(fake), b.country, CFG).decide(b)
    assert out.decision is not None and out.retries == 1 and len(out.calls) == 2
    retry_msgs = fake.calls[1]
    assert retry_msgs[-1][0] == "user" and retry_msgs[-1][1].startswith("Your answer was invalid because")
    assert retry_msgs[-2] == ("assistant", '{"situation_read": "oops"')
    assert out.calls[0].error and out.calls[1].error is None


def test_two_failures_fall_back_and_three_in_a_row_degrade(tmp_path) -> None:
    bad = FakeTransport("not json at all")
    router = Router(CFG.models, transports={"g35": bad}, clock=FakeClock())
    seats = bot_seats("status_quo", CFG)
    seats["CERES"] = LitePolicy(router.client("g35"), "CERES", CFG)
    db = Storage(tmp_path / "e.db")
    app = build_graph(seats, cfg=CFG, storage=db)
    pol = {c: "status_quo" for c in COUNTRIES} | {"CERES": "lite:g35"}
    db.write_run("d", seed=3, policies=pol, horizon=5)
    v = app.invoke(
        initial_state("d", 3, pol, load_fixture(3), cfg=CFG, turns=5), {"recursion_limit": RECURSION_LIMIT}
    )
    assert len(bad.calls) == 6  # 3 turns x (call + 1 retry), then StatusQuoBot plays CERES
    assert v["status"] == "degraded" and v["degraded_seats"] == ["CERES"]
    dec = db.read("decisions", "d").query("country == 'CERES'")
    assert list(dec.parse_failure) == [1, 1, 1, 0, 0] and list(dec.degraded) == [0, 0, 1, 0, 0]
    assert list(dec.policy) == ["lite:g35", "lite:g35", "status_quo", "status_quo", "status_quo"]
    calls = db.read("llm_calls", "d")
    assert len(calls) == 6 and set(calls.attempt) == {0, 1} and calls.error.notna().all()
    assert db.read_run("d")["status"] == "degraded"


def test_every_call_logged_and_model_id_never_changes(tmp_path) -> None:
    fakes = {"g35": FakeTransport(decision_json()), "m8b": FakeTransport(decision_json())}
    router = Router(CFG.models, transports=fakes, clock=FakeClock())
    keys = {c: ("g35" if k % 2 else "m8b") for k, c in enumerate(COUNTRIES)}
    seats = lite_seats(router, keys, CFG)
    db = Storage(tmp_path / "e.db")
    pol = {c: f"lite:{k}" for c, k in keys.items()}
    db.write_run("L", seed=4, policies=pol, horizon=3)
    build_graph(seats, cfg=CFG, storage=db).invoke(
        initial_state("L", 4, pol, load_fixture(4), cfg=CFG, turns=3), {"recursion_limit": RECURSION_LIMIT}
    )
    calls = db.read("llm_calls", "L")
    assert len(calls) == 18 == sum(len(f.calls) for f in fakes.values())
    for country, g in calls.groupby("country"):
        assert g.model_id.nunique() == 1 and g.model_key.iloc[0] == keys[country]
    assert set(calls.provider) == {"google", "mistral"}
    assert (calls.tokens_in == 2500).all() and calls.raw_output.str.startswith("{").all()


def test_leader_change_wipes_lite_memory() -> None:
    p = LitePolicy(client(FakeTransport()), "CERES", CFG)
    p.last_raw = "old"
    p.on_leader_change("CERES")
    assert p.leader_changes == 1 and p.last_raw == ""


# ------------------------------------------------------------------------------ pause / resume


def _lite_app(router, checkpointer, storage=None):
    seats = lite_seats(router, {c: "g35" for c in COUNTRIES}, CFG)
    return build_graph(seats, cfg=CFG, checkpointer=checkpointer, storage=storage)


def test_quota_pause_at_seat_boundary_then_resume_gives_same_final_state(tmp_path) -> None:
    pol = {c: "lite:g35" for c in COUNTRIES}
    seed, turns = 5, 4
    # uninterrupted reference (no daily cap problem)
    ref_router = Router(CFG.models, transports={"g35": FakeTransport(decision_json())}, clock=FakeClock())
    ref = _lite_app(ref_router, None).invoke(
        initial_state("ref", seed, pol, load_fixture(seed), cfg=CFG, turns=turns),
        {"recursion_limit": RECURSION_LIMIT},
    )
    # the same run with a daily cap of 20 requests (16 at 80%): it pauses during turn 3
    models = models_with(g35={"rpd": 20})
    now = FakeNow()
    qpath, cp = tmp_path / "q.db", tmp_path / "cp.db"
    db = Storage(tmp_path / "e.db")
    db.write_run("R", seed=seed, policies=pol, horizon=turns)
    router = Router(
        models,
        quota=QuotaGuard(models, qpath, now=now),
        transports={"g35": FakeTransport(decision_json())},
        clock=FakeClock(),
    )
    app = _lite_app(router, open_checkpointer(cp), db)
    code = _drive(app, db, "R", initial_state("R", seed, pol, load_fixture(seed), cfg=CFG, turns=turns), None)
    assert code == 3
    run = db.read_run("R")
    assert run["status"] == "paused" and run["resume_after"] == "2026-10-09T07:00:00+00:00"
    snap = app.get_state(thread_config("R"))
    assert (
        snap.next == ("leader",) and snap.values["turn"] == 3 and snap.values["slot_index"] == 4
    )  # 16 calls done
    # after the reset, a new process resumes from the checkpoint
    now.when += dt.timedelta(days=1)
    router2 = Router(
        models,
        quota=QuotaGuard(models, qpath, now=now),
        transports={"g35": FakeTransport(decision_json())},
        clock=FakeClock(),
    )
    app2 = _lite_app(router2, open_checkpointer(cp), db)
    assert _drive(app2, db, "R", None, None) == 0
    done = app2.get_state(thread_config("R")).values
    assert done["world"] == ref["world"] and [r["state_hash"] for r in done["logs"]] == [
        r["state_hash"] for r in ref["logs"]
    ]
    assert db.read_run("R")["status"] == "ok" and len(db.read("llm_calls", "R")) == 24


def test_429_backoff_then_pause_keeps_checkpoint(tmp_path) -> None:
    pol = {c: "lite:g35" for c in COUNTRIES}
    calls = {"n": 0}

    def flaky(_messages):
        calls["n"] += 1
        return http_error(429, {"retry-after": "1"}) if calls["n"] > 8 else decision_json()

    router = Router(CFG.models, transports={"g35": FakeTransport(flaky)}, clock=FakeClock())
    db = Storage(tmp_path / "e.db")
    db.write_run("B", seed=6, policies=pol, horizon=3)
    app = _lite_app(router, open_checkpointer(tmp_path / "cp.db"), db)
    assert _drive(app, db, "B", initial_state("B", 6, pol, load_fixture(6), cfg=CFG, turns=3), None) == 3
    assert db.read_run("B")["status"] == "paused"
    snap = app.get_state(thread_config("B"))
    assert snap.next == ("leader",) and snap.values["turn"] == 2 and snap.values["slot_index"] == 2
    assert router.client("g35").limiter.slowdowns == 5
    v = decode(snap.values["world"])
    assert v.turn == 2


def test_zero_allowance_stops_the_run(tmp_path) -> None:
    pol = {c: "lite:g35" for c in COUNTRIES}
    router = Router(
        CFG.models,
        transports={"g35": FakeTransport(http_error(429, {"x-ratelimit-limit-req-minute": "0"}))},
        clock=FakeClock(),
    )
    db = Storage(tmp_path / "e.db")
    db.write_run("Z", seed=6, policies=pol, horizon=3)
    app = _lite_app(router, open_checkpointer(tmp_path / "cp.db"), db)
    assert _drive(app, db, "Z", initial_state("Z", 6, pol, load_fixture(6), cfg=CFG, turns=3), None) == 2
    assert db.read_run("Z")["status"] == "blocked_model"
    assert drive  # imported for the API surface


# ------------------------------------------------------------------------------ runner


def test_runner_seat_spec_and_estimate(tmp_path) -> None:
    seats = seat_spec("all=tit_for_tat", "CERES=g35,DORNE=m8b")
    assert seats["CERES"] == "lite:g35" and seats["DORNE"] == "lite:m8b" and seats["FALKEN"] == "tit_for_tat"
    assert set(seat_spec("all=status_quo", "all=m8b").values()) == {"lite:m8b"}
    rows = estimate(
        seat_spec("all=status_quo", "all=g35"), 14, CFG, QuotaGuard(CFG.models, tmp_path / "q.db")
    )
    assert rows[0]["seats"] == 6 and rows[0]["calls"] == 93 and rows[0]["can_start"]
