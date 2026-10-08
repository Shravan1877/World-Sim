# Quota plan

Measured tokens per call, runs per day per model, and the E1 schedule (CLAUDE.md §11.7, §13.2).
Measured on 2026-10-08 (Phase 6 probe + live smoke games). Regenerate after any prompt change.

## 1. What was measured

All numbers are for one full lite-mode decision: the system prompt (~6.5k characters) + one briefing
+ the flat `TurnDecision` schema, `max_output_tokens` 2000, temperature 0.7. "Live" excludes cache
replays. Sources: `llm_calls` in `data/experiments.db` (runs `smoke-*`) and `data/probe_results.json`.

| Key | Model id | Method | Live calls | Tokens in | Tokens out | Thinking | Tokens / call | Latency | Parse failures | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| `m8b` | ministral-8b-2512 | json_schema | 42 | 3,652 | 1,316 | 0 | **4,968** | 16.1 s | 0 of 42 (0 retries) | verified |
| `g35` | gemini-3.5-flash-lite | json_schema | 44 | 3,103 | 691 | 0 | **3,794** | 16.6 s | 0 of 44 (3 retries, all recovered) | verified |
| `g31` | gemini-3.1-flash-lite | json_schema | 1 (probe) | 2,436* | 800 | 0 | ~3,900 (plan) | 5.1 s | 0 of 1 | verified |
| `m3b` | ministral-3b-2512 | json_schema (tentative) | 3 (probe) | – | – | – | 3,477–4,612 | ~4 s | validity not recorded | **unverified** |
| `m14b` | ministral-14b-2512 | function_calling (tentative) | 9 attempts (probe) | – | – | – | 4,699 (1 answer) | ~41 s; 7 timeouts at 60 s | validity not recorded | **unverified** |

\* probe prompt; the final prompt is ~650 input tokens longer (measured on g35), so g31 is planned at 3,900.

Other findings:
- Mistral response headers match the published limits for m8b: 188 requests/min, 625,000 tokens/min;
  `x-ratelimit-tokens-query-cost` = prompt + actual completion tokens.
- Gemini: `thinking_level: minimal` (the lowest level in ai.google.dev/gemini-api/docs/thinking for
  Flash-Lite) gives 0 thinking tokens. LangChain reports Gemini `output_tokens` including thinking.
- Gemini success responses carry no rate-limit headers. Whether g31 and g35 share one daily counter is
  **not confirmed** (open issue P6-3). They are planned as separate pools; the shared case is shown below.
- Mistral json_schema is non-strict in LangChain (strict=False), so a small model can leave out
  fields; the typed answer shape in the system prompt fixed this for m8b (D70).
- m8b fills every field of the flat action object; D69 ignores the unused ones (33/36 actions
  accepted in the 1 LLM + 5 bots game, 204/216 with 6 LLMs).

### Why m3b and m14b are out (for now)
The first full probe wrote its results only at the end; it was stopped when m14b's 60 s timeouts were
using up the 20-call probe budget, so per-method validity for m3b and m14b was lost (my bug, fixed:
results are saved after every call and the probe does no hidden retries). The quota ledger kept every
attempt: m3b answered all three methods in ~4 s (3.5k–4.6k tokens); m14b timed out on json_schema
(4×) and json_mode (3×) and answered once with function_calling after ~41 s. The HTTP timeout is now
120 s. **A re-probe needs the owner's approval:** 6 calls (m3b and m14b × 3 methods), at most 12.

## 2. Pace and runs per day

One run = 12 turns (the mean of the hidden horizon 10–14) × 6 seats × 1.1 for retries = **79 calls**
(66–92). Limits at 80% (D19). A lane makes one call at a time (sequential game).

| Key | Min spacing | Time per call | One run | Tokens per run | Binding limit | Runs per day |
|---|---|---|---|---|---|---|
| `m8b` | 0.48 s | 16.1 s (latency) | ~21 min | ~392k | model speed | ~65 if run all day; no daily cap |
| `g35` | 5.0 s | 16.6 s (latency) | ~22 min | ~300k | **400 requests/day** | **5** (395 calls) |
| `g31` | 5.0 s | ~5.1 s | ~7 min | ~308k | **400 requests/day** | **5** |
| `m3b` (if verified) | 0.23 s | ~4 s | ~5 min | ~320k | model speed | no daily cap |
| `m14b` (if verified) | 2.5 s | ~41 s | ~54 min | ~370k | model speed | ~25 |

Token windows never bind: g35 uses ~3.6 calls/min × 3.8k ≈ 14k tokens/min (cap 200k at 80%); m8b
~3.7 calls/min × 5k ≈ 19k (cap 500k). Mistral's monthly cap (1B, 800M at 80%) is not a constraint:
all of E1 on Mistral is ~11M tokens.
**Shared Gemini pool:** if g31 and g35 share one daily counter, each gets ~200 calls/day (2.5 runs).

## 3. Does the 4,000 tokens/call planning number hold? (§13.2, D21)

Mostly. Gemini is at 3.8–3.9k (below plan). m8b is at **5.0k (+24%)** because its answers are long
(1.3k completion tokens). The mean over measured models is ~4.2k. Proposed: plan per model with the
measured values (m8b 5,000; g31/g35 3,900; m3b/m14b 4,000 until measured). E1 total ≈ 17M tokens
(plan was 15.8M). Gemini is limited by requests per day, not tokens, so this changes no schedule.

## 4. E1 schedule (5 models × 10 seeds, `energy_crunch`)

Gemini resets at midnight Pacific = **12:30 IST until 1 Nov 2026, 13:30 IST after**. Start the Gemini
lanes right after the reset. Today (2026-10-08) g35 already used 56 calls and g31 1 for the smoke
tests and the probe.

Before day 1: the owner approves D69 and the m3b/m14b re-probe (≤ 12 calls); E0 (bots, no quota);
`docs/preregistration.md` committed (§13.5).

| Day | Gemini lanes (start 12:30 IST) | Mistral lanes (any time, in parallel) |
|---|---|---|
| 1 | g31 seeds 1–5 (~395 calls, ~35 min); g35 seeds 1–5 (~395 calls, ~1 h 50 min) | m8b seeds 1–10 (~3.5 h); m3b seeds 1–10 (~1 h, if verified); m14b seeds 1–5 (~4.5 h, if verified) |
| 2 | g31 seeds 6–10; g35 seeds 6–10 | m14b seeds 6–10 (~4.5 h) |
| 3 | E2b (m14b vs g35, ~400 g35 calls) | E2a (m3b, m8b, m14b) |

If g31 and g35 share one pool: days 1–4, alternating 2–3 runs per model per day (E1 Gemini done on
day 4, E2b on day 5). The runner prints the estimate and today's remaining quota before each run and
pauses (status `paused`) at a seat boundary if a run does not fit; `runner resume` continues it after
the reset.
