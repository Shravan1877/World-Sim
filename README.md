# WorldSim

WorldSim is a six-country simulated economy for studying how LLM agents behave in a simulated economy with scarce resources.
Each country is led by an LLM agent that only makes decisions: taxes, tariffs, sanctions, treaties and special powers.
A deterministic Python/numpy engine computes everything that happens: production, prices, trade, jobs, budgets, debt and stability.
Because the engine knows the ground truth, we can measure each agent's power gain against the collateral damage it causes, along with promise keeping, prediction skill, escalation and hallucination.
The engine is inspired by published economic and agent-simulation models (see CLAUDE.md §19). It does not replicate them.

The full design lives in [CLAUDE.md](CLAUDE.md).

## Quick start

```bash
uv sync
uv run pytest          # live LLM tests are skipped unless you pass --live
uv run ruff check
```

## Status

- **Phase 0 (repo setup): done.** uv project, folder layout (empty modules), all config files
  (`config/*.yaml`), validated config loader (`world/config.py`), config tests.
- **Phase 1 (engine core): done.** Seeded RNG streams (`world/rng.py`), double-entry ledger
  (`world/ledger.py`), vectorized world state + initial state + state hash
  (`world/engine/state.py`), production, demand and prices (`world/engine/`).
- **Phase 2 (full economy step): done.** Trade as one Armington market per good with home bias (D45),
  labor, fiscal, monetary, stability, the income step, the money loop (D37), a burn-in that runs until
  settled (D44/D47), per-country food floors (D43) and the balanced-benchmark calibration script
  (`scripts/calibrate_balance.py`, D41). Every §6.14 invariant holds, and the 60-turn × 20-seed flow
  check passes (`tests/test_flow.py`). Parameters that differ from the spec and open points:
  `docs/calibration.md` (pass 3) and `docs/open_issues.md`.
- **Phase 3 (firms, shocks, trust): done.** The 14-turn × 20-seed run with shocks and the
  energy_crunch scenario passes from the settled state.
- **Phase 4 (actions, treaties, bots, calibration): done.** Strict typed actions with a flat
  `ActionIn` wire format and the full `TurnDecision` schema (`world/actions.py`), the validator
  (Layer 1 rules and the Layer 2 fact check, `world/validator.py`), policy application and special
  powers (`world/engine/policy.py`), treaties with execution and violation detection inside `step()`
  (`world/engine/treaties.py`), briefings as data (`world/briefing.py`), six scripted bots
  (status quo, tit-for-tat, greedy, cooperative, random, aggressor) and a plain-loop game
  (`world/game.py`, `scripts/run_bot_game.py`). Required tests pass: 14-turn × 20-seed bot games,
  macro signs (Phillips, Okun, price convergence), no country dominating the cooperative baseline,
  roster characters and shock bite. Decisions D52–D60; numbers in `docs/calibration.md`.
- **Phase 5 (LangGraph loop): done.** The turn loop as a LangGraph `StateGraph` (`world/graph.py`:
  turn_start → leader → validate_apply ×6 → resolve → record) with SQLite checkpoints
  (`data/checkpoints.db`, thread = run id), an exact plain-data encoding of the game state
  (`world/serial.py`), the briefing as prompt text with all nine §11.2 sections
  (`world/llm/prompts/briefing.md.j2`), the experiment database (`world/storage.py`, §15) and a CLI
  (`python -m world.runner run | resume | fork`). The graph gives the same state every turn as the
  plain loop; interrupt/resume, crash/resume and forks work. Metrics use real GDP (D61).
  Decisions D61–D67; open points in `docs/open_issues.md`.
