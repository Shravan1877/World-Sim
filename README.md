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
- **Phase 2 (trade, labor, fiscal, monetary, stability, invariant checks): parts built** (committed
  together with Phase 3). `step()` was missing and is now in `world/engine/step.py`.
- **Phase 3 (firms, shocks, trust): built, NOT done.** Firms (markups, antitrust, breakups,
  startup entry, exit, HHI, profit payout), military, trust, all §7 shocks with timed multipliers,
  scenario incidents (`config/scenarios/energy_crunch.yaml`), leader change, move order, and the full
  `turn_start()` + `step()` in §6.1 order. **Blocked:** the 14-turn invariants test fails on turn 7 in
  all 20 seeds. Energy importers lose all energy after turn 1 (firm energy inputs come from stock,
  and nothing refills it), their GDP reaches 0, and the debt premium becomes infinite. This needs a
  CLAUDE.md decision (see the Phase 3 summary).
