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
- **Phase 2 (full economy step): code complete, NOT done.** Trade (all 5 goods, D40), labor,
  fiscal, monetary, stability, income step, money loop, burn-in until settled (D44), per-country food
  floor (D43), balanced-benchmark calibration script (`scripts/calibrate_balance.py`, D41). Every §6.14
  invariant holds. GDP, unemployment, bond market and treasuries are now in the owner's bands.
  **Blocked:** prices leave [0.5, 2.0], stability sinks below 40 from recurring shortages, and the
  burn-in does not settle in 40 turns. See `docs/calibration.md` (pass 2) and `docs/open_issues.md`.
- **Phase 3 (firms, shocks, trust): built.** The 14-turn × 20-seed invariant test with shocks and
  the energy_crunch scenario passes (from the settled state). It waits on the Phase 2 blocker above.
