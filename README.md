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
- **Phase 2 (full economy step): code complete, NOT done.** Trade, labor, fiscal, monetary,
  stability, invariant checks, `step()`, and the income step (`world/engine/income.py`: wages,
  energy inputs, profits and losses, taxes). Owner decisions D30–D33 and assistant choices D34–D35
  are wired in. Every §6.14 invariant holds over 14 turns × 20 seeds, and the income identity
  closes to 1e-9. **Blocked:** the economy does not settle under status-quo policies (food floor ≈ 4×
  world food output, money draining into the bond market, stability crash in turn 1), so the 60-turn
  flow test and the energy-starvation test fail. See `docs/open_issues.md`.
- **Phase 3 (firms, shocks, trust): built.** The 14-turn × 20-seed invariant test with shocks and
  the energy_crunch scenario now passes. It waits on the Phase 2 blocker above.
