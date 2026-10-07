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
