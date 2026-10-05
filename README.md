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
- Phase 1 onward: not started.
