"""Run one bot game with the plain loop (no LangGraph, no LLM) and print a per-turn summary.

    uv run python scripts/run_bot_game.py --bots greedy --seed 3
    uv run python scripts/run_bot_game.py --bots status_quo --seat DORNE=aggressor --turns 14
    uv run python scripts/run_bot_game.py --bots cooperative --scenario energy_crunch --no-shocks

Bots: status_quo, tit_for_tat, greedy, cooperative, random, aggressor. Without --turns the game
length is the hidden horizon (10-14 turns, HORIZON stream).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world.config import COUNTRIES, load_config, load_scenario  # noqa: E402
from world.game import BOTS, bot_seats, run_game  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bots", default="status_quo", choices=sorted(BOTS), help="bot for every seat")
    ap.add_argument("--seat", action="append", default=[], help="COUNTRY=bot override (repeatable)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--turns", type=int, default=None)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--no-shocks", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    spec = dict.fromkeys(COUNTRIES, args.bots)
    for item in args.seat:
        country, bot = item.split("=")
        spec[country.upper()] = bot
    scenario = load_scenario(args.scenario) if args.scenario else None
    r = run_game(
        bot_seats(spec, cfg),
        seed=args.seed,
        cfg=cfg,
        turns=args.turns,
        scenario=scenario,
        shocks_on=not args.no_shocks,
    )
    print(f"seed {r.seed}  horizon {r.horizon}  status {r.status} {r.error}")
    print("seats:", ", ".join(f"{c}={spec[c]}" for c in COUNTRIES))
    print(f"{'turn':>4}  {'order':<52} {'GDP (DORNE..EVERMERE)':<44} stability")
    for rec, s in zip(r.records, r.states[1:], strict=True):
        order = " ".join(c[:3] for c in rec.order)
        gdp = " ".join(f"{x:6.2f}" for x in s.gdp)
        stab = " ".join(f"{x:4.0f}" for x in s.stability)
        print(f"{rec.turn:>4}  {order:<52} {gdp:<44} {stab}")
        acts = sum(len(x.accepted) for x in rec.seats)
        rej = sum(len(x.rejected) for x in rec.seats)
        extra = [f"{acts} accepted / {rej} rejected actions"]
        extra += [f"shock {x}" for x in rec.shocks]
        extra += [f"violation {v} -> {w}: {why}" for v, w, why in rec.violations]
        extra += [f"leader change {c}" for c in rec.leader_changes]
        print("      " + "; ".join(extra))
    final = r.final
    print("final unemployment %:", np.round(100 * final.unemployment, 1).tolist())
    print("treaties:", {t.id: t.status for t in final.treaties} or "none")


if __name__ == "__main__":
    main()
