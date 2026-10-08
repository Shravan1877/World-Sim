# ruff: noqa: E501  (long f-strings in a print-only report script)
"""Print the decision log of one turn of a run, in plain text, from data/experiments.db.

uv run python scripts/show_turn.py RUN_ID TURN [--db data/experiments.db]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world.storage import Storage  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("turn", type=int)
    ap.add_argument("--db", default="data/experiments.db")
    a = ap.parse_args(argv)
    db = Storage(a.db)
    t = a.turn
    turn = db.read("turns", a.run_id).query(f"turn == {t}").iloc[0]
    print(f"Run {a.run_id}, quarter {t}. Move order: {', '.join(json.loads(turn.move_order))}")
    print(
        f"Shocks: {', '.join(json.loads(turn.shocks)) or 'none'}. Events: {'; '.join(json.loads(turn.events)) or 'none'}"
    )
    dec = db.read("decisions", a.run_id).query(f"turn == {t}").sort_values("seat")
    acts = db.read("actions", a.run_id).query(f"turn == {t}")
    calls = db.read("llm_calls", a.run_id).query(f"turn == {t}")
    preds = db.read("predictions", a.run_id).query(f"turn == {t}")
    comms = db.read("commitments", a.run_id).query(f"turn == {t}")
    for _, d in dec.iterrows():
        c = d.country
        print(f"\n--- seat {d.seat}: {c} ({d.policy}){' PARSE FAILURE -> wait' if d.parse_failure else ''}")
        for _, k in calls[calls.country == c].iterrows():
            print(f"    call: {k.provider}/{k.model_id} method={k.method} attempt={k.attempt} "
                  f"tokens in/out/thinking={k.tokens_in}/{k.tokens_out}/{k.tokens_reasoning} "
                  f"latency={k.latency_s:.1f}s cached={bool(k.cached)}{' error: ' + str(k.error)[:120] if isinstance(k.error, str) and k.error else ''}")  # fmt: skip
        if d.situation_read:
            print(f"  situation read : {d.situation_read}")
        print(f"  public         : {d.public_statement}")
        if d.private_plan:
            print(f"  private plan   : {d.private_plan}")
        if d.stance_power is not None and d.stance_power == d.stance_power:
            print(
                f"  stance         : power {d.stance_power:.2f}, citizens {d.stance_citizens:.2f}, world {d.stance_world:.2f}"
            )
        for _, p in preds[preds.country == c].iterrows():
            print(
                f"  predicts       : {p.target} -> {p.move} (p={p.probability:.2f}){'' if p.status == 'kept' else ' DROPPED: ' + p.reason}"
            )
        for _, x in acts[acts.country == c].iterrows():
            body = {k: v for k, v in json.loads(x.action_json).items() if v is not None and k != "type"}
            note = f"  [{x.reason}]" if x.reason else ""
            print(f"  action {x.status:8s}: {x.type} {body}{note}")
        for _, m in comms[comms.country == c].iterrows():
            print(f"  commits        : {m.kind} toward {m.target} for {m.turns} quarters ({m.status})")
        if d.forecast_gdp_growth_pct is not None and d.forecast_gdp_growth_pct == d.forecast_gdp_growth_pct:
            print(
                f"  forecast       : GDP growth {d.forecast_gdp_growth_pct:+.1f}%, stability {d.forecast_stability:.0f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
