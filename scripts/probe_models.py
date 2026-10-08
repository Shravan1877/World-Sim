"""Phase 6 probe (CLAUDE.md §11.7, PHASES.md Phase 6 step 0). LIVE: uses real API calls.

For each model key, send the real system prompt + a real turn-1 briefing (DORNE, seed 1) with the
full flat TurnDecision schema, once per structured-output method (json_schema, function_calling,
json_mode), and record: does the id work, valid TurnDecision JSON, Layer-1 accepted/rejected actions,
real prompt / completion / thinking tokens, latency, rate-limit headers. Hard cap: MAX_CALLS calls.
A model whose id fails is not retried with other methods (two attempts at most, then move on).
Every call goes through the model's ModelLimiter and the quota ledger (data/quota.db).
No cache, no hidden transient retries (each attempt is one counted call); results are saved after
every call.

    uv run python scripts/probe_models.py [--keys m3b,m8b] [--max-calls 20]
Results: data/probe_results.json (gitignored) and stdout. Keys are never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world.config import COUNTRIES, load_config  # noqa: E402
from world.engine.burn_in import settled_state  # noqa: E402
from world.engine.step import turn_start  # noqa: E402
from world.game import seat_briefing, shock_text  # noqa: E402
from world.llm.limiter import ModelLimiter, QuotaPause  # noqa: E402
from world.llm.prompting import render_system_prompt, render_turn_message  # noqa: E402
from world.llm.quota import QuotaGuard  # noqa: E402
from world.llm.router import ModelClient, RunStop, keys_present  # noqa: E402
from world.rng import RngBundle  # noqa: E402
from world.validator import Context, validate  # noqa: E402

METHODS = ("json_schema", "function_calling", "json_mode")
OUT = Path("data/probe_results.json")


def sample_messages(country: str = "DORNE", seed: int = 1):
    cfg = load_config()
    s, ts = turn_start(settled_state(cfg, seed), RngBundle(seed), cfg)
    order = ts.order
    b = seat_briefing(s, country, order, tuple(shock_text(e) for e in ts.fired), (), (), cfg)
    return cfg, s, b, [("system", render_system_prompt(country, cfg)), ("user", render_turn_message(b))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default="m3b,m8b,m14b,g31,g35")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--max-calls", type=int, default=20)
    a = ap.parse_args(argv)
    present = keys_present()
    print("keys present:", present)
    cfg, s, b, messages = sample_messages()
    print(f"prompt chars: system {len(messages[0][1])}, user {len(messages[1][1])}")
    models = cfg.models
    quota = QuotaGuard(models)
    results = json.loads(OUT.read_text()) if OUT.exists() else []
    calls = 0
    for key in a.keys.split(","):
        m = models.models[key]
        env = "MISTRAL_API_KEY" if m.provider == "mistral" else "GOOGLE_API_KEY"
        if not present[env]:
            print(f"{key}: {env} missing in .env, skipped")
            continue
        limiter = ModelLimiter(key, m, safety=models.defaults.safety,
                               max_output_tokens=models.defaults.max_output_tokens, quota=quota)  # fmt: skip
        failures = 0
        for method in a.methods.split(","):
            if calls >= a.max_calls:
                print("call cap reached")
                break
            client = ModelClient(key, models, limiter=limiter, method=method, transient_retries=0)
            row = {"key": key, "model_id": m.model_id, "method": method}
            calls += 1
            try:
                out = client.call(messages)
            except (RunStop, QuotaPause) as e:
                row.update(ok_id=False, error=f"{type(e).__name__}: {e}"[:600])
                results.append(row)
                print(json.dumps(row), flush=True)
                OUT.write_text(json.dumps(results, indent=1))
                failures += 1
                if isinstance(e, RunStop) or failures >= 2:
                    break
                continue
            except Exception as e:  # noqa: BLE001 - probe records every failure
                row.update(ok_id=None, error=f"{type(e).__name__}: {e}"[:600])
                results.append(row)
                print(json.dumps(row), flush=True)
                OUT.write_text(json.dumps(results, indent=1))
                failures += 1
                if failures >= 2:
                    break
                continue
            vr = (
                validate(out.parsed, Context(s, COUNTRIES.index("DORNE"), b.still_to_move, cfg))
                if out.parsed
                else None
            )
            hdr = {k: v for k, v in out.headers.items() if k.startswith("x-ratelimit") or k == "retry-after"}
            row.update(
                ok_id=True,
                valid=out.parsed is not None,
                parse_error=out.parse_error,
                tokens_in=out.tokens_in,
                tokens_out=out.tokens_out,
                tokens_reasoning=out.tokens_reasoning,
                latency_s=round(out.latency_s, 2),
                model_name=out.model_name,
                raw_chars=len(out.raw_text),
                accepted=len(vr.actions) if vr else None,
                rejected=[w for _, _, w in vr.rejected] if vr else None,
                headers=hdr,
                raw_head=out.raw_text[:300],
            )
            results.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "raw_head"}), flush=True)
            OUT.write_text(json.dumps(results, indent=1))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    print(f"live calls this probe: {calls}; results in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
