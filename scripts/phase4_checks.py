"""Phase 4 calibration checks (§6.15): figures to reports/figures/, tables for docs/calibration.md.

    uv run python scripts/phase4_checks.py

Figures (interactive HTML, plotly from the CDN):
  phase4_phillips.html        unemployment vs wage growth (status quo, shocks on, 20 seeds x 14 turns)
  phase4_okun.html            change in unemployment vs real GDP growth (same panel)
  phase4_price_convergence.html  mean |dP/P| per turn from the rough start (burn-in mechanics)
  phase4_cooperative_power.html  power shares over time in CooperativeBot self-play (mean of 10 seeds)
Table: every bot type in self-play, 14 turns, seeds 1-10, shocks on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.runs import CFG, run, run_bots  # noqa: E402
from world.config import COUNTRIES  # noqa: E402
from world.engine.state import initial_state  # noqa: E402
from world.engine.step import step, turn_start  # noqa: E402
from world.game import BOTS  # noqa: E402
from world.metrics import power  # noqa: E402
from world.rng import RngBundle  # noqa: E402

OUT = ROOT / "reports" / "figures"


def save(fig: go.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUT / name, include_plotlyjs="cdn")
    print(f"wrote reports/figures/{name}")


def scatter(x, y, who, title: str, xt: str, yt: str) -> go.Figure:
    fig = go.Figure()
    for c in COUNTRIES:
        m = who == c
        fig.add_trace(go.Scatter(x=x[m], y=y[m], mode="markers", name=c, marker={"size": 5, "opacity": 0.6}))
    fig.update_layout(title=title, xaxis_title=xt, yaxis_title=yt, template="plotly_white")
    return fig


def main() -> None:
    u, wg, du, rg, who = [], [], [], [], []
    for seed in range(1, 21):
        _, states, _ = run(seed, 14, shocks_on=True)
        for a, b in zip(states[:-1], states[1:], strict=True):
            u.append(100 * b.unemployment)
            wg.append(100 * (b.wage / a.wage - 1))
            du.append(100 * (b.unemployment - a.unemployment))
            rg.append(100 * ((b.gdp / b.cpi) / (a.gdp / a.cpi) - 1))
            who.append(np.array(COUNTRIES))
    u, wg, du, rg, who = map(np.concatenate, (u, wg, du, rg, who))
    save(scatter(u, wg, who, "Phillips: unemployment vs wage growth", "unemployment %", "wage growth %"),
         "phase4_phillips.html")  # fmt: skip
    save(scatter(du, rg, who, "Okun: change in unemployment vs real GDP growth", "change in u (points)",
                 "real GDP growth %"), "phase4_okun.html")  # fmt: skip

    s, rng, change = initial_state(CFG, 0), RngBundle(0), []
    for _ in range(40):
        a = s
        s, _ = turn_start(s, rng, CFG, shocks_on=False)
        s, _ = step(s, None, rng, CFG, burn_in=True)
        change.append(100 * np.abs(s.price / a.price - 1).mean())
    fig = go.Figure(go.Scatter(x=list(range(1, 41)), y=change, mode="lines+markers"))
    fig.update_layout(title="Price convergence from the rough start (no shocks)", xaxis_title="turn",
                      yaxis_title="mean |dP/P| %", yaxis_type="log", template="plotly_white")  # fmt: skip
    save(fig, "phase4_price_convergence.html")

    shares = np.mean(
        [[power(st, CFG) for st in run_bots("cooperative", sd, 14).states] for sd in range(1, 11)], 0
    )
    fig = go.Figure([go.Scatter(x=list(range(15)), y=shares[:, i], name=c) for i, c in enumerate(COUNTRIES)])
    fig.update_layout(title="CooperativeBot self-play: power share (mean of 10 seeds)", xaxis_title="turn",
                      yaxis_title="power share", template="plotly_white")  # fmt: skip
    save(fig, "phase4_cooperative_power.html")

    print(
        "\n| Bot (self-play) | world GDP end/start | min stability | leader falls "
        "| accepted / rejected actions |"
    )
    print("|---|---|---|---|---|")
    for name in sorted(BOTS):
        g, smin, falls, acc, rej = [], [], 0, 0, 0
        for seed in range(1, 11):
            r = run_bots(name, seed, 14)
            g.append(r.final.gdp.sum() / r.states[0].gdp.sum())
            smin.append(min(float(x.stability.min()) for x in r.states[1:]))
            falls += sum(len(rec.leader_changes) for rec in r.records)
            acc += sum(len(x.accepted) for rec in r.records for x in rec.seats)
            rej += sum(len(x.rejected) for rec in r.records for x in rec.seats)
        print(f"| {name} | {min(g):.3f}-{max(g):.3f} | {min(smin):.1f} | {falls} | {acc} / {rej} |")


if __name__ == "__main__":
    main()
