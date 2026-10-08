"""Experiment database (CLAUDE.md §15): the SQLite schema, writers and readers.

All writes go through this module; analysis and the dashboard read only from here. Every row
carries `run_id`. Per-turn rows are written by write_turn() in ONE transaction that first deletes any
rows of that (run_id, turn): recording a turn twice (a resume after a crash between the database
write and the checkpoint) gives the same rows, never duplicates.

Tables (§15): runs, turns, country_state, bilateral, firms, shocks, decisions, actions, treaties,
commitments, predictions, facts_checks, metrics, llm_calls, referee_flags.
Numbers are stored unrounded (briefings round; analysis must not). Sector vectors in country_state
are JSON lists in sector order FOOD, ENERGY, GOODS, TECH, SERVICES.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from world.briefing import country_numbers
from world.config import COUNTRIES, DEFAULT_CONFIG_DIR, SECTORS, Config
from world.engine.shocks import ShockEvent
from world.engine.state import WorldState
from world.engine.step import TurnLog
from world.history import SeatRecord, TurnRecord
from world.metrics import power, real_gdp
from world.policies.base import DecisionResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, experiment TEXT, seed INTEGER, config_hash TEXT, models_per_seat TEXT,
    scenario TEXT, shocks_on INTEGER, horizon INTEGER, git_commit TEXT, started_at TEXT, ended_at TEXT,
    status TEXT, error TEXT, degraded_seats TEXT, parent_run_id TEXT, fork_turn INTEGER
);
CREATE TABLE IF NOT EXISTS turns (
    run_id TEXT, turn INTEGER, move_order TEXT, shocks TEXT, events TEXT, violations TEXT,
    leader_changes TEXT, state_hash TEXT, PRIMARY KEY (run_id, turn)
);
CREATE TABLE IF NOT EXISTS country_state (
    run_id TEXT, turn INTEGER, country TEXT,
    gdp REAL, real_gdp REAL, gdp_growth_pct REAL, cpi REAL, inflation_pct REAL, unemployment REAL,
    wage REAL, labor_force REAL, population REAL, tax_rate REAL, welfare_share REAL, military_share REAL,
    subsidy_share REAL, subsidy_target TEXT, gov_revenue REAL, treasury REAL, debt REAL, debt_to_gdp REAL,
    policy_rate REAL, saving_rate REAL, household_cash REAL, y_disp REAL, stability REAL, military REAL,
    default_premium REAL, default_turns_left INTEGER, leader_changes INTEGER, leader_changed_turn INTEGER,
    power REAL, output TEXT, price TEXT, stock TEXT, shortage TEXT, demand TEXT, consumption TEXT,
    imports TEXT, exports TEXT, industry_subsidy TEXT,
    PRIMARY KEY (run_id, turn, country)
);
CREATE TABLE IF NOT EXISTS bilateral (
    run_id TEXT, turn INTEGER, importer TEXT, exporter TEXT, trust REAL, sanction INTEGER,
    tariff TEXT, export_cap TEXT, trade TEXT, PRIMARY KEY (run_id, turn, importer, exporter)
);
CREATE TABLE IF NOT EXISTS firms (
    run_id TEXT, turn INTEGER, country TEXT, sector TEXT, n_firms INTEGER, hhi REAL, markup REAL,
    nationalized INTEGER, dominance_age INTEGER, shares TEXT, PRIMARY KEY (run_id, turn, country, sector)
);
CREATE TABLE IF NOT EXISTS shocks (
    run_id TEXT, turn INTEGER, shock_id TEXT, country TEXT, sector TEXT, u REAL, hazard REAL,
    fired INTEGER, scheduled INTEGER
);
CREATE TABLE IF NOT EXISTS decisions (
    run_id TEXT, turn INTEGER, country TEXT, seat INTEGER, policy TEXT, parse_failure INTEGER,
    degraded INTEGER, situation_read TEXT, private_plan TEXT, public_statement TEXT,
    stance_power REAL, stance_citizens REAL, stance_world REAL, forecast_gdp_growth_pct REAL,
    forecast_stability REAL, decision_json TEXT, raw_text TEXT, parse_error TEXT,
    PRIMARY KEY (run_id, turn, country)
);
CREATE TABLE IF NOT EXISTS actions (
    run_id TEXT, turn INTEGER, country TEXT, idx INTEGER, type TEXT, status TEXT, reason TEXT,
    action_json TEXT
);
CREATE TABLE IF NOT EXISTS treaties (
    run_id TEXT, turn INTEGER, treaty_id TEXT, kind TEXT, proposer TEXT, addressee TEXT, terms TEXT,
    duration INTEGER, status TEXT, proposed_turn INTEGER, start_turn INTEGER, end_turn INTEGER, note TEXT
);
CREATE TABLE IF NOT EXISTS commitments (
    run_id TEXT, turn INTEGER, country TEXT, kind TEXT, target TEXT, turns INTEGER, max_rate REAL,
    status TEXT, reason TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    run_id TEXT, turn INTEGER, country TEXT, target TEXT, move TEXT, probability REAL, status TEXT,
    reason TEXT
);
CREATE TABLE IF NOT EXISTS facts_checks (
    run_id TEXT, turn INTEGER, country TEXT, fact_country TEXT, metric TEXT, stated REAL,
    true_value REAL, wrong INTEGER
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT, turn INTEGER, country TEXT, name TEXT, value REAL
);
CREATE TABLE IF NOT EXISTS llm_calls (
    run_id TEXT, turn INTEGER, country TEXT, policy TEXT, model TEXT, provider TEXT, tokens_in INTEGER,
    tokens_out INTEGER, latency_s REAL, retries INTEGER, raw_output TEXT, parsed_output TEXT, error TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS referee_flags (
    run_id TEXT, turn INTEGER, country TEXT, model TEXT, flag TEXT, detail TEXT
);
"""

TURN_TABLES = (
    "turns", "country_state", "bilateral", "firms", "shocks", "decisions", "actions", "treaties",
    "commitments", "predictions", "facts_checks", "metrics", "llm_calls", "referee_flags",
)  # fmt: skip


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _j(x: Any) -> str:
    if isinstance(x, np.ndarray):
        x = x.tolist()
    return json.dumps(x)


def config_hash(config_dir: Path = DEFAULT_CONFIG_DIR) -> str:
    """sha256 over every config yaml (path + content), so runs on different parameters differ."""
    h = hashlib.sha256()
    for p in sorted(config_dir.rglob("*.yaml")):
        h.update(str(p.relative_to(config_dir)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def git_commit() -> str:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class Storage:
    """One experiment database (data/experiments.db by default)."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL") if self.path != ":memory:" else None
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------------------- runs
    def write_run(
        self,
        run_id: str,
        *,
        seed: int,
        policies: dict[str, str],
        horizon: int,
        experiment: str = "",
        scenario: str | None = None,
        shocks_on: bool = True,
        parent_run_id: str | None = None,
        fork_turn: int | None = None,
        status: str = "running",
    ) -> None:
        """Create (or reset) the run row. Called once when a run starts or a fork is made."""
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id, experiment, seed, config_hash(), _j(policies), scenario, int(shocks_on), horizon,
                    git_commit(), _now(), None, status, "", "[]", parent_run_id, fork_turn,
                ),
            )  # fmt: skip

    def set_status(
        self,
        run_id: str,
        status: str,
        *,
        error: str = "",
        degraded_seats: Sequence[str] = (),
        ended: bool = False,
    ) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET status=?, error=?, degraded_seats=?, ended_at=? WHERE run_id=?",
                (status, error, _j(list(degraded_seats)), _now() if ended else None, run_id),
            )

    def copy_turns(self, src_run: str, dst_run: str, before_turn: int) -> None:
        """A fork starts with its parent's rows for turns < before_turn (the shared past)."""
        with self.conn:
            for table in TURN_TABLES:
                cols = [r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")]
                rest = ", ".join(c for c in cols if c != "run_id")
                self.conn.execute(f"DELETE FROM {table} WHERE run_id=?", (dst_run,))
                self.conn.execute(
                    f"INSERT INTO {table} (run_id, {rest}) SELECT ?, {rest} FROM {table} "
                    "WHERE run_id=? AND turn<?",
                    (dst_run, src_run, before_turn),
                )

    # ------------------------------------------------------------------------------ turns
    def write_turn(
        self,
        run_id: str,
        record: TurnRecord,
        state: WorldState,
        log: TurnLog,
        start: WorldState,
        cfg: Config,
        *,
        shock_events: Iterable[ShockEvent] = (),
        results: dict[str, DecisionResult] | None = None,
    ) -> None:
        """Everything about one resolved turn. `state` = the state after the turn, `start` = the game's
        starting (settled) state, used for real GDP and the power index (D61)."""
        t = record.turn
        rows: dict[str, list[tuple]] = {k: [] for k in TURN_TABLES}
        rows["turns"].append((
            run_id, t, _j(record.order), _j(record.shocks), _j(record.events), _j(record.violations),
            _j(record.leader_changes), state.state_hash(),
        ))  # fmt: skip
        rgdp = real_gdp(state, start)
        pw = power(state, cfg, start)
        cash, treas = state.household_cash, state.treasury
        for i, c in enumerate(COUNTRIES):
            n = country_numbers(state, i, cfg)
            rows["country_state"].append((
                run_id, t, c, float(state.gdp[i]), float(rgdp[i]), n["gdp_growth_pct"], float(state.cpi[i]),
                n["inflation_pct"], float(state.unemployment[i]), float(state.wage[i]),
                float(state.labor_force[i]), float(state.population[i]), float(state.tax_rate[i]),
                float(state.welfare_share[i]), float(state.military_share[i]), float(state.subsidy_share[i]),
                SECTORS[state.subsidy_target[i]] if state.subsidy_target[i] >= 0 else None,
                float(state.gov_revenue[i]), float(treas[i]), float(state.debt[i]), n["debt_to_gdp"],
                float(state.policy_rate[i]), float(state.saving_rate[i]), float(cash[i]),
                float(state.y_disp[i]),
                float(state.stability[i]), float(state.military[i]), float(state.default_premium[i]),
                int(state.default_turns_left[i]), int(state.leader_changes[i]),
                int(state.leader_changed_turn[i]), float(pw[i]), _j(state.output[i]), _j(state.price[i]),
                _j(state.stock[i]), _j(state.shortage[i]), _j(state.demand[i]), _j(log.consumption[i]),
                _j(log.imports[i]), _j(log.exports[i]), _j(state.industry_subsidy[i]),
            ))  # fmt: skip
            rows["metrics"] += [
                (run_id, t, c, "power", float(pw[i])),
                (run_id, t, c, "real_gdp", float(rgdp[i])),
            ]
            for j, d in enumerate(COUNTRIES):
                if i != j:
                    rows["bilateral"].append((
                        run_id, t, c, d, float(state.trust[i, j]), int(state.sanction[i, j]),
                        _j(state.tariff[i, j]), _j(state.export_cap[i, j]), _j(state.trade[i, j]),
                    ))  # fmt: skip
            for g, sec in enumerate(SECTORS):
                shares = [f.share for f in state.firms_of(i, g)]
                rows["firms"].append((
                    run_id, t, c, sec, int(state.n_firms[i, g]), float(log.hhi[i, g]),
                    float(state.markup[i, g]), int(state.nationalized[i, g]), int(state.dominance_age[i, g]),
                    _j(shares),
                ))  # fmt: skip
        for e in shock_events:
            rows["shocks"].append((
                run_id, t, e.shock_id, COUNTRIES[e.country] if e.country >= 0 else None,
                SECTORS[e.sector] if e.sector >= 0 else None, float(e.u), float(e.hazard), int(e.fired),
                int(e.scheduled),
            ))  # fmt: skip
        for tr in state.treaties:
            rows["treaties"].append((
                run_id, t, tr.id, tr.kind, COUNTRIES[tr.proposer], COUNTRIES[tr.addressee],
                _j({k: (v.item() if isinstance(v, np.generic) else v) for k, v in tr.terms}), tr.duration,
                tr.status, tr.proposed_turn, tr.start_turn, tr.end_turn, tr.note,
            ))  # fmt: skip
        for r in record.seats:
            _seat_rows(rows, run_id, t, r, (results or {}).get(r.country))
        with self.conn:
            for table in TURN_TABLES:
                self.conn.execute(f"DELETE FROM {table} WHERE run_id=? AND turn=?", (run_id, t))
                if rows[table]:
                    marks = ",".join("?" * len(rows[table][0]))
                    self.conn.executemany(f"INSERT INTO {table} VALUES ({marks})", rows[table])

    # ------------------------------------------------------------------------------ reads
    def read_run(self, run_id: str) -> dict[str, Any] | None:
        cur = self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
        row = cur.fetchone()
        if row is None:
            return None
        out = dict(zip([c[0] for c in cur.description], row, strict=True))
        out["models_per_seat"] = json.loads(out["models_per_seat"])
        out["degraded_seats"] = json.loads(out["degraded_seats"] or "[]")
        return out

    def read(self, table: str, run_id: str) -> pd.DataFrame:
        """All rows of one table for one run, ordered by turn when the table has one."""
        if table not in TURN_TABLES and table != "runs":
            raise ValueError(f"unknown table {table}")
        order = " ORDER BY turn, rowid" if table != "runs" else ""
        return pd.read_sql_query(f"SELECT * FROM {table} WHERE run_id=?{order}", self.conn, params=(run_id,))

    def run_ids(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT run_id FROM runs ORDER BY started_at, run_id")]


def _seat_rows(
    rows: dict[str, list[tuple]], run_id: str, t: int, r: SeatRecord, res: DecisionResult | None
) -> None:
    """decisions, actions, commitments, predictions, facts_checks and llm_calls rows of one seat."""
    key = (run_id, t, r.country)
    d = r.decision
    st, fc = (d.stance, d.forecast) if d else (None, None)
    rows["decisions"].append((
        *key, r.seat, r.policy, int(r.parse_failure), int(r.degraded),
        d.situation_read if d else None, d.private_plan if d else None, r.public_statement,
        st.power if st else None, st.citizens if st else None, st.world if st else None,
        fc.my_gdp_growth_pct if fc else None, fc.my_stability_next if fc else None,
        d.model_dump_json() if d else None, (res.raw_text or None) if res else None,
        res.parse_error if res else None,
    ))  # fmt: skip
    acts = [(a, "accepted", "") for a in r.accepted] + [(a, "rejected", why) for a, why in r.rejected]
    for k, (a, status, why) in enumerate(acts):
        rows["actions"].append((*key, k, a.type, status, why, a.model_dump_json()))
    cms = [(c, "kept", "") for c in r.commitments] + [(c, "dropped", why) for c, why in r.dropped_commitments]
    for c, status, why in cms:
        rows["commitments"].append((*key, c.kind, c.target, c.turns, c.max_rate, status, why))
    preds = [(p, "kept", "") for p in r.predictions] + [(p, "dropped", w) for p, w in r.dropped_predictions]
    for p, status, why in preds:
        rows["predictions"].append((*key, p.country, p.move, p.probability, status, why))
    for f in r.fact_checks:
        rows["facts_checks"].append((*key, f.country, f.metric, f.stated, f.true, int(f.wrong)))
    if res is not None and (res.raw_text or res.tokens_in or res.tokens_out):
        rows["llm_calls"].append((
            *key, r.policy, None, None, res.tokens_in, res.tokens_out, res.latency_s, res.retries,
            res.raw_text, d.model_dump_json() if d else None, res.parse_error, _now(),
        ))  # fmt: skip
