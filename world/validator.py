"""Checking a leader's TurnDecision (CLAUDE.md §11.3, §11.5 Layers 1 and 2, D29).

Layer 1 (code, always on), per action -> accepted or rejected(reason), never a crash:
  1. convert the flat ActionIn into its strict typed action: missing or extra fields for that type,
     wrong enums, out-of-range values, bad treaty terms -> rejected;
  2. the action is allowed for this country (shared actions + its special actions, countries.yaml);
  3. targets exist and are not the actor itself;
  4. treaty ids exist, are addressed to this country (accept/reject) or include it (renounce), and
     are still open (proposals) or active (renounce);
  5. loans: a lender other than AURELIA (loan_unlimited) may lend at most 10% of its treasury;
  6. in default: no spending plan with outlays above last turn's revenue;
  7. duplicates: only the first set_tax / set_spending / set_policy_rate counts;
  8. contradictory pairs in one turn (sanction on and off the same target, food ban on and off the
     same target, accept and reject the same treaty) -> both rejected.
  Predictions must name a country that is still to move (dropped and counted, never retried).
  Commitments must name another country; keep_treaty needs a treaty with it; tariff_cap needs a
  max_rate in the tariff range (invalid ones are dropped and counted).
Layer 2 (code, always on, a metric only): each facts_used entry against the briefing value; wrong if
  the relative error is above 5% or the (country, metric) is not in the briefing.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from world.actions import (
    ACTION_SPECS,
    Action,
    ActionIn,
    Commitment,
    Fact,
    Prediction,
    ProposeTreaty,
    TermsError,
    TurnDecision,
    parse_terms,
    short_error,
)
from world.config import COUNTRIES, Config
from world.engine import treaties as treaties_mod
from world.engine.state import WorldState
from world.history import FactCheck

ONCE_PER_TURN = ("set_tax", "set_spending", "set_policy_rate")
ON_OFF_PAIRS = ("set_sanction", "set_food_export_ban")
TREATY_ID_ACTIONS = ("accept_treaty", "reject_treaty", "renounce_treaty")
SPENDING_ACTIONS = ("set_spending", "subsidize_industry")


@dataclass(frozen=True)
class Context:
    state: WorldState  # the state when this country moves (earlier movers' actions applied)
    country: int
    still_to_move: tuple[str, ...]
    cfg: Config


@dataclass(frozen=True)
class ValidationResult:
    accepted: tuple[tuple[int, Action], ...]  # (index in decision.actions, strict action)
    rejected: tuple[tuple[int, ActionIn, str], ...]
    predictions: tuple[Prediction, ...]
    dropped_predictions: tuple[tuple[Prediction, str], ...]
    commitments: tuple[Commitment, ...]
    dropped_commitments: tuple[tuple[Commitment, str], ...]
    notes: tuple[str, ...] = ()  # one per accepted action: "" or "ignored unused fields: ..." (D69)

    @property
    def actions(self) -> list[Action]:
        return [a for _, a in self.accepted]


def allowed_actions(cfg: Config, country: int) -> frozenset[str]:
    c = cfg.countries.countries[country]
    return frozenset(cfg.countries.shared_actions) | frozenset(c.special_actions) | {"wait"}


# --------------------------------------------------------------------------- conversion


def unused_fields(a: ActionIn) -> list[str]:
    """Filled-in fields that this action type does not use (D69: ignored, noted, not rejected)."""
    _, required, optional = ACTION_SPECS[a.type]
    return sorted(a.given().keys() - required - optional)


def convert(a: ActionIn) -> Action | str:
    """ActionIn -> strict action, or a human-readable reason. Fields the type does not use are dropped
    (D69: small models fill every field of the flat wire object); a missing field or a bad value
    still rejects the action."""
    model, required, optional = ACTION_SPECS[a.type]
    given = {k: v for k, v in a.given().items() if k in required | optional}
    missing = sorted(required - given.keys())
    if missing:
        return f"{a.type}: missing {', '.join(missing)}"
    data = dict(given)
    if model is ProposeTreaty:
        try:
            data["terms"] = parse_terms(str(given["kind"]), given.get("terms"))  # type: ignore[arg-type]
        except TermsError as e:
            return f"propose_treaty: bad terms ({e})"
    try:
        return model.model_validate(data)  # type: ignore[return-value]
    except ValidationError as e:
        return f"{a.type}: {short_error(e)}"


# ------------------------------------------------------------------------------- Layer 1


def _check_one(a: Action, ctx: Context) -> str | None:
    """Rules 2-6 for one converted action. None = fine."""
    cfg, s, i = ctx.cfg, ctx.state, ctx.country
    me = COUNTRIES[i]
    if a.type not in allowed_actions(cfg, i):
        return f"{a.type} is not an action {me} can take"
    target = getattr(a, "target", None)
    if target == me:
        return f"{a.type}: target cannot be yourself"
    if a.type in TREATY_ID_ACTIONS:
        t = treaties_mod.by_id(s.treaties, a.treaty_id)  # type: ignore[union-attr]
        if t is None:
            return f"{a.type}: no treaty {a.treaty_id}"  # type: ignore[union-attr]
        if a.type == "renounce_treaty":
            if i not in t.parties():
                return f"renounce_treaty: {t.id} is not your treaty"
            if t.status != "active":
                return f"renounce_treaty: {t.id} is {t.status}, not active"
            return None
        if t.addressee != i:
            return f"{a.type}: {t.id} is not addressed to you"
        if not treaties_mod.proposal_open(t, s.turn, cfg.world.treaties.proposal_expiry_turns):
            return f"{a.type}: {t.id} is {t.status if t.status != 'proposed' else 'expired'}"
        if a.type == "accept_treaty" and t.kind == "loan" and int(t.term("lender")) == i:
            return _loan_limit(float(t.term("amount")), i, ctx)
        return None
    if isinstance(a, ProposeTreaty) and a.terms.kind == "loan" and a.terms.role == "lender":
        return _loan_limit(a.terms.amount, i, ctx)
    if a.type in SPENDING_ACTIONS and s.default_turns_left[i] > 0:
        return _default_rule(a, ctx)
    return None


def _loan_limit(amount: float, lender: int, ctx: Context) -> str | None:
    if ctx.cfg.countries.countries[lender].params.loan_unlimited:
        return None
    cap = ctx.cfg.world.treaties.loan_max_treasury_share * max(float(ctx.state.treasury[lender]), 0.0)
    if amount > cap + 1e-12:
        return f"loan of {amount:.3g} is above your limit of {cap:.3g} (10% of treasury)"
    return None


def _default_rule(a: Action, ctx: Context) -> str | None:
    """§6.7: in default, planned outlays may not exceed last turn's revenue."""
    s, i = ctx.state, ctx.country
    gdp = max(float(s.gdp[i]), 0.0)
    shares = {"welfare": s.welfare_share[i], "military": s.military_share[i], "subsidy": s.subsidy_share[i]}
    extra = float(s.industry_subsidy[i].sum())
    if a.type == "set_spending":
        shares = {"welfare": a.welfare, "military": a.military, "subsidy": a.subsidy}  # type: ignore[union-attr]
    else:
        extra += a.amount  # type: ignore[union-attr]
    ppy = ctx.cfg.world.periods_per_year
    interest = float(s.debt[i]) * (float(s.policy_rate[i]) + float(s.default_premium[i])) / ppy
    outlays = (sum(shares.values()) + extra) * gdp + interest
    if outlays > float(s.gov_revenue[i]) + 1e-12:
        return f"in default: planned outlays {outlays:.3g} exceed revenue {float(s.gov_revenue[i]):.3g}"
    return None


def _pair_key(a: Action) -> tuple[str, str] | None:
    if a.type in ON_OFF_PAIRS:
        return a.type, str(a.target)  # type: ignore[union-attr]
    if a.type in ("accept_treaty", "reject_treaty"):
        return "treaty_answer", str(a.treaty_id)  # type: ignore[union-attr]
    return None


def _pair_value(a: Action) -> object:
    return a.type if a.type in ("accept_treaty", "reject_treaty") else a.on  # type: ignore[union-attr]


def validate(decision: TurnDecision, ctx: Context) -> ValidationResult:
    ok: list[tuple[int, Action]] = []
    rejected: list[tuple[int, ActionIn, str]] = []
    seen_once: set[str] = set()
    for k, a_in in enumerate(decision.actions):
        a = convert(a_in)
        if isinstance(a, str):
            rejected.append((k, a_in, a))
            continue
        why = _check_one(a, ctx)
        if why is None and a.type in ONCE_PER_TURN:
            if a.type in seen_once:
                why = f"duplicate {a.type}: only the first one this turn counts"
            seen_once.add(a.type)
        if why is not None:
            rejected.append((k, a_in, why))
        else:
            ok.append((k, a))
    # Rule 8: contradictory pairs (on and off / accept and reject in the same turn) -> both rejected.
    values: dict[tuple[str, str], set[object]] = {}
    for _, a in ok:
        key = _pair_key(a)
        if key is not None:
            values.setdefault(key, set()).add(_pair_value(a))
    clash = {key for key, v in values.items() if len(v) > 1}
    accepted = []
    for k, a in ok:
        if _pair_key(a) in clash:
            rejected.append(
                (k, decision.actions[k], f"contradicts another {a.type} this turn (both rejected)")
            )
        else:
            accepted.append((k, a))
    rejected.sort(key=lambda r: r[0])
    preds, dropped_p = _predictions(decision.predictions, ctx)
    comms, dropped_c = _commitments(decision.commitments, ctx)
    notes = []
    for k, _ in accepted:
        extra = unused_fields(decision.actions[k])
        notes.append(f"ignored unused fields: {', '.join(extra)}" if extra else "")
    return ValidationResult(
        tuple(accepted), tuple(rejected), preds, dropped_p, comms, dropped_c, tuple(notes)
    )


def _predictions(preds: list[Prediction], ctx: Context):
    kept, dropped = [], []
    for p in preds:
        if p.country not in ctx.still_to_move:
            dropped.append((p, f"{p.country} is not still to move"))
        else:
            kept.append(p)
    return tuple(kept), tuple(dropped)


def _commitments(comms: list[Commitment], ctx: Context):
    i, s = ctx.country, ctx.state
    lo, hi = ctx.cfg.world.actions.tariff_rate
    kept, dropped = [], []
    for c in comms:
        j = COUNTRIES.index(c.target)
        if j == i:
            why: str | None = "target cannot be yourself"
        elif c.kind == "tariff_cap" and (c.max_rate is None or not lo <= c.max_rate <= hi):
            why = f"tariff_cap needs max_rate in [{lo}, {hi}]"
        elif c.kind == "keep_treaty" and not any(
            {i, j} == set(t.parties()) and t.status in ("active", "proposed") for t in s.treaties
        ):
            why = f"no treaty with {c.target} to keep"
        else:
            why = None
        if why is None:
            kept.append(c)
        else:
            dropped.append((c, why))
    return tuple(kept), tuple(dropped)


# ------------------------------------------------------------------------------- Layer 2


def fact_check(
    facts: list[Fact], truth: dict[tuple[str, str], float], rel_tol: float
) -> tuple[FactCheck, ...]:
    out = []
    for f in facts:
        true = truth.get((f.country, f.metric))
        # unknown (country, metric) counts as wrong; otherwise relative error above the tolerance
        wrong = true is None or abs(f.value - true) > rel_tol * max(abs(true), 1e-9)
        out.append(FactCheck(f.country, f.metric, f.value, true, wrong))
    return tuple(out)


def hallucination_rate(checks: tuple[FactCheck, ...] | list[FactCheck]) -> float:
    """Wrong facts / facts stated (0 if none stated)."""
    return sum(c.wrong for c in checks) / len(checks) if checks else 0.0
