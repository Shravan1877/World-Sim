"""Scripted bots (CLAUDE.md §11.1): the experiment baselines and the test leaders.

Every bot returns a full, valid TurnDecision (facts_used, predictions, stance, commitments, actions,
forecast), so a bot game runs the whole pipeline: validator, policy application, treaties, step.
Facts are copied from the briefing, so a bot's hallucination rate is 0 by construction.

  StatusQuoBot    no actions, ever.
  TitForTatBot    hits back at whoever hurt it in the last 2 turns (sanction for a sanction, a tariff
                  on all goods otherwise); lifts the retaliation once they stop; accepts proposals
                  from countries that did not hurt it.
  GreedyBot       tariffs on every other country, a bit more military each turn, DORNE also levies
                  its energy exports; ignores proposals.
  CooperativeBot  removes its tariffs, sanctions, bans and quotas, accepts every proposal, and offers
                  a no-sanction pact to one neighbour per turn.
  AggressorBot    sanctions every other country and spends the maximum on its military (the
                  shock-bite test's hostile leader, tests/test_shock_bite.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from world.actions import (
    ActionIn,
    Commitment,
    Fact,
    Forecast,
    Prediction,
    Stance,
    TurnDecision,
)
from world.briefing import Briefing
from world.config import COUNTRIES, Config, load_config
from world.policies.base import DecisionResult


def facts_from(b: Briefing) -> list[Fact]:
    """Two true facts from the briefing (own GDP and stability)."""
    mine = b.numbers[b.country]
    return [Fact(country=b.country, metric=m, value=mine[m]) for m in ("gdp", "stability")]  # type: ignore[arg-type]


def predict(b: Briefing, move: str = "status_quo", p: float = 0.7) -> list[Prediction]:
    return [Prediction(country=c, move=move, probability=p) for c in b.still_to_move]  # type: ignore[arg-type]


def make_decision(
    b: Briefing,
    actions: list[ActionIn],
    *,
    read: str,
    plan: str,
    statement: str,
    stance: tuple[float, float, float],
    predictions: list[Prediction] | None = None,
    commitments: list[Commitment] | None = None,
    max_actions: int = 6,
) -> TurnDecision:
    mine = b.numbers[b.country]
    return TurnDecision(
        situation_read=read,
        facts_used=facts_from(b),
        predictions=predict(b) if predictions is None else predictions,
        stance=Stance(power=stance[0], citizens=stance[1], world=stance[2]),
        private_plan=plan,
        public_statement=statement,
        commitments=commitments or [],
        actions=actions[:max_actions],
        forecast=Forecast(my_gdp_growth_pct=mine["gdp_growth_pct"], my_stability_next=mine["stability"]),
    )


def others(b: Briefing) -> list[str]:
    return [c for c in COUNTRIES if c != b.country]


@dataclass
class Bot:
    name: str = "bot"
    cfg: Config = field(default_factory=load_config)

    @property
    def max_actions(self) -> int:
        return self.cfg.world.actions.max_actions_per_turn

    def decide(self, briefing: Briefing) -> DecisionResult:
        return DecisionResult(decision=self.decision(briefing))

    def decision(self, b: Briefing) -> TurnDecision:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass
class StatusQuoBot(Bot):
    name: str = "status_quo"

    def decision(self, b: Briefing) -> TurnDecision:
        return make_decision(
            b,
            [],
            read="Steady quarter.",
            plan="Keep every policy as it is.",
            statement=f"{b.country} keeps its course.",
            stance=(0.34, 0.33, 0.33),
        )


@dataclass
class TitForTatBot(Bot):
    name: str = "tit_for_tat"
    retaliation_tariff: float = 0.2

    def decision(self, b: Briefing) -> TurnDecision:
        hurt: dict[str, set[str]] = {}
        for _turn, who, kind in b.who_hurt_you:
            hurt.setdefault(who, set()).add(kind)
        acts: list[ActionIn] = []
        for c, kinds in hurt.items():
            if "sanction" in kinds and c not in b.own["my_sanctions"]:  # type: ignore[operator]
                acts.append(ActionIn(type="set_sanction", target=c, on=True))  # type: ignore[arg-type]
            elif b.own["my_tariffs"][c] < self.retaliation_tariff:  # type: ignore[index]
                acts.append(ActionIn(type="set_tariff", target=c, good="ALL", rate=self.retaliation_tariff))  # type: ignore[arg-type]
        for c in others(b):
            if c in hurt:
                continue
            if c in b.own["my_sanctions"]:  # type: ignore[operator]
                acts.append(ActionIn(type="set_sanction", target=c, on=False))  # type: ignore[arg-type]
            if b.own["my_tariffs"][c] > 0:  # type: ignore[index]
                acts.append(ActionIn(type="set_tariff", target=c, good="ALL", rate=0.0))  # type: ignore[arg-type]
        for t in b.proposals_to_me:
            if t.proposer not in hurt:
                acts.append(ActionIn(type="accept_treaty", treaty_id=t.id))
        expect = [
            Prediction(country=c, move="raise_tariff_on_me" if c in hurt else "status_quo", probability=0.6)  # type: ignore[arg-type]
            for c in b.still_to_move
        ]
        return make_decision(
            b,
            acts,
            read=f"Hostile moves against us from: {', '.join(sorted(hurt)) or 'nobody'}.",
            plan="Answer every hostile move in kind; forgive those who stop.",
            statement="We treat others as they treat us.",
            stance=(0.4, 0.3, 0.3),
            predictions=expect,
            max_actions=self.max_actions,
        )


@dataclass
class GreedyBot(Bot):
    name: str = "greedy"
    tariff: float = 0.25
    military_step: float = 0.02
    military_cap: float = 0.15
    dorne_levy: float = 0.2

    def decision(self, b: Briefing) -> TurnDecision:
        acts: list[ActionIn] = []
        own = b.own
        mil = float(own["military_share"])  # type: ignore[arg-type]
        new_mil = min(mil + self.military_step, self.military_cap)
        welfare, subsidy = float(own["welfare_share"]), float(own["subsidy_share"])  # type: ignore[arg-type]
        if new_mil > mil and welfare + new_mil + subsidy <= self.cfg.world.actions.spending_sum_max:
            acts.append(ActionIn(type="set_spending", welfare=welfare, military=new_mil, subsidy=subsidy))
        if b.country == "DORNE" and float(own["energy_levy"]) < self.dorne_levy:  # type: ignore[arg-type]
            acts.append(ActionIn(type="set_energy_export_levy", rate=self.dorne_levy))
        for c in others(b):
            if own["my_tariffs"][c] < self.tariff:  # type: ignore[index]
                acts.append(ActionIn(type="set_tariff", target=c, good="ALL", rate=self.tariff))  # type: ignore[arg-type]
        return make_decision(
            b,
            acts,
            read="Others are rivals.",
            plan="Protect home industry, build the army, take what the market allows.",
            statement=f"{b.country} puts its own people first.",
            stance=(0.7, 0.2, 0.1),
            max_actions=self.max_actions,
        )


@dataclass
class CooperativeBot(Bot):
    name: str = "cooperative"
    pact_turns: int = 4

    def decision(self, b: Briefing) -> TurnDecision:
        acts: list[ActionIn] = [ActionIn(type="accept_treaty", treaty_id=t.id) for t in b.proposals_to_me]
        own = b.own
        for c in others(b):
            if c in own["my_sanctions"]:  # type: ignore[operator]
                acts.append(ActionIn(type="set_sanction", target=c, on=False))  # type: ignore[arg-type]
            if own["my_tariffs"][c] > 0:  # type: ignore[index]
                acts.append(ActionIn(type="set_tariff", target=c, good="ALL", rate=0.0))  # type: ignore[arg-type]
        if b.country == "DORNE" and float(own["energy_levy"]) > 0:  # type: ignore[arg-type]
            acts.append(ActionIn(type="set_energy_export_levy", rate=0.0))
        partners = {t.proposer for t in b.treaties_active} | {t.addressee for t in b.treaties_active}
        partners |= {t.addressee for t in b.my_proposals} | {t.proposer for t in b.proposals_to_me}
        ring = COUNTRIES[(COUNTRIES.index(b.country) + b.turn) % len(COUNTRIES)]
        if ring != b.country and ring not in partners:
            acts.append(
                ActionIn(
                    type="propose_treaty", target=ring, kind="no_sanction_pact", duration=self.pact_turns
                )  # type: ignore[arg-type]
            )
        commitments = [Commitment(kind="no_sanction", target=c, turns=2) for c in others(b)[:3]]  # type: ignore[arg-type]
        return make_decision(
            b,
            acts,
            read="Open trade helps everyone.",
            plan="Remove barriers, sign pacts, keep promises.",
            statement=f"{b.country} offers open trade and peace to all.",
            stance=(0.2, 0.4, 0.4),
            predictions=predict(b, "cooperate_with_others", 0.6),
            commitments=commitments,
            max_actions=self.max_actions,
        )


@dataclass
class AggressorBot(Bot):
    name: str = "aggressor"

    def decision(self, b: Briefing) -> TurnDecision:
        a = self.cfg.world.actions
        welfare = float(b.own["welfare_share"])  # type: ignore[arg-type]
        military = min(a.spending_each[1], a.spending_sum_max - welfare)
        subsidy = min(float(b.own["subsidy_share"]), a.spending_sum_max - welfare - military)  # type: ignore[arg-type]
        acts = [ActionIn(type="set_spending", welfare=welfare, military=military, subsidy=subsidy)]
        acts += [ActionIn(type="set_sanction", target=c, on=True) for c in others(b)]  # type: ignore[arg-type]
        return make_decision(
            b,
            acts,
            read="The world is against us.",
            plan="Cut every tie and arm to the limit.",
            statement=f"{b.country} stands alone and armed.",
            stance=(0.9, 0.05, 0.05),
            predictions=predict(b, "sanction_me", 0.5),
            max_actions=self.max_actions,
        )
