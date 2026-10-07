"""RandomBot: random actions from the BOT stream (CLAUDE.md §8, §11.1).

Each turn it draws 0-3 actions from the full action list, including actions it may not take and
values that may be out of range, so random games also exercise the validator's rejections. Its
dice are keyed (seed, turn, BOT, country), so the same seed gives the same random moves.
"""

from __future__ import annotations

from dataclasses import dataclass

from world.actions import ActionIn
from world.briefing import Briefing
from world.config import COUNTRIES, SECTORS
from world.policies.base import DecisionResult
from world.policies.bots import Bot, make_decision
from world.rng import Generator, Stream, make_rng

GOODS = (*SECTORS, "ALL")
KINDS = ("supply_contract", "tariff_cap", "no_sanction_pact", "loan")


@dataclass
class RandomBot(Bot):
    name: str = "random"
    max_random_actions: int = 3

    def decide(self, briefing: Briefing) -> DecisionResult:
        rng = make_rng(briefing.seed, briefing.turn, Stream.BOT, briefing.index)
        n = int(rng.integers(0, self.max_random_actions + 1))
        acts = [self._action(briefing, rng) for _ in range(n)]
        return DecisionResult(
            decision=make_decision(
                briefing,
                acts,
                read="Anything can happen.",
                plan="Roll the dice.",
                statement="We keep our options open.",
                stance=(0.34, 0.33, 0.33),
                max_actions=self.max_actions,
            )
        )

    def _action(self, b: Briefing, rng: Generator) -> ActionIn:  # noqa: C901
        others = [c for c in COUNTRIES if c != b.country]
        pick = lambda xs: xs[int(rng.integers(0, len(xs)))]  # noqa: E731
        a = self.cfg.world.actions
        kind = pick(
            (
                "set_tax", "set_spending", "set_tariff", "set_tariff", "set_sanction", "propose_treaty",
                "accept_treaty", "antitrust", "set_subsidy_target", "special", "wait",
            )
        )  # fmt: skip
        if kind == "set_tax":
            return ActionIn(type="set_tax", rate=float(rng.uniform(0.1, 0.4)))
        if kind == "set_spending":
            w, m, s = rng.uniform(0.0, 0.2, 3)
            return ActionIn(type="set_spending", welfare=float(w), military=float(m), subsidy=float(s))
        if kind == "set_tariff":
            return ActionIn(
                type="set_tariff",
                target=pick(others),
                good=pick(GOODS),
                rate=float(rng.uniform(0, a.tariff_rate[1])),
            )
        if kind == "set_sanction":
            return ActionIn(type="set_sanction", target=pick(others), on=bool(rng.random() < 0.3))
        if kind == "propose_treaty":
            k = pick(KINDS)
            terms = {
                "supply_contract": f"role=seller; good={pick(SECTORS[:4])}; "
                f"quantity={rng.uniform(0.05, 0.5):.3f}; price={rng.uniform(0.3, 1.5):.3f}",
                "tariff_cap": f"max_rate={rng.uniform(0, 0.2):.3f}",
                "no_sanction_pact": "",
                "loan": f"role=lender; amount={rng.uniform(0.01, 0.2):.3f}; rate={rng.uniform(0, 0.1):.3f}",
            }[k]
            return ActionIn(
                type="propose_treaty",
                target=pick(others),
                kind=k,
                duration=int(rng.integers(1, 5)),
                terms=terms,
            )
        if kind == "accept_treaty" and b.proposals_to_me:
            return ActionIn(type="accept_treaty", treaty_id=pick([t.id for t in b.proposals_to_me]))
        if kind == "antitrust":
            return ActionIn(type="antitrust", sector=pick(SECTORS))
        if kind == "set_subsidy_target":
            return ActionIn(type="set_subsidy_target", sector=pick(SECTORS))
        if kind == "special":
            return pick(
                [
                    ActionIn(
                        type="set_energy_export_quota",
                        target=pick([*others, "ALL"]),
                        rate=float(rng.uniform()),
                    ),
                    ActionIn(
                        type="set_energy_export_levy", rate=float(rng.uniform(0, a.energy_export_levy[1]))
                    ),
                    ActionIn(
                        type="set_food_export_ban", target=pick([*others, "ALL"]), on=bool(rng.random() < 0.5)
                    ),
                    ActionIn(
                        type="subsidize_industry", sector=pick(SECTORS), amount=float(rng.uniform(0, 0.05))
                    ),
                    ActionIn(type="nationalize", sector=pick(SECTORS)),
                    ActionIn(type="set_policy_rate", rate=float(rng.uniform(0, 0.1))),
                ]
            )
        return ActionIn(type="wait")
