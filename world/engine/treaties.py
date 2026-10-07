"""Treaties: objects, life cycle, execution and violation detection (CLAUDE.md §10, §6.1 step 15).

Life cycle
  proposed (turn t) -> accepted (active from that turn) | rejected | expired
  A proposal stays open through turn t + proposal_expiry_turns (1): the addressee can accept on its
  own move in turn t if it moves after the proposer, otherwise in turn t + 1. At the start of turn
  t + 2 an unanswered proposal becomes `expired` (expire_proposals, called by turn_start).
  An active treaty runs for `duration` turns: turns start .. start + duration - 1. A loan is paid out
  on its start turn and repaid over the next `duration` turns, so it runs through start + duration.
  Then it is `ended` ("completed"). FALKEN's renounce_treaty ends it at once ("renounced", no trust
  penalty, -5 stability for FALKEN; policy.py). A violation `suspends` it (final in v1).

Execution inside step()
  supply_contract: delivered in trade before the Armington market, at the contract price, up to the
                   quantity, the seller's supply and the export cap (trade.py). A shortfall caused
                   by the seller's quota/ban or a sanction is a violation by the seller.
  tariff_cap:      either side's tariff on the other above max_rate (any good) -> violation by it.
  no_sanction_pact: either side sanctioning the other -> violation by the sanctioner.
  loan:            the lender's treasury pays `amount` to the borrower's on the start turn; then the
                   borrower pays amount / duration + amount * rate / 4 each turn (equal principal
                   plus simple quarterly interest on the original amount, D57). Payments are ledger
                   transfers between treasuries. A sanction between the two blocks a payment: the
                   side whose sanction blocks it violates the treaty. An installment the borrower
                   cannot pay from its treasury (cash below the installment) is paid in part and is
                   NOT a violation ("missed while the treasury had funds" is the violation, §10).
Every active treaty with no violation this turn counts as honored (trust +0.02 both ways, §6.13).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from world.config import COUNTRIES, SECTORS
from world.engine.state import Treaty
from world.engine.trade import ContractDelivery, SupplyContract
from world.ledger import Ledger, government

EPS = 1e-12


def make_treaty(
    treaty_id: str, kind: str, proposer: int, addressee: int, terms: object, duration: int, turn: int
) -> Treaty:
    """Build a proposed Treaty from strict terms (world.actions.*Terms). Roles become indices."""
    d = terms.model_dump()  # type: ignore[attr-defined]
    d.pop("kind")
    out: dict[str, float | int | str] = {}
    if kind == "supply_contract":
        seller_is_proposer = d.pop("role") == "seller"
        out["seller"] = proposer if seller_is_proposer else addressee
        out["buyer"] = addressee if seller_is_proposer else proposer
        out["good"] = SECTORS.index(d.pop("good"))
    elif kind == "loan":
        lender_is_proposer = d.pop("role") == "lender"
        out["lender"] = proposer if lender_is_proposer else addressee
        out["borrower"] = addressee if lender_is_proposer else proposer
    out.update({k: float(v) for k, v in d.items()})
    return Treaty(
        id=treaty_id,
        kind=kind,
        proposer=proposer,
        addressee=addressee,
        terms=tuple(sorted(out.items())),
        duration=int(duration),
        proposed_turn=turn,
    )


def proposal_open(t: Treaty, turn: int, expiry_turns: int) -> bool:
    return t.status == "proposed" and turn <= t.proposed_turn + expiry_turns


def last_active_turn(t: Treaty) -> int:
    return t.start_turn + t.duration if t.kind == "loan" else t.start_turn + t.duration - 1


def expire_proposals(treaties: tuple[Treaty, ...], turn: int, expiry_turns: int) -> tuple[Treaty, ...]:
    """Start of `turn`: unanswered proposals older than expiry_turns full turns expire."""
    return tuple(
        replace(t, status="expired", end_turn=turn, note="not answered")
        if t.status == "proposed" and turn > t.proposed_turn + expiry_turns
        else t
        for t in treaties
    )


def supply_contracts(treaties: tuple[Treaty, ...]) -> tuple[SupplyContract, ...]:
    return tuple(
        SupplyContract(
            treaty_id=t.id,
            seller=int(t.term("seller")),
            buyer=int(t.term("buyer")),
            good=int(t.term("good")),
            quantity=float(t.term("quantity")),
            price=float(t.term("price")),
        )
        for t in treaties
        if t.status == "active" and t.kind == "supply_contract"
    )


def loan_installment(t: Treaty) -> float:
    amount, rate = float(t.term("amount")), float(t.term("rate"))
    return amount / t.duration + amount * rate / 4.0


@dataclass(frozen=True)
class LoanPayment:
    treaty_id: str
    kind: str  # "disbursement" | "installment"
    due: float
    paid: float
    blocked_by: tuple[int, ...]  # countries whose sanction blocked the payment


def pay_loans(
    ledger: Ledger, treaties: tuple[Treaty, ...], turn: int, sanction: np.ndarray
) -> tuple[Ledger, tuple[LoanPayment, ...]]:
    """Disbursements on the start turn, installments on the following `duration` turns."""
    led = ledger.copy()
    out = []
    for t in treaties:
        if t.status != "active" or t.kind != "loan":
            continue
        lender, borrower = int(t.term("lender")), int(t.term("borrower"))
        if turn == t.start_turn:
            kind, src, dst, due = "disbursement", lender, borrower, float(t.term("amount"))
        elif t.start_turn < turn <= last_active_turn(t):
            kind, src, dst, due = "installment", borrower, lender, loan_installment(t)
        else:
            continue
        blocked = tuple(c for c in (lender, borrower) if sanction[c, lender + borrower - c])
        if blocked:
            paid = 0.0
        elif kind == "disbursement":
            paid = due  # the lender may borrow for it; the size limit was checked when agreed
        else:
            paid = min(due, max(led.balance(government(src)), 0.0))
        if paid > 0:
            led.transfer(government(src), government(dst), paid, f"loan {kind} {t.id}")
        out.append(LoanPayment(t.id, kind, due, paid, blocked))
    return led, tuple(out)


@dataclass(frozen=True)
class Violation:
    treaty_id: str
    violator: int
    victim: int
    reason: str


def find_violations(
    treaties: tuple[Treaty, ...],
    tariff: np.ndarray,
    sanction: np.ndarray,
    deliveries: tuple[ContractDelivery, ...],
    loans: tuple[LoanPayment, ...],
) -> tuple[Violation, ...]:
    """§10: every way an active treaty can be broken this turn (state = after all moves)."""
    out: list[Violation] = []
    by_id = {t.id: t for t in treaties}
    for t in treaties:
        if t.status != "active":
            continue
        a, b = t.parties()
        if t.kind == "tariff_cap":
            cap = float(t.term("max_rate"))
            for x, y in ((a, b), (b, a)):
                if tariff[x, y].max() > cap + EPS:
                    out.append(Violation(t.id, x, y, f"tariff {tariff[x, y].max():.2f} above cap {cap:.2f}"))
        elif t.kind == "no_sanction_pact":
            for x, y in ((a, b), (b, a)):
                if sanction[x, y]:
                    out.append(Violation(t.id, x, y, "sanction under a no-sanction pact"))
    for d in deliveries:
        t = by_id.get(d.treaty_id)
        if t is not None and d.policy_caused:
            seller, buyer = int(t.term("seller")), int(t.term("buyer"))
            out.append(Violation(t.id, seller, buyer, f"supply shortfall {d.shortfall:.3g} by policy"))
    for p in loans:
        t = by_id[p.treaty_id]
        for x in p.blocked_by:
            y = t.addressee if x == t.proposer else t.proposer
            out.append(Violation(t.id, x, y, f"loan {p.kind} blocked by a sanction"))
    return tuple(out)


def settle(
    treaties: tuple[Treaty, ...], turn: int, violations: tuple[Violation, ...]
) -> tuple[tuple[Treaty, ...], tuple[tuple[int, int], ...]]:
    """End of step: suspend violated treaties, end completed ones. Returns (treaties, honored pairs)."""
    broken: dict[str, Violation] = {}
    for v in violations:
        broken.setdefault(v.treaty_id, v)
    new, honored = [], []
    for t in treaties:
        if t.status != "active":
            new.append(t)
            continue
        v = broken.get(t.id)
        if v is not None:
            note = f"violated by {COUNTRIES[v.violator]}"
            new.append(replace(t, status="suspended", end_turn=turn, note=note))
            continue
        honored.append(t.parties())
        if turn >= last_active_turn(t):
            new.append(replace(t, status="ended", end_turn=turn, note="completed"))
        else:
            new.append(t)
    return tuple(new), tuple(honored)


def by_id(treaties: tuple[Treaty, ...], treaty_id: str) -> Treaty | None:
    for t in treaties:
        if t.id == treaty_id:
            return t
    return None
