"""Double-entry money ledger with conservation checks (CLAUDE.md §6.12).

Every credit that moves goes through Ledger.transfer(from, to, amount, reason). Money is never
created or destroyed by a transfer, so the sum of all balances stays constant. The bond_market
account is the lender to governments and may go negative (it is the money issuer).

The ledger is the one mutable object in the engine. Engine functions that use it work on a
copy (Ledger.copy()), so callers' ledgers are never changed behind their backs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

N_COUNTRIES = 6
N_SECTORS = 5
DEFAULT_REL_TOL = 1e-9


class LedgerError(Exception):
    """Raised on a bad transfer or a failed conservation / reconciliation check."""


class Account(NamedTuple):
    kind: str  # "households" | "firms" | "government" | "bond_market"
    country: int = -1
    sector: int = -1

    def __str__(self) -> str:
        if self.kind == "bond_market":
            return "bond_market"
        if self.kind == "firms":
            return f"firms[{self.country},{self.sector}]"
        return f"{self.kind}[{self.country}]"


def _check_country(i: int) -> None:
    if not 0 <= i < N_COUNTRIES:
        raise LedgerError(f"country index out of range: {i}")


def households(i: int) -> Account:
    _check_country(i)
    return Account("households", i)


def government(i: int) -> Account:
    _check_country(i)
    return Account("government", i)


def firms(i: int, g: int) -> Account:
    _check_country(i)
    if not 0 <= g < N_SECTORS:
        raise LedgerError(f"sector index out of range: {g}")
    return Account("firms", i, g)


BOND_MARKET = Account("bond_market")


def all_accounts() -> list[Account]:
    """Every account in a fixed order: households, firms, government, bond_market."""
    accs = [households(i) for i in range(N_COUNTRIES)]
    accs += [firms(i, g) for i in range(N_COUNTRIES) for g in range(N_SECTORS)]
    accs += [government(i) for i in range(N_COUNTRIES)]
    accs.append(BOND_MARKET)
    return accs


class Transfer(NamedTuple):
    src: Account
    dst: Account
    amount: float
    reason: str


@dataclass
class Ledger:
    opening: dict[Account, float]
    balances: dict[Account, float] = field(init=False)
    log: list[Transfer] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        missing = set(all_accounts()) - set(self.opening)
        extra = set(self.opening) - set(all_accounts())
        if missing or extra:
            raise LedgerError(f"opening balances: missing {missing}, unknown {extra}")
        for acc, v in self.opening.items():
            if not math.isfinite(v):
                raise LedgerError(f"opening balance of {acc} is not finite: {v}")
        self.opening = {acc: float(self.opening[acc]) for acc in all_accounts()}
        self.balances = dict(self.opening)

    @classmethod
    def with_opening(cls, nonzero: dict[Account, float] | None = None) -> Ledger:
        """A ledger where every account opens at 0 except the ones given."""
        opening = {acc: 0.0 for acc in all_accounts()}
        for acc, v in (nonzero or {}).items():
            if acc not in opening:
                raise LedgerError(f"unknown account {acc}")
            opening[acc] = float(v)
        return cls(opening)

    def copy(self) -> Ledger:
        new = Ledger(dict(self.opening))
        new.balances = dict(self.balances)
        new.log = list(self.log)
        return new

    def rebased(self) -> Ledger:
        """The same balances as a new ledger that opens at them, with an empty log (D68: the graph state
        keeps balances only; each turn's transfers go to the experiment database)."""
        return Ledger(dict(self.balances))

    def transfer(self, src: Account, dst: Account, amount: float, reason: str) -> None:
        """Move `amount` credits from src to dst. Amount must be finite and >= 0."""
        amount = float(amount)
        if not math.isfinite(amount) or amount < 0:
            raise LedgerError(f"transfer amount must be finite and >= 0, got {amount} ({reason})")
        if src not in self.balances or dst not in self.balances:
            raise LedgerError(f"unknown account in transfer {src} -> {dst} ({reason})")
        if src == dst:
            raise LedgerError(f"transfer from an account to itself: {src} ({reason})")
        self.balances[src] -= amount
        self.balances[dst] += amount
        self.log.append(Transfer(src, dst, amount, reason))

    def balance(self, acc: Account) -> float:
        return self.balances[acc]

    def total(self) -> float:
        return math.fsum(self.balances.values())

    def opening_total(self) -> float:
        return math.fsum(self.opening.values())

    def _scale(self) -> float:
        """Size used for relative tolerances: total absolute money ever held or moved, at least 1."""
        gross = math.fsum(abs(v) for v in self.opening.values()) + math.fsum(t.amount for t in self.log)
        return max(1.0, gross)

    def conservation_error(self) -> float:
        """Relative drift of total money since opening."""
        return abs(self.total() - self.opening_total()) / self._scale()

    def check_conservation(self, rel_tol: float = DEFAULT_REL_TOL) -> None:
        err = self.conservation_error()
        if err > rel_tol:
            raise LedgerError(f"money not conserved: relative drift {err:.3e} > {rel_tol:.1e}")

    def reconcile(self) -> dict[Account, float]:
        """Per-account check: balance == opening + inflows - outflows, recomputed from the log.

        Returns {account: difference} for every account (differences should be ~0).
        """
        inflows: dict[Account, list[float]] = {acc: [] for acc in self.balances}
        outflows: dict[Account, list[float]] = {acc: [] for acc in self.balances}
        for t in self.log:
            outflows[t.src].append(t.amount)
            inflows[t.dst].append(t.amount)
        return {
            acc: self.balances[acc] - (self.opening[acc] + math.fsum(inflows[acc]) - math.fsum(outflows[acc]))
            for acc in self.balances
        }

    def check_reconciliation(self, rel_tol: float = DEFAULT_REL_TOL) -> None:
        scale = self._scale()
        bad = {str(a): d for a, d in self.reconcile().items() if abs(d) / scale > rel_tol}
        if bad:
            raise LedgerError(f"accounts do not reconcile: {bad}")
