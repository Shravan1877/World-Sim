"""Invariant checks run at the end of every step (CLAUDE.md §6.14). Raise InvariantError with details.

1. Money conservation (ledger) and per-account reconciliation.
2. World trade balance: for each good, sum of exports == sum of imports.
3. Stock-flow: stock' = stock + Q + imports - exports - consumption - energy_inputs - spoilage.
4. Non-negativity: stocks, prices, wages, labor, treasury, household cash; firm accounts end at 0.
5. Bounds: 0 <= u <= 1, 0 <= stability <= 100, 0 <= trust <= 1, shares sum to 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world.ledger import Ledger, LedgerError, firms

N_TRADED = 5


class InvariantError(Exception):
    """An invariant failed. The run must stop and be marked invariant_failed (§6.14)."""


@dataclass(frozen=True)
class StockFlow:
    """Quantities of one turn needed for the stock-flow check, traded goods only, shape (6, 4)."""

    stock_start: np.ndarray
    output: np.ndarray
    imports: np.ndarray
    exports: np.ndarray
    consumption: np.ndarray  # includes military GOODS purchases
    energy_inputs: np.ndarray  # only the ENERGY column is non-zero
    spoilage: np.ndarray
    stock_end: np.ndarray


def _tol(scale: np.ndarray | float, rel: float) -> np.ndarray:
    return rel * np.maximum(1.0, np.abs(scale))


def check_money(ledger: Ledger, rel_tol: float) -> list[str]:
    errors = []
    try:
        ledger.check_conservation(rel_tol)
        ledger.check_reconciliation(rel_tol)
    except LedgerError as e:
        errors.append(f"money: {e}")
    return errors


def check_firm_accounts_zero(ledger: Ledger, abs_scale: float, rel_tol: float) -> list[str]:
    bad = {
        f"firms[{i},{g}]": ledger.balance(firms(i, g))
        for i in range(6)
        for g in range(5)
        if abs(ledger.balance(firms(i, g))) > rel_tol * max(1.0, abs_scale)
    }
    return [f"firm accounts not zero: {bad}"] if bad else []


def check_trade_balance(flows: np.ndarray, rel_tol: float) -> list[str]:
    exports = flows.sum(axis=(0,))  # (exporter, good)
    imports = flows.sum(axis=(1,))  # (importer, good)
    diff = exports.sum(axis=0) - imports.sum(axis=0)
    bad = np.abs(diff) > _tol(exports.sum(axis=0), rel_tol)
    return [f"world exports != imports for goods {np.nonzero(bad)[0].tolist()}: {diff}"] if bad.any() else []


def check_stock_flow(sf: StockFlow, rel_tol: float) -> list[str]:
    expected = (
        sf.stock_start + sf.output + sf.imports - sf.exports - sf.consumption - sf.energy_inputs - sf.spoilage
    )
    scale = sf.stock_start + sf.output + sf.imports
    bad = np.abs(sf.stock_end - expected) > _tol(scale, rel_tol)
    if bad.any():
        idx = [tuple(int(x) for x in ix) for ix in np.argwhere(bad)]
        return [f"stock-flow broken at (country, good) {idx}: end {sf.stock_end[bad]} vs {expected[bad]}"]
    return []


def check_non_negative(named: dict[str, np.ndarray], abs_tol: float) -> list[str]:
    errors = []
    for name, v in named.items():
        if not np.all(np.isfinite(v)):
            errors.append(f"{name} has non-finite values")
        elif np.any(v < -abs_tol):
            errors.append(f"{name} negative: min {float(np.min(v))}")
    return errors


def check_bounds(named: dict[str, tuple[np.ndarray, float, float]], abs_tol: float) -> list[str]:
    errors = []
    for name, (v, lo, hi) in named.items():
        if np.any(v < lo - abs_tol) or np.any(v > hi + abs_tol):
            errors.append(f"{name} outside [{lo}, {hi}]: min {float(np.min(v))}, max {float(np.max(v))}")
    return errors


def check_shares(named: dict[str, np.ndarray], abs_tol: float) -> list[str]:
    errors = []
    for name, rows in named.items():
        sums = rows.sum(axis=-1)
        if np.any(np.abs(sums - 1.0) > abs_tol):
            errors.append(f"{name} shares do not sum to 1: {sums}")
    return errors


def raise_if_any(errors: list[str], turn: int) -> None:
    if errors:
        raise InvariantError(f"turn {turn}: " + "; ".join(errors))
