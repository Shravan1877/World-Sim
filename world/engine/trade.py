"""Armington-with-trust trade allocation, rationing and payments (CLAUDE.md §6.4).

For each traded good g (FOOD, ENERGY, GOODS, TECH):
  S_dom[i]  = stock[i] + Q[i]                       (stock is AFTER energy inputs, §6.2)
  X[j]      = max(S_dom[j] - D[j], 0)               exportable surplus
  M[i]      = max(D[i] - S_dom[i], 0)               import need
  export_cap[i, j] limits the share of j's surplus that may go to importer i
  (DORNE's energy quota, CERES's food ban). A sanction either way blocks the pair.
Step 1: supply contracts deliver first, at the contract price.
Step 2: Armington shares over eligible exporters:
        landed c[i,j] = P[j] (1 + levy[j]) (1 + tariff[i,j])
        score = trust[i,j]^kappa * c^(1 - sigma);  request = share * M_remaining[i]
Step 3: if requests to j exceed j's remaining surplus, scale them proportionally.
Unmet requests get a second pass (steps 2-3) over exporters with surplus left; the rest is shortage.

Payments (ledger): importer households pay c per unit; exporter firms get P[j]; the exporter's
government gets the levy part; the importer's government gets the tariff part. Contract
deliveries are paid at the contract price, all to the seller's firms (no tariff or levy).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world.ledger import Ledger, firms, government, households

N_TRADED = 4


@dataclass(frozen=True)
class SupplyContract:
    treaty_id: str
    seller: int
    buyer: int
    good: int  # 0..3
    quantity: float  # per turn
    price: float  # per unit


@dataclass(frozen=True)
class ContractDelivery:
    treaty_id: str
    delivered: float
    shortfall: float
    policy_caused: bool  # shortfall due to the seller's quota/ban or a sanction -> violation (§10)


@dataclass(frozen=True)
class TradeResult:
    flows: np.ndarray  # (6, 6, 4): flows[i, j, g] = units of g from exporter j to importer i
    imports: np.ndarray  # (6, 4)
    exports: np.ndarray  # (6, 4)
    export_requests: np.ndarray  # (6, 4): X_req, all requests over both passes, before rationing
    unmet_imports: np.ndarray  # (6, 4): import need left after both passes
    surplus: np.ndarray  # (6, 4): X before trade
    deliveries: tuple[ContractDelivery, ...]
    ledger: Ledger


def landed_cost(price: np.ndarray, levy: np.ndarray, tariff: np.ndarray) -> np.ndarray:
    """c[i, j, g] = P[j, g] (1 + levy[j, g]) (1 + tariff[i, j, g]); price, levy are (6, 4)."""
    return (price * (1.0 + levy))[None, :, :] * (1.0 + tariff)


def blocked_pairs(sanction: np.ndarray) -> np.ndarray:
    """(6, 6) bool: True where trade is impossible (a sanction either way, or i == j)."""
    blocked = sanction | sanction.T
    return blocked | np.eye(sanction.shape[0], dtype=bool)


def armington_shares(
    cost: np.ndarray, trust: np.ndarray, eligible: np.ndarray, kappa: float, sigma: float
) -> np.ndarray:
    """Row-wise shares: score = trust^kappa * cost^(1 - sigma) over eligible exporters.

    cost, trust, eligible have shape (n_importers, n_exporters). Rows with no eligible exporter
    get all-zero shares.
    """
    score = np.where(eligible, np.power(trust, kappa) * np.power(cost, 1.0 - sigma), 0.0)
    total = score.sum(axis=1, keepdims=True)
    return np.where(total > 0, score / np.where(total > 0, total, 1.0), 0.0)


def _pay_imports(
    ledger: Ledger, g: int, granted: np.ndarray, price: np.ndarray, levy: np.ndarray, cost: np.ndarray
) -> None:
    for i, j in zip(*np.nonzero(granted > 0), strict=True):
        q = float(granted[i, j])
        base = float(price[j]) * q
        levy_part = base * float(levy[j])
        tariff_part = float(cost[i, j]) * q - base - levy_part
        ledger.transfer(households(i), firms(j, g), base, f"import g{g} {j}->{i}")
        if levy_part > 0:
            ledger.transfer(households(i), government(j), levy_part, f"export levy g{g} {j}->{i}")
        if tariff_part > 0:
            ledger.transfer(households(i), government(i), tariff_part, f"tariff g{g} {j}->{i}")


def allocate_trade(
    *,
    stock: np.ndarray,
    output: np.ndarray,
    demand: np.ndarray,
    price: np.ndarray,
    levy: np.ndarray,
    tariff: np.ndarray,
    export_cap: np.ndarray,
    sanction: np.ndarray,
    trust: np.ndarray,
    kappa: float,
    sigma: float,
    passes: int,
    contracts: tuple[SupplyContract, ...] | list[SupplyContract],
    ledger: Ledger,
) -> TradeResult:
    """Run §6.4 for all traded goods. Array inputs are the traded columns: (6, 4) or (6, 6, 4).

    Returns a new ledger; the input ledger is not changed.
    """
    n = stock.shape[0]
    ledger = ledger.copy()
    s_dom = stock + output
    surplus = np.maximum(s_dom - demand, 0.0)
    need = np.maximum(demand - s_dom, 0.0)
    cost = landed_cost(price, levy, tariff)
    open_pair = ~blocked_pairs(sanction)

    flows = np.zeros((n, n, N_TRADED))
    requests = np.zeros((n, N_TRADED))
    unmet = np.zeros((n, N_TRADED))
    deliveries: list[ContractDelivery] = []

    for g in range(N_TRADED):
        pair_rem = export_cap[:, :, g] * surplus[None, :, g]  # [i, j]: what j may still send to i
        x_rem = surplus[:, g].copy()
        m_rem = need[:, g].copy()

        # Step 1: supply contracts.
        for c in (c for c in contracts if c.good == g):
            requests[c.seller, g] += c.quantity
            allowed = bool(open_pair[c.buyer, c.seller])
            q = min(c.quantity, x_rem[c.seller], pair_rem[c.buyer, c.seller]) if allowed else 0.0
            q = max(q, 0.0)
            without_policy = min(c.quantity, x_rem[c.seller])
            shortfall = c.quantity - q
            deliveries.append(
                ContractDelivery(c.treaty_id, q, shortfall, shortfall > 0 and q < without_policy)
            )
            if q > 0:
                flows[c.buyer, c.seller, g] += q
                x_rem[c.seller] -= q
                pair_rem[c.buyer, c.seller] -= q
                m_rem[c.buyer] = max(m_rem[c.buyer] - q, 0.0)
                ledger.transfer(
                    households(c.buyer), firms(c.seller, g), c.price * q, f"contract {c.treaty_id}"
                )

        # Steps 2-3, repeated for each pass.
        for _ in range(passes):
            eligible = open_pair & (pair_rem > 0) & (x_rem[None, :] > 0) & (m_rem[:, None] > 0)
            if not eligible.any():
                break
            shares = armington_shares(cost[:, :, g], trust, eligible, kappa, sigma)
            req = shares * m_rem[:, None]
            requests[:, g] += req.sum(axis=0)
            granted = np.minimum(req, pair_rem)
            total = granted.sum(axis=0)
            scale = np.where(total > x_rem, x_rem / np.where(total > 0, total, 1.0), 1.0)
            granted = granted * scale[None, :]
            flows[:, :, g] += granted
            x_rem = np.maximum(x_rem - granted.sum(axis=0), 0.0)
            pair_rem = np.maximum(pair_rem - granted, 0.0)
            m_rem = np.maximum(m_rem - granted.sum(axis=1), 0.0)
            _pay_imports(ledger, g, granted, price[:, g], levy[:, g], cost[:, :, g])
        unmet[:, g] = m_rem

    return TradeResult(
        flows=flows,
        imports=flows.sum(axis=1),
        exports=flows.sum(axis=0),
        export_requests=requests,
        unmet_imports=unmet,
        surplus=surplus,
        deliveries=tuple(deliveries),
        ledger=ledger,
    )
