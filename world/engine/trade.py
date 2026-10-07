"""Armington-with-trust-and-home-bias market for every good (CLAUDE.md §6.4, D45).

One market per good g (FOOD, ENERGY, GOODS, TECH, SERVICES; D40; the number of goods comes from the
array shapes). Every buyer i spreads its WHOLE demand D[i,g] over its own home sellers and every
eligible foreign seller j (no sanction either way, export cap > 0):
  c[i,i]   = P[i]                                   home price (no tariff, no levy)
  c[i,j]   = P[j] (1 + levy[j]) (1 + tariff[i,j])   landed price of an import
  score    = trust[i,s]^kappa * c[i,s]^(1 - sigma) * (theta if s == i)   trust[i,i] = 1
  request  = share * D_remaining[i]
Order inside one good:
  0. reserve: a seller first keeps its own reserved demand (the D30 firm energy stock-building
     part) from its own supply: "an energy exporter keeps what its own firms need".
  1. supply contracts deliver first, at the contract price.
  2. pass 1: every seller rations ALL requests (home and foreign) proportionally against its
     remaining supply S_dom = stock + Q (SERVICES: Q). Export caps (DORNE's quota, CERES's ban)
     limit only the foreign part: buyer i may get at most export_cap[i,j] x S_dom[j] from j.
  3. pass 2: unmet requests are spread again over sellers with supply left; then the rest is
     a shortage.
R_s = everything requested from seller s in step 0, 1 and pass 1 drives its price (§6.5).

Payments (ledger): imports are paid here: the buyer pays c per unit; exporter firms get P[j]; the
exporter's government gets the levy part; the importer's government gets the tariff part. Contract
deliveries are paid at the contract price, all to the seller's firms. Home purchases are paid by
step() (it applies the household budget, D35); this module only returns their quantities.

Who is "the buyer" (D34): the buyers of good g in country i split every payment by their share
of total demand: households (D_house), the government (military GOODS, D_gov) and the country's own
ENERGY firm account (the stock-building firm part of ENERGY demand, D30). `payer_weights[i, g]`
holds those three shares; without it, households pay for everything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world.ledger import Account, Ledger, firms, government, households

N_TRADED = 5
N_PAYERS = 3  # payer_weights[..., k]: 0 households, 1 government, 2 the importer's own firms[i, g]


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
    flows: np.ndarray  # (6, 6, G): flows[i, j, g] = units of g from foreign seller j to buyer i (diag 0)
    home: np.ndarray  # (6, G): units each country bought from its own sellers (incl. the reserve)
    imports: np.ndarray  # (6, G)
    exports: np.ndarray  # (6, G)
    requests: np.ndarray  # (6, G): R_s, all requests a seller received in reserve, contracts, pass 1
    unmet: np.ndarray  # (6, G): demand left after both passes (shortage before the budget rule)
    supply: np.ndarray  # (6, G): S_dom = stock + Q
    deliveries: tuple[ContractDelivery, ...]
    ledger: Ledger


def landed_cost(price: np.ndarray, levy: np.ndarray, tariff: np.ndarray) -> np.ndarray:
    """c[i, j, g] = P[j, g] (1 + levy[j, g]) (1 + tariff[i, j, g]); price, levy are (6, 4)."""
    return (price * (1.0 + levy))[None, :, :] * (1.0 + tariff)


def blocked_pairs(sanction: np.ndarray) -> np.ndarray:
    """(6, 6) bool: True where foreign trade is impossible (a sanction either way, or i == j)."""
    blocked = sanction | sanction.T
    return blocked | np.eye(sanction.shape[0], dtype=bool)


def armington_shares(
    cost: np.ndarray,
    trust: np.ndarray,
    eligible: np.ndarray,
    kappa: float,
    sigma: float,
    home_bias: float = 1.0,
) -> np.ndarray:
    """Row-wise shares: score = trust^kappa * cost^(1 - sigma) (* home_bias on the diagonal).

    cost, trust, eligible have shape (n_buyers, n_sellers). Rows with no eligible seller get
    all-zero shares.
    """
    bias = np.where(np.eye(cost.shape[0], cost.shape[1], dtype=bool), home_bias, 1.0)
    score = np.where(eligible, np.power(trust, kappa) * np.power(cost, 1.0 - sigma) * bias, 0.0)
    total = score.sum(axis=1, keepdims=True)
    return np.where(total > 0, score / np.where(total > 0, total, 1.0), 0.0)


def _payers(i: int, g: int) -> tuple[Account, Account, Account]:
    return households(i), government(i), firms(i, g)


def _pay_split(
    ledger: Ledger, i: int, g: int, weights: np.ndarray, dst: Account, amount: float, reason: str
) -> None:
    """The buyers of good g in country i pay `amount` to dst, split by `weights` (sums to 1).

    A payer that is also the receiver (the government's share of its own tariff) moves nothing.
    """
    for payer, w in zip(_payers(i, g), weights, strict=True):
        if w > 0 and amount * w > 0 and payer != dst:
            ledger.transfer(payer, dst, amount * float(w), reason)


def _pay_imports(
    ledger: Ledger,
    g: int,
    granted: np.ndarray,
    price: np.ndarray,
    levy: np.ndarray,
    cost: np.ndarray,
    weights: np.ndarray,
) -> None:
    for i, j in zip(*np.nonzero(granted > 0), strict=True):
        q = float(granted[i, j])
        base = float(price[j]) * q
        levy_part = base * float(levy[j])
        tariff_part = float(cost[i, j]) * q - base - levy_part
        _pay_split(ledger, i, g, weights[i], firms(j, g), base, f"import g{g} {j}->{i}")
        if levy_part > 0:
            _pay_split(ledger, i, g, weights[i], government(j), levy_part, f"export levy g{g} {j}->{i}")
        if tariff_part > 0:
            _pay_split(ledger, i, g, weights[i], government(i), tariff_part, f"tariff g{g} {j}->{i}")


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
    payer_weights: np.ndarray | None = None,
    home_bias: float = 1.0,
    reserve: np.ndarray | None = None,
) -> TradeResult:
    """Run §6.4 (D45) for all goods. Arrays are (6, G) or (6, 6, G).

    reserve (6, G): demand a country takes from its own supply before anyone else (the D30 firm
    energy part); it is part of `demand`. payer_weights (6, G, 3): who pays for the non-reserved
    purchases. Returns a new ledger; the input ledger is not changed.
    """
    n, n_goods = stock.shape
    ledger = ledger.copy()
    if payer_weights is None:
        payer_weights = np.zeros((n, n_goods, N_PAYERS))
        payer_weights[:, :, 0] = 1.0
    reserve = np.zeros((n, n_goods)) if reserve is None else reserve
    supply = stock + output
    cost_all = landed_cost(price, levy, tariff)
    eye = np.eye(n, dtype=bool)
    open_pair = ~blocked_pairs(sanction)  # foreign pairs that may trade
    trust_h = np.where(eye, 1.0, trust)

    flows = np.zeros((n, n, n_goods))
    home = np.zeros((n, n_goods))
    requests = np.zeros((n, n_goods))
    unmet = np.zeros((n, n_goods))
    deliveries: list[ContractDelivery] = []

    for g in range(n_goods):
        s_rem = supply[:, g].copy()
        d_rem = demand[:, g].copy()
        cost = np.where(eye, price[:, g][:, None], cost_all[:, :, g])
        cap_rem = np.where(eye, np.inf, export_cap[:, :, g] * supply[None, :, g])

        # Step 0: own reserve (home firm energy stock-building).
        kept = np.minimum(np.minimum(reserve[:, g], s_rem), d_rem)
        requests[:, g] += reserve[:, g]
        home[:, g] += kept
        s_rem -= kept
        d_rem -= kept

        # Step 1: supply contracts.
        for c in (c for c in contracts if c.good == g):
            requests[c.seller, g] += c.quantity
            allowed = bool(open_pair[c.buyer, c.seller])
            q = min(c.quantity, s_rem[c.seller], cap_rem[c.buyer, c.seller]) if allowed else 0.0
            q = max(q, 0.0)
            without_policy = min(c.quantity, s_rem[c.seller])
            shortfall = c.quantity - q
            deliveries.append(
                ContractDelivery(c.treaty_id, q, shortfall, shortfall > 0 and q < without_policy)
            )
            if q > 0:
                flows[c.buyer, c.seller, g] += q
                s_rem[c.seller] -= q
                cap_rem[c.buyer, c.seller] -= q
                d_rem[c.buyer] = max(d_rem[c.buyer] - q, 0.0)
                _pay_split(
                    ledger,
                    c.buyer,
                    g,
                    payer_weights[c.buyer, g],
                    firms(c.seller, g),
                    c.price * q,
                    f"contract {c.treaty_id}",
                )

        # Steps 2-3: two passes over home and foreign sellers.
        for k in range(passes):
            eligible = (eye | open_pair) & (cap_rem > 0) & (s_rem[None, :] > 0) & (d_rem[:, None] > 0)
            if not eligible.any():
                break
            shares = armington_shares(cost, trust_h, eligible, kappa, sigma, home_bias)
            req = shares * d_rem[:, None]
            if k == 0:
                requests[:, g] += req.sum(axis=0)
            granted = np.minimum(req, cap_rem)
            total = granted.sum(axis=0)
            scale = np.where(total > s_rem, s_rem / np.where(total > 0, total, 1.0), 1.0)
            granted = granted * scale[None, :]
            home[:, g] += np.diag(granted)
            foreign = np.where(eye, 0.0, granted)
            flows[:, :, g] += foreign
            s_rem = np.maximum(s_rem - granted.sum(axis=0), 0.0)
            cap_rem = np.maximum(cap_rem - granted, 0.0)
            d_rem = np.maximum(d_rem - granted.sum(axis=1), 0.0)
            _pay_imports(ledger, g, foreign, price[:, g], levy[:, g], cost_all[:, :, g], payer_weights[:, g])
        unmet[:, g] = d_rem

    return TradeResult(
        flows=flows,
        home=home,
        imports=flows.sum(axis=1),
        exports=flows.sum(axis=0),
        requests=requests,
        unmet=unmet,
        supply=supply,
        deliveries=tuple(deliveries),
        ledger=ledger,
    )
