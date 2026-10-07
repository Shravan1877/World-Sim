"""WorldState, CountryState, Firm, Treaty and the initial state built from config (CLAUDE.md §4, §5, §6).

WorldState is vectorized: every per-country-per-sector variable is one numpy array of shape
(6, 5) (countries x sectors), per-country variables are shape (6,), and bilateral variables are
(6, 6) or (6, 6, 4) with axis order (importer, exporter, good) as in §5. CountryState is a
read-only snapshot of one country, built from WorldState, for briefings and logs.

Money balances (treasury, household cash, firm cash) live in the ledger (§6.12), not in arrays,
so there is one source of truth for money.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields

import numpy as np

from world.config import COUNTRIES, SECTORS, Config
from world.engine.production import cobb_douglas
from world.ledger import BOND_MARKET, Account, Ledger, all_accounts, government, households

N_COUNTRIES = len(COUNTRIES)
N_SECTORS = len(SECTORS)
N_TRADED = 5  # D40: every good is traded (SERVICES is perishable: spoilage 1.0, stock always 0)
FOOD, ENERGY, GOODS, TECH, SERVICES = range(N_SECTORS)


@dataclass(frozen=True)
class Firm:
    id: str
    share: float
    owner: str = "private"  # "private" | "state"


@dataclass(frozen=True)
class Treaty:
    id: str
    kind: str  # supply_contract | tariff_cap | no_sanction_pact | loan
    proposer: int
    addressee: int
    terms: tuple[tuple[str, float | int | str], ...]  # sorted (key, value) pairs, hashable
    duration: int
    proposed_turn: int
    status: str = "proposed"  # proposed | active | rejected | expired | suspended | ended
    start_turn: int = -1  # turn it was accepted (becomes active)
    end_turn: int = -1  # turn it stopped being active or open (any final status)
    note: str = ""  # why it ended: "completed", "renounced", "violated by X", ...

    def term(self, key: str) -> float | int | str:
        return dict(self.terms)[key]

    def parties(self) -> tuple[int, int]:
        return self.proposer, self.addressee


@dataclass(frozen=True)
class ActiveShock:
    """A timed multiplier. turns_left counts the steps it still applies to (aged at the end of step)."""

    shock_id: str
    country: int  # -1 = world-wide
    sector: int  # -1 = not sector-specific
    multiplier: float
    turns_left: int  # -1 = permanent
    kind: str = "productivity"  # productivity (A) | labor_force (LF) | entry_hazard (startup hazard)


@dataclass(frozen=True)
class Event:
    """Something that happened in the engine (shock, firm entry, leader change). Logged only."""

    kind: str
    country: int = -1
    sector: int = -1
    detail: str = ""
    value: float = 0.0


@dataclass
class WorldState:
    # --- meta
    seed: int
    turn: int
    # --- (6, 5) country x sector
    productivity: np.ndarray  # base A
    shock_mult: np.ndarray  # product of active shock multipliers on A
    price: np.ndarray  # P
    stock: np.ndarray  # goods held at turn start (SERVICES column always 0)
    output: np.ndarray  # Q last turn
    revenue: np.ndarray  # R-hat: last turn's sales revenue
    labor: np.ndarray  # L used last turn
    energy_in: np.ndarray  # E used last turn
    subsidy: np.ndarray  # credits of subsidy per sector this turn (enters R-tilde)
    industry_subsidy: np.ndarray  # BRONTIA's subsidize_industry: extra targeted subsidy, share of GDP
    markup: np.ndarray  # mu
    n_firms: np.ndarray  # int
    nationalized: np.ndarray  # bool
    dominance_age: np.ndarray  # int: consecutive turns the largest firm share > s_max
    consumption_shares: np.ndarray  # fixed base-period budget shares (CPI weights)
    demand: np.ndarray  # D last turn (GOODS includes government purchases)
    excess_ema: np.ndarray  # D50: exponential average of each seller's relative excess demand
    shortage: np.ndarray  # unmet demand last turn
    # --- (6,) country
    labor_force_base: np.ndarray
    labor_force: np.ndarray  # after temporary shocks (pandemic)
    population: np.ndarray
    wage: np.ndarray
    unemployment: np.ndarray
    tax_rate: np.ndarray
    welfare_share: np.ndarray  # spending plan, shares of GDP
    military_share: np.ndarray
    subsidy_share: np.ndarray
    subsidy_target: np.ndarray  # int sector index, -1 = not set
    gov_revenue: np.ndarray  # last turn's government revenue: taxes, tariffs, levies, state profits
    debt: np.ndarray
    gdp: np.ndarray  # quarterly nominal value added, last turn
    gdp_prev: np.ndarray
    gdp_hist: np.ndarray  # (6, gdp_ma_turns): last quarterly GDPs, newest last (D32 moving average)
    gdp_start: np.ndarray  # starting GDP (D32 floor); Phase 4 burn-in resets it to the settled GDP
    cpi: np.ndarray
    cpi_prev: np.ndarray
    inflation_q: np.ndarray  # quarterly CPI inflation
    policy_rate: np.ndarray  # annual
    policy_rate_override: np.ndarray  # annual; NaN = follow the Taylor rule
    saving_rate: np.ndarray
    y_disp: np.ndarray  # household disposable income last turn
    stability: np.ndarray
    military: np.ndarray  # military stock
    default_premium: np.ndarray
    default_turns_left: np.ndarray  # int
    leader_changes: np.ndarray  # int: count of leader changes so far
    leader_changed_turn: np.ndarray  # int: turn of the last leader change, -1 = never (memory-wipe flag)
    # --- bilateral, axis order (importer, exporter[, good])
    trust: np.ndarray  # (6, 6): trust[i, j] = how much i trusts j
    sanction: np.ndarray  # (6, 6) bool: sanction[i, j] = i sanctions j
    tariff: np.ndarray  # (6, 6, 4): tariff[i, j, g] = i's tariff on good g from j
    export_cap: np.ndarray  # (6, 6, 4): share of j's surplus of g that may go to i
    levy: np.ndarray  # (6, 4): exporter's levy on its exports of g
    trade: np.ndarray  # (6, 6, 4): last turn's flows, importer i <- exporter j
    # --- objects
    firms: tuple[tuple[Firm, ...], ...]  # length 30, index i * 5 + g
    treaties: tuple[Treaty, ...]
    active_shocks: tuple[ActiveShock, ...]
    ledger: Ledger

    # ---- money views (the ledger is the source of truth)
    @property
    def treasury(self) -> np.ndarray:
        return np.array([self.ledger.balance(government(i)) for i in range(N_COUNTRIES)])

    @property
    def household_cash(self) -> np.ndarray:
        return np.array([self.ledger.balance(households(i)) for i in range(N_COUNTRIES)])

    def firms_of(self, i: int, g: int) -> tuple[Firm, ...]:
        return self.firms[i * N_SECTORS + g]

    def copy(self) -> WorldState:
        """Deep copy: arrays and ledger are copied; frozen objects are shared safely."""
        kwargs = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray | Ledger):
                v = v.copy()
            kwargs[f.name] = v
        return WorldState(**kwargs)

    def state_hash(self) -> str:
        """sha256 over every field in declaration order. Same state => same hash, across runs."""
        h = hashlib.sha256()
        for f in fields(self):
            v = getattr(self, f.name)
            h.update(f.name.encode())
            if isinstance(v, np.ndarray):
                a = np.ascontiguousarray(v)
                h.update(str(a.dtype).encode())
                h.update(str(a.shape).encode())
                h.update(a.tobytes())
            elif isinstance(v, Ledger):
                bal = np.array([v.balance(acc) for acc in all_accounts()], dtype=np.float64)
                h.update(bal.tobytes())
            else:
                h.update(repr(v).encode())
        return h.hexdigest()

    def country(self, i: int) -> CountryState:
        return CountryState.from_world(self, i)


@dataclass(frozen=True)
class CountryState:
    """Read-only snapshot of one country (scalars and 5-vectors in sector order)."""

    index: int
    name: str
    price: tuple[float, ...]
    stock: tuple[float, ...]
    output: tuple[float, ...]
    markup: tuple[float, ...]
    n_firms: tuple[int, ...]
    labor_force: float
    population: float
    wage: float
    unemployment: float
    tax_rate: float
    welfare_share: float
    military_share: float
    subsidy_share: float
    treasury: float
    household_cash: float
    debt: float
    gdp: float
    cpi: float
    inflation_q: float
    policy_rate: float
    stability: float
    military: float

    @classmethod
    def from_world(cls, w: WorldState, i: int) -> CountryState:
        def vec(a: np.ndarray) -> tuple:
            return tuple(a[i].tolist())

        return cls(
            index=i,
            name=COUNTRIES[i],
            price=vec(w.price),
            stock=vec(w.stock),
            output=vec(w.output),
            markup=vec(w.markup),
            n_firms=vec(w.n_firms),
            labor_force=float(w.labor_force[i]),
            population=float(w.population[i]),
            wage=float(w.wage[i]),
            unemployment=float(w.unemployment[i]),
            tax_rate=float(w.tax_rate[i]),
            welfare_share=float(w.welfare_share[i]),
            military_share=float(w.military_share[i]),
            subsidy_share=float(w.subsidy_share[i]),
            treasury=float(w.treasury[i]),
            household_cash=float(w.household_cash[i]),
            debt=float(w.debt[i]),
            gdp=float(w.gdp[i]),
            cpi=float(w.cpi[i]),
            inflation_q=float(w.inflation_q[i]),
            policy_rate=float(w.policy_rate[i]),
            stability=float(w.stability[i]),
            military=float(w.military[i]),
        )


# ----------------------------------------------------------------------------- building


def _per_country(cfg: Config, getter) -> np.ndarray:
    return np.array([getter(c) for c in cfg.countries.countries], dtype=float)


def _per_country_sector(cfg: Config, getter) -> np.ndarray:
    return np.array([[getter(c)[s] for s in SECTORS] for c in cfg.countries.countries], dtype=float)


def consumption_share_matrix(cfg: Config) -> np.ndarray:
    default = cfg.world.demand.consumption_shares_default
    return np.array(
        [[(c.consumption_shares or default)[s] for s in SECTORS] for c in cfg.countries.countries]
    )


def initial_state(cfg: Config, seed: int) -> WorldState:
    """Rough pre-burn-in state (§4.3, §6.15). The rules are documented in world.yaml: initial_state."""
    from world.engine.firms import base_markup  # local import: firms.py imports Firm/Event from here

    w = cfg.world
    init = w.initial_state
    beta = np.array([w.production.exponents[s].beta for s in SECTORS])
    gamma = np.array([w.production.exponents[s].gamma for s in SECTORS])

    lf = _per_country(cfg, lambda c: c.labor_force)
    pop = lf / w.labor_force_share_of_population
    A = _per_country_sector(cfg, lambda c: c.productivity)
    n_firms = _per_country_sector(cfg, lambda c: c.n_firms).astype(np.int64)
    shares = consumption_share_matrix(cfg)
    price = np.full((N_COUNTRIES, N_SECTORS), w.prices.initial_price)
    mu0 = base_markup(n_firms, w.firms.mu_dominant)

    # Rough but self-consistent factor allocation and output (see world.yaml initial_state):
    # E0 solves the energy-demand rule E = (1 - mu) gamma P Q / P_E with Q = A L0^beta E^gamma,
    # and the wage makes total labor demand (1 - mu) beta R / w equal the labor force.
    L0 = lf[:, None] * shares
    E0 = np.power(
        (1.0 - mu0) * gamma * A * np.power(L0, beta) * price / price[:, [ENERGY]], 1.0 / (1.0 - gamma)
    )
    Q0 = cobb_douglas(A, L0, E0, beta, gamma)
    revenue = price * Q0
    wage = ((1.0 - mu0) * beta * revenue).sum(axis=1) / lf
    gdp = revenue.sum(axis=1) - price[:, ENERGY] * E0.sum(axis=1)

    stock = init.stock_quarters * Q0
    stock[:, ENERGY] = init.energy_stock_turns * E0.sum(axis=1)  # D44
    stock[:, SERVICES] = 0.0

    tax = _per_country(cfg, lambda c: c.tax_rate)
    welfare = _per_country(cfg, lambda c: c.spending.welfare)
    military_share = _per_country(cfg, lambda c: c.spending.military)
    subsidy_share = _per_country(cfg, lambda c: c.spending.subsidy)
    treasury = _per_country(cfg, lambda c: c.treasury_to_quarterly_gdp) * gdp
    debt = _per_country(cfg, lambda c: c.debt_to_annual_gdp) * w.periods_per_year * gdp
    military = w.military.initial_stock_multiplier * military_share * gdp / price[:, GOODS]

    s0 = w.monetary.s0
    y_disp = gdp * (1.0 - tax) + welfare * gdp
    cash = s0 * y_disp / w.demand.wealth_spend_rate

    trust = np.full((N_COUNTRIES, N_COUNTRIES), w.trust.initial_default)
    for j, c in enumerate(cfg.countries.countries):
        trust[:, j] = c.initial_trust_received
    np.fill_diagonal(trust, init.trust_self)

    opening: dict[Account, float] = {households(i): cash[i] for i in range(N_COUNTRIES)}
    opening.update({government(i): treasury[i] for i in range(N_COUNTRIES)})
    opening[BOND_MARKET] = 0.0
    ledger = Ledger.with_opening(opening)

    firm_lists = []
    for i, cname in enumerate(COUNTRIES):
        for g, sname in enumerate(SECTORS):
            n = int(n_firms[i, g])
            firm_lists.append(tuple(Firm(f"{cname}-{sname}-{k}", 1.0 / n) for k in range(n)))

    cpi = (shares * price).sum(axis=1)
    zeros6 = np.zeros(N_COUNTRIES)
    zeros65 = np.zeros((N_COUNTRIES, N_SECTORS))

    return WorldState(
        seed=seed,
        turn=0,
        productivity=A,
        shock_mult=np.ones((N_COUNTRIES, N_SECTORS)),
        price=price,
        stock=stock,
        output=Q0,
        revenue=revenue,
        labor=L0,
        energy_in=E0,
        subsidy=zeros65.copy(),
        industry_subsidy=zeros65.copy(),
        markup=mu0,
        n_firms=n_firms,
        nationalized=np.zeros((N_COUNTRIES, N_SECTORS), dtype=bool),
        dominance_age=np.zeros((N_COUNTRIES, N_SECTORS), dtype=np.int64),
        consumption_shares=shares,
        demand=zeros65.copy(),
        excess_ema=zeros65.copy(),
        shortage=zeros65.copy(),
        labor_force_base=lf,
        labor_force=lf.copy(),
        population=pop,
        wage=wage,
        unemployment=zeros6.copy(),
        tax_rate=tax,
        welfare_share=welfare,
        military_share=military_share,
        subsidy_share=subsidy_share,
        subsidy_target=np.full(N_COUNTRIES, -1, dtype=np.int64),
        gov_revenue=tax * gdp,
        debt=debt,
        gdp=gdp,
        gdp_prev=gdp.copy(),
        gdp_hist=np.tile(gdp[:, None], (1, w.fiscal.gdp_ma_turns)),
        gdp_start=gdp.copy(),
        cpi=cpi,
        cpi_prev=cpi.copy(),
        inflation_q=zeros6.copy(),
        policy_rate=np.full(N_COUNTRIES, w.monetary.r_n + w.monetary.pi_target),
        policy_rate_override=np.full(N_COUNTRIES, np.nan),
        saving_rate=np.full(N_COUNTRIES, s0),
        y_disp=y_disp,
        stability=_per_country(cfg, lambda c: c.stability),
        military=military,
        default_premium=zeros6.copy(),
        default_turns_left=np.zeros(N_COUNTRIES, dtype=np.int64),
        leader_changes=np.zeros(N_COUNTRIES, dtype=np.int64),
        leader_changed_turn=np.full(N_COUNTRIES, -1, dtype=np.int64),
        trust=trust,
        sanction=np.zeros((N_COUNTRIES, N_COUNTRIES), dtype=bool),
        tariff=np.zeros((N_COUNTRIES, N_COUNTRIES, N_TRADED)),
        export_cap=np.full((N_COUNTRIES, N_COUNTRIES, N_TRADED), w.trade.export_cap_default),
        levy=np.zeros((N_COUNTRIES, N_TRADED)),
        trade=np.zeros((N_COUNTRIES, N_COUNTRIES, N_TRADED)),
        firms=tuple(firm_lists),
        treaties=(),
        active_shocks=(),
        ledger=ledger,
    )
