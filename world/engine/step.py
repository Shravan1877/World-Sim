"""turn_start() and step(): the functions that advance the world by one quarter (CLAUDE.md §6.1).

    turn_start: turn += 1, roll + apply shocks (§7), scripted incidents, move order (§4.2)
    [agents move; accepted actions change POLICY fields of the state (policy.py, a later phase)]
    step:       1 production   2 demand   3 trade   4 consumption   5 stocks   6 prices
                7 income/ledger (profits)  8 labor  9 fiscal  10 monetary  11 firms  12 military
                13 stability (unrest, leader change)  14 trust  15 treaties (later phase)
                16 snapshot: age shocks, invariant checks (§6.14)

Both functions are pure: they copy the state and return a new one. All money moves through the
ledger. Spending plans are shares of GDP, turned into credits with LAST turn's GDP (the latest
known value when the turn starts, floored at 0). Policies set this turn apply to this turn's
resolution; income earned this turn is spent next turn (D33).

Decisions wired here (CLAUDE.md §17):
- D30: ENERGY demand = households + firms' planned energy inputs (stock-building, not consumed).
- D31: taxes on household income, collected in the income step (income.py).
- D32: debt ratios use the floored 4-quarter GDP average; the premium is capped.
- D33: firm losses go to the owners, then the bond market; firm accounts end at 0.
- D34: every buyer of a good (households, government, the ENERGY firm account) pays its demand share
  of imports and home purchases; when a good is short, every buyer gets the same fill rate.
- D35: household spending this turn is capped at cash on hand + this turn's after-tax wages.
- Subsidy with no target (-1) is spread over sectors by last turn's revenue shares.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from world.config import COUNTRIES, SECTORS, Config, ScenarioCfg
from world.engine import demand as demand_mod
from world.engine import firms as firms_mod
from world.engine import fiscal, income, invariants, labor, military, monetary, prices, production, shocks
from world.engine import stability as stab_mod
from world.engine import trust as trust_mod
from world.engine.params import country_params
from world.engine.state import ENERGY, FOOD, GOODS, N_TRADED, SERVICES, ActiveShock, Event, WorldState
from world.engine.trade import ContractDelivery, SupplyContract, allocate_trade
from world.ledger import BOND_MARKET, LedgerError, government, households
from world.ledger import firms as firm_account
from world.order import move_order
from world.rng import RngBundle

# ------------------------------------------------------------------------------ turn start


@dataclass(frozen=True)
class TurnStart:
    turn: int
    shock_events: tuple[shocks.ShockEvent, ...]  # one per die rolled (fired or not) + scenario incidents
    order: tuple[str, ...]

    @property
    def fired(self) -> tuple[shocks.ShockEvent, ...]:
        return tuple(e for e in self.shock_events if e.fired)


def turn_start(
    state: WorldState,
    rng: RngBundle,
    cfg: Config,
    *,
    scenario: ScenarioCfg | None = None,
    shocks_on: bool = True,
) -> tuple[WorldState, TurnStart]:
    """Begin turn state.turn + 1: shocks, scripted incidents, move order. Agents then see these."""
    s = state.copy()
    s.turn = state.turn + 1
    events: list[shocks.ShockEvent] = []
    if shocks_on:
        s, ev = shocks.draw_shocks(s, rng, cfg)
        events += ev
    if scenario is not None:
        s, ev = shocks.apply_scenario(s, scenario, cfg)
        events += ev
    shocks.recompute_multipliers(s)
    return s, TurnStart(s.turn, tuple(events), move_order(rng, s.turn))


# ---------------------------------------------------------------------------------- step


@dataclass(frozen=True)
class TurnInputs:
    """What step() needs from this turn's accepted actions beyond the policy fields already set.

    Built by the policy/treaty layers (later phases). Defaults = a quiet turn.
    """

    antitrust: np.ndarray | None = None  # (6, 5) int: antitrust actions this turn per (country, sector)
    sanctions_imposed: tuple[tuple[int, int], ...] = ()  # (sanctioner, target), new this turn
    violations: tuple[tuple[int, int], ...] = ()  # (violator, victim)
    treaties_honored: tuple[tuple[int, int], ...] = ()
    renounced: tuple[tuple[int, int], ...] = ()  # (renouncer, counterpart)
    contracts: tuple[SupplyContract, ...] = ()


@dataclass(frozen=True)
class TurnLog:
    turn: int
    events: tuple[Event, ...]
    consumption: np.ndarray  # (6, 5)
    shortage: np.ndarray  # (6, 5)
    imports: np.ndarray  # (6, 4)
    exports: np.ndarray  # (6, 4)
    taxes: np.ndarray  # (6,)
    profit: np.ndarray  # (6, 5)
    interest: np.ndarray  # (6,)
    borrowing: np.ndarray  # (6,)
    hhi: np.ndarray  # (6, 5)
    deliveries: tuple[ContractDelivery, ...]
    firm_dice: np.ndarray  # (6, 5, 4)
    extra: dict[str, np.ndarray] = field(default_factory=dict)


def subsidy_matrix(subsidy_credits: np.ndarray, target: np.ndarray, revenue_last: np.ndarray) -> np.ndarray:
    """Subsidy credits per (country, sector): all to the target sector, or by revenue shares if -1."""
    n_c, n_s = revenue_last.shape
    out = np.zeros((n_c, n_s))
    for i in range(n_c):
        if target[i] >= 0:
            out[i, target[i]] = subsidy_credits[i]
        else:
            total = revenue_last[i].sum()
            w = revenue_last[i] / total if total > 0 else np.full(n_s, 1.0 / n_s)
            out[i] = subsidy_credits[i] * w
    return out


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    return a / np.maximum(b, eps)


N_PAYERS = 3  # households, government, the country's own firms in that sector (trade.N_PAYERS)


def payer_weights(
    d_house: np.ndarray, d_gov: np.ndarray, d_firm_energy: np.ndarray, D: np.ndarray
) -> np.ndarray:
    """(6, 5, 3) demand shares of each buyer of good g in country i (D34).

    0 households (D_house), 1 government (military GOODS), 2 the ENERGY firm account (D30 firm part).
    Where total demand is 0, households are the (notional) buyer.
    """
    n_c, n_s = D.shape
    w = np.zeros((n_c, n_s, N_PAYERS))
    has = D > 0
    safe = np.where(has, D, 1.0)
    w[:, :, 0] = np.where(has, d_house / safe, 1.0)
    w[:, GOODS, 1] = np.where(has[:, GOODS], d_gov / safe[:, GOODS], 0.0)
    w[:, ENERGY, 2] = np.where(has[:, ENERGY], d_firm_energy / safe[:, ENERGY], 0.0)
    return w


def step(
    state: WorldState, inputs: TurnInputs | None, rng: RngBundle, cfg: Config
) -> tuple[WorldState, TurnLog]:
    """Resolve the turn that turn_start() began (state.turn). Returns (new_state, log).

    Raises InvariantError if any §6.14 invariant fails, including a money transfer that is not
    finite (the ledger refuses it). The run must then stop and be marked invariant_failed.
    """
    try:
        return _step(state, inputs, rng, cfg)
    except LedgerError as e:
        raise invariants.InvariantError(f"turn {state.turn}: {e}") from e


def _step(
    state: WorldState, inputs: TurnInputs | None, rng: RngBundle, cfg: Config
) -> tuple[WorldState, TurnLog]:
    w = cfg.world
    cp = country_params(cfg)
    inputs = inputs or TurnInputs()
    eps = w.eps
    ppy = w.periods_per_year
    t = state.turn
    s = state.copy()
    led = s.ledger
    n_c, n_s = s.price.shape
    events: list[Event] = []
    beta = np.array([w.production.exponents[x].beta for x in SECTORS])
    gamma = np.array([w.production.exponents[x].gamma for x in SECTORS])
    P = state.price  # this turn's trading prices (updated in step 6)
    cash_start = state.household_cash
    stock_start = state.stock

    # Spending plans in credits, from last turn's GDP.
    gdp_last = np.maximum(state.gdp, 0.0)
    welfare_c = state.welfare_share * gdp_last
    military_c = state.military_share * gdp_last
    subsidy = subsidy_matrix(state.subsidy_share * gdp_last, state.subsidy_target, state.revenue)

    # ---- 1 production
    antitrust = inputs.antitrust if inputs.antitrust is not None else np.zeros((n_c, n_s), dtype=np.int64)
    mu = firms_mod.effective_markup(
        state.n_firms, state.nationalized, antitrust, cp.antitrust_strength, w.firms.mu_dominant
    )
    for i in range(n_c):
        for g in range(n_s):
            if subsidy[i, g] > 0:
                led.transfer(government(i), firm_account(i, g), subsidy[i, g], "subsidy")
    prod = production.produce(
        A=state.productivity,
        shock_mult=state.shock_mult,
        nationalized=state.nationalized,
        delta_nat=w.firms.nationalization_productivity_loss,
        revenue_last=state.revenue,
        subsidy=subsidy,
        eta=cp.subsidy_efficiency,
        mu=mu,
        beta=beta,
        gamma=gamma,
        wage=state.wage,
        price=P,
        labor_force=state.labor_force,
        stock=stock_start,
    )
    Q, L, E = prod.output, prod.labor, prod.energy
    wages_paid = state.wage[:, None] * L  # paid in the income step (7)
    energy_cost = P[:, ENERGY][:, None] * E

    # ---- 2 demand (D30: firms' planned energy inputs refill the ENERGY stock; D35: budget cap)
    y_spend = demand_mod.spendable_income(
        state.y_disp, state.saving_rate, cash_start, w.demand.wealth_spend_rate
    )
    budget = np.maximum(cash_start + wages_paid.sum(axis=1) * (1.0 - state.tax_rate), 0.0)
    y_spend = np.clip(y_spend, 0.0, budget)
    d_house, _ = demand_mod.household_demand(
        state.consumption_shares, y_spend, P, w.demand.f_min, state.population
    )
    d_gov = demand_mod.government_goods_demand(military_c, P[:, GOODS])
    d_firm_energy = prod.energy_demand.sum(axis=1)
    D = demand_mod.total_demand(d_house, d_gov, d_firm_energy)
    payers = payer_weights(d_house, d_gov, d_firm_energy, D)

    # ---- 3 trade
    tr = allocate_trade(
        stock=prod.stock_after_inputs[:, :N_TRADED],
        output=Q[:, :N_TRADED],
        demand=D[:, :N_TRADED],
        price=P[:, :N_TRADED],
        levy=state.levy,
        tariff=state.tariff,
        export_cap=state.export_cap,
        sanction=state.sanction,
        trust=state.trust,
        kappa=w.trade.kappa,
        sigma=w.trade.sigma_trade,
        passes=w.trade.allocation_passes,
        contracts=inputs.contracts,
        ledger=led,
        payer_weights=payers[:, :N_TRADED],
    )
    led = tr.ledger

    # ---- 4 consumption: every buyer of a good gets the same fill rate min(1, available / D) (D34).
    # The firm part of ENERGY demand is filled into stock, not consumed (D30).
    s_dom = prod.stock_after_inputs + Q
    available = s_dom.copy()
    available[:, :N_TRADED] += tr.imports - tr.exports
    available[:, SERVICES] = Q[:, SERVICES]
    available = np.maximum(available, 0.0)
    fill = np.where(D > 0, np.minimum(available / np.where(D > 0, D, 1.0), 1.0), 1.0)
    taken = D * fill
    shortage = D - taken
    firm_energy_filled = d_firm_energy * fill[:, ENERGY]
    consumption = taken.copy()
    consumption[:, ENERGY] -= firm_energy_filled
    gov_goods = d_gov * fill[:, GOODS]  # military goods actually received
    imports5 = np.zeros((n_c, n_s))
    imports5[:, :N_TRADED] = tr.imports
    domestic = np.maximum(taken - imports5, 0.0)  # units bought from home firms (imports go first)
    # D35: households buy home goods only with the budget left after paying for imports (landed
    # costs can exceed home prices). Units they cannot pay for stay in stock and count as shortage.
    house_dom = domestic * payers[:, :, 0]
    paid_for_imports = cash_start - np.array([led.balance(households(i)) for i in range(n_c)])
    left = np.maximum(budget - paid_for_imports, 0.0)
    want = (P * house_dom).sum(axis=1)
    afford = np.where(want > left, left / np.where(want > 0, want, 1.0), 1.0)
    unbought = house_dom * (1.0 - afford[:, None])
    consumption -= unbought
    shortage += unbought
    for i in range(n_c):
        for g in range(n_s):
            house = P[i, g] * house_dom[i, g] * afford[i]
            gov = P[i, g] * domestic[i, g] * payers[i, g, 1]
            if house > 0:
                led.transfer(households(i), firm_account(i, g), house, "domestic sales")
            if gov > 0:
                led.transfer(government(i), firm_account(i, g), gov, "military goods")
            # payers[i, g, 2]: the ENERGY firms keep their own stock for next turn's inputs; no payment

    # ---- 5 stocks (spoilage on what is left)
    spoil_rate = np.array([w.spoilage[x] for x in SECTORS[:N_TRADED]])
    held = prod.stock_after_inputs[:, :N_TRADED] + Q[:, :N_TRADED] + tr.imports - tr.exports
    held = held - consumption[:, :N_TRADED]
    spoilage = spoil_rate[None, :] * held
    new_stock = np.zeros((n_c, n_s))
    new_stock[:, :N_TRADED] = held - spoilage

    # ---- 6 prices, CPI, inflation
    d_eff, s_eff = prices.market_balance(D, s_dom, Q, tr.export_requests, tr.imports)
    new_price = prices.update_prices(P, d_eff, s_eff, w.prices.sigma_p, w.prices.max_change_per_turn, eps)
    new_cpi = prices.cpi(state.consumption_shares, new_price)
    infl_q = prices.inflation_quarterly(new_cpi, state.cpi)
    pi_annual = prices.annualize(infl_q, ppy)

    # ---- 7 income: wages, energy inputs, profits/losses to owners, taxes (income.py)
    inc = income.settle_income(
        led,
        wages_paid=wages_paid,
        energy_cost=energy_cost,
        subsidy=subsidy,
        nationalized=state.nationalized,
        tax_rate=state.tax_rate,
    )
    led, revenue = inc.ledger, inc.revenue

    # ---- 8 labor
    u = labor.unemployment_rate(prod.labor_demand, state.labor_force)
    new_wage = labor.update_wage(
        state.wage, prod.labor_demand, state.labor_force, w.labor.psi, w.labor.max_wage_change_per_turn
    )

    # ---- 9 fiscal
    gdp = fiscal.gdp_value_added(P, Q, E)
    gdp_hist = fiscal.push_gdp(state.gdp_hist, gdp)
    gdp_floor = w.fiscal.gdp_floor_share_of_start * state.gdp_start
    gdp_ref = fiscal.gdp_reference(gdp_hist, state.gdp_start, w.fiscal.gdp_floor_share_of_start)
    premium = fiscal.risk_premium(
        state.debt,
        gdp_ref,
        state.default_premium,
        slope=w.fiscal.premium_slope,
        threshold=w.fiscal.premium_threshold_debt_to_gdp,
        cap=w.fiscal.premium_cap,
        periods_per_year=ppy,
        eps=eps,
    )
    interest = fiscal.interest_due(state.debt, state.policy_rate, premium, ppy)
    sav_int = monetary.savings_interest(cash_start, state.policy_rate, ppy)
    for i in range(n_c):
        if welfare_c[i] > 0:
            led.transfer(government(i), households(i), welfare_c[i], "welfare")
        if interest[i] > 0:
            led.transfer(government(i), BOND_MARKET, interest[i], "interest on debt")
    led = monetary.pay_savings_interest(led, sav_int)
    # Backstop (D35): only if import bills alone exceeded the household budget can cash end below 0;
    # the bond market then tops households up to 0. Logged; zero in normal runs (tested).
    backstop = np.maximum(-np.array([led.balance(households(i)) for i in range(n_c)]), 0.0)
    for i in np.nonzero(backstop > 0)[0]:
        led.transfer(BOND_MARKET, households(int(i)), backstop[i], "household cash backstop")
    borrowing = np.zeros(n_c)
    for i in range(n_c):
        bal = led.balance(government(i))
        if bal < 0:
            borrowing[i] = -bal
            led.transfer(BOND_MARKET, government(i), -bal, "borrowing")
    debt = state.debt + borrowing
    dflt = fiscal.apply_default(
        debt,
        gdp_ref,
        state.stability,
        state.default_premium,
        state.default_turns_left,
        threshold=w.fiscal.default_threshold_debt_to_gdp,
        haircut=w.fiscal.default_haircut,
        stability_hit=w.fiscal.default_stability_hit,
        premium_add=w.fiscal.default_premium_add,
        premium_turns=w.fiscal.default_premium_turns,
        periods_per_year=ppy,
        eps=eps,
    )
    events += [Event("default", i, detail="debt haircut") for i in np.nonzero(dflt.defaulted)[0]]
    y_disp = fiscal.disposable_income(inc.wages, inc.profits_to_households, inc.taxes, welfare_c, sav_int)

    # ---- 10 monetary
    m = w.monetary
    taylor = monetary.taylor_rate(
        pi_annual, u, r_n=m.r_n, pi_target=m.pi_target, a=m.a, b=m.b, u_n=m.u_n, floor=m.rate_floor
    )
    rate = monetary.policy_rate(taylor, state.policy_rate_override)
    saving = monetary.saving_rate(rate, s0=m.s0, k_r=m.k_r, r_n=m.r_n, s_min=m.saving_min, s_max=m.saving_max)

    # ---- 11 firms (FIRMS stream)
    fu = firms_mod.update_firms(
        firm_lists=state.firms,
        nationalized=state.nationalized,
        dominance_age=state.dominance_age,
        mu=state.markup,
        entry_boost=shocks.entry_boost(state),
        breakup_omega=cp.breakup_omega,
        entry_mult=cp.entry_mult,
        p=w.firms,
        rng=rng,
        turn=t,
        country_names=COUNTRIES,
        sector_names=SECTORS,
    )
    events += fu.events

    # ---- 12 military
    mil = military.update_military(state.military, gov_goods, cp.military_efficiency, w.military.depreciation)

    # ---- 13 stability, unrest, leader change
    renounce_count = np.bincount([r for r, _ in inputs.renounced], minlength=n_c).astype(float)
    sanction_cost = state.sanction.sum(axis=1) * cp.sanction_self_cost
    d_stab = stab_mod.stability_change(
        stability=dflt.stability,
        u=u,
        pi_annual=pi_annual,
        food_shortage_frac=_safe_div(shortage[:, FOOD], D[:, FOOD], eps),
        energy_shortage_frac=_safe_div(shortage[:, ENERGY], D[:, ENERGY], eps),
        welfare_to_gdp=_safe_div(welfare_c, np.maximum(gdp, gdp_floor), eps),
        sanction_cost=sanction_cost,
        shock_effects=renounce_count * cp.renounce_stability_cost,
        u_n=m.u_n,
        pi_target=m.pi_target,
        p=w.stability,
    )
    stab = stab_mod.update_stability(dflt.stability, d_stab, w.stability)
    ur = stab_mod.unrest_and_leader_fall(stab, cp.leader_fall_prob, w.stability, rng, t)
    stab, new_shocks = ur.stability, []
    leader_changes, leader_turn = state.leader_changes, state.leader_changed_turn
    for i in np.nonzero(ur.unrest)[0]:
        new_shocks.append(ActiveShock("unrest", int(i), -1, 1.0 - w.stability.unrest_output_loss, 1))
        events.append(Event("unrest", int(i), detail="output -5% next turn"))
    for i in np.nonzero(ur.leader_falls)[0]:
        lc = stab_mod.leader_change(stab, leader_changes, leader_turn, int(i), t, w.stability)
        stab, leader_changes, leader_turn = lc.stability, lc.leader_changes, lc.leader_changed_turn
        label = cfg.countries.countries[i].params.leader_fall_label or "leader fall"
        events.append(Event("leader_change", int(i), detail=label))

    # ---- 14 trust
    contract_violations = tuple(
        (c.seller, c.buyer)
        for c, d in zip(_goods_order(inputs.contracts), tr.deliveries, strict=True)
        if d.policy_caused
    )
    new_trust = trust_mod.update_trust(
        state.trust,
        trust_mod.TrustEvents(
            violations=inputs.violations + contract_violations,
            sanctions=inputs.sanctions_imposed,
            honored=inputs.treaties_honored,
            renounced=inputs.renounced,
        ),
        w.trust,
    )

    # ---- 15 treaties: execution, expiry and violation detection come with treaties.py (later phase)

    # ---- 16 snapshot
    s.ledger = led
    s.price, s.stock, s.output, s.revenue = new_price, new_stock, Q, revenue
    s.labor, s.energy_in, s.subsidy = L, E, subsidy
    s.markup, s.n_firms, s.firms, s.dominance_age = fu.markup, fu.n_firms, fu.firms, fu.dominance_age
    s.demand, s.shortage = D, shortage
    s.wage, s.unemployment = new_wage, u
    s.debt, s.default_premium, s.default_turns_left = dflt.debt, dflt.default_premium, dflt.default_turns_left
    s.gdp_prev, s.gdp, s.gdp_hist = state.gdp, gdp, gdp_hist
    s.cpi_prev, s.cpi, s.inflation_q = state.cpi, new_cpi, infl_q
    s.policy_rate, s.saving_rate, s.y_disp = rate, saving, y_disp
    s.stability, s.military = stab, mil
    s.leader_changes, s.leader_changed_turn = leader_changes, leader_turn
    s.trust, s.trade = new_trust, tr.flows
    s.active_shocks = shocks.age_shocks(state.active_shocks) + tuple(new_shocks)
    shocks.recompute_multipliers(s)

    check_invariants(
        s,
        cfg,
        invariants.StockFlow(
            stock_start=stock_start[:, :N_TRADED],
            output=Q[:, :N_TRADED],
            imports=tr.imports,
            exports=tr.exports,
            consumption=consumption[:, :N_TRADED],
            energy_inputs=_energy_inputs(E),
            spoilage=spoilage,
            stock_end=new_stock[:, :N_TRADED],
        ),
        flows=tr.flows,
    )
    log = TurnLog(
        turn=t,
        events=tuple(events),
        consumption=consumption,
        shortage=shortage,
        imports=tr.imports,
        exports=tr.exports,
        taxes=inc.taxes,
        profit=inc.profit,
        interest=interest,
        borrowing=borrowing,
        hhi=fu.hhi,
        deliveries=tr.deliveries,
        firm_dice=fu.dice,
        extra={
            "military_goods": gov_goods,
            "labor_demand": prod.labor_demand,
            "energy_demand": prod.energy_demand,
            "energy_used": E,
            "firm_energy_filled": firm_energy_filled,
            "surplus": tr.surplus,
            "y_spend": y_spend,
            "welfare": welfare_c,
            "subsidy": subsidy.sum(axis=1),
            "savings_interest": sav_int,
            "state_profit": inc.state_profit,
            "loss_bond_market": inc.loss_bond_market,
            "household_backstop": backstop,
            "household_unbought": unbought,
            "gdp_ref": gdp_ref,
            "unrest": ur.unrest,
            "leader_falls": ur.leader_falls,
        },
    )
    return s, log


def _energy_inputs(E: np.ndarray) -> np.ndarray:
    out = np.zeros((E.shape[0], N_TRADED))
    out[:, ENERGY] = E.sum(axis=1)
    return out


def _goods_order(contracts: tuple[SupplyContract, ...]) -> list[SupplyContract]:
    """allocate_trade reports deliveries good by good; return the contracts in that same order."""
    return [c for g in range(N_TRADED) for c in contracts if c.good == g]


# ---------------------------------------------------------------------------- invariants


def check_invariants(s: WorldState, cfg: Config, sf: invariants.StockFlow, flows: np.ndarray) -> None:
    """§6.14: raise InvariantError (the run must stop and be marked invariant_failed)."""
    tol = cfg.world.invariants
    errors = invariants.check_money(s.ledger, tol.rel_tol)
    errors += invariants.check_firm_accounts_zero(s.ledger, float(np.sum(np.abs(s.gdp))), tol.rel_tol)
    errors += invariants.check_trade_balance(flows, tol.rel_tol)
    errors += invariants.check_stock_flow(sf, tol.rel_tol)
    errors += invariants.check_non_negative(
        {
            "stock": s.stock,
            "price": s.price,
            "wage": s.wage,
            "labor": s.labor,
            "treasury": s.treasury,
            "household_cash": s.household_cash,
            "debt": s.debt,
            "military": s.military,
        },
        tol.abs_tol * max(1.0, float(np.sum(np.abs(s.gdp)))),
    )
    errors += invariants.check_bounds(
        {
            "unemployment": (s.unemployment, 0.0, 1.0),
            "stability": (s.stability, 0.0, 100.0),
            "trust": (s.trust, 0.0, 1.0),
        },
        tol.abs_tol,
    )
    firm_shares = np.zeros((len(s.firms), max(len(fl) for fl in s.firms)))
    for k, fl in enumerate(s.firms):
        firm_shares[k, : len(fl)] = [f.share for f in fl]
    errors += invariants.check_shares(
        {"firm shares": firm_shares, "consumption shares": s.consumption_shares}, tol.abs_tol
    )
    if np.any(s.n_firms != np.array([len(fl) for fl in s.firms]).reshape(s.n_firms.shape)):
        errors.append("n_firms out of sync with firm lists")
    invariants.raise_if_any(errors, s.turn)
