"""Trade (§6.4): Armington checks, tariffs, sanctions, caps, rationing, balance, payments, contracts."""

import numpy as np
import pytest

from world.engine.trade import SupplyContract, allocate_trade, armington_shares
from world.ledger import Ledger, firms, government, households

N = 6


def _setup(**over):
    """Importer 0 needs 100 FOOD; exporters 1, 2, 3 have 1000 surplus at prices 1.0, 1.1, 1.3."""
    stock = np.zeros((N, 4))
    output = np.zeros((N, 4))
    demand = np.zeros((N, 4))
    demand[0, 0] = 100.0
    output[1:4, 0] = 1000.0
    price = np.ones((N, 4))
    price[1:4, 0] = [1.0, 1.1, 1.3]
    args = dict(
        stock=stock,
        output=output,
        demand=demand,
        price=price,
        levy=np.zeros((N, 4)),
        tariff=np.zeros((N, N, 4)),
        export_cap=np.ones((N, N, 4)),
        sanction=np.zeros((N, N), dtype=bool),
        trust=np.full((N, N), 0.7),
        kappa=0.0,
        sigma=3.0,
        passes=2,
        contracts=(),
        ledger=Ledger.with_opening({households(i): 1e6 for i in range(N)}),
    )
    args.update(over)
    return args


def test_armington_claude_md_shares() -> None:
    r = allocate_trade(**_setup())
    shares = r.flows[0, 1:4, 0] / r.flows[0, :, 0].sum()
    np.testing.assert_allclose(np.round(shares * 100), [41, 34, 24])
    assert r.imports[0, 0] == pytest.approx(100.0)


def test_tariff_on_seller_1_shifts_shares() -> None:
    tariff = np.zeros((N, N, 4))
    tariff[0, 1, 0] = 0.30
    r = allocate_trade(**_setup(tariff=tariff))
    shares = r.flows[0, 1:4, 0] / r.flows[0, :, 0].sum()
    np.testing.assert_allclose(np.round(shares * 100), [29, 41, 29])


def test_trust_weighting() -> None:
    cost = np.ones((1, 2))
    trust = np.array([[0.7, 0.35]])
    s = armington_shares(cost, trust, np.ones((1, 2), dtype=bool), kappa=1.0, sigma=3.0)
    np.testing.assert_allclose(s, [[2 / 3, 1 / 3]])


@pytest.mark.parametrize("direction", ["importer_sanctions", "exporter_sanctions"])
def test_sanction_blocks_both_ways(direction: str) -> None:
    sanction = np.zeros((N, N), dtype=bool)
    if direction == "importer_sanctions":
        sanction[0, 1] = True
    else:
        sanction[1, 0] = True
    # Also give country 0 surplus of GOODS that country 1 needs: the reverse flow must be 0 too.
    a = _setup(sanction=sanction)
    a["output"][0, 2] = 500.0
    a["demand"][1, 2] = 50.0
    r = allocate_trade(**a)
    assert r.flows[0, 1].sum() == 0.0
    assert r.flows[1, 0].sum() == 0.0
    assert r.imports[0, 0] == pytest.approx(100.0)  # others fill the need


def test_rationing_never_exports_more_than_surplus() -> None:
    a = _setup()
    a["demand"][[0, 4, 5], 0] = 900.0  # 2700 needed, 1300 + 0 surplus offered below
    a["output"][1:4, 0] = [500.0, 500.0, 300.0]
    r = allocate_trade(**a)
    assert np.all(r.exports <= r.surplus + 1e-9)
    np.testing.assert_allclose(r.exports[1:4, 0], [500.0, 500.0, 300.0])  # all surplus sold
    assert r.unmet_imports[:, 0].sum() == pytest.approx(2700.0 - 1300.0)
    # requests include the cut parts, so X_req exceeds what was shipped
    assert np.all(r.export_requests[1:4, 0] > r.exports[1:4, 0])


def test_world_exports_equal_imports_random_cases() -> None:
    from world.rng import Stream, make_rng

    for k in range(25):
        rng = make_rng(k, 0, Stream.BOT)
        a = _setup(
            stock=rng.uniform(0, 50, (N, 4)),
            output=rng.uniform(0, 100, (N, 4)),
            demand=rng.uniform(0, 150, (N, 4)),
            price=rng.uniform(0.5, 2, (N, 4)),
            levy=rng.uniform(0, 0.5, (N, 4)),
            tariff=rng.uniform(0, 1, (N, N, 4)),
            export_cap=rng.uniform(0, 1, (N, N, 4)),
            sanction=rng.random((N, N)) < 0.15,
            trust=rng.uniform(0, 1, (N, N)),
            kappa=1.0,
        )
        r = allocate_trade(**a)
        np.testing.assert_allclose(r.imports.sum(axis=0), r.exports.sum(axis=0))
        assert np.all(r.exports <= r.surplus + 1e-9)
        assert np.all(r.flows >= 0)
        assert np.all(np.diagonal(r.flows, axis1=0, axis2=1) == 0)
        r.ledger.check_conservation()


def test_payments_split_into_price_levy_tariff() -> None:
    levy = np.zeros((N, 4))
    levy[1, 0] = 0.1
    tariff = np.zeros((N, N, 4))
    tariff[0, 1, 0] = 0.2
    a = _setup(levy=levy, tariff=tariff)
    a["output"][2:4, 0] = 0.0  # only exporter 1
    r = allocate_trade(**a)
    L = r.ledger
    assert L.balance(firms(1, 0)) == pytest.approx(100.0)  # P * q
    assert L.balance(government(1)) == pytest.approx(10.0)  # levy
    assert L.balance(government(0)) == pytest.approx(100 * 1.1 * 0.2)  # tariff on the levied price
    assert L.balance(households(0)) == pytest.approx(1e6 - 100 * 1.1 * 1.2)
    L.check_conservation()


def test_export_cap_limits_target_and_second_pass_refills() -> None:
    cap = np.ones((N, N, 4))
    cap[0, 1, 0] = 0.0  # exporter 1 bans exports to 0
    r = allocate_trade(**_setup(export_cap=cap))
    assert r.flows[0, 1, 0] == 0.0
    assert r.imports[0, 0] == pytest.approx(100.0)


def test_second_pass_uses_remaining_surplus() -> None:
    a = _setup()
    a["output"][1:4, 0] = [10.0, 10.0, 1000.0]  # cheap sellers run out in pass 1
    r = allocate_trade(**a)
    assert r.imports[0, 0] == pytest.approx(100.0)
    np.testing.assert_allclose(r.flows[0, 1:3, 0], [10.0, 10.0])


def test_contract_delivers_first_and_detects_policy_shortfall() -> None:
    c = SupplyContract("T1", seller=1, buyer=0, good=0, quantity=60.0, price=0.9)
    r = allocate_trade(**_setup(contracts=(c,)))
    assert r.deliveries[0].delivered == pytest.approx(60.0)
    assert r.deliveries[0].shortfall == 0.0
    assert r.imports[0, 0] == pytest.approx(100.0)  # 60 contract + 40 market
    assert r.ledger.balance(firms(1, 0)) >= 60 * 0.9

    cap = np.ones((N, N, 4))
    cap[0, 1, 0] = 0.0
    r2 = allocate_trade(**_setup(contracts=(c,), export_cap=cap))
    assert r2.deliveries[0].delivered == 0.0
    assert r2.deliveries[0].policy_caused


def test_contract_shortfall_from_low_supply_is_not_a_violation() -> None:
    a = _setup(contracts=(SupplyContract("T2", 1, 0, 0, 60.0, 1.0),))
    a["output"][1, 0] = 20.0
    r = allocate_trade(**a)
    assert r.deliveries[0].delivered == pytest.approx(20.0)
    assert not r.deliveries[0].policy_caused


def test_input_ledger_not_mutated() -> None:
    a = _setup()
    before = dict(a["ledger"].balances)
    allocate_trade(**a)
    assert a["ledger"].balances == before
