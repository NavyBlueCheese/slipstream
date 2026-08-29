from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from slipstream.contracts import get_spec
from slipstream.costs import (
    CommissionLayer,
    DeterministicLatency,
    FinancingLayer,
    ImpactLayer,
    LatencyLayer,
    LinearParticipationImpact,
    RealisticCostModel,
    RollCostLayer,
    SpreadLayer,
    SquareRootImpact,
    TradeContext,
    ZeroCostModel,
    conditional_spread_distribution,
)
from slipstream.data import SyntheticMarketConfig, SyntheticMarketGenerator

ES = get_spec("ES")


def make_ctx(qty=2.0, side=1, signal_mid=5000.0, arrival_mid=5000.0, spread=0.25, **kw):
    ts = pd.Timestamp("2025-06-02 14:00", tz="UTC")
    defaults = dict(
        spec=ES,
        side=side,
        qty=qty,
        signal_ts=ts,
        arrival_ts=ts + pd.Timedelta(milliseconds=250),
        signal_mid=signal_mid,
        arrival_mid=arrival_mid,
        spread=spread,
        top_depth=40.0,
        adv=1_200_000.0,
        daily_vol=0.011,
        horizon_volume=500.0,
    )
    defaults.update(kw)
    return TradeContext(**defaults)


def test_commission_layer_is_deterministic_per_contract():
    layer = CommissionLayer(0.85, 1.40, 0.10, 0.02)
    assert layer.per_contract() == pytest.approx(2.37)
    assert layer.cost(make_ctx(qty=3)) == pytest.approx(7.11)
    assert layer.cost(make_ctx(qty=-3)) == pytest.approx(7.11)


def test_spread_layer_charges_half_spread():
    layer = SpreadLayer(weight=1.0)
    assert layer.cost(make_ctx(qty=2, spread=0.25)) == pytest.approx(2 * 50 * 0.125)


def test_spread_layer_uses_observed_spread_not_constant():
    layer = SpreadLayer(weight=1.0)
    narrow = layer.cost(make_ctx(spread=0.25))
    wide = layer.cost(make_ctx(spread=0.75))
    assert wide == pytest.approx(3 * narrow)


def test_sqrt_impact_cost_scales_as_three_halves_power():
    layer = ImpactLayer(SquareRootImpact(coefficient=0.7))
    small = layer.cost(make_ctx(qty=100))
    big = layer.cost(make_ctx(qty=400))
    assert big / small == pytest.approx(4.0**1.5)


def test_linear_participation_impact_cost_is_quadratic_in_qty():
    layer = ImpactLayer(LinearParticipationImpact(coefficient=0.5))
    small = layer.cost(make_ctx(qty=10))
    big = layer.cost(make_ctx(qty=20))
    assert big / small == pytest.approx(4.0)


def test_latency_layer_signs_drift_by_side():
    layer = LatencyLayer(DeterministicLatency(0.5))
    adverse_buy = layer.cost(make_ctx(side=1, signal_mid=5000.0, arrival_mid=5000.5))
    favorable_buy = layer.cost(make_ctx(side=1, signal_mid=5000.0, arrival_mid=4999.5))
    adverse_sell = layer.cost(make_ctx(side=-1, signal_mid=5000.0, arrival_mid=4999.5))
    assert adverse_buy == pytest.approx(2 * 50 * 0.5)
    assert favorable_buy == pytest.approx(-2 * 50 * 0.5)
    assert adverse_sell == pytest.approx(2 * 50 * 0.5)


def test_latency_sampling_is_seeded():
    model = RealisticCostModel("ES")
    a = [model.sample_latency_seconds(np.random.default_rng(5)) for _ in range(3)]
    b = [model.sample_latency_seconds(np.random.default_rng(5)) for _ in range(3)]
    assert a == b
    assert all(x > 0 for x in a)


def test_roll_cost_charges_calendar_spread_not_two_outrights():
    layer = RollCostLayer(calendar_spread_ticks=1.0)
    cost = layer.cost(ES, qty=4)
    assert cost == pytest.approx(4 * 12.50 * 0.5)
    two_outright_spreads = 2 * 4 * 12.50 * 0.5
    assert cost < two_outright_spreads


def test_financing_cost_accrues_on_margin():
    layer = FinancingLayer(margin_rate=0.0, collateral_opportunity_rate=0.045)
    cost = layer.cost(ES, margin_posted=28000.0, days=30.0)
    assert cost == pytest.approx(28000 * 0.045 * 30 / 360)


def test_zero_cost_model_is_explicit_and_zero_everywhere():
    model = ZeroCostModel()
    ctx = make_ctx(arrival_mid=5001.0)
    breakdown = model.trade_breakdown(ctx)
    assert set(breakdown) == {"commission", "spread", "impact", "latency"}
    assert all(v == 0.0 for v in breakdown.values())
    assert model.roll_cost(ES, 10) == 0.0
    assert model.financing_cost(ES, 50000, 90) == 0.0
    assert model.sample_latency_seconds(np.random.default_rng(0)) == 0.0


def test_realistic_model_breakdown_attributes_by_layer():
    model = RealisticCostModel("ES")
    ctx = make_ctx(qty=2, spread=0.25, signal_mid=5000.0, arrival_mid=5000.0)
    breakdown = model.trade_breakdown(ctx)
    assert breakdown["commission"] == pytest.approx(2 * 2.37)
    assert breakdown["spread"] == pytest.approx(12.5)
    assert breakdown["latency"] == 0.0
    assert breakdown["impact"] > 0.0


def test_layer_swap_isolates_single_source():
    model = RealisticCostModel("ES")
    ctx = make_ctx(qty=2)
    without_commission = model.replaced(commission=CommissionLayer())
    diff = model.trade_cost(ctx) - without_commission.trade_cost(ctx)
    assert diff == pytest.approx(model.commission.cost(ctx))


def test_unknown_commission_root_rejected():
    with pytest.raises(KeyError):
        RealisticCostModel("CL")


def test_conditional_spread_distribution_by_time_and_vol():
    config = SyntheticMarketConfig(session_start=time(13, 30), session_end=time(17, 30))
    quotes, _ = SyntheticMarketGenerator(config, seed=21).generate_day(date(2025, 6, 2))
    stats = conditional_spread_distribution(quotes, time_bucket="1h", vol_window=120)
    assert {"mean", "median", "std", "count", "p90"} <= set(stats.columns)
    by_vol = stats.groupby(level="vol_bucket", observed=True)["mean"].mean()
    assert by_vol["vol_q3"] > by_vol["vol_q1"]
