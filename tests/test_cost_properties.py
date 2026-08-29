import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from slipstream.contracts import get_spec
from slipstream.costs import (
    CommissionLayer,
    ImpactLayer,
    LinearParticipationImpact,
    RollCostLayer,
    FinancingLayer,
    SpreadLayer,
    SquareRootImpact,
    TradeContext,
)

ES = get_spec("ES")

ctx_strategy = st.builds(
    TradeContext,
    spec=st.just(ES),
    side=st.sampled_from((-1, 1)),
    qty=st.floats(min_value=0.0, max_value=10_000.0, allow_nan=False),
    signal_ts=st.just(pd.Timestamp("2025-06-02 14:00", tz="UTC")),
    arrival_ts=st.just(pd.Timestamp("2025-06-02 14:00:00.2", tz="UTC")),
    signal_mid=st.floats(min_value=1.0, max_value=10_000.0),
    arrival_mid=st.floats(min_value=1.0, max_value=10_000.0),
    spread=st.floats(min_value=0.0, max_value=10.0),
    top_depth=st.floats(min_value=1.0, max_value=5_000.0),
    adv=st.floats(min_value=1.0, max_value=5_000_000.0),
    daily_vol=st.floats(min_value=0.0, max_value=0.2),
    horizon_volume=st.floats(min_value=1.0, max_value=100_000.0),
)


@settings(max_examples=200, deadline=None)
@given(ctx=ctx_strategy)
def test_fee_spread_and_impact_costs_never_negative(ctx):
    assert CommissionLayer(0.85, 1.40, 0.10, 0.02).cost(ctx) >= 0.0
    assert SpreadLayer(weight=1.0).cost(ctx) >= 0.0
    assert ImpactLayer(SquareRootImpact(0.7)).cost(ctx) >= 0.0
    assert ImpactLayer(LinearParticipationImpact(0.5)).cost(ctx) >= 0.0


@settings(max_examples=200, deadline=None)
@given(
    qty=st.floats(min_value=-1_000.0, max_value=1_000.0, allow_nan=False),
    ticks=st.floats(min_value=0.0, max_value=8.0),
    margin=st.floats(min_value=0.0, max_value=1e7),
    days=st.floats(min_value=0.0, max_value=365.0),
    rate=st.floats(min_value=0.0, max_value=0.2),
)
def test_roll_and_financing_costs_never_negative(qty, ticks, margin, days, rate):
    assert RollCostLayer(ticks).cost(ES, qty) >= 0.0
    assert FinancingLayer(0.0, rate).cost(ES, margin, days) >= 0.0


@settings(max_examples=100, deadline=None)
@given(
    ctx=ctx_strategy,
    scale=st.floats(min_value=1.0, max_value=10.0),
)
def test_impact_cost_monotone_in_size(ctx, scale):
    layer = ImpactLayer(SquareRootImpact(0.7))
    bigger = TradeContext(**{**ctx.__dict__, "qty": ctx.qty * scale})
    assert layer.cost(bigger) >= layer.cost(ctx)
