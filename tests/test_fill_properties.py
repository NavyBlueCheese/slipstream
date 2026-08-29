import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from slipstream.contracts import get_spec
from slipstream.costs import RealisticCostModel, ZeroCostModel
from slipstream.engine import MarketState, Order, OrderType
from slipstream.fills import (
    LimitOrderFillModel,
    MarketOrderFillModel,
    QueuePositionFillModel,
)

ES = get_spec("ES")
T0 = pd.Timestamp("2025-06-02 14:00", tz="UTC")


def tickify(price):
    return round(price / ES.tick_size) * ES.tick_size


def make_state(bid, bid_size=20.0, ask_size=20.0):
    return MarketState(
        ts=T0, bid=bid, ask=bid + ES.tick_size, bid_size=bid_size, ask_size=ask_size
    )


def make_order(order_id, side, qty, limit=None):
    order = Order(
        order_id=order_id,
        symbol="ES",
        side=side,
        qty=qty,
        order_type=OrderType.MARKET if limit is None else OrderType.LIMIT,
        signal_ts=T0,
        signal_mid=5000.0,
        limit_price=limit,
    )
    order.arrival_ts = T0
    return order


trade_stream = st.lists(
    st.tuples(
        st.integers(min_value=-8, max_value=8),
        st.floats(min_value=1.0, max_value=50.0),
        st.sampled_from((-1, 1)),
    ),
    min_size=1,
    max_size=40,
)


@settings(max_examples=100, deadline=None)
@given(
    side=st.sampled_from((-1, 1)),
    qty=st.floats(min_value=1.0, max_value=500.0),
    depth=st.floats(min_value=1.0, max_value=100.0),
    use_impact=st.booleans(),
)
def test_market_fill_never_better_than_far_touch(side, qty, depth, use_impact):
    model = MarketOrderFillModel()
    cost_model = RealisticCostModel("ES") if use_impact else ZeroCostModel()
    model.bind(ES, cost_model, np.random.default_rng(0))
    state = make_state(5000.0, bid_size=depth, ask_size=depth)
    outcome = model.on_order_arrival(make_order(1, side, qty), state)[0]
    assert side * (outcome.fill_price - state.far_touch(side)) >= -1e-9


@settings(max_examples=100, deadline=None)
@given(
    side=st.sampled_from((-1, 1)),
    limit_ticks=st.integers(min_value=1, max_value=6),
    trades=trade_stream,
)
def test_limit_fills_only_at_limit_and_only_through(side, limit_ticks, trades):
    model = LimitOrderFillModel(trade_through_ticks=1)
    model.bind(ES, ZeroCostModel(), np.random.default_rng(0))
    state = make_state(5000.0)
    limit = tickify(state.near_touch(side) - side * (limit_ticks - 1) * ES.tick_size)
    order = make_order(1, side, 3, limit=limit)
    model.on_order_arrival(order, state)
    for tick_offset, size, aggressor in trades:
        price = tickify(5000.0 + tick_offset * ES.tick_size)
        fills = model.on_trade(price, size, aggressor, state)
        for fill in fills:
            assert fill.fill_price == limit
            assert side * (price - limit) <= -ES.tick_size + 1e-9


@settings(max_examples=100, deadline=None)
@given(trades=trade_stream, quotes=st.lists(st.floats(min_value=1.0, max_value=60.0), max_size=10))
def test_queue_position_monotone_without_new_orders(trades, quotes):
    model = QueuePositionFillModel(cancel_to_trade_ratio=1.5)
    model.bind(ES, ZeroCostModel(), np.random.default_rng(0))
    state = make_state(5000.0, bid_size=45.0)
    order = make_order(1, 1, 5, limit=5000.0)
    model.on_order_arrival(order, state)
    last = model.queue_ahead(1)
    for tick_offset, size, aggressor in trades:
        price = tickify(5000.0 + tick_offset * ES.tick_size)
        model.on_trade(price, size, aggressor, state)
        if 1 not in model.open_order_ids():
            return
        current = model.queue_ahead(1)
        assert current <= last + 1e-9
        last = current
    for displayed in quotes:
        model.on_quote(make_state(5000.0, bid_size=displayed))
        current = model.queue_ahead(1)
        assert current <= last + 1e-9
        last = current


@settings(max_examples=100, deadline=None)
@given(
    side=st.sampled_from((-1, 1)),
    qty=st.floats(min_value=1.0, max_value=300.0),
    signal_mid=st.floats(min_value=4990.0, max_value=5010.0),
)
def test_market_slippage_decomposition_is_exact(side, qty, signal_mid):
    model = MarketOrderFillModel()
    model.bind(ES, RealisticCostModel("ES"), np.random.default_rng(0))
    state = make_state(5000.0)
    order = make_order(1, side, qty)
    order.signal_mid = signal_mid
    outcome = model.on_order_arrival(order, state)[0]
    total = side * (outcome.fill_price - signal_mid) * qty * ES.multiplier
    assert abs(outcome.slippage_total() - total) < 1e-6 * max(1.0, abs(total))
