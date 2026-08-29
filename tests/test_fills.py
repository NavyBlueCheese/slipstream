import numpy as np
import pandas as pd
import pytest

from slipstream.contracts import get_spec
from slipstream.costs import RealisticCostModel, ZeroCostModel
from slipstream.engine import MarketState, Order, OrderState, OrderType
from slipstream.fills import (
    LimitOrderFillModel,
    MarketOrderFillModel,
    QueuePositionFillModel,
    adverse_selection_report,
    attach_markouts,
)
from slipstream.data import QuoteSeries

ES = get_spec("ES")
T0 = pd.Timestamp("2025-06-02 14:00", tz="UTC")


def market_state(bid=4999.75, ask=5000.0, bid_size=40.0, ask_size=40.0, ts=T0):
    return MarketState(ts=ts, bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size)


def make_order(side=1, qty=2.0, order_type=OrderType.MARKET, limit=None, signal_mid=4999.875):
    order = Order(
        order_id=1,
        symbol="ES",
        side=side,
        qty=qty,
        order_type=order_type,
        signal_ts=T0 - pd.Timedelta(milliseconds=250),
        signal_mid=signal_mid,
        limit_price=limit,
    )
    order.arrival_ts = T0
    return order


def bind(model, cost_model=None, seed=0):
    model.bind(ES, cost_model or ZeroCostModel(), np.random.default_rng(seed))
    return model


def test_market_order_fills_at_far_touch():
    model = bind(MarketOrderFillModel())
    outcome = model.on_order_arrival(make_order(side=1, qty=2), market_state())[0]
    assert outcome.filled
    assert not outcome.passive
    assert outcome.fill_price == pytest.approx(5000.0)
    assert outcome.slippage["spread"] == pytest.approx(1 * 0.125 * 50 * 2)
    assert outcome.slippage["book_walk"] == 0.0


def test_market_sell_fills_at_bid():
    model = bind(MarketOrderFillModel())
    outcome = model.on_order_arrival(make_order(side=-1, qty=1), market_state())[0]
    assert outcome.fill_price == pytest.approx(4999.75)


def test_market_order_walks_book_when_size_exceeds_depth():
    model = bind(MarketOrderFillModel(depth_decay=1.0))
    outcome = model.on_order_arrival(make_order(side=1, qty=100), market_state())[0]
    expected_vwap = (40 * 5000.0 + 40 * 5000.25 + 20 * 5000.50) / 100
    assert outcome.fill_price == pytest.approx(expected_vwap)
    assert outcome.slippage["book_walk"] > 0.0


def test_market_order_slippage_components_sum_to_total():
    model = bind(MarketOrderFillModel(), cost_model=RealisticCostModel("ES"))
    order = make_order(side=1, qty=50, signal_mid=4999.5)
    outcome = model.on_order_arrival(order, market_state())[0]
    total = outcome.side * (outcome.fill_price - outcome.intended_price) * 50 * ES.multiplier
    assert outcome.slippage_total() == pytest.approx(total)
    assert outcome.slippage["impact"] > 0.0
    assert outcome.slippage["latency_drift"] > 0.0


def test_limit_order_does_not_fill_on_touch():
    model = bind(LimitOrderFillModel(trade_through_ticks=1))
    order = make_order(side=1, order_type=OrderType.LIMIT, limit=4999.75)
    assert model.on_order_arrival(order, market_state()) == []
    fills = model.on_trade(4999.75, 50.0, -1, market_state())
    assert fills == []
    assert order.is_open()


def test_limit_order_fills_on_trade_through():
    model = bind(LimitOrderFillModel(trade_through_ticks=1))
    order = make_order(side=1, order_type=OrderType.LIMIT, limit=4999.75)
    model.on_order_arrival(order, market_state())
    fills = model.on_trade(4999.50, 5.0, -1, market_state(bid=4999.25, ask=4999.75))
    assert len(fills) == 1
    assert fills[0].fill_price == pytest.approx(4999.75)
    assert fills[0].passive
    assert order.state == OrderState.FILLED


def test_limit_order_requires_min_traded_volume():
    model = bind(LimitOrderFillModel(trade_through_ticks=1, min_traded_volume=10.0))
    order = make_order(side=1, order_type=OrderType.LIMIT, limit=4999.75)
    model.on_order_arrival(order, market_state())
    assert model.on_trade(4999.50, 4.0, -1, market_state()) == []
    fills = model.on_trade(4999.50, 7.0, -1, market_state())
    assert len(fills) == 1


def test_marketable_limit_fills_aggressively():
    model = bind(LimitOrderFillModel())
    order = make_order(side=1, order_type=OrderType.LIMIT, limit=5000.25)
    fills = model.on_order_arrival(order, market_state())
    assert len(fills) == 1
    assert fills[0].fill_price == pytest.approx(5000.0)
    assert not fills[0].passive


def test_limit_cancel_removes_resting_order():
    model = bind(LimitOrderFillModel())
    order = make_order(side=1, order_type=OrderType.LIMIT, limit=4999.75)
    model.on_order_arrival(order, market_state())
    model.cancel(order.order_id)
    assert model.open_order_ids() == []
    assert model.on_trade(4999.0, 50.0, -1, market_state()) == []


def test_queue_position_tracks_and_fills_after_queue_clears():
    model = bind(QueuePositionFillModel(cancel_to_trade_ratio=0.0))
    order = make_order(side=1, order_type=OrderType.LIMIT, limit=4999.75, qty=5)
    model.on_order_arrival(order, market_state(bid_size=30))
    assert model.queue_ahead(order.order_id) == 30.0
    assert model.on_trade(4999.75, 10.0, -1, market_state()) == []
    assert model.queue_ahead(order.order_id) == 20.0
    assert model.on_trade(4999.75, 20.0, -1, market_state()) == []
    assert model.queue_ahead(order.order_id) == 0.0
    fills = model.on_trade(4999.75, 3.0, -1, market_state())
    assert len(fills) == 1
    assert fills[0].filled_qty == 3.0
    assert order.state == OrderState.PARTIALLY_FILLED
    fills = model.on_trade(4999.75, 10.0, -1, market_state())
    assert fills[0].filled_qty == 2.0
    assert order.state == OrderState.FILLED


def test_queue_cancellations_ahead_accelerate_progress():
    slow = bind(QueuePositionFillModel(cancel_to_trade_ratio=0.0))
    fast = bind(QueuePositionFillModel(cancel_to_trade_ratio=2.0))
    for model in (slow, fast):
        order = make_order(side=1, order_type=OrderType.LIMIT, limit=4999.75, qty=1)
        model.on_order_arrival(order, market_state(bid_size=30))
        model.on_trade(4999.75, 5.0, -1, market_state())
    assert fast.queue_ahead(1) < slow.queue_ahead(1)


def test_queue_trade_through_fills_remaining():
    model = bind(QueuePositionFillModel())
    order = make_order(side=1, order_type=OrderType.LIMIT, limit=4999.75, qty=4)
    model.on_order_arrival(order, market_state(bid_size=50))
    fills = model.on_trade(4999.50, 2.0, -1, market_state())
    assert len(fills) == 1
    assert fills[0].filled_qty == 4.0
    assert fills[0].fill_price == pytest.approx(4999.75)


def test_queue_shrinks_with_displayed_size():
    model = bind(QueuePositionFillModel())
    order = make_order(side=1, order_type=OrderType.LIMIT, limit=4999.75, qty=1)
    model.on_order_arrival(order, market_state(bid_size=40))
    model.on_quote(market_state(bid_size=12))
    assert model.queue_ahead(order.order_id) == 12.0


def test_queue_ignores_same_side_aggressor_trades():
    model = bind(QueuePositionFillModel())
    order = make_order(side=1, order_type=OrderType.LIMIT, limit=4999.75, qty=1)
    model.on_order_arrival(order, market_state(bid_size=10))
    model.on_trade(4999.75, 50.0, 1, market_state())
    assert model.queue_ahead(order.order_id) == 10.0


def test_markouts_separate_passive_and_aggressive():
    ts = pd.date_range(T0, periods=120, freq="1s", tz="UTC")
    quotes = QuoteSeries.from_frame(
        pd.DataFrame(
            {
                "ts": ts,
                "bid": np.linspace(4999.75, 4990.75, 120),
                "ask": np.linspace(5000.0, 4991.0, 120),
                "bid_size": 10.0,
                "ask_size": 10.0,
            }
        )
    )
    passive_model = bind(QueuePositionFillModel(cancel_to_trade_ratio=5.0))
    order = make_order(side=1, order_type=OrderType.LIMIT, limit=4999.75, qty=1)
    passive_model.on_order_arrival(order, market_state(bid_size=1))
    passive_fills = passive_model.on_trade(4999.75, 5.0, -1, market_state())
    aggressive_model = bind(MarketOrderFillModel())
    aggressive_fills = aggressive_model.on_order_arrival(
        make_order(side=-1, qty=1), market_state()
    )
    outcomes = attach_markouts(passive_fills + aggressive_fills, quotes)
    report = adverse_selection_report(outcomes)
    assert set(report.index) == {"passive", "aggressive"}
    assert report.loc["passive", "10s"] < 0
    assert report.loc["aggressive", "10s"] > 0
    assert {"1s", "10s", "60s"} <= set(report.columns)
