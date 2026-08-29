from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from slipstream.contracts import RollEvent, get_spec
from slipstream.costs import (
    FinancingLayer,
    RealisticCostModel,
    RollCostLayer,
    ZeroCostModel,
)
from slipstream.data import (
    BarSeries,
    SyntheticMarketConfig,
    SyntheticMarketGenerator,
    bars_from_trades,
)
from slipstream.engine import (
    LookaheadError,
    MarketDataBundle,
    OrderState,
    run_backtest,
)
from slipstream.fills import LimitOrderFillModel, MarketOrderFillModel
from slipstream.strategies import Strategy, TimeSeriesMomentum

ES = get_spec("ES")


def daily_bars(closes, start="2025-06-02"):
    closes = np.asarray(closes, dtype=float)
    ts = pd.bdate_range(start, periods=len(closes), tz="UTC") + pd.Timedelta(hours=20)
    opens = np.concatenate(([closes[0]], closes[:-1]))
    frame = pd.DataFrame(
        {
            "ts": ts,
            "open": opens,
            "high": np.maximum(opens, closes) + 0.25,
            "low": np.minimum(opens, closes) - 0.25,
            "close": closes,
            "volume": np.full(len(closes), 1000.0),
        }
    )
    return BarSeries.from_frame(frame)


class BuyOnce(Strategy):
    def __init__(self):
        self.done = False

    def on_bar(self, view, broker):
        if not self.done:
            broker.submit_market(1)
            self.done = True


class Peeker(Strategy):
    def __init__(self):
        self.now_peek_raised = False
        self.future_peek_raised = False
        self.past_read_ok = False

    def on_bar(self, view, broker):
        try:
            view.mid_at(view.now)
        except LookaheadError:
            self.now_peek_raised = True
        try:
            view.mid_at(view.now + pd.Timedelta(minutes=1))
        except LookaheadError:
            self.future_peek_raised = True
        if view.bar_count() > 1:
            past = view.mid_at(view.now - pd.Timedelta(days=1))
            self.past_read_ok = np.isfinite(past)


def test_lookahead_peek_raises():
    strategy = Peeker()
    result = run_backtest(
        strategy,
        MarketDataBundle.from_bars(daily_bars([100, 101, 102, 103])),
        ZeroCostModel(),
        MarketOrderFillModel(),
        spec=ES,
        seed=1,
    )
    assert strategy.now_peek_raised
    assert strategy.future_peek_raised
    assert strategy.past_read_ok
    assert len(result.equity) > 0


def test_constant_drift_pnl_matches_hand_calculation():
    closes = [100 + i for i in range(10)]
    result = run_backtest(
        BuyOnce(),
        MarketDataBundle.from_bars(daily_bars(closes)),
        ZeroCostModel(),
        MarketOrderFillModel(),
        spec=ES,
        seed=1,
    )
    fill_price = 100.0 + ES.tick_size / 2.0
    expected = (109.0 - fill_price) * ES.multiplier
    assert result.net_pnl() == pytest.approx(expected)
    assert result.account.position == 1.0


def test_commission_reduces_net_pnl_exactly():
    closes = [100 + i for i in range(10)]
    zero = run_backtest(
        BuyOnce(),
        MarketDataBundle.from_bars(daily_bars(closes)),
        ZeroCostModel(),
        MarketOrderFillModel(),
        spec=ES,
        seed=1,
    )
    realistic_commission = ZeroCostModel().replaced(
        commission=RealisticCostModel("ES").commission
    )
    with_commission = run_backtest(
        BuyOnce(),
        MarketDataBundle.from_bars(daily_bars(closes)),
        realistic_commission,
        MarketOrderFillModel(),
        spec=ES,
        seed=1,
    )
    assert zero.net_pnl() - with_commission.net_pnl() == pytest.approx(2.37)


def test_same_seed_is_byte_identical():
    config = SyntheticMarketConfig(session_start=time(13, 30), session_end=time(15, 0))
    quotes, trades = SyntheticMarketGenerator(config, seed=3).generate_day(date(2025, 6, 2))
    bundle = MarketDataBundle.from_ticks(quotes, trades, bars=bars_from_trades(trades, "1min"))

    def run_once():
        return run_backtest(
            TimeSeriesMomentum(lookback_bars=5),
            bundle,
            RealisticCostModel("ES"),
            MarketOrderFillModel(),
            spec=ES,
            seed=99,
        )

    first = run_once()
    second = run_once()
    assert first.equity.equals(second.equity)
    assert len(first.outcomes) == len(second.outcomes)
    for a, b in zip(first.outcomes, second.outcomes):
        assert a.__dict__ == b.__dict__


def test_order_lifecycle_states():
    class LifecycleStrategy(Strategy):
        def __init__(self):
            self.limit_id = None
            self.market_id = None
            self.cancelled = False

        def on_bar(self, view, broker):
            if self.limit_id is None:
                self.limit_id = broker.submit_limit(1, view.bid() - 10 * 0.25)
                self.market_id = broker.submit_market(1)
                return
            if not self.cancelled:
                broker.cancel(self.limit_id)
                self.cancelled = True

    strategy = LifecycleStrategy()
    result = run_backtest(
        strategy,
        MarketDataBundle.from_bars(daily_bars([100] * 6)),
        ZeroCostModel(),
        LimitOrderFillModel(),
        spec=ES,
        seed=1,
        ack_latency_s=0.05,
    )
    orders = {order.order_id: order for order in result.orders}
    market_order = orders[strategy.market_id]
    limit_order = orders[strategy.limit_id]
    assert market_order.state is OrderState.FILLED
    assert limit_order.state is OrderState.CANCELLED
    assert market_order.ack_ts == market_order.signal_ts + pd.Timedelta(seconds=0.05)
    assert limit_order.arrival_ts is not None


def test_oversized_order_is_rejected_on_margin():
    class Greedy(Strategy):
        def __init__(self):
            self.order_id = None

        def on_bar(self, view, broker):
            if self.order_id is None:
                self.order_id = broker.submit_market(1000)

    strategy = Greedy()
    result = run_backtest(
        strategy,
        MarketDataBundle.from_bars(daily_bars([5000] * 4)),
        ZeroCostModel(),
        MarketOrderFillModel(),
        spec=ES,
        seed=1,
        initial_cash=100_000.0,
    )
    orders = {order.order_id: order for order in result.orders}
    assert orders[strategy.order_id].state is OrderState.REJECTED
    assert result.account.position == 0.0


def test_roll_does_not_create_spurious_pnl():
    closes = [100.0] * 5 + [105.0] * 5
    bars = daily_bars(closes)
    roll_ts = bars.frame["ts"].iloc[4]
    rolls = (RollEvent(ts=roll_ts, from_symbol="ESU5", to_symbol="ESZ5", gap=5.0, ratio=1.05),)
    result = run_backtest(
        BuyOnce(),
        MarketDataBundle.from_bars(bars, rolls=rolls),
        ZeroCostModel(),
        MarketOrderFillModel(),
        spec=ES,
        seed=1,
    )
    equity = result.equity
    assert equity.iloc[-1] == pytest.approx(equity.iloc[1])
    assert len(result.account.roll_records) == 1


def test_roll_cost_is_charged_when_holding_through_roll():
    closes = [100.0] * 5 + [105.0] * 5
    bars = daily_bars(closes)
    roll_ts = bars.frame["ts"].iloc[4]
    rolls = (RollEvent(ts=roll_ts, from_symbol="ESU5", to_symbol="ESZ5", gap=5.0, ratio=1.05),)
    cost_model = ZeroCostModel().replaced(roll=RollCostLayer(calendar_spread_ticks=1.0))
    result = run_backtest(
        BuyOnce(),
        MarketDataBundle.from_bars(bars, rolls=rolls),
        cost_model,
        MarketOrderFillModel(),
        spec=ES,
        seed=1,
    )
    assert result.account.roll_costs_paid == pytest.approx(12.50 * 0.5)
    assert result.equity.iloc[-1] == pytest.approx(result.equity.iloc[1] - 6.25)


def test_financing_accrues_daily_on_posted_margin():
    closes = [5000.0] * 6
    cost_model = ZeroCostModel().replaced(
        financing=FinancingLayer(margin_rate=0.36, collateral_opportunity_rate=0.0)
    )
    result = run_backtest(
        BuyOnce(),
        MarketDataBundle.from_bars(daily_bars(closes)),
        cost_model,
        MarketOrderFillModel(),
        spec=ES,
        seed=1,
    )
    expected_per_day = ES.initial_margin * 0.36 / 360.0
    assert result.account.financing_paid == pytest.approx(expected_per_day * 7)


def test_margin_call_liquidates_position():
    closes = [5000.0, 4950.0, 4800.0, 4800.0, 4800.0]
    result = run_backtest(
        BuyOnce(),
        MarketDataBundle.from_bars(daily_bars(closes)),
        ZeroCostModel(),
        MarketOrderFillModel(),
        spec=ES,
        seed=1,
        initial_cash=20_000.0,
    )
    assert len(result.margin_calls) >= 1
    assert result.account.position == 0.0


def test_cost_model_type_is_enforced():
    with pytest.raises(TypeError, match="ZeroCostModel"):
        run_backtest(
            BuyOnce(),
            MarketDataBundle.from_bars(daily_bars([100, 101])),
            None,
            MarketOrderFillModel(),
            spec=ES,
            seed=1,
        )


def test_summary_contains_standard_fields():
    result = run_backtest(
        TimeSeriesMomentum(lookback_bars=3),
        MarketDataBundle.from_bars(daily_bars([100 + i for i in range(15)])),
        RealisticCostModel("ES"),
        MarketOrderFillModel(),
        spec=ES,
        seed=7,
    )
    summary = result.summary()
    assert {"net_pnl", "sharpe", "max_drawdown", "turnover_per_day", "fills"} <= set(summary)
    assert summary["fills"] >= 1
