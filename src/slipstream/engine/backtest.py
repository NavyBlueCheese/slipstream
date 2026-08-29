from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from slipstream.contracts.continuous import RollEvent
from slipstream.contracts.specs import ContractSpec
from slipstream.costs.base import CostModel
from slipstream.data.schema import BarSeries, QuoteSeries, TradeSeries
from slipstream.engine.accounting import Account
from slipstream.engine.margin import (
    MarginCallHandler,
    MarginTracker,
    liquidating_margin_call_handler,
)
from slipstream.engine.orders import Order, OrderState, OrderType
from slipstream.engine.state import MarketState
from slipstream.engine.view import MarketView

if TYPE_CHECKING:
    from slipstream.fills.base import FillModel
    from slipstream.fills.outcome import FillOutcome

QUOTE, TRADE, BAR, ROLL = 0, 1, 2, 3
NS_PER_SECOND = 1_000_000_000
NS_PER_DAY = 86_400 * NS_PER_SECOND


@dataclass(frozen=True)
class MarketDataBundle:
    quotes: QuoteSeries | None = None
    trades: TradeSeries | None = None
    bars: BarSeries | None = None
    rolls: tuple[RollEvent, ...] = ()
    adv: float = 1_200_000.0
    daily_vol: float = 0.011

    @staticmethod
    def from_ticks(
        quotes: QuoteSeries,
        trades: TradeSeries,
        bars: BarSeries | None = None,
        adv: float = 1_200_000.0,
        daily_vol: float = 0.011,
    ) -> "MarketDataBundle":
        return MarketDataBundle(
            quotes=quotes, trades=trades, bars=bars, adv=adv, daily_vol=daily_vol
        )

    @staticmethod
    def from_bars(
        bars: BarSeries,
        rolls: tuple[RollEvent, ...] = (),
        adv: float = 1_200_000.0,
        daily_vol: float = 0.011,
    ) -> "MarketDataBundle":
        return MarketDataBundle(bars=bars, rolls=rolls, adv=adv, daily_vol=daily_vol)


@dataclass
class BacktestResult:
    equity: pd.Series
    outcomes: "list[FillOutcome]"
    orders: list[Order]
    account: Account
    spec: ContractSpec
    seed: int
    margin_calls: list[pd.Timestamp]

    def initial_cash(self) -> float:
        return self.account.initial_cash

    def net_pnl(self) -> float:
        if len(self.equity) == 0:
            return 0.0
        return float(self.equity.iloc[-1] - self.account.initial_cash)

    def daily_returns(self) -> pd.Series:
        if len(self.equity) == 0:
            return pd.Series(dtype=float)
        daily = self.equity.resample("1D").last().dropna()
        return daily.diff().dropna() / self.account.initial_cash

    def sharpe(self) -> float:
        returns = self.daily_returns()
        if len(returns) < 2 or returns.std() == 0.0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(252.0))

    def max_drawdown(self) -> float:
        if len(self.equity) == 0:
            return 0.0
        running_peak = self.equity.cummax()
        return float((self.equity - running_peak).min())

    def span_days(self) -> float:
        if len(self.equity) < 2:
            return 1.0
        span = (self.equity.index[-1] - self.equity.index[0]).total_seconds() / 86_400
        return max(span, 1.0)

    def turnover_per_day(self) -> float:
        return self.account.contracts_traded / self.span_days()

    def fill_count(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.filled)

    def standard_report(
        self,
        output_dir: object = None,
        quotes: object = None,
        name: str = "backtest",
    ) -> dict:
        from slipstream.diagnostics.report import standard_report

        return standard_report(self, output_dir=output_dir, quotes=quotes, name=name)

    def summary(self) -> dict[str, float]:
        return {
            "net_pnl": self.net_pnl(),
            "sharpe": self.sharpe(),
            "max_drawdown": self.max_drawdown(),
            "turnover_per_day": self.turnover_per_day(),
            "fills": float(self.fill_count()),
            "contracts_traded": self.account.contracts_traded,
            "commissions": self.account.commissions_paid,
            "roll_costs": self.account.roll_costs_paid,
            "financing": self.account.financing_paid,
            "margin_calls": float(len(self.margin_calls)),
        }


class Broker:
    def __init__(self, engine: "_Engine") -> None:
        self._engine = engine

    def submit_market(self, qty: float) -> int:
        return self._engine.submit_order(qty, OrderType.MARKET, None)

    def submit_limit(self, qty: float, price: float) -> int:
        return self._engine.submit_order(qty, OrderType.LIMIT, price)

    def cancel(self, order_id: int) -> None:
        self._engine.request_cancel(order_id)

    def cancel_all(self) -> None:
        for order_id in self.open_order_ids():
            self.cancel(order_id)

    def flatten(self) -> None:
        position = self.position()
        if position != 0.0:
            self.submit_market(-position)

    def position(self) -> float:
        return self._engine.account.position

    def pending_position(self) -> float:
        return self._engine.pending_position()

    def equity(self) -> float:
        return self._engine.current_equity()

    def open_order_ids(self) -> list[int]:
        return [
            order.order_id
            for order in self._engine.orders.values()
            if order.is_open()
        ]

    def order(self, order_id: int) -> Order:
        return self._engine.orders[order_id]


@dataclass
class _EngineConfig:
    initial_cash: float
    ack_latency_s: float
    entry_delay_s: float
    mark_every_quote: bool


class _Engine:
    def __init__(
        self,
        strategy: object,
        bundle: MarketDataBundle,
        cost_model: CostModel,
        fill_model: "FillModel",
        spec: ContractSpec,
        seed: int,
        config: _EngineConfig,
        margin_call_handler: MarginCallHandler,
    ) -> None:
        self.strategy = strategy
        self.bundle = bundle
        self.cost_model = cost_model
        self.fill_model = fill_model
        self.spec = spec
        self.seed = seed
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.account = Account(spec=spec, initial_cash=config.initial_cash)
        self.margin = MarginTracker(spec, margin_call_handler)
        self.view = MarketView(spec)
        self.broker = Broker(self)
        self.market: MarketState | None = None
        self.orders: dict[int, Order] = {}
        self.outcomes: "list[FillOutcome]" = []
        self.pending: list[tuple[int, int, str, int]] = []
        self.next_order_id = 1
        self.next_pending_seq = 0
        self.equity_ts: list[pd.Timestamp] = []
        self.equity_values: list[float] = []
        self.recent_trades: deque[tuple[int, float]] = deque()
        self.last_day: int | None = None
        self.in_margin_check = False
        fill_model.bind(spec, cost_model, self.rng)

    def submit_order(self, qty: float, order_type: OrderType, price: float | None) -> int:
        if self.market is None:
            raise RuntimeError("cannot submit orders before any market data")
        if qty == 0.0:
            raise ValueError("order quantity must be nonzero")
        side = 1 if qty > 0 else -1
        order = Order(
            order_id=self.next_order_id,
            symbol=self.spec.symbol,
            side=side,
            qty=abs(qty),
            order_type=order_type,
            signal_ts=self.market.ts,
            signal_mid=self.market.mid(),
            limit_price=price,
        )
        self.next_order_id += 1
        projected = self.pending_position() + side * order.qty
        required = self.margin.posted_initial(projected)
        self.orders[order.order_id] = order
        if required > self.current_equity() and abs(projected) > abs(self.account.position):
            order.state = OrderState.REJECTED
            return order.order_id
        latency_s = self.config.entry_delay_s + self.cost_model.sample_latency_seconds(
            self.rng
        )
        arrival_ns = self.market.ts.value + int(latency_s * NS_PER_SECOND)
        ack_ns = arrival_ns + int(self.config.ack_latency_s * NS_PER_SECOND)
        self._push_pending(arrival_ns, "arrival", order.order_id)
        self._push_pending(ack_ns, "ack", order.order_id)
        return order.order_id

    def request_cancel(self, order_id: int) -> None:
        if self.market is None:
            return
        latency_s = self.cost_model.sample_latency_seconds(self.rng)
        cancel_ns = self.market.ts.value + int(latency_s * NS_PER_SECOND)
        self._push_pending(cancel_ns, "cancel", order_id)

    def pending_position(self) -> float:
        open_qty = sum(
            order.side * order.remaining()
            for order in self.orders.values()
            if order.is_open()
        )
        return self.account.position + open_qty

    def current_equity(self) -> float:
        if self.market is None:
            return self.account.initial_cash
        return self.account.equity(self.market.mid())

    def _push_pending(self, ts_ns: int, action: str, order_id: int) -> None:
        heapq.heappush(self.pending, (ts_ns, self.next_pending_seq, action, order_id))
        self.next_pending_seq += 1

    def _commission_for(self, qty: float) -> float:
        per_contract = getattr(self.cost_model.commission, "per_contract", None)
        if per_contract is None:
            return 0.0
        return per_contract() * abs(qty)

    def _process_outcomes(self, outcomes: "list[FillOutcome]") -> None:
        for outcome in outcomes:
            if outcome.filled and outcome.fill_price is not None:
                self.account.apply_fill(
                    outcome.side,
                    outcome.filled_qty,
                    outcome.fill_price,
                    self._commission_for(outcome.filled_qty),
                )
            self.outcomes.append(outcome)
            handler = getattr(self.strategy, "on_fill", None)
            if handler is not None:
                handler(self.view, self.broker, outcome)

    def _drain_pending(self, up_to_ns: int) -> None:
        while self.pending and self.pending[0][0] <= up_to_ns:
            ts_ns, _, action, order_id = heapq.heappop(self.pending)
            order = self.orders[order_id]
            ts = pd.Timestamp(ts_ns, tz="UTC")
            if action == "arrival":
                if not order.is_open():
                    continue
                order.arrival_ts = ts
                arrival_market = self._market_at(ts)
                self._process_outcomes(
                    self.fill_model.on_order_arrival(order, arrival_market)
                )
            elif action == "ack":
                if order.state is OrderState.NEW:
                    order.state = OrderState.ACKNOWLEDGED
                order.ack_ts = ts
            elif action == "cancel":
                if order.is_open():
                    self.fill_model.cancel(order_id)
                    order.state = OrderState.CANCELLED

    def _market_at(self, ts: pd.Timestamp) -> MarketState:
        market = self.market
        assert market is not None
        return MarketState(
            ts=ts,
            bid=market.bid,
            ask=market.ask,
            bid_size=market.bid_size,
            ask_size=market.ask_size,
            last_trade_price=market.last_trade_price,
            last_trade_size=market.last_trade_size,
            adv=market.adv,
            daily_vol=market.daily_vol,
            recent_horizon_volume=market.recent_horizon_volume,
        )

    def _update_horizon_volume(self, ts_ns: int, size: float | None) -> float:
        if size is not None:
            self.recent_trades.append((ts_ns, size))
        cutoff = ts_ns - 60 * NS_PER_SECOND
        while self.recent_trades and self.recent_trades[0][0] < cutoff:
            self.recent_trades.popleft()
        total = sum(entry[1] for entry in self.recent_trades)
        return max(total, 1.0)

    def _accrue_day_boundary(self, ts_ns: int) -> None:
        day = ts_ns // NS_PER_DAY
        if self.last_day is None:
            self.last_day = day
            return
        if day == self.last_day:
            return
        elapsed_days = day - self.last_day
        self.last_day = day
        margin_posted = self.margin.posted_initial(self.account.position)
        if margin_posted > 0.0:
            self.account.pay_financing(
                self.cost_model.financing_cost(self.spec, margin_posted, elapsed_days)
            )

    def _mark(self, ts: pd.Timestamp, mark_price: float) -> None:
        self.equity_ts.append(ts)
        self.equity_values.append(self.account.equity(mark_price))
        if self.in_margin_check:
            return
        self.in_margin_check = True
        self.margin.check(ts, self.account, self.broker, mark_price)
        self.in_margin_check = False

    def _call_strategy(self, hook: str, *args: object) -> None:
        handler = getattr(self.strategy, hook, None)
        if handler is not None:
            handler(self.view, self.broker, *args)

    def run(self) -> BacktestResult:
        events = self._static_events()
        bar_mode_quotes = self.bundle.quotes is None
        columns = self._column_arrays()
        started = False
        for ts_ns, kind, index in events:
            self._drain_pending(ts_ns)
            self._accrue_day_boundary(ts_ns)
            ts = pd.Timestamp(ts_ns, tz="UTC")
            if kind == QUOTE:
                self._apply_quote(
                    ts,
                    columns["bid"][index],
                    columns["ask"][index],
                    columns["bid_size"][index],
                    columns["ask_size"][index],
                )
                if not started:
                    started = True
                    self._call_strategy("on_start")
                self._call_strategy("on_quote")
                if self.config.mark_every_quote:
                    self._mark(ts, self.market.mid())
            elif kind == TRADE:
                self._apply_trade(
                    ts,
                    columns["price"][index],
                    columns["size"][index],
                    int(columns["aggressor"][index]),
                )
                self._call_strategy("on_trade")
            elif kind == BAR:
                close = columns["close"][index]
                volume = columns["volume"][index]
                if bar_mode_quotes:
                    half = self.spec.tick_size / 2.0
                    depth = max(volume, 1.0)
                    self._apply_quote(ts, close - half, close + half, depth, depth)
                    direction = 1 if close >= columns["open"][index] else -1
                    self._apply_trade(ts, close, depth, direction)
                self.view._append_bar(
                    ts,
                    columns["open"][index],
                    columns["high"][index],
                    columns["low"][index],
                    close,
                    volume,
                )
                if not started:
                    started = True
                    self._call_strategy("on_start")
                self._call_strategy("on_bar")
                self._drain_pending(ts_ns)
                self._mark(ts, close)
            elif kind == ROLL:
                roll = self.bundle.rolls[index]
                if self.account.position != 0.0:
                    qty = abs(self.account.position)
                    cost = self.cost_model.roll_cost(self.spec, qty)
                    cost += 2.0 * self._commission_for(qty)
                    self.account.apply_roll(ts, roll.gap, cost)
        self._call_strategy("on_end")
        if self.market is not None:
            self._mark(self.market.ts, self.market.mid())
        equity = pd.Series(
            self.equity_values, index=pd.DatetimeIndex(self.equity_ts, name="ts")
        )
        equity = equity[~equity.index.duplicated(keep="last")]
        return BacktestResult(
            equity=equity,
            outcomes=self.outcomes,
            orders=list(self.orders.values()),
            account=self.account,
            spec=self.spec,
            seed=self.seed,
            margin_calls=self.margin.margin_calls,
        )

    def _apply_quote(
        self, ts: pd.Timestamp, bid: float, ask: float, bid_size: float, ask_size: float
    ) -> None:
        horizon_volume = self._update_horizon_volume(ts.value, None)
        if self.market is None:
            self.market = MarketState(
                ts=ts,
                bid=bid,
                ask=ask,
                bid_size=bid_size,
                ask_size=ask_size,
                adv=self.bundle.adv,
                daily_vol=self.bundle.daily_vol,
                recent_horizon_volume=horizon_volume,
            )
        else:
            self.market.ts = ts
            self.market.bid = bid
            self.market.ask = ask
            self.market.bid_size = bid_size
            self.market.ask_size = ask_size
            self.market.recent_horizon_volume = horizon_volume
        self.view._set_quote(ts, bid, ask, bid_size, ask_size)
        self._process_outcomes(self.fill_model.on_quote(self.market))

    def _apply_trade(self, ts: pd.Timestamp, price: float, size: float, aggressor: int) -> None:
        if self.market is None:
            return
        self.market.ts = ts
        self.market.last_trade_price = price
        self.market.last_trade_size = size
        self.market.recent_horizon_volume = self._update_horizon_volume(ts.value, size)
        self.view._set_trade(ts, price, size)
        self._process_outcomes(
            self.fill_model.on_trade(price, size, aggressor, self.market)
        )

    def _static_events(self) -> list[tuple[int, int, int]]:
        events: list[tuple[int, int, int]] = []
        streams = (
            (self.bundle.quotes, QUOTE),
            (self.bundle.trades, TRADE),
            (self.bundle.bars, BAR),
        )
        for series, kind in streams:
            if series is None:
                continue
            for i, value in enumerate(series.frame["ts"].astype("int64").to_numpy()):
                events.append((int(value), kind, i))
        for i, roll in enumerate(self.bundle.rolls):
            events.append((int(roll.ts.value), ROLL, i))
        events.sort(key=lambda event: (event[0], event[1]))
        return events

    def _column_arrays(self) -> dict[str, np.ndarray]:
        columns: dict[str, np.ndarray] = {}
        if self.bundle.quotes is not None:
            for name in ("bid", "ask", "bid_size", "ask_size"):
                columns[name] = self.bundle.quotes.frame[name].to_numpy()
        if self.bundle.trades is not None:
            for name in ("price", "size", "aggressor"):
                columns[name] = self.bundle.trades.frame[name].to_numpy()
        if self.bundle.bars is not None:
            for name in ("open", "high", "low", "close", "volume"):
                columns[name] = self.bundle.bars.frame[name].to_numpy()
        return columns


def run_backtest(
    strategy: object,
    data: MarketDataBundle,
    cost_model: CostModel,
    fill_model: "FillModel",
    *,
    spec: ContractSpec,
    seed: int,
    initial_cash: float = 1_000_000.0,
    ack_latency_s: float = 0.05,
    entry_delay_s: float = 0.0,
    mark_every_quote: bool = True,
    margin_call_handler: MarginCallHandler | None = None,
) -> BacktestResult:
    if not isinstance(cost_model, CostModel):
        raise TypeError("cost_model must be a CostModel, use ZeroCostModel() explicitly for frictionless runs")
    config = _EngineConfig(
        initial_cash=initial_cash,
        ack_latency_s=ack_latency_s,
        entry_delay_s=entry_delay_s,
        mark_every_quote=mark_every_quote,
    )
    engine = _Engine(
        strategy=strategy,
        bundle=data,
        cost_model=cost_model,
        fill_model=fill_model,
        spec=spec,
        seed=seed,
        config=config,
        margin_call_handler=margin_call_handler or liquidating_margin_call_handler,
    )
    return engine.run()
