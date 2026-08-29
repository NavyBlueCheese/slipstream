from __future__ import annotations

from abc import ABC

import numpy as np
import pandas as pd

from slipstream.contracts.specs import ContractSpec
from slipstream.costs.base import CostModel, TradeContext
from slipstream.engine.orders import Order
from slipstream.engine.state import MarketState
from slipstream.fills.outcome import FillOutcome


class FillModel(ABC):
    def bind(
        self, spec: ContractSpec, cost_model: CostModel, rng: np.random.Generator
    ) -> None:
        self.spec = spec
        self.cost_model = cost_model
        self.rng = rng
        self.reset()

    def reset(self) -> None:
        pass

    def on_order_arrival(self, order: Order, market: MarketState) -> list[FillOutcome]:
        return []

    def on_trade(
        self,
        price: float,
        size: float,
        aggressor: int,
        market: MarketState,
    ) -> list[FillOutcome]:
        return []

    def on_quote(self, market: MarketState) -> list[FillOutcome]:
        return []

    def cancel(self, order_id: int) -> None:
        pass

    def open_order_ids(self) -> list[int]:
        return []

    def aggressive_touch_fill(self, order: Order, market: MarketState) -> FillOutcome:
        side = order.side
        qty = order.remaining()
        touch = market.far_touch(side)
        scale = self.spec.multiplier * qty
        arrival_mid = market.mid()
        order.register_fill(qty)
        return FillOutcome(
            order_id=order.order_id,
            symbol=order.symbol,
            side=side,
            requested_qty=order.qty,
            filled_qty=qty,
            filled=True,
            passive=False,
            signal_ts=order.signal_ts,
            arrival_ts=market.ts,
            fill_ts=market.ts,
            intended_price=order.signal_mid,
            fill_price=touch,
            slippage={
                "latency_drift": side * (arrival_mid - order.signal_mid) * scale,
                "spread": side * (touch - arrival_mid) * scale,
                "book_walk": 0.0,
                "impact": 0.0,
            },
        )

    def make_context(self, order: Order, market: MarketState) -> TradeContext:
        return TradeContext(
            spec=self.spec,
            side=order.side,
            qty=order.remaining(),
            signal_ts=order.signal_ts,
            arrival_ts=market.ts,
            signal_mid=order.signal_mid,
            arrival_mid=market.mid(),
            spread=market.spread(),
            top_depth=market.touch_depth(order.side),
            adv=market.adv,
            daily_vol=market.daily_vol,
            horizon_volume=market.recent_horizon_volume,
        )


def unfilled_outcome(order: Order, reason_ts: pd.Timestamp | None) -> FillOutcome:
    return FillOutcome(
        order_id=order.order_id,
        symbol=order.symbol,
        side=order.side,
        requested_qty=order.qty,
        filled_qty=order.filled_qty,
        filled=False,
        passive=order.limit_price is not None,
        signal_ts=order.signal_ts,
        arrival_ts=order.arrival_ts,
        fill_ts=reason_ts,
        intended_price=order.signal_mid,
        fill_price=None,
    )
