from __future__ import annotations

import numpy as np

from slipstream.engine.orders import Order
from slipstream.engine.state import MarketState
from slipstream.fills.base import FillModel
from slipstream.fills.outcome import FillOutcome


class MarketOrderFillModel(FillModel):
    def __init__(self, book_levels: int = 10, depth_decay: float = 0.85) -> None:
        self.book_levels = book_levels
        self.depth_decay = depth_decay

    def walk_book(
        self, side: int, qty: float, touch: float, top_depth: float
    ) -> float:
        tick = self.spec.tick_size
        remaining = qty
        notional = 0.0
        level_price = touch
        for level in range(self.book_levels):
            level_size = max(np.round(top_depth * self.depth_decay**level), 1.0)
            take = min(remaining, level_size)
            notional += take * level_price
            remaining -= take
            if remaining <= 1e-12:
                break
            level_price += side * tick
        if remaining > 1e-12:
            notional += remaining * level_price
        return notional / qty

    def on_order_arrival(self, order: Order, market: MarketState) -> list[FillOutcome]:
        side = order.side
        qty = order.remaining()
        touch = market.far_touch(side)
        vwap = self.walk_book(side, qty, touch, market.touch_depth(side))
        ctx = self.make_context(order, market)
        impact = self.cost_model.price_impact(ctx)
        fill_price = vwap + side * impact
        scale = self.spec.multiplier * qty
        arrival_mid = market.mid()
        slippage = {
            "latency_drift": side * (arrival_mid - order.signal_mid) * scale,
            "spread": side * (touch - arrival_mid) * scale,
            "book_walk": side * (vwap - touch) * scale,
            "impact": impact * scale,
        }
        order.register_fill(qty)
        outcome = FillOutcome(
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
            fill_price=fill_price,
            slippage=slippage,
        )
        return [outcome]
