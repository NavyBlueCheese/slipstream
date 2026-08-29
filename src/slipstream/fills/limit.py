from __future__ import annotations

from slipstream.engine.orders import Order, OrderState, OrderType
from slipstream.engine.state import MarketState
from slipstream.fills.base import FillModel
from slipstream.fills.outcome import FillOutcome

PRICE_EPS = 1e-9


class LimitOrderFillModel(FillModel):
    def __init__(self, trade_through_ticks: float = 1.0, min_traded_volume: float = 1.0) -> None:
        self.trade_through_ticks = trade_through_ticks
        self.min_traded_volume = min_traded_volume

    def reset(self) -> None:
        self.resting: dict[int, dict] = {}

    def open_order_ids(self) -> list[int]:
        return list(self.resting)

    def cancel(self, order_id: int) -> None:
        self.resting.pop(order_id, None)

    def on_order_arrival(self, order: Order, market: MarketState) -> list[FillOutcome]:
        if order.order_type is OrderType.MARKET:
            return [self.aggressive_touch_fill(order, market)]
        if order.limit_price is None:
            raise ValueError("LimitOrderFillModel requires a limit price")
        side = order.side
        marketable = (
            side > 0 and order.limit_price >= market.ask - PRICE_EPS
        ) or (side < 0 and order.limit_price <= market.bid + PRICE_EPS)
        if marketable:
            return [self.aggressive_touch_fill(order, market)]
        self.resting[order.order_id] = {
            "order": order,
            "arrival_mid": market.mid(),
            "through_volume": 0.0,
        }
        return []

    def on_trade(
        self, price: float, size: float, aggressor: int, market: MarketState
    ) -> list[FillOutcome]:
        fills: list[FillOutcome] = []
        through_distance = self.trade_through_ticks * self.spec.tick_size
        for order_id in list(self.resting):
            entry = self.resting[order_id]
            order: Order = entry["order"]
            side = order.side
            through = (
                price <= order.limit_price - through_distance + PRICE_EPS
                if side > 0
                else price >= order.limit_price + through_distance - PRICE_EPS
            )
            if not through:
                continue
            entry["through_volume"] += size
            if entry["through_volume"] < self.min_traded_volume - PRICE_EPS:
                continue
            qty = order.remaining()
            scale = self.spec.multiplier * qty
            slippage = {
                "latency_drift": side * (entry["arrival_mid"] - order.signal_mid) * scale,
                "spread": side * (order.limit_price - entry["arrival_mid"]) * scale,
                "book_walk": 0.0,
                "impact": 0.0,
            }
            order.register_fill(qty)
            fills.append(
                FillOutcome(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=side,
                    requested_qty=order.qty,
                    filled_qty=qty,
                    filled=True,
                    passive=True,
                    signal_ts=order.signal_ts,
                    arrival_ts=order.arrival_ts,
                    fill_ts=market.ts,
                    intended_price=order.signal_mid,
                    fill_price=order.limit_price,
                    slippage=slippage,
                )
            )
            del self.resting[order_id]
            assert order.state == OrderState.FILLED
        return fills
