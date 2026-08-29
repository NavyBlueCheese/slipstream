from __future__ import annotations

from slipstream.engine.orders import Order, OrderType
from slipstream.engine.state import MarketState
from slipstream.fills.base import FillModel
from slipstream.fills.outcome import FillOutcome

PRICE_EPS = 1e-9


class QueuePositionFillModel(FillModel):
    def __init__(
        self,
        cancel_to_trade_ratio: float = 2.0,
        behind_touch_depth_factor: float = 1.0,
    ) -> None:
        self.cancel_to_trade_ratio = cancel_to_trade_ratio
        self.behind_touch_depth_factor = behind_touch_depth_factor

    def reset(self) -> None:
        self.resting: dict[int, dict] = {}

    def open_order_ids(self) -> list[int]:
        return list(self.resting)

    def cancel(self, order_id: int) -> None:
        self.resting.pop(order_id, None)

    def queue_ahead(self, order_id: int) -> float:
        return self.resting[order_id]["queue_ahead"]

    def on_order_arrival(self, order: Order, market: MarketState) -> list[FillOutcome]:
        if order.order_type is OrderType.MARKET:
            return [self.aggressive_touch_fill(order, market)]
        if order.limit_price is None:
            raise ValueError("QueuePositionFillModel requires a limit price")
        side = order.side
        limit = order.limit_price
        marketable = (side > 0 and limit >= market.ask - PRICE_EPS) or (
            side < 0 and limit <= market.bid + PRICE_EPS
        )
        if marketable:
            return [self.aggressive_touch_fill(order, market)]
        same_touch = market.near_touch(side)
        at_touch = abs(limit - same_touch) < PRICE_EPS
        inside_spread = limit > same_touch if side > 0 else limit < same_touch
        if inside_spread:
            queue_ahead = 0.0
        elif at_touch:
            queue_ahead = market.same_side_depth(side)
        else:
            queue_ahead = market.same_side_depth(side) * self.behind_touch_depth_factor
        self.resting[order.order_id] = {
            "order": order,
            "arrival_mid": market.mid(),
            "queue_ahead": queue_ahead,
        }
        return []

    def on_quote(self, market: MarketState) -> list[FillOutcome]:
        for entry in self.resting.values():
            order: Order = entry["order"]
            side = order.side
            same_touch = market.near_touch(side)
            if abs(order.limit_price - same_touch) < PRICE_EPS:
                displayed = market.same_side_depth(side)
                entry["queue_ahead"] = min(entry["queue_ahead"], displayed)
        return []

    def on_trade(
        self, price: float, size: float, aggressor: int, market: MarketState
    ) -> list[FillOutcome]:
        fills: list[FillOutcome] = []
        for order_id in list(self.resting):
            entry = self.resting[order_id]
            order: Order = entry["order"]
            side = order.side
            if aggressor == side:
                continue
            at_level = abs(price - order.limit_price) < PRICE_EPS
            through_level = (
                price < order.limit_price - PRICE_EPS
                if side > 0
                else price > order.limit_price + PRICE_EPS
            )
            if through_level:
                fills.append(self._passive_fill(entry, order.remaining(), market))
                del self.resting[order_id]
                continue
            if not at_level:
                continue
            consumed_from_queue = min(entry["queue_ahead"], size)
            entry["queue_ahead"] -= consumed_from_queue
            available = size - consumed_from_queue
            if available > PRICE_EPS:
                fill_qty = min(available, order.remaining())
                fills.append(self._passive_fill(entry, fill_qty, market))
                if not order.is_open():
                    del self.resting[order_id]
            entry_still_open = order_id in self.resting
            if entry_still_open:
                cancelled_ahead = self.cancel_to_trade_ratio * size
                entry["queue_ahead"] = max(entry["queue_ahead"] - cancelled_ahead, 0.0)
        return fills

    def _passive_fill(self, entry: dict, qty: float, market: MarketState) -> FillOutcome:
        order: Order = entry["order"]
        side = order.side
        scale = self.spec.multiplier * qty
        slippage = {
            "latency_drift": side * (entry["arrival_mid"] - order.signal_mid) * scale,
            "spread": side * (order.limit_price - entry["arrival_mid"]) * scale,
            "book_walk": 0.0,
            "impact": 0.0,
        }
        order.register_fill(qty)
        return FillOutcome(
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
