from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderState(Enum):
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    order_id: int
    symbol: str
    side: int
    qty: float
    order_type: OrderType
    signal_ts: pd.Timestamp
    signal_mid: float
    limit_price: float | None = None
    state: OrderState = OrderState.NEW
    arrival_ts: pd.Timestamp | None = None
    ack_ts: pd.Timestamp | None = None
    filled_qty: float = 0.0
    tags: dict = field(default_factory=dict)

    def remaining(self) -> float:
        return self.qty - self.filled_qty

    def is_open(self) -> bool:
        return self.state in (
            OrderState.NEW,
            OrderState.ACKNOWLEDGED,
            OrderState.PARTIALLY_FILLED,
        )

    def register_fill(self, qty: float) -> None:
        self.filled_qty += qty
        if self.filled_qty >= self.qty - 1e-9:
            self.state = OrderState.FILLED
        else:
            self.state = OrderState.PARTIALLY_FILLED

    def unregister_fill(self, qty: float) -> None:
        self.filled_qty = max(self.filled_qty - qty, 0.0)
        if self.filled_qty > 0.0:
            self.state = OrderState.PARTIALLY_FILLED
        elif self.ack_ts is not None:
            self.state = OrderState.ACKNOWLEDGED
        else:
            self.state = OrderState.NEW
