from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MarketState:
    ts: pd.Timestamp
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last_trade_price: float | None = None
    last_trade_size: float | None = None
    adv: float = 1_200_000.0
    daily_vol: float = 0.011
    recent_horizon_volume: float = 500.0

    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    def spread(self) -> float:
        return self.ask - self.bid

    def far_touch(self, side: int) -> float:
        return self.ask if side > 0 else self.bid

    def near_touch(self, side: int) -> float:
        return self.bid if side > 0 else self.ask

    def touch_depth(self, side: int) -> float:
        return self.ask_size if side > 0 else self.bid_size

    def same_side_depth(self, side: int) -> float:
        return self.bid_size if side > 0 else self.ask_size
