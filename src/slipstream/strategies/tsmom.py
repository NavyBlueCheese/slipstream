from __future__ import annotations

import numpy as np

from slipstream.engine.backtest import Broker
from slipstream.engine.view import MarketView
from slipstream.strategies.base import Strategy


class TimeSeriesMomentum(Strategy):
    def __init__(self, lookback_bars: int = 20, target_contracts: float = 1.0) -> None:
        self.lookback_bars = lookback_bars
        self.target_contracts = target_contracts

    def on_bar(self, view: MarketView, broker: Broker) -> None:
        closes = view.closes(self.lookback_bars + 1)
        if len(closes) < self.lookback_bars + 1:
            return
        signal = np.sign(closes[-1] - closes[0])
        target = signal * self.target_contracts
        delta = target - broker.pending_position()
        if delta != 0.0:
            broker.submit_market(delta)
