from __future__ import annotations

import numpy as np

from slipstream.engine.backtest import Broker
from slipstream.engine.view import MarketView
from slipstream.strategies.base import Strategy


class IntradayMeanReversion(Strategy):
    def __init__(
        self,
        window: int = 30,
        entry_z: float = 1.5,
        exit_z: float = 0.3,
        target_contracts: float = 1.0,
    ) -> None:
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.target_contracts = target_contracts
        self.last_bar_date = None

    def on_bar(self, view: MarketView, broker: Broker) -> None:
        bar_date = view.now.date()
        position = broker.pending_position()
        if self.last_bar_date is not None and bar_date != self.last_bar_date:
            if position != 0.0:
                broker.submit_market(-position)
                position = 0.0
        self.last_bar_date = bar_date
        closes = view.closes(self.window)
        if len(closes) < self.window:
            return
        std = closes.std()
        if std <= 0.0:
            return
        z = (closes[-1] - closes.mean()) / std
        if position == 0.0 and abs(z) > self.entry_z:
            broker.submit_market(-np.sign(z) * self.target_contracts)
        elif position != 0.0 and abs(z) < self.exit_z:
            broker.submit_market(-position)

    def on_end(self, view: MarketView, broker: Broker) -> None:
        if broker.pending_position() != 0.0:
            broker.submit_market(-broker.pending_position())
