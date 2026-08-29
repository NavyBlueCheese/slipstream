from __future__ import annotations

from slipstream.engine.backtest import Broker
from slipstream.engine.view import MarketView
from slipstream.fills.outcome import FillOutcome


class Strategy:
    def on_start(self, view: MarketView, broker: Broker) -> None:
        pass

    def on_quote(self, view: MarketView, broker: Broker) -> None:
        pass

    def on_trade(self, view: MarketView, broker: Broker) -> None:
        pass

    def on_bar(self, view: MarketView, broker: Broker) -> None:
        pass

    def on_fill(self, view: MarketView, broker: Broker, outcome: FillOutcome) -> None:
        pass

    def on_end(self, view: MarketView, broker: Broker) -> None:
        pass
