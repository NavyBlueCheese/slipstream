from __future__ import annotations

from slipstream.engine.backtest import Broker
from slipstream.engine.view import MarketView
from slipstream.strategies.base import Strategy


class PassiveMarketMaker(Strategy):
    def __init__(
        self,
        quote_size: float = 1.0,
        max_inventory: float = 5.0,
        requote_every_n_quotes: int = 20,
    ) -> None:
        self.quote_size = quote_size
        self.max_inventory = max_inventory
        self.requote_every_n_quotes = requote_every_n_quotes
        self.quote_counter = 0

    def on_quote(self, view: MarketView, broker: Broker) -> None:
        self.quote_counter += 1
        if self.quote_counter % self.requote_every_n_quotes != 0:
            return
        broker.cancel_all()
        inventory = broker.position()
        if inventory < self.max_inventory:
            broker.submit_limit(self.quote_size, view.bid())
        if inventory > -self.max_inventory:
            broker.submit_limit(-self.quote_size, view.ask())

    def on_end(self, view: MarketView, broker: Broker) -> None:
        broker.cancel_all()
        if broker.position() != 0.0:
            broker.submit_market(-broker.position())
