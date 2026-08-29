from __future__ import annotations

from typing import Callable, Protocol

import pandas as pd

from slipstream.contracts.specs import ContractSpec
from slipstream.engine.accounting import Account


class BrokerLike(Protocol):
    def flatten(self) -> None: ...

    def position(self) -> float: ...


MarginCallHandler = Callable[[pd.Timestamp, Account, BrokerLike], None]


def liquidating_margin_call_handler(
    ts: pd.Timestamp, account: Account, broker: BrokerLike
) -> None:
    broker.flatten()


class MarginTracker:
    def __init__(self, spec: ContractSpec, handler: MarginCallHandler) -> None:
        self.spec = spec
        self.handler = handler
        self.margin_calls: list[pd.Timestamp] = []

    def required_maintenance(self, position: float) -> float:
        return abs(position) * self.spec.maintenance_margin

    def posted_initial(self, position: float) -> float:
        return abs(position) * self.spec.initial_margin

    def check(
        self,
        ts: pd.Timestamp,
        account: Account,
        broker: BrokerLike,
        mark_price: float,
    ) -> bool:
        if account.position == 0.0:
            return False
        if account.equity(mark_price) >= self.required_maintenance(account.position):
            return False
        self.margin_calls.append(ts)
        self.handler(ts, account, broker)
        return True
