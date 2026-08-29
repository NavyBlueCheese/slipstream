from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FillOutcome:
    order_id: int
    symbol: str
    side: int
    requested_qty: float
    filled_qty: float
    filled: bool
    passive: bool
    signal_ts: pd.Timestamp
    arrival_ts: pd.Timestamp | None
    fill_ts: pd.Timestamp | None
    intended_price: float
    fill_price: float | None
    slippage: dict[str, float] = field(default_factory=dict)
    markouts: dict[str, float] = field(default_factory=dict)

    def slippage_total(self) -> float:
        return float(sum(self.slippage.values()))


def total_slippage(outcomes: list[FillOutcome]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for outcome in outcomes:
        for source, value in outcome.slippage.items():
            totals[source] = totals.get(source, 0.0) + value
    return totals
