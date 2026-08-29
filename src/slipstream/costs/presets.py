from __future__ import annotations

from slipstream.costs.base import CostModel
from slipstream.costs.layers import (
    CommissionLayer,
    DeterministicLatency,
    FinancingLayer,
    ImpactLayer,
    LatencyLayer,
    LognormalLatency,
    NoImpact,
    RollCostLayer,
    SpreadLayer,
    SquareRootImpact,
)


class ZeroCostModel(CostModel):
    def __init__(self) -> None:
        super().__init__(
            commission=CommissionLayer(0.0, 0.0, 0.0, 0.0),
            spread=SpreadLayer(weight=0.0),
            impact=ImpactLayer(NoImpact()),
            latency=LatencyLayer(DeterministicLatency(0.0), drift_weight=0.0),
            roll=RollCostLayer(calendar_spread_ticks=0.0),
            financing=FinancingLayer(0.0, 0.0),
        )


RETAIL_COMMISSIONS: dict[str, CommissionLayer] = {
    "ES": CommissionLayer(
        broker_commission=0.85, exchange_fee=1.40, clearing_fee=0.10, nfa_fee=0.02
    ),
    "MES": CommissionLayer(
        broker_commission=0.25, exchange_fee=0.37, clearing_fee=0.05, nfa_fee=0.02
    ),
    "ZN": CommissionLayer(
        broker_commission=0.85, exchange_fee=0.87, clearing_fee=0.10, nfa_fee=0.02
    ),
    "ZF": CommissionLayer(
        broker_commission=0.85, exchange_fee=0.87, clearing_fee=0.10, nfa_fee=0.02
    ),
}


class RealisticCostModel(CostModel):
    def __init__(self, root: str = "ES") -> None:
        if root not in RETAIL_COMMISSIONS:
            raise KeyError(f"no commission schedule for {root}")
        super().__init__(
            commission=RETAIL_COMMISSIONS[root],
            spread=SpreadLayer(weight=1.0),
            impact=ImpactLayer(SquareRootImpact(coefficient=0.7)),
            latency=LatencyLayer(LognormalLatency(median_seconds=0.25, sigma=0.8)),
            roll=RollCostLayer(calendar_spread_ticks=1.0),
            financing=FinancingLayer(margin_rate=0.0, collateral_opportunity_rate=0.045),
        )
