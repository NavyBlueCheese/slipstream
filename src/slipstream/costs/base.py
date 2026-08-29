from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from slipstream.contracts.specs import ContractSpec


@dataclass(frozen=True)
class TradeContext:
    spec: ContractSpec
    side: int
    qty: float
    signal_ts: pd.Timestamp
    arrival_ts: pd.Timestamp
    signal_mid: float
    arrival_mid: float
    spread: float
    top_depth: float
    adv: float
    daily_vol: float
    horizon_volume: float


def with_qty(ctx: TradeContext, qty: float) -> TradeContext:
    return replace(ctx, qty=qty)


class CostLayer(ABC):
    name: str = "layer"

    @abstractmethod
    def cost(self, ctx: TradeContext) -> float:
        raise NotImplementedError


class RollLayer(ABC):
    name: str = "roll"

    @abstractmethod
    def cost(self, spec: ContractSpec, qty: float) -> float:
        raise NotImplementedError


class FinancingChargeLayer(ABC):
    name: str = "financing"

    @abstractmethod
    def cost(self, spec: ContractSpec, margin_posted: float, days: float) -> float:
        raise NotImplementedError


class CostModel:
    def __init__(
        self,
        commission: CostLayer,
        spread: CostLayer,
        impact: CostLayer,
        latency: CostLayer,
        roll: RollLayer,
        financing: FinancingChargeLayer,
    ) -> None:
        self.commission = commission
        self.spread = spread
        self.impact = impact
        self.latency = latency
        self.roll = roll
        self.financing = financing

    def trade_layers(self) -> dict[str, CostLayer]:
        return {
            "commission": self.commission,
            "spread": self.spread,
            "impact": self.impact,
            "latency": self.latency,
        }

    def trade_breakdown(self, ctx: TradeContext) -> dict[str, float]:
        return {name: layer.cost(ctx) for name, layer in self.trade_layers().items()}

    def trade_cost(self, ctx: TradeContext) -> float:
        return float(sum(self.trade_breakdown(ctx).values()))

    def roll_cost(self, spec: ContractSpec, qty: float) -> float:
        return self.roll.cost(spec, qty)

    def financing_cost(self, spec: ContractSpec, margin_posted: float, days: float) -> float:
        return self.financing.cost(spec, margin_posted, days)

    def sample_latency_seconds(self, rng: np.random.Generator) -> float:
        sampler = getattr(self.latency, "sample_seconds", None)
        if sampler is None:
            return 0.0
        return float(sampler(rng))

    def price_impact(self, ctx: TradeContext) -> float:
        fn = getattr(self.impact, "price_impact", None)
        if fn is None:
            return 0.0
        return float(fn(ctx))

    def replaced(self, **layers: object) -> "CostModel":
        return CostModel(
            commission=layers.get("commission", self.commission),
            spread=layers.get("spread", self.spread),
            impact=layers.get("impact", self.impact),
            latency=layers.get("latency", self.latency),
            roll=layers.get("roll", self.roll),
            financing=layers.get("financing", self.financing),
        )
