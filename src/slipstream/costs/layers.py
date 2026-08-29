from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from slipstream.contracts.specs import ContractSpec
from slipstream.costs.base import CostLayer, FinancingChargeLayer, RollLayer, TradeContext
from slipstream.data.schema import QuoteSeries


@dataclass(frozen=True)
class CommissionLayer(CostLayer):
    broker_commission: float = 0.0
    exchange_fee: float = 0.0
    clearing_fee: float = 0.0
    nfa_fee: float = 0.0
    name: str = "commission"

    def per_contract(self) -> float:
        return self.broker_commission + self.exchange_fee + self.clearing_fee + self.nfa_fee

    def cost(self, ctx: TradeContext) -> float:
        return abs(ctx.qty) * self.per_contract()


@dataclass(frozen=True)
class SpreadLayer(CostLayer):
    weight: float = 1.0
    name: str = "spread"

    def cost(self, ctx: TradeContext) -> float:
        half_spread = ctx.spread / 2.0
        return abs(ctx.qty) * ctx.spec.multiplier * half_spread * self.weight


def conditional_spread_distribution(
    quotes: QuoteSeries,
    time_bucket: str = "30min",
    vol_window: int = 300,
    vol_quantiles: tuple[float, ...] = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0),
) -> pd.DataFrame:
    frame = quotes.frame.set_index("ts")
    spread = frame["ask"] - frame["bid"]
    mid = (frame["ask"] + frame["bid"]) / 2.0
    realized_vol = np.log(mid).diff().rolling(vol_window).std()
    edges = realized_vol.quantile(list(vol_quantiles)).to_numpy()
    edges[0] -= 1e-12
    labels = [f"vol_q{i + 1}" for i in range(len(edges) - 1)]
    vol_bucket = pd.cut(realized_vol, bins=edges, labels=labels)
    tod = frame.index.floor(time_bucket).time
    grouped = spread.groupby([pd.Index(tod, name="time_of_day"), vol_bucket.rename("vol_bucket")], observed=True)
    stats = grouped.agg(["mean", "median", "std", "count"])
    stats["p90"] = grouped.quantile(0.9)
    return stats


class ImpactFunction(ABC):
    @abstractmethod
    def price_impact(self, ctx: TradeContext) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class NoImpact(ImpactFunction):
    def price_impact(self, ctx: TradeContext) -> float:
        return 0.0


@dataclass(frozen=True)
class SquareRootImpact(ImpactFunction):
    coefficient: float = 0.7

    def price_impact(self, ctx: TradeContext) -> float:
        participation = abs(ctx.qty) / max(ctx.adv, 1.0)
        return self.coefficient * ctx.daily_vol * ctx.arrival_mid * float(np.sqrt(participation))


@dataclass(frozen=True)
class LinearParticipationImpact(ImpactFunction):
    coefficient: float = 0.5

    def price_impact(self, ctx: TradeContext) -> float:
        participation = abs(ctx.qty) / max(ctx.horizon_volume, 1.0)
        return self.coefficient * ctx.daily_vol * ctx.arrival_mid * participation


@dataclass(frozen=True)
class ImpactLayer(CostLayer):
    impact_function: ImpactFunction = NoImpact()
    name: str = "impact"

    def price_impact(self, ctx: TradeContext) -> float:
        return max(self.impact_function.price_impact(ctx), 0.0)

    def cost(self, ctx: TradeContext) -> float:
        return abs(ctx.qty) * ctx.spec.multiplier * self.price_impact(ctx)


class LatencyDistribution(ABC):
    @abstractmethod
    def sample_seconds(self, rng: np.random.Generator) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class DeterministicLatency(LatencyDistribution):
    seconds: float = 0.0

    def sample_seconds(self, rng: np.random.Generator) -> float:
        return self.seconds


@dataclass(frozen=True)
class LognormalLatency(LatencyDistribution):
    median_seconds: float = 0.05
    sigma: float = 0.6

    def sample_seconds(self, rng: np.random.Generator) -> float:
        return float(rng.lognormal(np.log(self.median_seconds), self.sigma))


@dataclass(frozen=True)
class LatencyLayer(CostLayer):
    distribution: LatencyDistribution = DeterministicLatency(0.0)
    drift_weight: float = 1.0
    name: str = "latency"

    def sample_seconds(self, rng: np.random.Generator) -> float:
        return self.distribution.sample_seconds(rng)

    def cost(self, ctx: TradeContext) -> float:
        drift = ctx.side * (ctx.arrival_mid - ctx.signal_mid)
        return abs(ctx.qty) * ctx.spec.multiplier * drift * self.drift_weight


@dataclass(frozen=True)
class RollCostLayer(RollLayer):
    calendar_spread_ticks: float = 1.0
    name: str = "roll"

    def cost(self, spec: ContractSpec, qty: float) -> float:
        return abs(qty) * spec.tick_value * self.calendar_spread_ticks / 2.0


@dataclass(frozen=True)
class FinancingLayer(FinancingChargeLayer):
    margin_rate: float = 0.0
    collateral_opportunity_rate: float = 0.0
    name: str = "financing"

    def cost(self, spec: ContractSpec, margin_posted: float, days: float) -> float:
        annual_rate = self.margin_rate + self.collateral_opportunity_rate
        return abs(margin_posted) * annual_rate * days / 360.0
