from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from slipstream.contracts.specs import ContractSpec
from slipstream.costs.base import CostModel
from slipstream.data.schema import QuoteSeries
from slipstream.engine.backtest import BacktestResult
from slipstream.engine.orders import Order, OrderState
from slipstream.engine.state import MarketState
from slipstream.fills.base import FillModel
from slipstream.fills.markout import attach_markouts
from slipstream.fills.outcome import FillOutcome


class _DroppingFillModel(FillModel):
    def __init__(self, inner: FillModel) -> None:
        self.inner = inner
        self.tracked_orders: dict[int, Order] = {}
        self.dropped: list[FillOutcome] = []

    def bind(self, spec: ContractSpec, cost_model: CostModel, rng: np.random.Generator) -> None:
        self.spec = spec
        self.cost_model = cost_model
        self.rng = rng
        self.tracked_orders = {}
        self.dropped = []
        self.inner.bind(spec, cost_model, rng)

    def on_order_arrival(self, order: Order, market: MarketState) -> list[FillOutcome]:
        self.tracked_orders[order.order_id] = order
        return self._filter(self.inner.on_order_arrival(order, market))

    def on_trade(
        self, price: float, size: float, aggressor: int, market: MarketState
    ) -> list[FillOutcome]:
        return self._filter(self.inner.on_trade(price, size, aggressor, market))

    def on_quote(self, market: MarketState) -> list[FillOutcome]:
        return self._filter(self.inner.on_quote(market))

    def cancel(self, order_id: int) -> None:
        self.inner.cancel(order_id)

    def open_order_ids(self) -> list[int]:
        return self.inner.open_order_ids()

    def should_drop(self, outcome: FillOutcome) -> bool:
        raise NotImplementedError

    def _filter(self, outcomes: list[FillOutcome]) -> list[FillOutcome]:
        kept: list[FillOutcome] = []
        for outcome in outcomes:
            if outcome.filled and self.should_drop(outcome):
                self._undo(outcome)
                self.dropped.append(outcome)
            else:
                kept.append(outcome)
        return kept

    def _undo(self, outcome: FillOutcome) -> None:
        order = self.tracked_orders.get(outcome.order_id)
        if order is None:
            return
        order.unregister_fill(outcome.filled_qty)
        self.inner.cancel(order.order_id)
        order.state = OrderState.CANCELLED


class DropRandomFillModel(_DroppingFillModel):
    def __init__(self, inner: FillModel, drop_rate: float, seed: int) -> None:
        super().__init__(inner)
        self.drop_rate = drop_rate
        self.drop_rng = np.random.default_rng(seed)

    def should_drop(self, outcome: FillOutcome) -> bool:
        return bool(self.drop_rng.random() < self.drop_rate)


class DropAdverseFillModel(_DroppingFillModel):
    def __init__(self, inner: FillModel, drop_order_ids: set[int]) -> None:
        super().__init__(inner)
        self.drop_order_ids = drop_order_ids

    def should_drop(self, outcome: FillOutcome) -> bool:
        return outcome.passive and outcome.order_id in self.drop_order_ids


@dataclass(frozen=True)
class FillStressResult:
    baseline_pnl: float
    drop_rates: tuple[float, ...]
    random_pnls: tuple[float, ...]
    adverse_pnls: tuple[float, ...]

    def adverse_selection_tax(self) -> dict[float, float]:
        return {
            rate: random_pnl - adverse_pnl
            for rate, random_pnl, adverse_pnl in zip(
                self.drop_rates, self.random_pnls, self.adverse_pnls
            )
        }


def _fill_quality(outcome: FillOutcome) -> float:
    if outcome.markouts:
        first_key = sorted(outcome.markouts, key=lambda k: float(k.rstrip("s")))[0]
        return outcome.markouts[first_key]
    return -outcome.slippage.get("spread", 0.0)


def fill_rate_stress(
    run_with_fill_model: Callable[[FillModel], BacktestResult],
    fill_model_factory: Callable[[], FillModel],
    quotes: QuoteSeries | None = None,
    drop_rates: tuple[float, ...] = (0.1, 0.3, 0.5),
    seed: int = 0,
) -> FillStressResult:
    baseline = run_with_fill_model(fill_model_factory())
    if quotes is not None:
        attach_markouts(baseline.outcomes, quotes, horizons_s=(10.0,))
    passive = [o for o in baseline.outcomes if o.filled and o.passive]
    ranked = sorted(passive, key=_fill_quality, reverse=True)
    random_pnls: list[float] = []
    adverse_pnls: list[float] = []
    for k, rate in enumerate(drop_rates):
        random_result = run_with_fill_model(
            DropRandomFillModel(fill_model_factory(), rate, seed=seed + 1000 + k)
        )
        random_pnls.append(random_result.net_pnl())
        n_drop = int(round(rate * len(ranked)))
        drop_ids = {o.order_id for o in ranked[:n_drop]}
        adverse_result = run_with_fill_model(
            DropAdverseFillModel(fill_model_factory(), drop_ids)
        )
        adverse_pnls.append(adverse_result.net_pnl())
    return FillStressResult(
        baseline_pnl=baseline.net_pnl(),
        drop_rates=tuple(drop_rates),
        random_pnls=tuple(random_pnls),
        adverse_pnls=tuple(adverse_pnls),
    )
