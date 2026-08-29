from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np

from slipstream.contracts.specs import ContractSpec
from slipstream.costs.base import CostModel
from slipstream.costs.layers import CommissionLayer
from slipstream.engine.backtest import BacktestResult

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class BreakEvenResult:
    ticks_grid: tuple[float, ...]
    sharpes: tuple[float, ...]
    break_even_ticks: float
    break_even_bps: float
    realistic_cost_ticks: float
    reference_price: float


def with_extra_ticks_per_trade(
    cost_model: CostModel, spec: ContractSpec, extra_ticks: float
) -> CostModel:
    base: CommissionLayer = cost_model.commission
    extra_per_contract = extra_ticks * spec.tick_value
    boosted = CommissionLayer(
        broker_commission=base.broker_commission + extra_per_contract,
        exchange_fee=base.exchange_fee,
        clearing_fee=base.clearing_fee,
        nfa_fee=base.nfa_fee,
    )
    return cost_model.replaced(commission=boosted)


def realistic_cost_in_ticks(cost_model: CostModel, spec: ContractSpec) -> float:
    commission_ticks = cost_model.commission.per_contract() / spec.tick_value
    half_spread_ticks = 0.5 * getattr(cost_model.spread, "weight", 0.0)
    return commission_ticks + half_spread_ticks


def break_even_cost_curve(
    run_with_cost_model: Callable[[CostModel], BacktestResult],
    base_cost_model: CostModel,
    spec: ContractSpec,
    reference_price: float,
    ticks_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
) -> BreakEvenResult:
    sharpes: list[float] = []
    for extra_ticks in ticks_grid:
        model = with_extra_ticks_per_trade(base_cost_model, spec, extra_ticks)
        sharpes.append(run_with_cost_model(model).sharpe())
    break_even = _first_zero_crossing(np.asarray(ticks_grid), np.asarray(sharpes))
    bps = break_even * spec.tick_size / reference_price * 1e4
    return BreakEvenResult(
        ticks_grid=tuple(ticks_grid),
        sharpes=tuple(sharpes),
        break_even_ticks=break_even,
        break_even_bps=bps,
        realistic_cost_ticks=realistic_cost_in_ticks(base_cost_model, spec),
        reference_price=reference_price,
    )


def _first_zero_crossing(ticks: np.ndarray, sharpes: np.ndarray) -> float:
    if sharpes[0] <= 0.0:
        return 0.0
    for i in range(1, len(ticks)):
        if sharpes[i] <= 0.0:
            prev_ticks, prev_sharpe = ticks[i - 1], sharpes[i - 1]
            span = prev_sharpe - sharpes[i]
            if span == 0.0:
                return float(ticks[i])
            return float(prev_ticks + (ticks[i] - prev_ticks) * prev_sharpe / span)
    return float("inf")


def plot_break_even(result: BreakEvenResult, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.ticks_grid, result.sharpes, marker="o", color="steelblue")
    ax.axhline(0.0, color="black", linewidth=0.8)
    if np.isfinite(result.break_even_ticks):
        ax.axvline(
            result.break_even_ticks,
            color="indianred",
            linestyle="--",
            label=f"break-even {result.break_even_ticks:.2f} ticks ({result.break_even_bps:.2f} bps)",
        )
    ax.axvline(
        result.realistic_cost_ticks,
        color="seagreen",
        linestyle=":",
        label=f"realistic estimate {result.realistic_cost_ticks:.2f} ticks",
    )
    ax.set_xlabel("extra cost per trade (ticks)")
    ax.set_ylabel("sharpe")
    ax.set_title("break-even cost curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
