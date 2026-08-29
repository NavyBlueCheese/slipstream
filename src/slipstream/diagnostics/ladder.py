from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import pandas as pd

from slipstream.contracts.specs import ContractSpec
from slipstream.costs.base import CostModel
from slipstream.data.resample import bars_from_trades
from slipstream.data.schema import QuoteSeries, TradeSeries
from slipstream.engine.backtest import BacktestResult, MarketDataBundle, run_backtest
from slipstream.fills.base import FillModel

matplotlib.use("Agg")

import matplotlib.pyplot as plt

RESOLUTION_FREQ = {"1d": "1D", "1h": "1h", "5min": "5min", "1min": "1min"}


@dataclass(frozen=True)
class ResolutionLadderResult:
    table: pd.DataFrame
    results: dict[str, BacktestResult]


def resolution_ladder(
    strategy_factory: Callable[[str], object],
    quotes: QuoteSeries,
    trades: TradeSeries,
    cost_model_factory: Callable[[], CostModel],
    fill_model_factory: Callable[[], FillModel],
    spec: ContractSpec,
    seed: int,
    resolutions: tuple[str, ...] = ("1d", "1h", "5min", "1min", "tick"),
    tick_signal_freq: str = "1min",
) -> ResolutionLadderResult:
    rows: dict[str, dict[str, float]] = {}
    results: dict[str, BacktestResult] = {}
    for resolution in resolutions:
        if resolution == "tick":
            bars = bars_from_trades(trades, tick_signal_freq)
            bundle = MarketDataBundle.from_ticks(quotes, trades, bars=bars)
        else:
            bars = bars_from_trades(trades, RESOLUTION_FREQ[resolution])
            bundle = MarketDataBundle.from_bars(bars)
        result = run_backtest(
            strategy_factory(resolution),
            bundle,
            cost_model_factory(),
            fill_model_factory(),
            spec=spec,
            seed=seed,
        )
        results[resolution] = result
        rows[resolution] = {
            "sharpe": result.sharpe(),
            "turnover_per_day": result.turnover_per_day(),
            "max_drawdown": result.max_drawdown(),
            "net_pnl": result.net_pnl(),
            "fills": float(result.fill_count()),
        }
    table = pd.DataFrame(rows).T.loc[list(resolutions)]
    return ResolutionLadderResult(table=table, results=results)


def plot_resolution_ladder(result: ResolutionLadderResult, path: str | Path) -> None:
    table = result.table
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, column, color in zip(
        axes, ("sharpe", "turnover_per_day", "max_drawdown"), ("steelblue", "seagreen", "indianred")
    ):
        ax.bar(table.index, table[column], color=color)
        ax.set_title(column)
        ax.tick_params(axis="x", rotation=30)
        ax.axhline(0.0, color="black", linewidth=0.8)
    fig.suptitle("identical strategy across data resolutions")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
