from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from slipstream.engine.backtest import BacktestResult
from slipstream.fills.outcome import total_slippage

matplotlib.use("Agg")

import matplotlib.pyplot as plt

SLIPPAGE_ORDER = ("latency_drift", "spread", "book_walk", "impact")


def cost_attribution_waterfall(result: BacktestResult) -> pd.Series:
    slippage = total_slippage(result.outcomes)
    account = result.account
    net = result.net_pnl()
    steps: dict[str, float] = {}
    gross = net
    for source in SLIPPAGE_ORDER:
        value = slippage.get(source, 0.0)
        steps[source] = -value
        gross += value
    steps["commission"] = -account.commissions_paid
    steps["roll_cost"] = -account.roll_costs_paid
    steps["financing"] = -account.financing_paid
    gross += account.commissions_paid + account.roll_costs_paid + account.financing_paid
    ordered = {"gross_pnl": gross}
    ordered.update(steps)
    ordered["net_pnl"] = net
    return pd.Series(ordered)


def plot_waterfall(waterfall: pd.Series, path: str | Path) -> None:
    labels = list(waterfall.index)
    values = waterfall.to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    running = 0.0
    for i, label in enumerate(labels):
        value = values[i]
        if label in ("gross_pnl", "net_pnl"):
            ax.bar(i, value, bottom=0.0, color="steelblue")
            running = value
        else:
            bottom = running + min(value, 0.0) if value < 0 else running
            ax.bar(i, abs(value), bottom=bottom, color="indianred" if value < 0 else "seagreen")
            running += value
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("pnl")
    ax.set_title("cost attribution waterfall")
    ax.axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
