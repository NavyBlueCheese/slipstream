from __future__ import annotations

import json
from pathlib import Path

import matplotlib

from slipstream.data.schema import QuoteSeries
from slipstream.diagnostics.rollcost import roll_cost_decomposition
from slipstream.diagnostics.waterfall import cost_attribution_waterfall, plot_waterfall
from slipstream.engine.backtest import BacktestResult
from slipstream.fills.markout import adverse_selection_report, attach_markouts

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def standard_report(
    result: BacktestResult,
    output_dir: str | Path | None = None,
    quotes: QuoteSeries | None = None,
    name: str = "backtest",
) -> dict:
    waterfall = cost_attribution_waterfall(result)
    rolls = roll_cost_decomposition(result)
    payload: dict = {
        "summary": result.summary(),
        "waterfall": waterfall.to_dict(),
        "roll_decomposition": {
            "spot_pnl": rolls.spot_pnl,
            "roll_pnl": rolls.roll_pnl,
            "roll_cost": rolls.roll_cost,
            "gross_pnl": rolls.gross_pnl,
            "net_pnl": rolls.net_pnl,
            "n_rolls": len(rolls.per_roll),
        },
    }
    if quotes is not None:
        attach_markouts(result.outcomes, quotes)
        markout_table = adverse_selection_report(result.outcomes)
        payload["adverse_selection"] = (
            markout_table.to_dict() if len(markout_table) else {}
        )
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        plot_waterfall(waterfall, directory / f"{name}_waterfall.png")
        _plot_equity(result, directory / f"{name}_equity.png")
        with open(directory / f"{name}_report.json", "w") as handle:
            json.dump(payload, handle, indent=2, default=str)
    return payload


def _plot_equity(result: BacktestResult, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(result.equity.index, result.equity.to_numpy(), color="steelblue", linewidth=1.0)
    ax.set_title("equity curve")
    ax.set_ylabel("equity")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
