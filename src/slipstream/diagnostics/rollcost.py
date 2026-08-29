from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from slipstream.diagnostics.waterfall import cost_attribution_waterfall
from slipstream.engine.backtest import BacktestResult


@dataclass(frozen=True)
class RollDecomposition:
    per_roll: pd.DataFrame
    spot_pnl: float
    roll_pnl: float
    roll_cost: float
    gross_pnl: float
    net_pnl: float


def roll_cost_decomposition(result: BacktestResult) -> RollDecomposition:
    multiplier = result.spec.multiplier
    records = result.account.roll_records
    rows = [
        {
            "ts": record.ts,
            "position": record.position,
            "gap": record.gap,
            "roll_pnl": -record.gap * record.position * multiplier,
            "roll_cost": record.cost,
        }
        for record in records
    ]
    per_roll = pd.DataFrame(rows)
    roll_pnl = float(per_roll["roll_pnl"].sum()) if len(per_roll) else 0.0
    roll_cost = float(per_roll["roll_cost"].sum()) if len(per_roll) else 0.0
    waterfall = cost_attribution_waterfall(result)
    gross = float(waterfall["gross_pnl"])
    return RollDecomposition(
        per_roll=per_roll,
        spot_pnl=gross - roll_pnl,
        roll_pnl=roll_pnl,
        roll_cost=roll_cost,
        gross_pnl=gross,
        net_pnl=result.net_pnl(),
    )
