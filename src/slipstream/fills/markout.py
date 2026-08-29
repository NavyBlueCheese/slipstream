from __future__ import annotations

import numpy as np
import pandas as pd

from slipstream.data.schema import QuoteSeries
from slipstream.fills.outcome import FillOutcome


def attach_markouts(
    outcomes: list[FillOutcome],
    quotes: QuoteSeries,
    horizons_s: tuple[float, ...] = (1.0, 10.0, 60.0),
) -> list[FillOutcome]:
    ts_values = quotes.frame["ts"].dt.tz_convert(None).to_numpy()
    mids = quotes.mid().to_numpy()
    for outcome in outcomes:
        if outcome.fill_price is None or outcome.fill_ts is None:
            continue
        for horizon in horizons_s:
            target = np.datetime64(
                (outcome.fill_ts + pd.Timedelta(seconds=horizon)).tz_convert(None)
            )
            idx = int(np.searchsorted(ts_values, target))
            if idx >= len(ts_values):
                continue
            outcome.markouts[f"{horizon:g}s"] = float(
                outcome.side * (mids[idx] - outcome.fill_price)
            )
    return outcomes


def adverse_selection_report(outcomes: list[FillOutcome]) -> pd.DataFrame:
    rows = []
    for outcome in outcomes:
        if not outcome.markouts:
            continue
        row = {"passive": outcome.passive, "qty": outcome.filled_qty}
        row.update(outcome.markouts)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    horizon_cols = [c for c in frame.columns if c.endswith("s")]
    grouped = frame.groupby("passive")[horizon_cols].mean()
    grouped["fills"] = frame.groupby("passive")["qty"].count()
    grouped["total_qty"] = frame.groupby("passive")["qty"].sum()
    grouped.index = grouped.index.map({True: "passive", False: "aggressive"})
    return grouped
