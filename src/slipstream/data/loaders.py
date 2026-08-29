from __future__ import annotations

from pathlib import Path

import pandas as pd

from slipstream.data.schema import BarSeries, QuoteSeries, TradeSeries


def _read_with_utc_ts(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return frame


def load_quotes_csv(path: str | Path) -> QuoteSeries:
    return QuoteSeries.from_frame(_read_with_utc_ts(path))


def load_trades_csv(path: str | Path) -> TradeSeries:
    return TradeSeries.from_frame(_read_with_utc_ts(path))


def load_bars_csv(path: str | Path) -> BarSeries:
    return BarSeries.from_frame(_read_with_utc_ts(path))
