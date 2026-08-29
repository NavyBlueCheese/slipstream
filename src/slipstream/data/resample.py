from __future__ import annotations

import pandas as pd

from slipstream.data.schema import BarSeries, QuoteSeries, TradeSeries


def bars_from_trades(trades: TradeSeries, freq: str) -> BarSeries:
    frame = trades.frame.set_index("ts")
    grouped = frame["price"].resample(freq, label="right", closed="left")
    bars = pd.DataFrame(
        {
            "open": grouped.first(),
            "high": grouped.max(),
            "low": grouped.min(),
            "close": grouped.last(),
            "volume": frame["size"].resample(freq, label="right", closed="left").sum(),
        }
    ).dropna(subset=["open"])
    bars = bars.reset_index().rename(columns={"index": "ts"})
    return BarSeries.from_frame(bars)


def quotes_to_bars(quotes: QuoteSeries, freq: str) -> BarSeries:
    frame = quotes.frame.set_index("ts")
    mid = (frame["bid"] + frame["ask"]) / 2.0
    grouped = mid.resample(freq, label="right", closed="left")
    bars = pd.DataFrame(
        {
            "open": grouped.first(),
            "high": grouped.max(),
            "low": grouped.min(),
            "close": grouped.last(),
            "volume": grouped.count().astype(float),
        }
    ).dropna(subset=["open"])
    bars = bars.reset_index().rename(columns={"index": "ts"})
    return BarSeries.from_frame(bars)
