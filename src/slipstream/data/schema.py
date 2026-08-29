from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

QUOTE_COLUMNS = ("ts", "bid", "ask", "bid_size", "ask_size")
TRADE_COLUMNS = ("ts", "price", "size", "aggressor")
BAR_COLUMNS = ("ts", "open", "high", "low", "close", "volume")


class SchemaError(ValueError):
    pass


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise SchemaError(f"missing columns {missing}")


def _require_utc_timestamps(ts: pd.Series) -> None:
    if not isinstance(ts.dtype, pd.DatetimeTZDtype):
        raise SchemaError("ts must be timezone-aware datetimes")
    if str(ts.dt.tz) != "UTC":
        raise SchemaError("ts must be UTC")


def _require_strictly_increasing(ts: pd.Series) -> None:
    if len(ts) == 0:
        raise SchemaError("empty series")
    deltas = ts.diff().dropna()
    if (deltas <= pd.Timedelta(0)).any():
        raise SchemaError("timestamps must be strictly increasing with no duplicates")


def _require_positive(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        values = frame[column].to_numpy()
        if not np.all(np.isfinite(values)):
            raise SchemaError(f"{column} contains non-finite values")
        if np.any(values <= 0):
            raise SchemaError(f"{column} must be strictly positive")


@dataclass(frozen=True)
class QuoteSeries:
    frame: pd.DataFrame

    @staticmethod
    def from_frame(frame: pd.DataFrame) -> "QuoteSeries":
        _require_columns(frame, QUOTE_COLUMNS)
        frame = frame.loc[:, list(QUOTE_COLUMNS)].reset_index(drop=True)
        _require_utc_timestamps(frame["ts"])
        _require_strictly_increasing(frame["ts"])
        _require_positive(frame, ("bid", "ask", "bid_size", "ask_size"))
        if np.any(frame["bid"].to_numpy() >= frame["ask"].to_numpy()):
            raise SchemaError("book is crossed or locked")
        return QuoteSeries(frame)

    def __len__(self) -> int:
        return len(self.frame)

    def mid(self) -> pd.Series:
        return (self.frame["bid"] + self.frame["ask"]) / 2.0

    def spread(self) -> pd.Series:
        return self.frame["ask"] - self.frame["bid"]


@dataclass(frozen=True)
class TradeSeries:
    frame: pd.DataFrame

    @staticmethod
    def from_frame(frame: pd.DataFrame) -> "TradeSeries":
        _require_columns(frame, TRADE_COLUMNS)
        frame = frame.loc[:, list(TRADE_COLUMNS)].reset_index(drop=True)
        _require_utc_timestamps(frame["ts"])
        _require_strictly_increasing(frame["ts"])
        _require_positive(frame, ("price", "size"))
        aggressor = frame["aggressor"].to_numpy()
        if not np.all(np.isin(aggressor, (-1, 1))):
            raise SchemaError("aggressor must be -1 or 1")
        return TradeSeries(frame)

    def __len__(self) -> int:
        return len(self.frame)


@dataclass(frozen=True)
class BarSeries:
    frame: pd.DataFrame

    @staticmethod
    def from_frame(frame: pd.DataFrame) -> "BarSeries":
        _require_columns(frame, BAR_COLUMNS)
        keep = list(BAR_COLUMNS) + [c for c in ("open_interest",) if c in frame.columns]
        frame = frame.loc[:, keep].reset_index(drop=True)
        _require_utc_timestamps(frame["ts"])
        _require_strictly_increasing(frame["ts"])
        _require_positive(frame, ("open", "high", "low", "close"))
        volume = frame["volume"].to_numpy()
        if np.any(volume < 0) or not np.all(np.isfinite(volume)):
            raise SchemaError("volume must be non-negative and finite")
        high = frame["high"].to_numpy()
        low = frame["low"].to_numpy()
        opens = frame["open"].to_numpy()
        close = frame["close"].to_numpy()
        if np.any(high < np.maximum(opens, close)) or np.any(low > np.minimum(opens, close)):
            raise SchemaError("high/low must bound open/close")
        return BarSeries(frame)

    def __len__(self) -> int:
        return len(self.frame)
