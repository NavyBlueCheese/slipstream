import numpy as np
import pandas as pd
import pytest

from slipstream.data import (
    BarSeries,
    QuoteSeries,
    SchemaError,
    TradeSeries,
    load_quotes_csv,
)


def make_quote_frame(n=5):
    ts = pd.date_range("2025-06-02 13:30", periods=n, freq="1s", tz="UTC")
    return pd.DataFrame(
        {
            "ts": ts,
            "bid": np.full(n, 5000.0),
            "ask": np.full(n, 5000.25),
            "bid_size": np.full(n, 10.0),
            "ask_size": np.full(n, 12.0),
        }
    )


def test_valid_quotes_accepted():
    quotes = QuoteSeries.from_frame(make_quote_frame())
    assert len(quotes) == 5
    assert (quotes.spread() == 0.25).all()


def test_crossed_book_rejected():
    frame = make_quote_frame()
    frame.loc[2, "bid"] = 5001.0
    with pytest.raises(SchemaError, match="crossed or locked"):
        QuoteSeries.from_frame(frame)


def test_locked_book_rejected():
    frame = make_quote_frame()
    frame.loc[2, "bid"] = frame.loc[2, "ask"]
    with pytest.raises(SchemaError):
        QuoteSeries.from_frame(frame)


def test_duplicate_timestamps_rejected():
    frame = make_quote_frame()
    frame.loc[2, "ts"] = frame.loc[1, "ts"]
    with pytest.raises(SchemaError, match="strictly increasing"):
        QuoteSeries.from_frame(frame)


def test_non_monotonic_timestamps_rejected():
    frame = make_quote_frame()
    frame.loc[3, "ts"] = frame.loc[0, "ts"]
    with pytest.raises(SchemaError):
        QuoteSeries.from_frame(frame)


def test_naive_timestamps_rejected():
    frame = make_quote_frame()
    frame["ts"] = frame["ts"].dt.tz_localize(None)
    with pytest.raises(SchemaError, match="timezone"):
        QuoteSeries.from_frame(frame)


def test_nonpositive_sizes_rejected():
    frame = make_quote_frame()
    frame.loc[1, "bid_size"] = 0.0
    with pytest.raises(SchemaError, match="strictly positive"):
        QuoteSeries.from_frame(frame)


def test_missing_column_rejected():
    frame = make_quote_frame().drop(columns=["ask"])
    with pytest.raises(SchemaError, match="missing columns"):
        QuoteSeries.from_frame(frame)


def test_trade_aggressor_validated():
    ts = pd.date_range("2025-06-02 13:30", periods=3, freq="1s", tz="UTC")
    frame = pd.DataFrame({"ts": ts, "price": 5000.0, "size": 2.0, "aggressor": [1, 0, -1]})
    with pytest.raises(SchemaError, match="aggressor"):
        TradeSeries.from_frame(frame)


def test_bar_bounds_validated():
    ts = pd.date_range("2025-06-02", periods=3, freq="1D", tz="UTC")
    frame = pd.DataFrame(
        {
            "ts": ts,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 100.5, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 100.2, 102.5],
            "volume": [10.0, 12.0, 9.0],
        }
    )
    with pytest.raises(SchemaError, match="bound"):
        BarSeries.from_frame(frame)


def test_quote_csv_roundtrip(tmp_path):
    frame = make_quote_frame()
    path = tmp_path / "quotes.csv"
    frame.to_csv(path, index=False)
    quotes = load_quotes_csv(path)
    assert len(quotes) == len(frame)
    assert quotes.frame["bid"].equals(frame["bid"])
