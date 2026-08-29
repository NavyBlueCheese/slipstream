from datetime import date

import numpy as np
import pandas as pd
import pytest

from slipstream.contracts import (
    Adjustment,
    AdjustmentArithmeticError,
    build_continuous,
    calendar_roll_dates,
    open_interest_crossover_roll_dates,
    percentage_returns,
    volume_crossover_roll_dates,
)
from slipstream.data import BarSeries
from slipstream.data.synthetic import ChainConfig, generate_futures_chain


def make_bars(start, closes, volumes=None, open_interest=None):
    n = len(closes)
    ts = pd.bdate_range(start, periods=n, tz="UTC")
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate(([closes[0]], closes[:-1]))
    frame = pd.DataFrame(
        {
            "ts": ts,
            "open": opens,
            "high": np.maximum(opens, closes) + 0.5,
            "low": np.minimum(opens, closes) - 0.5,
            "close": closes,
            "volume": volumes if volumes is not None else np.full(n, 100.0),
        }
    )
    if open_interest is not None:
        frame["open_interest"] = open_interest
    return BarSeries.from_frame(frame)


def two_contract_chain():
    front_closes = [100, 101, 102, 103, 104, 105, 106, 107]
    back_closes = [c + 5 for c in [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]]
    front = make_bars(
        "2025-06-02",
        front_closes,
        volumes=[100, 100, 100, 100, 80, 60, 40, 20],
        open_interest=[200, 200, 200, 180, 150, 100, 60, 30],
    )
    back = make_bars(
        "2025-06-02",
        back_closes,
        volumes=[5, 5, 10, 30, 70, 90, 120, 140, 150, 150],
        open_interest=[10, 20, 40, 90, 140, 170, 190, 200, 210, 210],
    )
    return {"ESU5": front, "ESZ5": back}


ORDER = ("ESU5", "ESZ5")


def test_calendar_roll_trigger():
    chain = two_contract_chain()
    expiry = chain["ESU5"].frame["ts"].iloc[-1]
    rolls = calendar_roll_dates(chain, ORDER, {"ESU5": expiry}, days_before=3)
    assert rolls["ESU5"] == chain["ESU5"].frame["ts"].iloc[-4]


def test_volume_crossover_trigger():
    chain = two_contract_chain()
    rolls = volume_crossover_roll_dates(chain, ORDER)
    assert rolls["ESU5"] == chain["ESU5"].frame["ts"].iloc[5]


def test_open_interest_crossover_trigger():
    chain = two_contract_chain()
    rolls = open_interest_crossover_roll_dates(chain, ORDER)
    assert rolls["ESU5"] == chain["ESU5"].frame["ts"].iloc[5]


def test_panama_adjustment_is_continuous_at_roll():
    chain = two_contract_chain()
    roll_ts = chain["ESU5"].frame["ts"].iloc[5]
    continuous = build_continuous(chain, ORDER, {"ESU5": roll_ts})
    adjusted = continuous.prices(Adjustment.PANAMA).series
    diffs = adjusted.diff().dropna()
    assert np.allclose(diffs.to_numpy(), 1.0)
    assert continuous.rolls[0].gap == pytest.approx(5.0)


def test_ratio_adjustment_matches_gap():
    chain = two_contract_chain()
    roll_ts = chain["ESU5"].frame["ts"].iloc[5]
    continuous = build_continuous(chain, ORDER, {"ESU5": roll_ts})
    prices = continuous.prices(Adjustment.RATIO)
    rets = percentage_returns(prices).dropna()
    assert np.all(np.abs(rets.to_numpy()) < 0.02)


def test_panama_percentage_returns_raise():
    chain = two_contract_chain()
    roll_ts = chain["ESU5"].frame["ts"].iloc[5]
    continuous = build_continuous(chain, ORDER, {"ESU5": roll_ts})
    prices = continuous.prices(Adjustment.PANAMA)
    with pytest.raises(AdjustmentArithmeticError):
        prices.pct_change()
    with pytest.raises(AdjustmentArithmeticError):
        percentage_returns(prices)


def test_unadjusted_percentage_returns_raise():
    chain = two_contract_chain()
    roll_ts = chain["ESU5"].frame["ts"].iloc[5]
    continuous = build_continuous(chain, ORDER, {"ESU5": roll_ts})
    with pytest.raises(AdjustmentArithmeticError):
        continuous.prices(Adjustment.UNADJUSTED).pct_change()


def test_panama_prices_can_go_negative_but_diff_is_safe():
    front = make_bars("2025-06-02", [12.0, 13.0, 15.0, 17.0, 16.0])
    back = make_bars("2025-06-02", [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    chain = {"A": front, "B": back}
    roll_ts = front.frame["ts"].iloc[3]
    continuous = build_continuous(chain, ("A", "B"), {"A": roll_ts})
    adjusted = continuous.prices(Adjustment.PANAMA)
    assert (adjusted.series < 0).any()
    assert np.isfinite(adjusted.diff().dropna().to_numpy()).all()


def test_stitched_returns_have_no_roll_jump():
    chain = two_contract_chain()
    roll_ts = chain["ESU5"].frame["ts"].iloc[5]
    continuous = build_continuous(chain, ORDER, {"ESU5": roll_ts})
    rets = continuous.returns().dropna()
    day_after_roll = continuous.frame[continuous.frame["ts"] > roll_ts].iloc[0]
    expected = 111.0 / 110.0 - 1.0
    assert rets.loc[day_after_roll["ts"]] == pytest.approx(expected)
    assert np.all(rets.to_numpy() < 0.02)


def test_roll_requires_overlap():
    front = make_bars("2025-06-02", [10, 11, 12])
    back = make_bars("2025-06-09", [15, 16, 17])
    with pytest.raises(ValueError, match="roll date"):
        build_continuous(
            {"A": front, "B": back},
            ("A", "B"),
            {"A": front.frame["ts"].iloc[-1]},
        )


def test_synthetic_chain_end_to_end():
    config = ChainConfig(
        symbols=("ESU5", "ESZ5"),
        expiries=(date(2025, 9, 19), date(2025, 12, 19)),
        start=date(2025, 8, 1),
        n_days=35,
    )
    chain = generate_futures_chain(config, seed=4)
    rolls = volume_crossover_roll_dates(chain, ("ESU5", "ESZ5"))
    continuous = build_continuous(chain, ("ESU5", "ESZ5"), rolls)
    assert len(continuous.rolls) == 1
    assert continuous.returns().dropna().abs().max() < 0.2
    adjusted = continuous.prices(Adjustment.PANAMA).series
    raw_jump = adjusted.diff().abs().max()
    assert np.isfinite(raw_jump)
