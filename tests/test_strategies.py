from datetime import date, time

import numpy as np
import pandas as pd

from slipstream.contracts import get_spec
from slipstream.costs import RealisticCostModel, ZeroCostModel
from slipstream.data import BarSeries, SyntheticMarketConfig, SyntheticMarketGenerator
from slipstream.engine import MarketDataBundle, run_backtest
from slipstream.fills import (
    LimitOrderFillModel,
    MarketOrderFillModel,
    QueuePositionFillModel,
)
from slipstream.strategies import (
    IntradayMeanReversion,
    PassiveMarketMaker,
    TimeSeriesMomentum,
)

ES = get_spec("ES")


def daily_bars(closes, start="2025-06-02"):
    closes = np.asarray(closes, dtype=float)
    ts = pd.bdate_range(start, periods=len(closes), tz="UTC") + pd.Timedelta(hours=20)
    opens = np.concatenate(([closes[0]], closes[:-1]))
    frame = pd.DataFrame(
        {
            "ts": ts,
            "open": opens,
            "high": np.maximum(opens, closes) + 0.25,
            "low": np.minimum(opens, closes) - 0.25,
            "close": closes,
            "volume": np.full(len(closes), 1000.0),
        }
    )
    return BarSeries.from_frame(frame)


def test_tsmom_goes_long_in_uptrend_and_short_in_downtrend():
    up = run_backtest(
        TimeSeriesMomentum(lookback_bars=5),
        MarketDataBundle.from_bars(daily_bars([100 + i for i in range(12)])),
        ZeroCostModel(),
        MarketOrderFillModel(),
        spec=ES,
        seed=1,
    )
    down = run_backtest(
        TimeSeriesMomentum(lookback_bars=5),
        MarketDataBundle.from_bars(daily_bars([200 - i for i in range(12)])),
        ZeroCostModel(),
        MarketOrderFillModel(),
        spec=ES,
        seed=1,
    )
    assert up.account.position == 1.0
    assert down.account.position == -1.0
    assert up.net_pnl() > 0
    assert down.net_pnl() > 0


def test_tsmom_survives_realistic_costs_on_strong_trend():
    closes = [5000 + 5 * i for i in range(40)]
    result = run_backtest(
        TimeSeriesMomentum(lookback_bars=10),
        MarketDataBundle.from_bars(daily_bars(closes)),
        RealisticCostModel("ES"),
        MarketOrderFillModel(),
        spec=ES,
        seed=1,
    )
    assert result.net_pnl() > 0
    assert result.account.contracts_traded <= 3


def oscillating_minute_bars():
    n = 400
    ts = pd.date_range("2025-06-02 14:00", periods=n, freq="1min", tz="UTC")
    closes = 5000 + 4.0 * np.sin(np.arange(n) / 1.6)
    closes = np.round(closes / 0.25) * 0.25
    opens = np.concatenate(([closes[0]], closes[:-1]))
    frame = pd.DataFrame(
        {
            "ts": ts,
            "open": opens,
            "high": np.maximum(opens, closes) + 0.25,
            "low": np.minimum(opens, closes) - 0.25,
            "close": closes,
            "volume": np.full(n, 500.0),
        }
    )
    return BarSeries.from_frame(frame)


def test_meanrev_profits_frictionless_but_dies_under_costs():
    bars = oscillating_minute_bars()
    frictionless = run_backtest(
        IntradayMeanReversion(window=30, entry_z=1.2, exit_z=0.3),
        MarketDataBundle.from_bars(bars),
        ZeroCostModel(),
        MarketOrderFillModel(),
        spec=ES,
        seed=2,
    )
    realistic = run_backtest(
        IntradayMeanReversion(window=30, entry_z=1.2, exit_z=0.3),
        MarketDataBundle.from_bars(bars),
        RealisticCostModel("ES"),
        MarketOrderFillModel(),
        spec=ES,
        seed=2,
    )
    assert frictionless.net_pnl() > 0
    assert realistic.net_pnl() < frictionless.net_pnl()
    assert frictionless.account.contracts_traded >= 10


def test_meanrev_goes_flat_at_day_boundary():
    day_one = oscillating_minute_bars().frame
    day_two = day_one.copy()
    day_two["ts"] = day_two["ts"] + pd.Timedelta(days=1)
    bars = BarSeries.from_frame(pd.concat([day_one, day_two], ignore_index=True))
    result = run_backtest(
        IntradayMeanReversion(window=30, entry_z=1.2, exit_z=0.3),
        MarketDataBundle.from_bars(bars),
        ZeroCostModel(),
        MarketOrderFillModel(),
        spec=ES,
        seed=2,
    )
    assert result.account.position == 0.0


def market_making_setup(seed=31):
    config = SyntheticMarketConfig(session_start=time(13, 30), session_end=time(14, 30))
    quotes, trades = SyntheticMarketGenerator(config, seed=seed).generate_day(
        date(2025, 6, 2)
    )
    return MarketDataBundle.from_ticks(quotes, trades)


def test_market_maker_looks_great_on_touch_fills_and_worse_in_queue():
    bundle = market_making_setup()

    def run_with(fill_model):
        return run_backtest(
            PassiveMarketMaker(quote_size=1, max_inventory=3, requote_every_n_quotes=20),
            bundle,
            ZeroCostModel(),
            fill_model,
            spec=ES,
            seed=7,
        )

    naive_touch = run_with(LimitOrderFillModel(trade_through_ticks=0, min_traded_volume=0))
    conservative_queue = run_with(QueuePositionFillModel(cancel_to_trade_ratio=2.0))
    assert naive_touch.fill_count() > 0
    assert naive_touch.net_pnl() > conservative_queue.net_pnl()


def test_market_maker_passive_fills_dominate_under_queue_model():
    bundle = market_making_setup(seed=41)
    result = run_backtest(
        PassiveMarketMaker(quote_size=1, max_inventory=3, requote_every_n_quotes=20),
        bundle,
        ZeroCostModel(),
        QueuePositionFillModel(cancel_to_trade_ratio=1.0),
        spec=ES,
        seed=7,
    )
    filled = [o for o in result.outcomes if o.filled]
    passive = [o for o in filled if o.passive]
    assert len(filled) == 0 or len(passive) / max(len(filled), 1) > 0.5
