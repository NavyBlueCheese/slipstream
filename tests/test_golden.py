import json
import os
from datetime import date, time
from pathlib import Path

import pytest

from slipstream.contracts import get_spec
from slipstream.costs import RealisticCostModel
from slipstream.data import SyntheticMarketConfig, SyntheticMarketGenerator, bars_from_trades
from slipstream.engine import MarketDataBundle, run_backtest
from slipstream.fills import MarketOrderFillModel, QueuePositionFillModel
from slipstream.strategies import PassiveMarketMaker, TimeSeriesMomentum

GOLDEN_DIR = Path(__file__).parent / "golden"
ES = get_spec("ES")


def rounded_summary(result):
    return {key: round(value, 6) for key, value in result.summary().items()}


def tsmom_result():
    config = SyntheticMarketConfig(session_start=time(13, 30), session_end=time(16, 0))
    quotes, trades = SyntheticMarketGenerator(config, seed=101).generate_days(
        date(2025, 6, 2), 3
    )
    bars = bars_from_trades(trades, "5min")
    return run_backtest(
        TimeSeriesMomentum(lookback_bars=12),
        MarketDataBundle.from_ticks(quotes, trades, bars=bars),
        RealisticCostModel("ES"),
        MarketOrderFillModel(),
        spec=ES,
        seed=202,
    )


def market_maker_result():
    config = SyntheticMarketConfig(session_start=time(13, 30), session_end=time(14, 30))
    quotes, trades = SyntheticMarketGenerator(config, seed=303).generate_day(
        date(2025, 6, 2)
    )
    return run_backtest(
        PassiveMarketMaker(quote_size=1, max_inventory=3, requote_every_n_quotes=25),
        MarketDataBundle.from_ticks(quotes, trades),
        RealisticCostModel("ES"),
        QueuePositionFillModel(cancel_to_trade_ratio=2.0),
        spec=ES,
        seed=404,
    )


CASES = {
    "tsmom_tick.json": tsmom_result,
    "market_maker_queue.json": market_maker_result,
}


@pytest.mark.parametrize("filename", sorted(CASES))
def test_golden_regression(filename):
    summary = rounded_summary(CASES[filename]())
    path = GOLDEN_DIR / filename
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN_DIR.mkdir(exist_ok=True)
        with open(path, "w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        pytest.skip("golden file regenerated")
    with open(path) as handle:
        expected = json.load(handle)
    assert summary == expected
