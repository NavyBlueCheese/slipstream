from datetime import date, time

import numpy as np
import pandas as pd

from slipstream.data import SyntheticMarketConfig, SyntheticMarketGenerator, bars_from_trades
from slipstream.data.synthetic import ChainConfig, generate_futures_chain

CONFIG = SyntheticMarketConfig(session_start=time(13, 30), session_end=time(16, 30))


def test_same_seed_is_byte_identical():
    first_quotes, first_trades = SyntheticMarketGenerator(CONFIG, seed=7).generate_day(
        date(2025, 6, 2)
    )
    second_quotes, second_trades = SyntheticMarketGenerator(CONFIG, seed=7).generate_day(
        date(2025, 6, 2)
    )
    assert first_quotes.frame.equals(second_quotes.frame)
    assert first_trades.frame.equals(second_trades.frame)


def test_different_seed_differs():
    a, _ = SyntheticMarketGenerator(CONFIG, seed=1).generate_day(date(2025, 6, 2))
    b, _ = SyntheticMarketGenerator(CONFIG, seed=2).generate_day(date(2025, 6, 2))
    assert not a.frame["bid"].equals(b.frame["bid"])


def test_intraday_vol_is_u_shaped():
    quotes, _ = SyntheticMarketGenerator(CONFIG, seed=11).generate_day(date(2025, 6, 2))
    mid = quotes.mid().to_numpy()
    rets = np.diff(np.log(mid))
    n = len(rets)
    edge = np.concatenate([rets[: n // 10], rets[-n // 10 :]])
    middle = rets[4 * n // 10 : 6 * n // 10]
    assert edge.std() > 1.4 * middle.std()


def test_spread_widens_at_open_and_close():
    quotes, _ = SyntheticMarketGenerator(CONFIG, seed=11).generate_day(date(2025, 6, 2))
    spread = quotes.spread().to_numpy()
    n = len(spread)
    edge = np.concatenate([spread[: n // 10], spread[-n // 10 :]])
    middle = spread[4 * n // 10 : 6 * n // 10]
    assert edge.mean() > middle.mean()


def test_volume_is_clustered():
    _, trades = SyntheticMarketGenerator(CONFIG, seed=11).generate_day(date(2025, 6, 2))
    counts = (
        trades.frame.set_index("ts")["size"].resample("30s").count().to_numpy().astype(float)
    )
    lagged = np.corrcoef(counts[:-1], counts[1:])[0, 1]
    assert lagged > 0.2


def test_multi_day_generation_skips_weekends():
    quotes, trades = SyntheticMarketGenerator(CONFIG, seed=3).generate_days(date(2025, 6, 6), 3)
    days = quotes.frame["ts"].dt.dayofweek.unique()
    assert all(day < 5 for day in days)
    assert len(trades) > 0


def test_trades_resample_to_valid_bars():
    _, trades = SyntheticMarketGenerator(CONFIG, seed=5).generate_day(date(2025, 6, 2))
    bars = bars_from_trades(trades, "1min")
    assert len(bars) > 100
    assert (bars.frame["volume"] > 0).all()


def test_futures_chain_volume_migrates():
    config = ChainConfig(
        symbols=("ESU5", "ESZ5"),
        expiries=(date(2025, 9, 19), date(2025, 12, 19)),
        start=date(2025, 8, 1),
        n_days=35,
    )
    chain = generate_futures_chain(config, seed=9)
    front = chain["ESU5"].frame.set_index("ts")
    back = chain["ESZ5"].frame.set_index("ts")
    joined = pd.concat({"front": front["volume"], "back": back["volume"]}, axis=1).dropna()
    assert (joined["back"] > joined["front"]).any()
    assert joined["front"].iloc[0] > joined["back"].iloc[0]
    assert "open_interest" in front.columns
