from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from slipstream.contracts import RollEvent, get_spec
from slipstream.costs import RealisticCostModel, ZeroCostModel
from slipstream.data import (
    BarSeries,
    QuoteSeries,
    SyntheticMarketConfig,
    SyntheticMarketGenerator,
)
from slipstream.diagnostics import (
    break_even_cost_curve,
    cost_attribution_waterfall,
    fill_rate_stress,
    plot_break_even,
    plot_resolution_ladder,
    plot_signal_decay,
    plot_waterfall,
    resolution_ladder,
    roll_cost_decomposition,
    signal_decay_curve,
    standard_report,
)
from slipstream.engine import MarketDataBundle, run_backtest
from slipstream.fills import MarketOrderFillModel, QueuePositionFillModel
from slipstream.strategies import PassiveMarketMaker, Strategy, TimeSeriesMomentum

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


class BuyOnce(Strategy):
    def __init__(self):
        self.done = False

    def on_bar(self, view, broker):
        if not self.done:
            broker.submit_market(1)
            self.done = True


class Churner(Strategy):
    def __init__(self):
        self.count = 0

    def on_bar(self, view, broker):
        if self.count % 2 == 0:
            broker.submit_market(1)
        else:
            broker.submit_market(-1)
        self.count += 1

    def on_end(self, view, broker):
        position = broker.pending_position()
        if position != 0.0:
            broker.submit_market(-position)


class BuyFirstQuote(Strategy):
    def __init__(self):
        self.done = False

    def on_quote(self, view, broker):
        if not self.done:
            broker.submit_market(1)
            self.done = True

    def on_end(self, view, broker):
        if broker.pending_position() != 0.0:
            broker.submit_market(-broker.pending_position())


def test_waterfall_reconciles_gross_to_net():
    result = run_backtest(
        TimeSeriesMomentum(lookback_bars=3),
        MarketDataBundle.from_bars(daily_bars([100 + i + (i % 3) for i in range(20)])),
        RealisticCostModel("ES"),
        MarketOrderFillModel(),
        spec=ES,
        seed=5,
    )
    waterfall = cost_attribution_waterfall(result)
    assert waterfall["net_pnl"] == pytest.approx(result.net_pnl())
    reconstructed = waterfall["gross_pnl"] + sum(
        waterfall[k]
        for k in waterfall.index
        if k not in ("gross_pnl", "net_pnl")
    )
    assert reconstructed == pytest.approx(waterfall["net_pnl"])
    assert waterfall["commission"] == pytest.approx(-result.account.commissions_paid)


def test_waterfall_plot_writes_file(tmp_path):
    result = run_backtest(
        BuyOnce(),
        MarketDataBundle.from_bars(daily_bars([100, 101, 102])),
        ZeroCostModel(),
        MarketOrderFillModel(),
        spec=ES,
        seed=1,
    )
    path = tmp_path / "waterfall.png"
    plot_waterfall(cost_attribution_waterfall(result), path)
    assert path.exists() and path.stat().st_size > 0


def churn_closes():
    closes = [5000.0]
    for i in range(59):
        closes.append(closes[-1] + (1.5 if i % 2 == 0 else 0.5))
    return closes


def test_break_even_cost_curve_finds_crossing(tmp_path):
    bars = daily_bars(churn_closes())

    def run_with_cost_model(cost_model):
        return run_backtest(
            Churner(),
            MarketDataBundle.from_bars(bars),
            cost_model,
            MarketOrderFillModel(),
            spec=ES,
            seed=2,
        )

    result = break_even_cost_curve(
        run_with_cost_model, ZeroCostModel(), ES, reference_price=5000.0
    )
    assert result.sharpes[0] > 0
    assert np.isfinite(result.break_even_ticks)
    assert 0.5 < result.break_even_ticks < 8.0
    assert result.break_even_bps == pytest.approx(
        result.break_even_ticks * ES.tick_size / 5000.0 * 1e4
    )
    path = tmp_path / "breakeven.png"
    plot_break_even(result, path)
    assert path.exists()


def ramp_quotes():
    n = 900
    ts = pd.date_range("2025-06-02 14:00", periods=n, freq="1s", tz="UTC")
    seconds = np.arange(n, dtype=float)
    mid = 5000.0 + np.minimum(seconds, 300.0) * 0.01
    bid = np.floor((mid - 0.125) / 0.25) * 0.25
    ask = bid + 0.25
    return QuoteSeries.from_frame(
        pd.DataFrame(
            {"ts": ts, "bid": bid, "ask": ask, "bid_size": 50.0, "ask_size": 50.0}
        )
    )


def test_signal_decay_flags_execution_critical(tmp_path):
    quotes = ramp_quotes()

    def run_with_entry_delay(delay_s):
        return run_backtest(
            BuyFirstQuote(),
            MarketDataBundle(quotes=quotes),
            ZeroCostModel(),
            MarketOrderFillModel(),
            spec=ES,
            seed=3,
            entry_delay_s=delay_s,
        )

    result = signal_decay_curve(
        run_with_entry_delay, delays_s=(0.0, 1.0, 5.0, 30.0, 60.0, 300.0)
    )
    assert result.edges[0] > result.edges[-1]
    assert 60.0 < result.half_life_s < 250.0
    assert result.execution_critical
    path = tmp_path / "decay.png"
    plot_signal_decay(result, path)
    assert path.exists()


def test_fill_rate_stress_reports_adverse_selection_tax():
    config = SyntheticMarketConfig(session_start=time(13, 30), session_end=time(14, 10))
    quotes, trades = SyntheticMarketGenerator(config, seed=17).generate_day(
        date(2025, 6, 2)
    )
    bundle = MarketDataBundle.from_ticks(quotes, trades)

    def run_with_fill_model(fill_model):
        return run_backtest(
            PassiveMarketMaker(quote_size=1, max_inventory=3, requote_every_n_quotes=30),
            bundle,
            ZeroCostModel(),
            fill_model,
            spec=ES,
            seed=11,
        )

    stress = fill_rate_stress(
        run_with_fill_model,
        lambda: QueuePositionFillModel(cancel_to_trade_ratio=1.0),
        quotes=quotes,
        drop_rates=(0.3, 0.5),
        seed=11,
    )
    tax = stress.adverse_selection_tax()
    assert set(tax) == {0.3, 0.5}
    assert all(np.isfinite(v) for v in tax.values())
    assert np.isfinite(stress.baseline_pnl)


def test_resolution_ladder_runs_identical_strategy_across_resolutions(tmp_path):
    config = SyntheticMarketConfig(session_start=time(13, 30), session_end=time(16, 30))
    quotes, trades = SyntheticMarketGenerator(config, seed=23).generate_days(
        date(2025, 6, 2), 2
    )
    ladder = resolution_ladder(
        lambda resolution: TimeSeriesMomentum(lookback_bars=5),
        quotes,
        trades,
        cost_model_factory=lambda: RealisticCostModel("ES"),
        fill_model_factory=lambda: MarketOrderFillModel(),
        spec=ES,
        seed=29,
        resolutions=("1h", "5min", "1min", "tick"),
    )
    assert list(ladder.table.index) == ["1h", "5min", "1min", "tick"]
    assert {"sharpe", "turnover_per_day", "max_drawdown", "net_pnl", "fills"} <= set(
        ladder.table.columns
    )
    assert ladder.table["fills"].sum() > 0
    path = tmp_path / "ladder.png"
    plot_resolution_ladder(ladder, path)
    assert path.exists()


def test_roll_decomposition_matches_hand_calculation():
    closes = [100.0] * 5 + [105.0] * 5
    bars = daily_bars(closes)
    roll_ts = bars.frame["ts"].iloc[4]
    rolls = (RollEvent(ts=roll_ts, from_symbol="ESU5", to_symbol="ESZ5", gap=5.0, ratio=1.05),)
    result = run_backtest(
        BuyOnce(),
        MarketDataBundle.from_bars(bars, rolls=rolls),
        ZeroCostModel(),
        MarketOrderFillModel(),
        spec=ES,
        seed=1,
    )
    decomposition = roll_cost_decomposition(result)
    assert len(decomposition.per_roll) == 1
    assert decomposition.roll_pnl == pytest.approx(-5.0 * 1 * 50.0)
    assert decomposition.gross_pnl == pytest.approx(0.0)
    assert decomposition.spot_pnl == pytest.approx(250.0)
    assert decomposition.roll_cost == 0.0


def test_standard_report_is_default_output(tmp_path):
    result = run_backtest(
        TimeSeriesMomentum(lookback_bars=3),
        MarketDataBundle.from_bars(daily_bars([100 + i for i in range(12)])),
        RealisticCostModel("ES"),
        MarketOrderFillModel(),
        spec=ES,
        seed=13,
    )
    payload = result.standard_report(output_dir=tmp_path, name="check")
    assert {"summary", "waterfall", "roll_decomposition"} <= set(payload)
    assert (tmp_path / "check_waterfall.png").exists()
    assert (tmp_path / "check_equity.png").exists()
    assert (tmp_path / "check_report.json").exists()


def test_standard_report_without_output_dir():
    result = run_backtest(
        BuyOnce(),
        MarketDataBundle.from_bars(daily_bars([100, 101, 102])),
        ZeroCostModel(),
        MarketOrderFillModel(),
        spec=ES,
        seed=1,
    )
    payload = standard_report(result)
    assert payload["waterfall"]["net_pnl"] == pytest.approx(result.net_pnl())
