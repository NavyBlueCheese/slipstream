import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datetime import date, time
from pathlib import Path

import pandas as pd

from slipstream.contracts import get_spec
from slipstream.costs import RealisticCostModel, ZeroCostModel
from slipstream.data import SyntheticMarketConfig, SyntheticMarketGenerator, bars_from_trades
from slipstream.diagnostics import (
    break_even_cost_curve,
    fill_rate_stress,
    plot_break_even,
    plot_signal_decay,
    signal_decay_curve,
)
from slipstream.engine import MarketDataBundle, run_backtest
from slipstream.fills import MarketOrderFillModel
from slipstream.strategies import IntradayMeanReversion

OUTPUT = Path(__file__).resolve().parent.parent / "output"
SEED = 41


def build_data():
    config = SyntheticMarketConfig(session_start=time(13, 30), session_end=time(17, 30))
    quotes, trades = SyntheticMarketGenerator(config, seed=SEED).generate_days(
        date(2025, 6, 2), 3
    )
    bars = bars_from_trades(trades, "1min")
    return quotes, trades, bars


def make_strategy():
    return IntradayMeanReversion(window=30, entry_z=1.5, exit_z=0.3)


def main():
    OUTPUT.mkdir(exist_ok=True)
    spec = get_spec("ES")
    quotes, trades, bars = build_data()
    bundle = MarketDataBundle.from_ticks(quotes, trades, bars=bars)

    def run_with(cost_model, fill_model, entry_delay_s=0.0):
        return run_backtest(
            make_strategy(),
            bundle,
            cost_model,
            fill_model,
            spec=spec,
            seed=SEED,
            entry_delay_s=entry_delay_s,
        )

    realistic = run_with(RealisticCostModel("ES"), MarketOrderFillModel())
    report = realistic.standard_report(output_dir=OUTPUT, quotes=quotes, name="meanrev")
    waterfall = report["waterfall"]
    print("intraday mean reversion, 1-minute bars, tick execution")
    print(f"  gross pnl at decision price {waterfall['gross_pnl']:10.2f}")
    for layer in ("latency_drift", "spread", "book_walk", "impact", "commission"):
        print(f"  {layer:<27} {waterfall[layer]:10.2f}")
    print(f"  net pnl                     {waterfall['net_pnl']:10.2f}")
    print()

    breakeven = break_even_cost_curve(
        lambda cost_model: run_with(cost_model, MarketOrderFillModel()),
        RealisticCostModel("ES"),
        spec,
        reference_price=float(bars.frame["close"].iloc[-1]),
        ticks_grid=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    )
    plot_break_even(breakeven, OUTPUT / "meanrev_breakeven.png")
    print(
        f"break-even {breakeven.break_even_ticks:.2f} ticks, "
        f"realistic estimate {breakeven.realistic_cost_ticks:.2f} ticks"
    )

    decay = signal_decay_curve(
        lambda delay: run_with(ZeroCostModel(), MarketOrderFillModel(), entry_delay_s=delay),
        delays_s=(0.0, 1.0, 5.0, 30.0, 60.0, 300.0),
    )
    plot_signal_decay(decay, OUTPUT / "meanrev_decay.png")
    flag = "execution-critical" if decay.execution_critical else "latency-tolerant"
    print(f"signal half-life {decay.half_life_s:.0f}s ({flag})")

    stress = fill_rate_stress(
        lambda fill_model: run_backtest(
            make_strategy(), bundle, RealisticCostModel("ES"), fill_model, spec=spec, seed=SEED
        ),
        lambda: MarketOrderFillModel(),
        quotes=quotes,
        seed=SEED,
    )
    print("fill-rate stress (net pnl)")
    for rate, random_pnl, adverse_pnl in zip(
        stress.drop_rates, stress.random_pnls, stress.adverse_pnls
    ):
        print(f"  drop {rate:.0%}  random {random_pnl:10.2f}  adverse {adverse_pnl:10.2f}")
    print()
    print(pd.Series(realistic.summary()).round(3).to_string())


if __name__ == "__main__":
    main()
