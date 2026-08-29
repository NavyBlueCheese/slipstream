import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datetime import date
from pathlib import Path

import pandas as pd

from slipstream.contracts import build_continuous, get_spec, volume_crossover_roll_dates
from slipstream.costs import RealisticCostModel
from slipstream.data import BarSeries
from slipstream.data.synthetic import ChainConfig, generate_futures_chain
from slipstream.diagnostics import (
    break_even_cost_curve,
    fill_rate_stress,
    plot_break_even,
    roll_cost_decomposition,
)
from slipstream.engine import MarketDataBundle, run_backtest
from slipstream.fills import MarketOrderFillModel
from slipstream.strategies import TimeSeriesMomentum

OUTPUT = Path(__file__).resolve().parent.parent / "output"
SEED = 7


def build_bundle():
    chain_config = ChainConfig(
        symbols=("ESU5", "ESZ5"),
        expiries=(date(2025, 9, 19), date(2025, 12, 19)),
        start=date(2025, 7, 1),
        n_days=120,
        daily_vol=0.009,
    )
    chain = generate_futures_chain(chain_config, seed=SEED)
    order = ("ESU5", "ESZ5")
    rolls = volume_crossover_roll_dates(chain, order)
    continuous = build_continuous(chain, order, rolls)
    bar_columns = ["ts", "open", "high", "low", "close", "volume"]
    bars = BarSeries.from_frame(continuous.frame[bar_columns])
    return MarketDataBundle.from_bars(bars, rolls=continuous.rolls)


def make_strategy():
    return TimeSeriesMomentum(lookback_bars=20, target_contracts=1.0)


def main():
    OUTPUT.mkdir(exist_ok=True)
    spec = get_spec("ES")
    bundle = build_bundle()
    result = run_backtest(
        make_strategy(),
        bundle,
        RealisticCostModel("ES"),
        MarketOrderFillModel(),
        spec=spec,
        seed=SEED,
    )
    report = result.standard_report(output_dir=OUTPUT, name="tsmom")
    print("tsmom summary")
    print(pd.Series(report["summary"]).round(3).to_string())
    print()
    rolls = roll_cost_decomposition(result)
    print("roll decomposition")
    print(f"  spot pnl  {rolls.spot_pnl:12.2f}")
    print(f"  roll pnl  {rolls.roll_pnl:12.2f}")
    print(f"  roll cost {rolls.roll_cost:12.2f}")
    print(f"  net pnl   {rolls.net_pnl:12.2f}")
    print()

    def run_with_cost_model(cost_model):
        return run_backtest(
            make_strategy(),
            bundle,
            cost_model,
            MarketOrderFillModel(),
            spec=spec,
            seed=SEED,
        )

    reference_price = float(bundle.bars.frame["close"].iloc[-1])
    breakeven = break_even_cost_curve(
        run_with_cost_model, RealisticCostModel("ES"), spec, reference_price
    )
    plot_break_even(breakeven, OUTPUT / "tsmom_breakeven.png")
    print(
        f"break-even {breakeven.break_even_ticks:.2f} ticks "
        f"({breakeven.break_even_bps:.2f} bps), "
        f"realistic estimate {breakeven.realistic_cost_ticks:.2f} ticks"
    )

    def run_with_fill_model(fill_model):
        return run_backtest(
            make_strategy(),
            bundle,
            RealisticCostModel("ES"),
            fill_model,
            spec=spec,
            seed=SEED,
        )

    stress = fill_rate_stress(
        run_with_fill_model, lambda: MarketOrderFillModel(), seed=SEED
    )
    print("fill-rate stress (net pnl)")
    for rate, random_pnl, adverse_pnl in zip(
        stress.drop_rates, stress.random_pnls, stress.adverse_pnls
    ):
        print(f"  drop {rate:.0%}  random {random_pnl:10.2f}  adverse {adverse_pnl:10.2f}")


if __name__ == "__main__":
    main()
