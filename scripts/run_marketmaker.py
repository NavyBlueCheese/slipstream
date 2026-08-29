import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datetime import date, time
from pathlib import Path

import pandas as pd

from slipstream.contracts import get_spec
from slipstream.costs import RealisticCostModel
from slipstream.data import SyntheticMarketConfig, SyntheticMarketGenerator
from slipstream.diagnostics import fill_rate_stress
from slipstream.engine import MarketDataBundle, run_backtest
from slipstream.fills import (
    LimitOrderFillModel,
    QueuePositionFillModel,
    adverse_selection_report,
    attach_markouts,
)
from slipstream.strategies import PassiveMarketMaker

OUTPUT = Path(__file__).resolve().parent.parent / "output"
SEED = 19


def build_data():
    config = SyntheticMarketConfig(session_start=time(13, 30), session_end=time(17, 30))
    return SyntheticMarketGenerator(config, seed=SEED).generate_day(date(2025, 6, 2))


def make_strategy():
    return PassiveMarketMaker(quote_size=1, max_inventory=4, requote_every_n_quotes=20)


def main():
    OUTPUT.mkdir(exist_ok=True)
    spec = get_spec("ES")
    quotes, trades = build_data()
    bundle = MarketDataBundle.from_ticks(quotes, trades)

    def run_with(fill_model):
        return run_backtest(
            make_strategy(),
            bundle,
            RealisticCostModel("ES"),
            fill_model,
            spec=spec,
            seed=SEED,
        )

    naive = run_with(LimitOrderFillModel(trade_through_ticks=0, min_traded_volume=0))
    queue = run_with(QueuePositionFillModel(cancel_to_trade_ratio=2.0))
    queue.standard_report(output_dir=OUTPUT, quotes=quotes, name="marketmaker")
    print("passive market maker, one synthetic session")
    print(f"  naive touch fills   net pnl {naive.net_pnl():10.2f}  fills {naive.fill_count()}")
    print(f"  queue position sim  net pnl {queue.net_pnl():10.2f}  fills {queue.fill_count()}")
    print(
        "  queue realism costs "
        f"{naive.net_pnl() - queue.net_pnl():10.2f}"
    )
    print()
    attach_markouts(queue.outcomes, quotes)
    markouts = adverse_selection_report(queue.outcomes)
    if len(markouts):
        print("realized markout by fill style (price points per contract)")
        print(markouts.round(4).to_string())
    print()

    stress = fill_rate_stress(
        run_with,
        lambda: QueuePositionFillModel(cancel_to_trade_ratio=2.0),
        quotes=quotes,
        seed=SEED,
    )
    print("fill-rate stress (net pnl)")
    for rate, random_pnl, adverse_pnl in zip(
        stress.drop_rates, stress.random_pnls, stress.adverse_pnls
    ):
        tax = random_pnl - adverse_pnl
        print(
            f"  drop {rate:.0%}  random {random_pnl:10.2f}  adverse {adverse_pnl:10.2f}"
            f"  adverse selection tax {tax:10.2f}"
        )
    print()
    print(pd.Series(queue.summary()).round(3).to_string())


if __name__ == "__main__":
    main()
