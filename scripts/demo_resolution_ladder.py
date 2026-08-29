import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datetime import date, time
from pathlib import Path

from slipstream.contracts import get_spec
from slipstream.costs import RealisticCostModel
from slipstream.data import SyntheticMarketConfig, SyntheticMarketGenerator
from slipstream.diagnostics import plot_resolution_ladder, resolution_ladder
from slipstream.fills import MarketOrderFillModel
from slipstream.strategies import IntradayMeanReversion

OUTPUT = Path(__file__).resolve().parent.parent / "output"
SEED = 5


def main():
    OUTPUT.mkdir(exist_ok=True)
    spec = get_spec("ES")
    config = SyntheticMarketConfig(session_start=time(13, 30), session_end=time(17, 30))
    quotes, trades = SyntheticMarketGenerator(config, seed=SEED).generate_days(
        date(2025, 6, 2), 8
    )
    ladder = resolution_ladder(
        lambda resolution: IntradayMeanReversion(window=30, entry_z=1.5, exit_z=0.3),
        quotes,
        trades,
        cost_model_factory=lambda: RealisticCostModel("ES"),
        fill_model_factory=lambda: MarketOrderFillModel(),
        spec=spec,
        seed=SEED,
    )
    plot_resolution_ladder(ladder, OUTPUT / "resolution_ladder.png")
    ladder.table.to_csv(OUTPUT / "resolution_ladder.csv")
    print("identical strategy, five data resolutions, realistic costs")
    print()
    print(ladder.table.round(3).to_string())
    print()
    print(f"plots written to {OUTPUT}")


if __name__ == "__main__":
    main()
