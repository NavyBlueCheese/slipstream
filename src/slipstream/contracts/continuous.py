from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from slipstream.data.schema import BarSeries


class Adjustment(Enum):
    UNADJUSTED = "unadjusted"
    PANAMA = "panama"
    RATIO = "ratio"


class AdjustmentArithmeticError(TypeError):
    pass


@dataclass(frozen=True)
class RollEvent:
    ts: pd.Timestamp
    from_symbol: str
    to_symbol: str
    gap: float
    ratio: float


@dataclass(frozen=True)
class AdjustedPrices:
    series: pd.Series
    adjustment: Adjustment

    def pct_change(self) -> pd.Series:
        if self.adjustment is Adjustment.PANAMA:
            raise AdjustmentArithmeticError(
                "panama-adjusted prices can cross zero, percentage returns are undefined, "
                "use ContinuousContract.returns() which stitches unadjusted per-contract returns"
            )
        if self.adjustment is Adjustment.UNADJUSTED:
            raise AdjustmentArithmeticError(
                "unadjusted stitched prices jump at rolls, "
                "use ContinuousContract.returns() which handles roll gaps"
            )
        return self.series.pct_change()

    def diff(self) -> pd.Series:
        return self.series.diff()


def percentage_returns(prices: AdjustedPrices) -> pd.Series:
    return prices.pct_change()


@dataclass(frozen=True)
class ContinuousContract:
    frame: pd.DataFrame
    rolls: tuple[RollEvent, ...]

    def prices(self, adjustment: Adjustment) -> AdjustedPrices:
        close = self.frame.set_index("ts")["close"].copy()
        if adjustment is Adjustment.UNADJUSTED:
            return AdjustedPrices(close, adjustment)
        ts_values = self.frame["ts"]
        if adjustment is Adjustment.PANAMA:
            shift = np.zeros(len(close))
            for roll in self.rolls:
                mask = (ts_values <= roll.ts).to_numpy()
                shift[mask] += roll.gap
            return AdjustedPrices(close + shift, adjustment)
        factor = np.ones(len(close))
        for roll in self.rolls:
            mask = (ts_values <= roll.ts).to_numpy()
            factor[mask] *= roll.ratio
        return AdjustedPrices(close * factor, adjustment)

    def returns(self) -> pd.Series:
        return self.frame.set_index("ts")["ret"].copy()

    def roll_dates(self) -> tuple[pd.Timestamp, ...]:
        return tuple(roll.ts for roll in self.rolls)


def _aligned_closes(chain: dict[str, BarSeries]) -> dict[str, pd.Series]:
    return {
        symbol: bars.frame.set_index("ts")["close"] for symbol, bars in chain.items()
    }


def _first_crossover(
    front: pd.Series, back: pd.Series
) -> pd.Timestamp | None:
    joined = pd.concat({"front": front, "back": back}, axis=1).dropna()
    ahead = joined[joined["back"] > joined["front"]]
    if len(ahead) == 0:
        return None
    return ahead.index[0]


def _field(chain: dict[str, BarSeries], symbol: str, column: str) -> pd.Series:
    frame = chain[symbol].frame
    if column not in frame.columns:
        raise KeyError(f"{symbol} bars lack column {column}")
    return frame.set_index("ts")[column]


def calendar_roll_dates(
    chain: dict[str, BarSeries],
    order: tuple[str, ...],
    expiries: dict[str, pd.Timestamp],
    days_before: int,
) -> dict[str, pd.Timestamp]:
    rolls: dict[str, pd.Timestamp] = {}
    for symbol in order[:-1]:
        target = expiries[symbol] - pd.tseries.offsets.BDay(days_before)
        available = _field(chain, symbol, "close").index
        candidates = available[available >= target]
        rolls[symbol] = candidates[0] if len(candidates) else available[-1]
    return rolls


def volume_crossover_roll_dates(
    chain: dict[str, BarSeries], order: tuple[str, ...]
) -> dict[str, pd.Timestamp]:
    rolls: dict[str, pd.Timestamp] = {}
    for front, back in zip(order[:-1], order[1:]):
        crossover = _first_crossover(
            _field(chain, front, "volume"), _field(chain, back, "volume")
        )
        rolls[front] = (
            crossover if crossover is not None else _field(chain, front, "close").index[-1]
        )
    return rolls


def open_interest_crossover_roll_dates(
    chain: dict[str, BarSeries], order: tuple[str, ...]
) -> dict[str, pd.Timestamp]:
    rolls: dict[str, pd.Timestamp] = {}
    for front, back in zip(order[:-1], order[1:]):
        crossover = _first_crossover(
            _field(chain, front, "open_interest"), _field(chain, back, "open_interest")
        )
        rolls[front] = (
            crossover if crossover is not None else _field(chain, front, "close").index[-1]
        )
    return rolls


def build_continuous(
    chain: dict[str, BarSeries],
    order: tuple[str, ...],
    roll_dates: dict[str, pd.Timestamp],
) -> ContinuousContract:
    closes = _aligned_closes(chain)
    rows: list[pd.DataFrame] = []
    rolls: list[RollEvent] = []
    start: pd.Timestamp | None = None
    for i, symbol in enumerate(order):
        frame = chain[symbol].frame
        is_last = i == len(order) - 1
        segment_end = None if is_last else roll_dates[symbol]
        mask = pd.Series(True, index=frame.index)
        if start is not None:
            mask &= frame["ts"] > start
        if segment_end is not None:
            mask &= frame["ts"] <= segment_end
        segment = frame[mask].copy()
        if len(segment) == 0:
            continue
        segment["contract"] = symbol
        own_close = closes[symbol]
        prev_close = own_close.shift(1)
        segment["ret"] = (
            segment["close"].to_numpy()
            / prev_close.reindex(segment["ts"]).to_numpy()
            - 1.0
        )
        rows.append(segment)
        if segment_end is not None:
            next_symbol = order[i + 1]
            old_close = closes[symbol].reindex([segment_end])
            new_close = closes[next_symbol].reindex([segment_end])
            if old_close.isna().any() or new_close.isna().any():
                raise ValueError(
                    f"both {symbol} and {next_symbol} must trade on roll date {segment_end}"
                )
            gap = float(new_close.iloc[0] - old_close.iloc[0])
            ratio = float(new_close.iloc[0] / old_close.iloc[0])
            rolls.append(RollEvent(segment_end, symbol, next_symbol, gap, ratio))
            start = segment_end
    stitched = pd.concat(rows, ignore_index=True)
    return ContinuousContract(stitched, tuple(rolls))
