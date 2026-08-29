from __future__ import annotations

import numpy as np
import pandas as pd

from slipstream.contracts.specs import ContractSpec


class LookaheadError(RuntimeError):
    pass


class MarketView:
    def __init__(self, spec: ContractSpec) -> None:
        self.__spec = spec
        self.__now: pd.Timestamp | None = None
        self.__bid = float("nan")
        self.__ask = float("nan")
        self.__bid_size = float("nan")
        self.__ask_size = float("nan")
        self.__last_trade_price: float | None = None
        self.__last_trade_size: float | None = None
        self.__mid_ts: list[int] = []
        self.__mid_history: list[float] = []
        self.__bar_ts: list[pd.Timestamp] = []
        self.__bar_rows: list[tuple[float, float, float, float, float]] = []

    @property
    def spec(self) -> ContractSpec:
        return self.__spec

    @property
    def now(self) -> pd.Timestamp:
        if self.__now is None:
            raise LookaheadError("no market data observed yet")
        return self.__now

    def bid(self) -> float:
        return self.__bid

    def ask(self) -> float:
        return self.__ask

    def mid(self) -> float:
        return (self.__bid + self.__ask) / 2.0

    def spread(self) -> float:
        return self.__ask - self.__bid

    def bid_size(self) -> float:
        return self.__bid_size

    def ask_size(self) -> float:
        return self.__ask_size

    def last_trade(self) -> tuple[float, float] | None:
        if self.__last_trade_price is None:
            return None
        return self.__last_trade_price, self.__last_trade_size

    def bar_count(self) -> int:
        return len(self.__bar_ts)

    def closes(self, n: int) -> np.ndarray:
        rows = self.__bar_rows[-n:]
        return np.array([row[3] for row in rows])

    def bars(self, n: int) -> pd.DataFrame:
        rows = self.__bar_rows[-n:]
        ts = self.__bar_ts[-n:]
        return pd.DataFrame(
            rows, columns=["open", "high", "low", "close", "volume"], index=pd.Index(ts, name="ts")
        )

    def mid_at(self, ts: pd.Timestamp) -> float:
        if self.__now is None or ts >= self.__now:
            raise LookaheadError(
                f"requested market data at {ts}, current time is {self.__now}"
            )
        idx = int(np.searchsorted(np.asarray(self.__mid_ts), ts.value, side="right")) - 1
        if idx < 0:
            raise LookaheadError(f"no market data before {ts}")
        return self.__mid_history[idx]

    def _advance(self, ts: pd.Timestamp) -> None:
        self.__now = ts

    def _set_quote(
        self, ts: pd.Timestamp, bid: float, ask: float, bid_size: float, ask_size: float
    ) -> None:
        self.__now = ts
        self.__bid = bid
        self.__ask = ask
        self.__bid_size = bid_size
        self.__ask_size = ask_size
        self.__mid_ts.append(ts.value)
        self.__mid_history.append((bid + ask) / 2.0)

    def _set_trade(self, ts: pd.Timestamp, price: float, size: float) -> None:
        self.__now = ts
        self.__last_trade_price = price
        self.__last_trade_size = size

    def _append_bar(
        self,
        ts: pd.Timestamp,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        self.__now = ts
        self.__bar_ts.append(ts)
        self.__bar_rows.append((open_, high, low, close, volume))
