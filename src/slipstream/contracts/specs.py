from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time


@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    exchange: str
    tick_size: float
    tick_value: float
    multiplier: float
    currency: str
    session_open: time
    session_close: time
    holidays: tuple[date, ...]
    initial_margin: float
    maintenance_margin: float

    def price_to_currency(self, price_change: float, qty: float) -> float:
        return price_change * self.multiplier * qty

    def ticks_to_currency(self, ticks: float, qty: float) -> float:
        return ticks * self.tick_value * qty


CME_HOLIDAYS_2025_2026 = (
    date(2025, 1, 1),
    date(2025, 4, 18),
    date(2025, 5, 26),
    date(2025, 7, 4),
    date(2025, 9, 1),
    date(2025, 11, 27),
    date(2025, 12, 25),
    date(2026, 1, 1),
    date(2026, 4, 3),
    date(2026, 5, 25),
    date(2026, 7, 3),
    date(2026, 9, 7),
    date(2026, 11, 26),
    date(2026, 12, 25),
)

SPECS: dict[str, ContractSpec] = {
    "ES": ContractSpec(
        symbol="ES",
        exchange="CME",
        tick_size=0.25,
        tick_value=12.50,
        multiplier=50.0,
        currency="USD",
        session_open=time(13, 30),
        session_close=time(20, 0),
        holidays=CME_HOLIDAYS_2025_2026,
        initial_margin=15500.0,
        maintenance_margin=14000.0,
    ),
    "MES": ContractSpec(
        symbol="MES",
        exchange="CME",
        tick_size=0.25,
        tick_value=1.25,
        multiplier=5.0,
        currency="USD",
        session_open=time(13, 30),
        session_close=time(20, 0),
        holidays=CME_HOLIDAYS_2025_2026,
        initial_margin=1550.0,
        maintenance_margin=1400.0,
    ),
    "ZN": ContractSpec(
        symbol="ZN",
        exchange="CBOT",
        tick_size=0.015625,
        tick_value=15.625,
        multiplier=1000.0,
        currency="USD",
        session_open=time(12, 20),
        session_close=time(20, 0),
        holidays=CME_HOLIDAYS_2025_2026,
        initial_margin=2200.0,
        maintenance_margin=2000.0,
    ),
    "ZF": ContractSpec(
        symbol="ZF",
        exchange="CBOT",
        tick_size=0.0078125,
        tick_value=7.8125,
        multiplier=1000.0,
        currency="USD",
        session_open=time(12, 20),
        session_close=time(20, 0),
        holidays=CME_HOLIDAYS_2025_2026,
        initial_margin=1500.0,
        maintenance_margin=1350.0,
    ),
}


def get_spec(symbol: str) -> ContractSpec:
    root = symbol.rstrip("0123456789")
    root = root[:-1] if root and root[-1] in "FGHJKMNQUVXZ" and root not in SPECS else root
    if root in SPECS:
        return SPECS[root]
    if symbol in SPECS:
        return SPECS[symbol]
    raise KeyError(f"no contract spec for {symbol}")
