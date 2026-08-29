from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from slipstream.contracts.specs import ContractSpec


@dataclass
class RollRecord:
    ts: pd.Timestamp
    position: float
    gap: float
    cost: float


@dataclass
class Account:
    spec: ContractSpec
    initial_cash: float
    position: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    commissions_paid: float = 0.0
    roll_costs_paid: float = 0.0
    financing_paid: float = 0.0
    roll_records: list[RollRecord] = field(default_factory=list)
    contracts_traded: float = 0.0

    def apply_fill(self, side: int, qty: float, price: float, commission: float) -> None:
        self.commissions_paid += commission
        self.contracts_traded += qty
        signed = side * qty
        if self.position == 0.0 or self.position * signed > 0:
            new_position = self.position + signed
            self.avg_price = (
                self.avg_price * self.position + price * signed
            ) / new_position
            self.position = new_position
            return
        closing = min(abs(signed), abs(self.position))
        direction = 1.0 if self.position > 0 else -1.0
        self.realized_pnl += (price - self.avg_price) * direction * closing * self.spec.multiplier
        self.position += signed
        if self.position == 0.0:
            self.avg_price = 0.0
        elif direction * self.position < 0:
            self.avg_price = price

    def apply_roll(self, ts: pd.Timestamp, gap: float, cost: float) -> None:
        if self.position == 0.0:
            return
        self.avg_price += gap
        self.roll_costs_paid += cost
        self.roll_records.append(RollRecord(ts=ts, position=self.position, gap=gap, cost=cost))

    def pay_financing(self, amount: float) -> None:
        self.financing_paid += amount

    def unrealized_pnl(self, mark_price: float) -> float:
        return (mark_price - self.avg_price) * self.position * self.spec.multiplier

    def total_costs(self) -> float:
        return self.commissions_paid + self.roll_costs_paid + self.financing_paid

    def equity(self, mark_price: float) -> float:
        return (
            self.initial_cash
            + self.realized_pnl
            + self.unrealized_pnl(mark_price)
            - self.total_costs()
        )
