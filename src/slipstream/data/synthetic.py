from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time, timedelta

import numpy as np
import pandas as pd

from slipstream.data.schema import BarSeries, QuoteSeries, TradeSeries


@dataclass(frozen=True)
class SyntheticMarketConfig:
    symbol: str = "ES"
    tick_size: float = 0.25
    base_price: float = 5000.0
    daily_vol: float = 0.011
    drift_per_day: float = 0.0
    session_start: time = time(13, 30)
    session_end: time = time(20, 0)
    quote_interval_s: float = 1.0
    u_shape_strength: float = 1.6
    base_spread_ticks: float = 1.0
    edge_spread_extra_ticks: float = 2.5
    depth_mean: float = 40.0
    depth_sigma: float = 0.5
    gaps_per_day: float = 0.7
    gap_vol_multiple: float = 6.0
    trades_per_second: float = 1.4
    cluster_persistence: float = 0.97
    cluster_vol: float = 0.35
    trade_size_mean: float = 3.0
    mean_reversion: float = 0.02


def _u_shape(frac: np.ndarray, strength: float) -> np.ndarray:
    centered = (2.0 * frac - 1.0) ** 2
    shape = 1.0 + strength * (centered - 1.0 / 3.0)
    return np.clip(shape, 0.15, None)


class SyntheticMarketGenerator:
    def __init__(self, config: SyntheticMarketConfig, seed: int) -> None:
        self.config = config
        self._rng = np.random.default_rng(seed)

    def generate_day(self, session_date: date) -> tuple[QuoteSeries, TradeSeries]:
        quotes_frame, trades_frame, _ = self._simulate_day(session_date, self.config.base_price)
        return QuoteSeries.from_frame(quotes_frame), TradeSeries.from_frame(trades_frame)

    def generate_days(self, start_date: date, n_days: int) -> tuple[QuoteSeries, TradeSeries]:
        quote_frames = []
        trade_frames = []
        price = self.config.base_price
        current = start_date
        for _ in range(n_days):
            while current.weekday() >= 5:
                current += timedelta(days=1)
            quotes_frame, trades_frame, price = self._simulate_day(current, price)
            quote_frames.append(quotes_frame)
            trade_frames.append(trades_frame)
            current += timedelta(days=1)
        quotes = pd.concat(quote_frames, ignore_index=True)
        trades = pd.concat(trade_frames, ignore_index=True)
        return QuoteSeries.from_frame(quotes), TradeSeries.from_frame(trades)

    def _simulate_day(
        self, session_date: date, open_price: float
    ) -> tuple[pd.DataFrame, pd.DataFrame, float]:
        cfg = self.config
        rng = self._rng
        start = pd.Timestamp.combine(session_date, cfg.session_start).tz_localize("UTC")
        end = pd.Timestamp.combine(session_date, cfg.session_end).tz_localize("UTC")
        n = int((end - start).total_seconds() / cfg.quote_interval_s)
        ts = start + pd.to_timedelta(np.arange(n) * cfg.quote_interval_s, unit="s")
        frac = np.arange(n) / max(n - 1, 1)
        vol_mult = _u_shape(frac, cfg.u_shape_strength)

        step_vol = cfg.daily_vol / np.sqrt(n) * vol_mult / np.sqrt(np.mean(vol_mult**2))
        shocks = rng.standard_normal(n) * step_vol
        n_gaps = rng.poisson(cfg.gaps_per_day)
        if n_gaps > 0:
            gap_idx = rng.integers(1, n, size=n_gaps)
            gap_sign = rng.choice((-1.0, 1.0), size=n_gaps)
            shocks[gap_idx] += gap_sign * cfg.gap_vol_multiple * cfg.daily_vol / np.sqrt(n)
        log_price = np.log(open_price) + cfg.drift_per_day * frac
        increments = np.empty(n)
        level = 0.0
        for i in range(n):
            level += shocks[i] - cfg.mean_reversion * level
            increments[i] = level
        mid = np.exp(log_price + increments)

        spread_lambda = cfg.edge_spread_extra_ticks * np.clip(vol_mult - 0.5, 0.05, None)
        extra_ticks = rng.poisson(spread_lambda)
        spread_ticks = np.maximum(1, np.round(cfg.base_spread_ticks) + extra_ticks)
        half_spread = spread_ticks * cfg.tick_size / 2.0
        bid = np.floor((mid - half_spread) / cfg.tick_size) * cfg.tick_size
        ask = np.ceil((mid + half_spread) / cfg.tick_size) * cfg.tick_size
        ask = np.where(ask <= bid, bid + cfg.tick_size, ask)

        depth_scale = cfg.depth_mean / vol_mult
        bid_size = np.maximum(1, np.round(rng.lognormal(np.log(depth_scale), cfg.depth_sigma)))
        ask_size = np.maximum(1, np.round(rng.lognormal(np.log(depth_scale), cfg.depth_sigma)))

        quotes_frame = pd.DataFrame(
            {"ts": ts, "bid": bid, "ask": ask, "bid_size": bid_size, "ask_size": ask_size}
        )

        cluster = np.empty(n)
        state = 0.0
        noise = rng.standard_normal(n) * cfg.cluster_vol
        for i in range(n):
            state = cfg.cluster_persistence * state + noise[i]
            cluster[i] = np.exp(state - cfg.cluster_vol**2 / (2 * (1 - cfg.cluster_persistence**2)))
        intensity = cfg.trades_per_second * cfg.quote_interval_s * vol_mult * cluster
        counts = rng.poisson(intensity)
        total_trades = int(counts.sum())
        step_index = np.repeat(np.arange(n), counts)
        offsets = step_index * cfg.quote_interval_s + rng.uniform(
            0.0, cfg.quote_interval_s * 0.999, size=total_trades
        )
        order = np.argsort(offsets, kind="stable")
        offsets = offsets[order]
        step_index = step_index[order]
        trade_ts = (
            start
            + pd.to_timedelta(offsets, unit="s")
            + pd.to_timedelta(np.arange(total_trades), unit="ns")
        )
        aggressor = rng.choice((-1, 1), size=total_trades)
        price = np.where(aggressor > 0, ask[step_index], bid[step_index])
        size = 1 + rng.geometric(1.0 / cfg.trade_size_mean, size=total_trades)
        trades_frame = pd.DataFrame(
            {"ts": trade_ts, "price": price, "size": size, "aggressor": aggressor}
        )
        return quotes_frame, trades_frame, float(mid[-1])


@dataclass(frozen=True)
class ChainConfig:
    symbols: tuple[str, ...] = ("ESU5", "ESZ5")
    expiries: tuple[date, ...] = field(default_factory=tuple)
    start: date = date(2025, 6, 2)
    n_days: int = 60
    base_price: float = 5000.0
    daily_vol: float = 0.011
    carry_per_contract: float = 12.0
    migration_window_days: int = 10
    base_volume: float = 1_200_000.0
    base_open_interest: float = 1_800_000.0


def _migration_ramp(days_to_expiry: np.ndarray, window_days: int) -> np.ndarray:
    return np.clip((2.0 * window_days - days_to_expiry) / (2.0 * window_days), 0.0, 1.0)


def generate_futures_chain(config: ChainConfig, seed: int) -> dict[str, BarSeries]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(config.start, periods=config.n_days, tz="UTC")
    n = len(dates)
    shocks = rng.standard_normal(n) * config.daily_vol
    spot = config.base_price * np.exp(np.cumsum(shocks))
    days_to_each_expiry = [
        np.maximum((pd.Timestamp(expiry, tz="UTC") - dates).days, 0)
        for expiry in config.expiries
    ]
    chain: dict[str, BarSeries] = {}
    for k, symbol in enumerate(config.symbols):
        days_to_expiry = days_to_each_expiry[k]
        basis = config.carry_per_contract * days_to_expiry / 90.0
        close = spot + basis
        opens = np.concatenate(([close[0]], close[:-1]))
        high = np.maximum(opens, close) * (1 + np.abs(rng.standard_normal(n)) * 0.002)
        low = np.minimum(opens, close) * (1 - np.abs(rng.standard_normal(n)) * 0.002)
        own_decline = _migration_ramp(days_to_expiry, config.migration_window_days)
        if k == 0:
            share = 1.0 - own_decline
        else:
            rise = _migration_ramp(days_to_each_expiry[k - 1], config.migration_window_days)
            share = rise * (1.0 - own_decline)
        volume = np.maximum(config.base_volume * share * rng.uniform(0.7, 1.3, size=n), 1.0)
        open_interest = np.maximum(config.base_open_interest * share, 1.0)
        frame = pd.DataFrame(
            {
                "ts": dates,
                "open": opens,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.round(volume),
                "open_interest": np.round(open_interest),
            }
        )
        expiry = pd.Timestamp(config.expiries[k], tz="UTC")
        frame = frame[frame["ts"] <= expiry].reset_index(drop=True)
        chain[symbol] = BarSeries.from_frame(frame)
    return chain
