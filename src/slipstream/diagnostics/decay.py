from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np

from slipstream.engine.backtest import BacktestResult

matplotlib.use("Agg")

import matplotlib.pyplot as plt

EXECUTION_CRITICAL_HALF_LIFE_S = 300.0


@dataclass(frozen=True)
class SignalDecayResult:
    delays_s: tuple[float, ...]
    edges: tuple[float, ...]
    half_life_s: float
    execution_critical: bool


def signal_decay_curve(
    run_with_entry_delay: Callable[[float], BacktestResult],
    delays_s: tuple[float, ...] = (0.0, 1.0, 5.0, 30.0, 60.0, 300.0, 1800.0),
) -> SignalDecayResult:
    edges = tuple(run_with_entry_delay(delay).net_pnl() for delay in delays_s)
    half_life = _edge_half_life(np.asarray(delays_s), np.asarray(edges))
    return SignalDecayResult(
        delays_s=tuple(delays_s),
        edges=edges,
        half_life_s=half_life,
        execution_critical=half_life <= EXECUTION_CRITICAL_HALF_LIFE_S,
    )


def _edge_half_life(delays: np.ndarray, edges: np.ndarray) -> float:
    base = edges[0]
    if base <= 0.0:
        return 0.0
    threshold = base / 2.0
    for i in range(1, len(delays)):
        if edges[i] <= threshold:
            prev_delay, prev_edge = delays[i - 1], edges[i - 1]
            span = prev_edge - edges[i]
            if span == 0.0:
                return float(delays[i])
            return float(
                prev_delay + (delays[i] - prev_delay) * (prev_edge - threshold) / span
            )
    return float("inf")


def plot_signal_decay(result: SignalDecayResult, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    delays = [max(d, 0.5) for d in result.delays_s]
    ax.plot(delays, result.edges, marker="o", color="steelblue")
    ax.set_xscale("log")
    ax.axhline(0.0, color="black", linewidth=0.8)
    if np.isfinite(result.half_life_s) and result.half_life_s > 0:
        ax.axvline(
            max(result.half_life_s, 0.5),
            color="indianred",
            linestyle="--",
            label=f"half-life {result.half_life_s:.0f}s",
        )
        ax.legend()
    flag = "execution-critical" if result.execution_critical else "latency-tolerant"
    ax.set_xlabel("entry delay (s, log scale)")
    ax.set_ylabel("net pnl")
    ax.set_title(f"signal decay curve ({flag})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
