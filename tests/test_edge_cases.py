"""End-to-end edge cases through the full engine + metrics pipeline.

These guard against crashes and silently-wrong output on the degenerate price
paths every backtester eventually meets: dead-flat markets, perfectly trending
markets, and one-observation inputs.
"""

from __future__ import annotations

import math

import pandas as pd

from src.backtest import metrics
from src.backtest.engine import run_backtest
from src.backtest.strategies import MovingAverageCrossStrategy


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B")


def test_flat_market_produces_zero_pnl() -> None:
    """A perfectly flat market: no crossover, no P&L, no drawdown."""
    prices = pd.Series([100.0] * 60, index=_index(60))
    result = run_backtest(prices, MovingAverageCrossStrategy(5, 20))
    assert result.returns.sum() == 0.0
    assert result.equity_curve.iloc[-1] == 1.0
    assert metrics.max_drawdown(result.returns).max_drawdown == 0.0


def test_monotonic_increase_goes_and_stays_long() -> None:
    """A strictly rising market: the fast MA leads, signal is long throughout."""
    prices = pd.Series([100.0 * (1.01**i) for i in range(60)], index=_index(60))
    strat = MovingAverageCrossStrategy(5, 20)
    result = run_backtest(prices, strat)
    # After warm-up, a monotone series keeps fast above slow -> all long.
    active = result.signals.iloc[strat.slow_window :]
    assert (active == 1.0).all()
    # Long in a rising market makes money.
    assert result.equity_curve.iloc[-1] > 1.0


def test_single_data_point_does_not_crash() -> None:
    """One observation: no return is computable, output is flat and finite."""
    prices = pd.Series([100.0], index=_index(1))
    # A 1-bar series is shorter than any sane window; signal is flat warm-up.
    result = run_backtest(prices, MovingAverageCrossStrategy(2, 3))
    assert result.returns.tolist() == [0.0]
    assert result.equity_curve.tolist() == [1.0]
    assert not math.isnan(result.returns.iloc[0])
