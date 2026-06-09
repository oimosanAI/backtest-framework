"""Tests for the backtest engine -- the project's centre of gravity.

The headline test is :class:`TestNoLookAhead`: it *proves*, not merely
asserts, that the engine and its strategies cannot see the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import (
    EXECUTION_LAG,
    BacktestConfig,
    LookAheadError,
    assert_causal,
    run_backtest,
)
from src.backtest.strategies import MovingAverageCrossStrategy


def _index(n: int) -> pd.DatetimeIndex:
    """A simple business-day index of length ``n``."""
    return pd.date_range("2020-01-01", periods=n, freq="B")


class _AlwaysLong:
    """Trivial strategy: always fully long. Causal by construction."""

    name = "always-long"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        return pd.Series(1.0, index=prices.index)


class _PerfectForesight:
    """A *deliberately* non-causal strategy used to prove our tests bite.

    It longs whenever the NEXT bar's price is higher -- the canonical
    look-ahead cheat. The causality check must reject this.
    """

    name = "perfect-foresight"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        # prices.shift(-1) pulls tomorrow's price back to today: future info.
        next_price = prices.shift(-1)
        return (next_price > prices).astype(float)


class TestNoLookAhead:
    """The single most important property: no dependence on the future."""

    def test_future_perturbation_does_not_change_past_signals(self) -> None:
        """Changing future prices must not change earlier signals.

        This is the operational definition of "no look-ahead bias". We build a
        price path, record the MA-cross signals, then rewrite every price after
        a cut point and confirm the earlier signals are byte-for-byte identical.
        """
        rng = np.random.default_rng(42)
        n = 200
        prices = (
            pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=_index(n)).abs() + 1.0
        )
        strat = MovingAverageCrossStrategy(fast_window=10, slow_window=30)

        original = strat.generate_signals(prices)

        cut = 120
        tampered = prices.copy()
        tampered.iloc[cut + 1 :] = tampered.iloc[cut + 1 :] * 5.0  # wreck the future

        after = strat.generate_signals(tampered)

        pd.testing.assert_series_equal(original.iloc[: cut + 1], after.iloc[: cut + 1])

    def test_assert_causal_passes_for_causal_strategy(self) -> None:
        """The perturbation harness accepts a genuinely causal strategy."""
        rng = np.random.default_rng(7)
        prices = (
            pd.Series(100 + np.cumsum(rng.normal(0, 1, 150)), index=_index(150)).abs()
            + 1.0
        )
        assert_causal(MovingAverageCrossStrategy(5, 20), prices, n_trials=12)

    def test_assert_causal_rejects_forward_looking_strategy(self) -> None:
        """The harness must catch a strategy that peeks one bar ahead.

        If this fails, our look-ahead test is asleep at the wheel and the
        passing case above would be meaningless.
        """
        rng = np.random.default_rng(7)
        prices = (
            pd.Series(100 + np.cumsum(rng.normal(0, 1, 150)), index=_index(150)).abs()
            + 1.0
        )
        with pytest.raises(LookAheadError):
            assert_causal(_PerfectForesight(), prices, n_trials=20)

    def test_engine_lags_execution_by_one_bar(self) -> None:
        """A signal at t must only earn the return at t+1, never at t.

        With an always-long signal, the position on the very first bar must be
        flat (the decision was 'made' at bar 0 but cannot be acted on until
        bar 1), and from bar 1 onward it is fully invested.
        """
        prices = pd.Series([100.0, 110.0, 121.0], index=_index(3))
        result = run_backtest(prices, _AlwaysLong())

        assert EXECUTION_LAG == 1
        # Position held during bar 0 is flat: no prior decision was actionable.
        assert result.positions.iloc[0] == 0.0
        assert result.positions.iloc[1] == 1.0
        # Bar-0 P&L is therefore zero regardless of the bar-0 price move.
        assert result.returns.iloc[0] == 0.0
        # Bar-1 return is the actual 10% price move, fully captured.
        assert result.returns.iloc[1] == pytest.approx(0.10)


class TestExecutionAndCosts:
    """Mechanics of position holding and transaction costs."""

    def test_gross_return_matches_buy_and_hold_when_always_long(self) -> None:
        """Always-long, frictionless = buy & hold (minus the lagged first bar)."""
        prices = pd.Series([100.0, 105.0, 102.0, 110.0], index=_index(4))
        result = run_backtest(prices, _AlwaysLong())
        # From bar 1 onward the strategy is fully invested, so its compounded
        # wealth equals the price ratio measured from bar 0's close.
        expected_final = prices.iloc[-1] / prices.iloc[0]
        assert result.equity_curve.iloc[-1] == pytest.approx(expected_final)

    def test_costs_reduce_returns(self) -> None:
        """Adding transaction costs must lower net returns versus frictionless.

        Verifies the *direction* (costs are a drag) and the *magnitude* (the
        drag equals turnover times the cost rate on the entry bar).
        """
        prices = pd.Series([100.0, 110.0, 121.0], index=_index(3))
        free = run_backtest(prices, _AlwaysLong(), BacktestConfig())
        commission = 0.001
        costed = run_backtest(
            prices, _AlwaysLong(), BacktestConfig(commission=commission)
        )

        assert costed.returns.sum() < free.returns.sum()
        # The only turnover is the single entry on bar 1 (0 -> 1), so the total
        # cost is exactly one unit of turnover times the commission.
        assert costed.costs.sum() == pytest.approx(commission)
        # And it bites precisely on the entry bar.
        assert costed.costs.iloc[1] == pytest.approx(commission)

    def test_turnover_counts_initial_entry(self) -> None:
        """Entering from a flat book is turnover and must be charged."""
        prices = pd.Series([100.0, 101.0, 102.0], index=_index(3))
        result = run_backtest(prices, _AlwaysLong())
        # Position path is [0, 1, 1] -> turnover [0, 1, 0]: one entry, no exit.
        assert result.turnover.tolist() == [0.0, 1.0, 0.0]

    def test_position_size_scales_exposure(self) -> None:
        """Half-sizing halves the P&L of a fully-invested signal."""
        prices = pd.Series([100.0, 110.0, 121.0], index=_index(3))
        full = run_backtest(prices, _AlwaysLong(), BacktestConfig(position_size=1.0))
        half = run_backtest(prices, _AlwaysLong(), BacktestConfig(position_size=0.5))
        assert half.returns.iloc[1] == pytest.approx(full.returns.iloc[1] * 0.5)


class TestInputHandling:
    """The engine should accept flexible, realistic inputs."""

    def test_accepts_dataframe_with_close_column(self) -> None:
        df = pd.DataFrame(
            {"Open": [100.0, 101.0], "Close": [100.0, 110.0]}, index=_index(2)
        )
        result = run_backtest(df, _AlwaysLong())
        assert result.returns.iloc[1] == pytest.approx(0.10)

    def test_rejects_dataframe_without_close(self) -> None:
        df = pd.DataFrame({"Open": [100.0, 101.0]}, index=_index(2))
        with pytest.raises(KeyError):
            run_backtest(df, _AlwaysLong())
