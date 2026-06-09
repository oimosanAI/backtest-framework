"""Tests for performance metrics on series simple enough to verify by hand."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.backtest import metrics


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2022-01-01", periods=n, freq="B")


class TestMaxDrawdown:
    """Drawdown depth, dates and duration on a hand-traced equity path."""

    def test_simple_drawdown(self) -> None:
        # returns -> wealth: [1.0, 0.5, 1.0]. Worst point is a 50% drawdown
        # from the bar-0 peak to the bar-1 trough.
        returns = pd.Series([0.0, -0.5, 1.0], index=_index(3))
        dd = metrics.max_drawdown(returns)
        assert dd.max_drawdown == pytest.approx(-0.5)
        assert dd.peak_date == returns.index[0]
        assert dd.trough_date == returns.index[1]
        assert dd.duration == 1

    def test_monotonic_increase_has_no_drawdown(self) -> None:
        returns = pd.Series([0.01] * 10, index=_index(10))
        assert metrics.max_drawdown(returns).max_drawdown == 0.0


class TestAnnualizedReturn:
    def test_doubling_over_two_years(self) -> None:
        # 504 bars == 2 trading years; a single +100% bar doubles wealth.
        # CAGR = 2 ** (1/2) - 1 ~ 0.41421.
        returns = pd.Series([1.0] + [0.0] * 503, index=_index(504))
        assert metrics.annualized_return(returns) == pytest.approx(math.sqrt(2) - 1.0)

    def test_flat_series_is_zero(self) -> None:
        returns = pd.Series([0.0] * 50, index=_index(50))
        assert metrics.annualized_return(returns) == pytest.approx(0.0)

    def test_total_wipeout_is_minus_one(self) -> None:
        returns = pd.Series([0.0, -1.0, 0.5], index=_index(3))
        assert metrics.annualized_return(returns) == -1.0


class TestAnnualizedVolatility:
    def test_matches_std_times_sqrt_252(self) -> None:
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0, 0.01, 300), index=_index(300))
        expected = returns.std(ddof=1) * math.sqrt(metrics.TRADING_DAYS_PER_YEAR)
        assert metrics.annualized_volatility(returns) == pytest.approx(expected)

    def test_constant_series_has_zero_vol(self) -> None:
        returns = pd.Series([0.01] * 20, index=_index(20))
        assert metrics.annualized_volatility(returns) == pytest.approx(0.0)

    def test_single_point_is_nan(self) -> None:
        returns = pd.Series([0.01], index=_index(1))
        assert math.isnan(metrics.annualized_volatility(returns))


class TestSharpeRatio:
    def test_hand_computed_value(self) -> None:
        # returns [0.01, 0.03]: mean 0.02, sample std sqrt(0.0002).
        # Sharpe = 0.02 / sqrt(0.0002) * sqrt(252).
        returns = pd.Series([0.01, 0.03], index=_index(2))
        expected = 0.02 / math.sqrt(0.0002) * math.sqrt(252)
        assert metrics.sharpe_ratio(returns) == pytest.approx(expected)

    def test_zero_mean_gives_zero_sharpe(self) -> None:
        returns = pd.Series([0.01, -0.01, 0.01, -0.01], index=_index(4))
        assert metrics.sharpe_ratio(returns) == pytest.approx(0.0)

    def test_risk_free_rate_is_deannualised(self) -> None:
        # A constant daily return exactly equal to the per-period risk-free
        # rate yields zero excess return every day -> zero std -> nan Sharpe.
        daily = 0.0002
        annual_rf = daily * metrics.TRADING_DAYS_PER_YEAR
        returns = pd.Series([daily] * 30, index=_index(30))
        assert math.isnan(metrics.sharpe_ratio(returns, risk_free_rate=annual_rf))

    def test_constant_returns_have_nan_sharpe(self) -> None:
        returns = pd.Series([0.01] * 10, index=_index(10))
        assert math.isnan(metrics.sharpe_ratio(returns))


class TestWinRateAndPayoff:
    def test_win_rate_excludes_flat_periods(self) -> None:
        returns = pd.Series([0.01, -0.02, 0.0, 0.03, -0.01], index=_index(5))
        # Non-zero bars: 4, of which 2 are positive -> 0.5.
        assert metrics.win_rate(returns) == pytest.approx(0.5)

    def test_profit_loss_ratio(self) -> None:
        returns = pd.Series([0.01, -0.02, 0.03, -0.01], index=_index(4))
        # avg win 0.02, avg loss magnitude 0.015 -> 1.3333...
        assert metrics.profit_loss_ratio(returns) == pytest.approx(0.02 / 0.015)

    def test_win_rate_nan_when_all_flat(self) -> None:
        returns = pd.Series([0.0] * 5, index=_index(5))
        assert math.isnan(metrics.win_rate(returns))

    def test_payoff_nan_without_losses(self) -> None:
        returns = pd.Series([0.01, 0.02], index=_index(2))
        assert math.isnan(metrics.profit_loss_ratio(returns))


class TestEdgeCases:
    """Degenerate inputs must not raise and must return sensible sentinels."""

    def test_single_point(self) -> None:
        returns = pd.Series([0.05], index=_index(1))
        assert metrics.max_drawdown(returns).max_drawdown == 0.0
        assert math.isnan(metrics.annualized_volatility(returns))
        assert math.isnan(metrics.sharpe_ratio(returns))

    def test_summary_keys_present(self) -> None:
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(0, 0.01, 100), index=_index(100))
        keys = metrics.summary(returns).keys()
        assert {
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "max_drawdown_duration",
            "win_rate",
            "profit_loss_ratio",
        } <= set(keys)
