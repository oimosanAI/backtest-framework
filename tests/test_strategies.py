"""Tests for strategy signal generation against hand-computed truth."""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.strategies import MovingAverageCrossStrategy


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2021-01-01", periods=n, freq="B")


class TestMovingAverageCross:
    """Crossover signals must match theory on a known series.

    Series: [1, 2, 3, 4, 3, 2, 1, 2, 3], fast=SMA2, slow=SMA3.

    Hand computation (fast / slow):
        idx0: NaN / NaN   -> flat (warm-up)
        idx1: 1.5 / NaN   -> flat (warm-up)
        idx2: 2.5 / 2.000 -> long  (fast > slow)
        idx3: 3.5 / 3.000 -> long
        idx4: 3.5 / 3.333 -> long
        idx5: 2.5 / 3.000 -> short (fast < slow)
        idx6: 1.5 / 2.000 -> short
        idx7: 1.5 / 1.667 -> short
        idx8: 2.5 / 2.000 -> long
    """

    PRICES = [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0]
    EXPECTED_LONGSHORT = [0, 0, 1, 1, 1, -1, -1, -1, 1]
    EXPECTED_LONGONLY = [0, 0, 1, 1, 1, 0, 0, 0, 1]

    def test_long_short_signals_match_theory(self) -> None:
        prices = pd.Series(self.PRICES, index=_index(len(self.PRICES)))
        signals = MovingAverageCrossStrategy(2, 3).generate_signals(prices)
        assert signals.tolist() == [float(s) for s in self.EXPECTED_LONGSHORT]

    def test_long_only_suppresses_shorts(self) -> None:
        prices = pd.Series(self.PRICES, index=_index(len(self.PRICES)))
        signals = MovingAverageCrossStrategy(2, 3, long_only=True).generate_signals(
            prices
        )
        assert signals.tolist() == [float(s) for s in self.EXPECTED_LONGONLY]

    def test_warmup_period_is_flat(self) -> None:
        """No signal may be emitted before the slow average is defined."""
        prices = pd.Series(self.PRICES, index=_index(len(self.PRICES)))
        signals = MovingAverageCrossStrategy(2, 3).generate_signals(prices)
        # slow_window - 1 = 2 leading bars must be flat.
        assert signals.iloc[0] == 0.0
        assert signals.iloc[1] == 0.0

    def test_signals_align_to_price_index(self) -> None:
        prices = pd.Series(self.PRICES, index=_index(len(self.PRICES)))
        signals = MovingAverageCrossStrategy(2, 3).generate_signals(prices)
        pd.testing.assert_index_equal(signals.index, prices.index)


class TestValidation:
    """Constructor guards against nonsensical window configurations."""

    def test_fast_must_be_less_than_slow(self) -> None:
        with pytest.raises(ValueError):
            MovingAverageCrossStrategy(30, 10)

    def test_equal_windows_rejected(self) -> None:
        with pytest.raises(ValueError):
            MovingAverageCrossStrategy(20, 20)

    def test_non_positive_window_rejected(self) -> None:
        with pytest.raises(ValueError):
            MovingAverageCrossStrategy(0, 10)

    def test_name_is_descriptive(self) -> None:
        assert MovingAverageCrossStrategy(20, 50).name == "MA(20,50)"
        assert (
            MovingAverageCrossStrategy(20, 50, long_only=True).name
            == "MA(20,50,long-only)"
        )
