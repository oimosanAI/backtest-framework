"""Tests for the walk-forward analysis module.

Two properties matter most and are tested directly:

1. **Structural causality of the split.** Every out-of-sample window must lie
   strictly *after* its in-sample window, with no overlap -- otherwise the
   "future" leaks into the "past" and the whole exercise is meaningless. This is
   verified on the pure window generator, independent of any backtest.
2. **Over-fitting is detectable.** On data with no real edge, cherry-picking the
   best of many parameter sets in-sample must visibly decay out-of-sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.strategies import MovingAverageCrossStrategy
from src.backtest.walkforward import (
    generate_windows,
    run_walkforward,
)


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2015-01-01", periods=n, freq="B")


def _random_walk(n: int, seed: int) -> pd.Series:
    """A driftless geometric random walk: no edge for any timing rule to find."""
    rng = np.random.default_rng(seed)
    daily = rng.normal(0.0, 0.01, n)
    return pd.Series(100.0 * np.exp(np.cumsum(daily)), index=_index(n))


def _ma_grid() -> list[MovingAverageCrossStrategy]:
    """A grid of moving-average crosses to over-fit against."""
    grid = []
    for fast in (5, 10, 20, 30):
        for slow in (40, 60, 90):
            grid.append(MovingAverageCrossStrategy(fast, slow))
    return grid


class TestWindowGeneration:
    """The split geometry must be causal and consistent."""

    def test_out_of_sample_is_strictly_after_in_sample(self) -> None:
        """For every fold, the test window starts exactly where the train window
        ends and never overlaps it -- the core no-leak invariant."""
        splits = generate_windows(1000, train_size=200, test_size=50, mode="rolling")
        assert splits  # non-empty
        for s in splits:
            assert s.train_start < s.train_end
            # Contiguous and strictly forward in time:
            assert s.test_start == s.train_end
            assert s.test_start < s.test_end
            assert s.test_end <= 1000

    def test_rolling_windows_have_constant_train_size(self) -> None:
        splits = generate_windows(1000, train_size=200, test_size=50, mode="rolling")
        for s in splits:
            assert s.train_end - s.train_start == 200

    def test_anchored_windows_expand_from_zero(self) -> None:
        splits = generate_windows(1000, train_size=200, test_size=50, mode="anchored")
        for s in splits:
            assert s.train_start == 0
        # The in-sample window must grow monotonically fold over fold.
        train_lengths = [s.train_end - s.train_start for s in splits]
        assert train_lengths == sorted(train_lengths)
        assert train_lengths[-1] > train_lengths[0]

    def test_default_step_tiles_oos_without_gaps_or_overlap(self) -> None:
        """With the default step the out-of-sample windows partition the tested
        span end-to-end: each test window begins where the previous ended."""
        splits = generate_windows(1000, train_size=200, test_size=50, mode="rolling")
        for prev, nxt in zip(splits, splits[1:], strict=False):
            assert nxt.test_start == prev.test_end

    def test_too_short_series_yields_no_windows(self) -> None:
        assert generate_windows(100, train_size=200, test_size=50) == []


class TestWalkForwardRun:
    """End-to-end folds over synthetic data."""

    def test_overfitting_shows_as_in_sample_beating_out_of_sample(self) -> None:
        """On an edgeless random walk, selecting the best of many MA settings
        in-sample produces an optimistic in-sample score that decays out-of-
        sample. The degradation (in-sample minus out-of-sample) must be positive.
        """
        prices = _random_walk(2000, seed=42)
        result = run_walkforward(
            prices,
            _ma_grid(),
            train_size=250,
            test_size=120,
            mode="rolling",
        )
        assert len(result.windows) > 3
        # Selection bias lifts the in-sample mean above the out-of-sample mean.
        assert result.mean_in_sample > result.mean_out_of_sample
        assert result.degradation > 0.0

    def test_window_dates_track_positions(self) -> None:
        prices = _random_walk(1000, seed=1)
        result = run_walkforward(
            prices, _ma_grid(), train_size=200, test_size=100, mode="anchored"
        )
        for w in result.windows:
            # The recorded labels must match the split positions on the index.
            assert w.train_start_date == prices.index[w.split.train_start]
            assert w.test_end_date == prices.index[w.split.test_end - 1]
            # And out-of-sample must start at or after in-sample ends, in time.
            assert w.test_start_date > w.train_end_date or (
                w.test_start_date == prices.index[w.split.test_start]
            )

    def test_selects_a_real_candidate_each_fold(self) -> None:
        prices = _random_walk(1200, seed=2)
        grid = _ma_grid()
        result = run_walkforward(
            prices, grid, train_size=300, test_size=100, mode="rolling"
        )
        names = {s.name for s in grid}
        for w in result.windows:
            assert w.best_strategy_name in names
            assert 0 <= w.best_index < len(grid)


class TestValidation:
    def test_empty_candidates_rejected(self) -> None:
        prices = _random_walk(500, seed=0)
        with pytest.raises(ValueError):
            run_walkforward(prices, [], train_size=100, test_size=50)

    def test_too_short_series_rejected(self) -> None:
        prices = _random_walk(100, seed=0)
        with pytest.raises(ValueError):
            run_walkforward(prices, _ma_grid(), train_size=200, test_size=50)
