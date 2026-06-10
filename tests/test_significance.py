"""Tests for the statistical-significance module.

These are validated against *synthetic* data with a known ground truth -- never
real market data -- so each assertion has a correct answer we control:

- a strategy with **no** edge must not look significant (high p-value),
- a strategy with an **obvious** edge must look significant (low p-value),
- a bootstrap interval on a hand-chosen series must bracket the true value,
- every result must be exactly reproducible from its seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.significance import (
    best_strategy_permutation_test,
    bonferroni_adjust,
    bootstrap_statistic,
    permutation_test,
    sidak_adjust,
)


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


class TestPermutationTest:
    """A permutation test must tell luck and edge apart."""

    def test_random_signal_is_not_significant(self) -> None:
        """Positions independent of returns -> the edge is indistinguishable
        from luck, so the p-value should be large (not significant)."""
        rng = np.random.default_rng(0)
        n = 500
        returns = pd.Series(rng.normal(0.0, 0.01, n), index=_index(n))
        # Positions drawn with no reference whatsoever to the returns.
        positions = pd.Series(rng.choice([-1.0, 0.0, 1.0], size=n), index=_index(n))

        result = permutation_test(positions, returns, n_permutations=1000, seed=1)
        assert result.p_value > 0.10

    def test_obvious_edge_is_significant(self) -> None:
        """Positions aligned with the contemporaneous return are a clear edge;
        luck almost never reproduces it, so the p-value should be tiny."""
        rng = np.random.default_rng(0)
        n = 500
        returns = pd.Series(rng.normal(0.0, 0.01, n), index=_index(n))
        # A strong (noisy) edge: mostly hold the sign of the return.
        sign = np.sign(returns.to_numpy())
        flip = rng.random(n) < 0.2  # 20% of bets are wrong -> still a real edge
        positions = pd.Series(np.where(flip, -sign, sign), index=_index(n))

        result = permutation_test(positions, returns, n_permutations=1000, seed=1)
        assert result.p_value < 0.01
        # The null distribution should sit well below the observed score.
        assert result.observed > float(np.quantile(result.null_distribution, 0.99))

    def test_p_value_respects_add_one_floor(self) -> None:
        """The add-one correction makes the smallest possible p-value
        (1)/(1 + n_permutations), never zero."""
        rng = np.random.default_rng(0)
        n = 300
        returns = pd.Series(rng.normal(0.0, 0.01, n), index=_index(n))
        positions = pd.Series(np.sign(returns.to_numpy()), index=_index(n))

        n_perm = 200
        result = permutation_test(positions, returns, n_permutations=n_perm, seed=2)
        assert result.p_value >= 1.0 / (1.0 + n_perm)

    def test_reproducible_from_seed(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        returns = pd.Series(rng.normal(0.0, 0.01, n), index=_index(n))
        positions = pd.Series(rng.choice([-1.0, 1.0], n), index=_index(n))

        a = permutation_test(positions, returns, n_permutations=500, seed=7)
        b = permutation_test(positions, returns, n_permutations=500, seed=7)
        assert a.p_value == b.p_value
        assert np.array_equal(a.null_distribution, b.null_distribution)


class TestBootstrap:
    """Bootstrap intervals must bracket a known truth and behave sensibly."""

    def test_interval_brackets_known_mean(self) -> None:
        """For an alternating series the true mean is exactly 0.015; a bootstrap
        interval of the mean should comfortably contain it."""
        values = [0.01, 0.02] * 150  # mean exactly 0.015
        returns = pd.Series(values, index=_index(len(values)))

        res = bootstrap_statistic(
            returns,
            statistic=lambda r: float(r.mean()),
            n_resamples=2000,
            seed=3,
        )
        assert res.lower <= 0.015 <= res.upper
        assert res.point_estimate == 0.015
        # A non-degenerate series must have a positive sampling error.
        assert res.standard_error > 0.0

    def test_point_estimate_inside_interval(self) -> None:
        rng = np.random.default_rng(0)
        n = 400
        returns = pd.Series(rng.normal(0.001, 0.01, n), index=_index(n))
        res = bootstrap_statistic(returns, n_resamples=1000, seed=4)
        assert res.lower <= res.point_estimate <= res.upper
        assert res.distribution.size > 0

    def test_block_bootstrap_runs_and_widens_for_autocorrelated_series(self) -> None:
        """The moving-block bootstrap must run and, on a positively
        autocorrelated series, give a sampling error no smaller than the naive
        i.i.d. bootstrap (which ignores the dependence)."""
        rng = np.random.default_rng(0)
        n = 600
        # AR(1) with positive autocorrelation.
        eps = rng.normal(0.0, 0.01, n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.5 * x[t - 1] + eps[t]
        returns = pd.Series(x, index=_index(n))

        iid = bootstrap_statistic(
            returns, statistic=lambda r: float(r.mean()), n_resamples=1500, seed=5
        )
        block = bootstrap_statistic(
            returns,
            statistic=lambda r: float(r.mean()),
            n_resamples=1500,
            block_size=20,
            seed=5,
        )
        assert block.standard_error >= iid.standard_error

    def test_reproducible_from_seed(self) -> None:
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0.001, 0.01, 200), index=_index(200))
        a = bootstrap_statistic(returns, n_resamples=500, seed=9)
        b = bootstrap_statistic(returns, n_resamples=500, seed=9)
        assert (a.lower, a.upper) == (b.lower, b.upper)
        assert np.array_equal(a.distribution, b.distribution)

    def test_block_size_exceeding_sample_is_rejected(self) -> None:
        """A block longer than the sample would make every resample identical
        to the original series; the call must fail loudly, not degenerate."""
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0.0, 0.01, 50), index=_index(50))
        with pytest.raises(ValueError, match="block_size"):
            bootstrap_statistic(returns, n_resamples=10, block_size=51, seed=0)

    def test_block_size_equal_to_sample_is_allowed(self) -> None:
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0.0, 0.01, 50), index=_index(50))
        res = bootstrap_statistic(
            returns,
            statistic=lambda r: float(r.mean()),
            n_resamples=10,
            block_size=50,
            seed=0,
        )
        assert np.isfinite(res.point_estimate)


class TestMultipleTesting:
    """Corrections for having tried many strategies."""

    def test_bonferroni_and_sidak_inflate_pvalues(self) -> None:
        p = 0.01
        assert bonferroni_adjust(p, 10) == pytest.approx(0.10)
        # Šidák is always <= Bonferroni and >= the raw p.
        s = sidak_adjust(p, 10)
        assert p <= s <= bonferroni_adjust(p, 10)

    def test_bonferroni_caps_at_one(self) -> None:
        assert bonferroni_adjust(0.5, 100) == 1.0

    def test_best_of_random_strategies_is_not_significant(self) -> None:
        """The best of several *edgeless* strategies must not look significant
        once the selection is accounted for: the max-statistic null already
        contains that cherry-picking, so the p-value stays large."""
        rng = np.random.default_rng(0)
        n = 500
        returns = pd.Series(rng.normal(0.0, 0.01, n), index=_index(n))
        positions_list = [
            pd.Series(rng.choice([-1.0, 0.0, 1.0], n), index=_index(n))
            for _ in range(15)
        ]
        res = best_strategy_permutation_test(
            positions_list, returns, n_permutations=800, seed=6
        )
        assert res.p_value > 0.10
        assert 0 <= res.best_index < 15

    def test_all_undefined_candidates_raise_clearly(self) -> None:
        """All-flat candidates have an undefined Sharpe on every P&L; the test
        must explain that, not crash with an "All-NaN slice" error."""
        rng = np.random.default_rng(0)
        n = 100
        returns = pd.Series(rng.normal(0.0, 0.01, n), index=_index(n))
        flat = [pd.Series(0.0, index=_index(n)) for _ in range(3)]
        with pytest.raises(ValueError, match="undefined"):
            best_strategy_permutation_test(flat, returns, n_permutations=10, seed=0)
