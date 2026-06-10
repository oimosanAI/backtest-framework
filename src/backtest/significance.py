"""Statistical-significance tests for backtest results.

A clean equity curve is not evidence. The two questions this module exists to
answer are the ones that separate a *believable* backtest from a lucky one:

1. **Could this result be chance?** A strategy that is really just a coin flip
   will still, sometimes, post a high Sharpe ratio over a finite sample. The
   :func:`permutation_test` shuffles away the strategy's timing edge and asks
   how often pure luck reproduces a result this good -- that frequency is a
   p-value.
2. **How uncertain is the number itself?** A point estimate of the Sharpe ratio
   with no error bar is a half-truth. :func:`bootstrap_statistic` resamples the
   realised returns to put an honest confidence interval around it.

The third question -- **did I fool myself by trying many strategies?** -- is the
quiet killer of quant research (data snooping / multiple testing). Picking the
best of *N* attempts inflates its apparent significance even if none of the *N*
has real edge. :func:`bonferroni_adjust`, :func:`sidak_adjust` and the
multiple-strategy :func:`best_strategy_permutation_test` (a permutation Reality
Check) make that inflation explicit and correct for it.

Every function takes an explicit ``seed`` so that every reported p-value and
interval is exactly reproducible -- a result you cannot reproduce is not a
result.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.backtest.metrics import sharpe_ratio

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Number of resamples / permutations. The Monte-Carlo error on a p-value or an
# interval endpoint shrinks as 1/sqrt(n); a couple of thousand draws keeps that
# error comfortably below the second decimal place while staying fast enough to
# run in a notebook. Both are exposed as arguments for tighter or looser runs.
DEFAULT_N_RESAMPLES: int = 2000
DEFAULT_N_PERMUTATIONS: int = 2000

# 95% is the conventional default confidence level for the bootstrap interval.
DEFAULT_CONFIDENCE_LEVEL: float = 0.95

# Default reproducibility seed. Every public function accepts its own ``seed``.
DEFAULT_SEED: int = 0

# A *statistic* maps a return series to a single number (the "score" whose
# significance we are testing). The default below is the annualised Sharpe
# ratio, but mean return, total return, etc. are all valid choices.
Statistic = Callable[[pd.Series], float]

# Tail of interest for a permutation p-value. ``"greater"`` is the natural
# default for a trading edge: we ask how often luck beats the strategy, not how
# often it merely differs from it.
Alternative = Literal["greater", "less", "two-sided"]


def _default_statistic(returns: pd.Series) -> float:
    """Annualised Sharpe ratio -- the default score whose significance we test."""
    return sharpe_ratio(returns)


def _as_series(values: np.ndarray) -> pd.Series:
    """Wrap a resampled NumPy array in a plain-indexed float Series.

    The statistic callables (and the :mod:`metrics` functions they reuse) operate
    on a :class:`~pandas.Series`. Resampling/permuting destroys the original time
    ordering anyway, so a fresh ``RangeIndex`` is the honest index to attach.
    """
    return pd.Series(values, dtype=float)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """A bootstrap point estimate with a confidence interval and its sampling law.

    Attributes
    ----------
    point_estimate:
        The statistic evaluated once on the *original* (un-resampled) returns.
    lower, upper:
        The percentile-method confidence-interval endpoints at
        ``confidence_level``.
    confidence_level:
        The nominal coverage of ``[lower, upper]`` (e.g. ``0.95``).
    standard_error:
        Standard deviation of the bootstrap distribution -- a direct estimate of
        the statistic's sampling error.
    distribution:
        The full array of bootstrap statistics (length ``n_resamples`` minus any
        draws that produced an undefined statistic). Useful for plotting.
    n_resamples:
        Number of bootstrap resamples requested.
    seed:
        The seed used, so the run is reproducible.
    """

    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    standard_error: float
    distribution: np.ndarray
    n_resamples: int
    seed: int


def _resample_indices(rng: np.random.Generator, n: int, block_size: int) -> np.ndarray:
    """Return ``n`` resampled positions, i.i.d. or in moving blocks.

    With ``block_size == 1`` this is the ordinary i.i.d. bootstrap: each position
    is drawn independently with replacement. With ``block_size > 1`` it is the
    *moving-block* bootstrap, which resamples contiguous runs of observations and
    so preserves short-range autocorrelation -- important because daily strategy
    returns are rarely independent, and an i.i.d. bootstrap would understate the
    uncertainty of a serially-correlated Sharpe ratio.
    """
    if block_size <= 1:
        return rng.integers(0, n, size=n)

    # Number of blocks needed to cover n observations, then trim to exactly n.
    n_blocks = int(np.ceil(n / block_size))
    # A block may start anywhere that leaves room for a full block; wrapping is
    # avoided so every block is a genuine contiguous slice of the sample.
    max_start = max(n - block_size, 0)
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    offsets = np.arange(block_size)
    idx = (starts[:, None] + offsets[None, :]).ravel()[:n]
    return idx


def bootstrap_statistic(
    returns: pd.Series,
    statistic: Statistic = _default_statistic,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    block_size: int = 1,
    seed: int = DEFAULT_SEED,
) -> BootstrapResult:
    """Bootstrap a confidence interval for ``statistic`` of a return series.

    The procedure resamples the realised returns *with replacement*
    ``n_resamples`` times, recomputes ``statistic`` on each resample, and reports
    the percentile-method interval of that bootstrap distribution. This answers
    "given only this one sample of returns, how precisely do we know the
    statistic?" without assuming normality.

    Parameters
    ----------
    returns:
        Periodic (e.g. daily) strategy returns.
    statistic:
        Function mapping a return series to a scalar score. Defaults to the
        annualised Sharpe ratio.
    n_resamples:
        Number of bootstrap resamples.
    confidence_level:
        Nominal coverage of the returned interval, in ``(0, 1)``.
    block_size:
        Block length for the moving-block bootstrap. ``1`` (default) is the
        i.i.d. bootstrap; values ``> 1`` preserve serial correlation.
    seed:
        Seed for reproducibility.

    Returns
    -------
    BootstrapResult
        Point estimate, interval, standard error and the bootstrap distribution.

    Raises
    ------
    ValueError
        If ``returns`` has fewer than two observations, ``n_resamples`` is not
        positive, ``confidence_level`` is outside ``(0, 1)``, or ``block_size``
        is not positive.

    Notes
    -----
    We use the transparent *percentile* method rather than BCa: it makes no
    distributional assumption and is easy to verify by eye against the plotted
    distribution. For strongly skewed statistics a bias-corrected interval would
    be tighter, but the percentile interval never silently hides skew.
    """
    if confidence_level <= 0.0 or confidence_level >= 1.0:
        raise ValueError("confidence_level must lie strictly in (0, 1)")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    r = returns.astype(float).dropna()
    n = len(r)
    if n < 2:
        raise ValueError("need at least two observations to bootstrap")

    point = float(statistic(r))
    values = r.to_numpy()
    rng = np.random.default_rng(seed)

    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = _resample_indices(rng, n, block_size)
        samples[i] = statistic(_as_series(values[idx]))

    # A resample can yield an undefined statistic (e.g. a Sharpe on a constant
    # draw); drop those NaNs rather than letting them poison the percentiles.
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        raise ValueError(
            "every bootstrap resample produced an undefined statistic; "
            "the statistic may be ill-defined for this series"
        )

    tail = (1.0 - confidence_level) / 2.0
    lower = float(np.quantile(finite, tail))
    upper = float(np.quantile(finite, 1.0 - tail))
    standard_error = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")

    return BootstrapResult(
        point_estimate=point,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        standard_error=standard_error,
        distribution=finite,
        n_resamples=n_resamples,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Permutation (Monte-Carlo) significance test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PermutationResult:
    """Outcome of a permutation test of a strategy's edge.

    Attributes
    ----------
    observed:
        The statistic of the *true* strategy P&L (positions aligned to returns).
    p_value:
        Probability, under the no-edge null, of a statistic at least as extreme
        as ``observed`` in the direction set by ``alternative``. Computed with
        the add-one correction ``(1 + hits) / (1 + n_permutations)`` so it can
        never be exactly zero -- you have not *proven* impossibility, only failed
        to reproduce the result by luck in finitely many tries.
    n_permutations:
        Number of random permutations drawn for the null distribution.
    alternative:
        Which tail defines "as extreme": ``"greater"``, ``"less"`` or
        ``"two-sided"``.
    null_distribution:
        The array of statistics under permutation -- the empirical null.
    seed:
        Seed used, for reproducibility.
    """

    observed: float
    p_value: float
    n_permutations: int
    alternative: Alternative
    null_distribution: np.ndarray
    seed: int


def _p_value_from_null(
    observed: float, null: np.ndarray, alternative: Alternative
) -> float:
    """Add-one-corrected p-value of ``observed`` against an empirical ``null``.

    The ``(1 + hits) / (1 + n)`` form counts the observed statistic itself as one
    valid draw from the null. This is the standard finite-sample correction; it
    keeps the p-value strictly positive, honestly reflecting that a finite Monte
    Carlo can never certify a probability of exactly zero.
    """
    n = null.size
    if alternative == "greater":
        hits = int(np.sum(null >= observed))
    elif alternative == "less":
        hits = int(np.sum(null <= observed))
    elif alternative == "two-sided":
        # Centre on the null mean and compare absolute deviations: a result is
        # "as extreme" if it is as far from typical luck as the observed value,
        # on either side.
        centre = float(np.mean(null))
        hits = int(np.sum(np.abs(null - centre) >= abs(observed - centre)))
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown alternative {alternative!r}")
    return (1 + hits) / (1 + n)


def permutation_test(
    positions: pd.Series,
    returns: pd.Series,
    statistic: Statistic = _default_statistic,
    *,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    alternative: Alternative = "greater",
    seed: int = DEFAULT_SEED,
) -> PermutationResult:
    """Monte-Carlo permutation test of a strategy's timing edge.

    A strategy earns money by holding the right *position* at the right *time*:
    its P&L each bar is ``position * return``. The null hypothesis is that the
    position carries no information about the contemporaneous return -- i.e. the
    pairing of positions to returns is arbitrary. We simulate that null directly
    by randomly permuting the positions against the (fixed, in-order) returns and
    recomputing the statistic. If the real strategy's score sits deep in the tail
    of this null distribution, luck rarely reproduces it and the edge is
    significant.

    Parameters
    ----------
    positions:
        The position actually *held* each bar -- typically
        :attr:`BacktestResult.positions`, which is already execution-lagged, so
        no look-ahead can leak in through this test.
    returns:
        The per-bar asset returns the positions were exposed to (e.g.
        ``close.pct_change()``). Must align to ``positions``.
    statistic:
        Score whose significance is tested, applied to the per-bar P&L series.
        Defaults to the annualised Sharpe ratio.
    n_permutations:
        Number of permutations forming the null distribution.
    alternative:
        Tail of interest; ``"greater"`` (default) asks how often luck *beats* the
        strategy.
    seed:
        Seed for reproducibility.

    Returns
    -------
    PermutationResult
        The observed score, its p-value and the empirical null distribution.

    Raises
    ------
    ValueError
        If the inputs do not align, are too short, or ``n_permutations`` is not
        positive.

    Notes
    -----
    Permuting positions destroys the time ordering of the P&L. For an
    order-invariant statistic like the Sharpe ratio (a function of the mean and
    standard deviation only) this is exactly what we want. A *path-dependent*
    statistic such as maximum drawdown would also have its path scrambled, so
    interpret such choices with care.
    """
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")

    pos = positions.astype(float)
    ret = returns.astype(float)
    if len(pos) != len(ret):
        raise ValueError(
            f"positions and returns must align (got {len(pos)} vs {len(ret)})"
        )
    if len(pos) < 2:
        raise ValueError("need at least two observations for a permutation test")

    pos_values = pos.to_numpy()
    ret_values = ret.to_numpy()

    observed = float(statistic(_as_series(pos_values * ret_values)))

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        shuffled = rng.permutation(pos_values)
        null[i] = statistic(_as_series(shuffled * ret_values))

    # Undefined null draws (NaN) are excluded from the comparison set; this only
    # happens for degenerate statistics and keeps the p-value well defined.
    null = null[np.isfinite(null)]
    p_value = _p_value_from_null(observed, null, alternative)

    return PermutationResult(
        observed=observed,
        p_value=p_value,
        n_permutations=n_permutations,
        alternative=alternative,
        null_distribution=null,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Multiple-testing / data-snooping corrections
# ---------------------------------------------------------------------------


def bonferroni_adjust(p_value: float, n_trials: int) -> float:
    """Bonferroni-correct a p-value for having tried ``n_trials`` strategies.

    If you searched ``n_trials`` strategies/parameters and reported the best, its
    raw p-value overstates significance. The Bonferroni bound multiplies it by
    ``n_trials`` (capped at 1.0). It controls the family-wise error rate under
    *any* dependence structure, which makes it conservative -- often too
    conservative when the trials are highly correlated -- but it is the simplest
    honest adjustment and a good first sanity check.

    Parameters
    ----------
    p_value:
        The raw, unadjusted p-value of the best result.
    n_trials:
        Number of strategies/parameter settings actually tried.

    Returns
    -------
    float
        The adjusted p-value, in ``[0, 1]``.
    """
    if not 0.0 <= p_value <= 1.0:
        raise ValueError("p_value must lie in [0, 1]")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    return min(1.0, p_value * n_trials)


def sidak_adjust(p_value: float, n_trials: int) -> float:
    """Šidák-correct a p-value for ``n_trials`` *independent* trials.

    The Šidák adjustment ``1 - (1 - p) ** n_trials`` is the exact family-wise
    correction when the trials are independent, and is marginally less
    conservative than Bonferroni. Independence rarely holds for nearby
    parameters, so treat it as the optimistic bookend to Bonferroni's
    pessimistic one; the truth lies between, and
    :func:`best_strategy_permutation_test` estimates it directly.

    Parameters
    ----------
    p_value:
        The raw, unadjusted p-value of the best result.
    n_trials:
        Number of trials.

    Returns
    -------
    float
        The adjusted p-value, in ``[0, 1]``.
    """
    if not 0.0 <= p_value <= 1.0:
        raise ValueError("p_value must lie in [0, 1]")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    return 1.0 - (1.0 - p_value) ** n_trials


@dataclass(frozen=True)
class MultipleTestingResult:
    """Outcome of a permutation test on the *best* of several strategies.

    Attributes
    ----------
    observed_best:
        The best (maximum) statistic across the candidate strategies.
    best_index:
        Index of the winning candidate in the input sequence.
    p_value:
        Probability that the *best of this many* candidates beats
        ``observed_best`` purely by luck, with the add-one correction. Because
        the null is the distribution of the maximum across candidates, this p
        already accounts for the multiplicity -- no separate Bonferroni step is
        needed.
    n_candidates:
        Number of candidate strategies compared.
    n_permutations:
        Number of permutations drawn.
    null_distribution:
        The empirical null of the best-across-candidates statistic.
    seed:
        Seed used, for reproducibility.
    """

    observed_best: float
    best_index: int
    p_value: float
    n_candidates: int
    n_permutations: int
    null_distribution: np.ndarray
    seed: int


def best_strategy_permutation_test(
    positions_list: Sequence[pd.Series],
    returns: pd.Series,
    statistic: Statistic = _default_statistic,
    *,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> MultipleTestingResult:
    """Permutation Reality Check: is the *best* of many strategies real?

    This is the direct, data-snooping-aware answer to "I tried many strategies
    and kept the best one". For each permutation we apply **the same** shuffle to
    every candidate's positions and record the maximum statistic across
    candidates. The resulting null is the distribution of the best score you
    would expect from this many candidates *if none had any edge*. Comparing the
    real best score against that null yields a multiplicity-corrected p-value --
    the spirit of White's Reality Check.

    Sharing one permutation across candidates (rather than shuffling each
    independently) preserves the contemporaneous correlation between the
    candidates' P&Ls. That matters: nearby parameter settings are highly
    correlated, and independent shuffles would inflate the null maximum and make
    the test too lenient.

    Parameters
    ----------
    positions_list:
        One execution-lagged position series per candidate strategy, each aligned
        to ``returns``.
    returns:
        Per-bar asset returns shared by all candidates.
    statistic:
        Score applied to each candidate's per-bar P&L. Defaults to the annualised
        Sharpe ratio.
    n_permutations:
        Number of shared permutations forming the null.
    seed:
        Seed for reproducibility.

    Returns
    -------
    MultipleTestingResult
        The best observed score, which candidate won, and the corrected p-value.

    Raises
    ------
    ValueError
        If no candidates are supplied, any candidate fails to align to
        ``returns``, or ``n_permutations`` is not positive.
    """
    if len(positions_list) == 0:
        raise ValueError("need at least one candidate strategy")
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")

    ret_values = returns.astype(float).to_numpy()
    n = len(ret_values)
    pos_matrix = np.empty((len(positions_list), n), dtype=float)
    for j, pos in enumerate(positions_list):
        if len(pos) != n:
            raise ValueError(
                f"candidate {j} does not align to returns " f"(got {len(pos)} vs {n})"
            )
        pos_matrix[j] = pos.astype(float).to_numpy()

    observed_scores = np.array(
        [
            statistic(_as_series(pos_matrix[j] * ret_values))
            for j in range(len(positions_list))
        ]
    )
    best_index = int(np.nanargmax(observed_scores))
    observed_best = float(observed_scores[best_index])

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        order = rng.permutation(n)
        permuted_returns = ret_values[order]
        # Same permutation applied to every candidate (via the shared return
        # ordering), then take the best score across candidates.
        scores = [
            statistic(_as_series(pos_matrix[j] * permuted_returns))
            for j in range(len(positions_list))
        ]
        null[i] = np.nanmax(scores)

    null = null[np.isfinite(null)]
    p_value = _p_value_from_null(observed_best, null, "greater")

    return MultipleTestingResult(
        observed_best=observed_best,
        best_index=best_index,
        p_value=p_value,
        n_candidates=len(positions_list),
        n_permutations=n_permutations,
        null_distribution=null,
        seed=seed,
    )
