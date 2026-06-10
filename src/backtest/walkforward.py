"""Walk-forward analysis: the structural test against over-fitting.

The most seductive lie a backtest tells is the *over-fitted* one: a parameter
set tuned on the very data it is then evaluated on. Such a strategy looks
brilliant in-sample and falls apart the moment it meets data it was not fitted
to. Walk-forward analysis exposes this by construction. It repeatedly:

1. fits/selects parameters on an **in-sample** window (the past), then
2. evaluates the chosen parameters on the **out-of-sample** window that comes
   *strictly after* it (the future the strategy had not yet seen),

sliding both windows forward through time. The gap between in-sample and
out-of-sample performance *is* the over-fitting, measured directly.

Two windowing schemes are provided:

- **Anchored** (expanding): the in-sample window always starts at the beginning
  and grows, mimicking a researcher who re-fits on all history to date.
- **Rolling** (fixed): the in-sample window is a constant width that slides
  forward, mimicking a strategy that adapts to a moving, recent regime.

**Causality within every window.** Parameter selection on a window touches only
that window's in-sample slice; the out-of-sample slice that scores it lies
entirely later in time and is never consulted during selection. The underlying
:func:`~src.backtest.engine.run_backtest` still enforces the one-bar execution
lag inside each slice, so the look-ahead guarantee of the base engine holds
unchanged here -- out-of-sample information cannot leak into the in-sample fit.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestConfig, _SignalStrategy, run_backtest
from src.backtest.metrics import sharpe_ratio

# A *score* ranks candidate strategies on a return series; the best in-sample
# score is selected and then measured out-of-sample. Defaults to the Sharpe
# ratio, the standard risk-adjusted yardstick.
Score = Callable[[pd.Series], float]

# Windowing scheme: an expanding "anchored" in-sample window, or a fixed-width
# "rolling" one. See the module docstring for the modelling intuition.
WindowMode = Literal["anchored", "rolling"]


def _default_score(returns: pd.Series) -> float:
    """Annualised Sharpe ratio -- the default selection/evaluation score."""
    return sharpe_ratio(returns)


@dataclass(frozen=True)
class WindowSplit:
    """A single train/test split, expressed as half-open integer position ranges.

    All four fields are integer positions into the price series, using Python's
    half-open ``[start, end)`` convention. The defining invariants -- verified in
    :func:`generate_windows` and relied on for the no-leak guarantee -- are::

        train_start < train_end == test_start < test_end

    so the out-of-sample range begins exactly where the in-sample range ends and
    never overlaps it: the test data is always strictly in the future of the
    train data.

    Attributes
    ----------
    train_start, train_end:
        Half-open position range of the in-sample (training) window.
    test_start, test_end:
        Half-open position range of the out-of-sample (test) window.
    """

    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def __post_init__(self) -> None:
        # These are guaranteed by the generator, but a frozen split that violates
        # them would silently corrupt every downstream guarantee, so we assert.
        if not (self.train_start < self.train_end == self.test_start < self.test_end):
            raise ValueError(
                "invalid window split: require "
                "train_start < train_end == test_start < test_end, got "
                f"{(self.train_start, self.train_end, self.test_start, self.test_end)}"
            )


def generate_windows(
    n: int,
    *,
    train_size: int,
    test_size: int,
    mode: WindowMode = "rolling",
    step: int | None = None,
) -> list[WindowSplit]:
    """Generate the sequence of walk-forward train/test splits.

    Parameters
    ----------
    n:
        Total number of observations available.
    train_size:
        Length of the in-sample window. For ``mode="anchored"`` this is the
        *initial* in-sample length; the window then expands by ``step`` each
        fold. For ``mode="rolling"`` it is the fixed in-sample length.
    test_size:
        Length of each out-of-sample window.
    mode:
        ``"rolling"`` (fixed-width in-sample) or ``"anchored"`` (expanding
        in-sample anchored at position 0).
    step:
        How far the scheme advances each fold. Defaults to ``test_size``, which
        tiles the out-of-sample windows end-to-end with no gaps or overlaps --
        the canonical walk-forward layout where every out-of-sample bar is used
        exactly once.

    Returns
    -------
    list[WindowSplit]
        Splits in chronological order. Empty only if the data is too short to
        form even one split.

    Raises
    ------
    ValueError
        If any size argument is non-positive.
    """
    if train_size < 1 or test_size < 1:
        raise ValueError("train_size and test_size must be >= 1")
    advance = test_size if step is None else step
    if advance < 1:
        raise ValueError("step must be >= 1")

    splits: list[WindowSplit] = []
    fold = 0
    while True:
        if mode == "anchored":
            # In-sample always starts at 0 and grows by `advance` each fold.
            train_start = 0
            train_end = train_size + fold * advance
        elif mode == "rolling":
            # In-sample is a fixed window sliding forward by `advance` each fold.
            train_start = fold * advance
            train_end = train_start + train_size
        else:  # pragma: no cover - guarded by the Literal type
            raise ValueError(f"unknown mode {mode!r}")

        test_start = train_end
        test_end = test_start + test_size
        if test_end > n:
            # The out-of-sample window would run past the available data; stop.
            break

        splits.append(
            WindowSplit(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold += 1

    return splits


@dataclass(frozen=True)
class WalkForwardWindow:
    """Result of one walk-forward fold: what was chosen and how it then did.

    Attributes
    ----------
    split:
        The integer-position :class:`WindowSplit` for this fold.
    train_start_date, train_end_date, test_start_date, test_end_date:
        The corresponding index labels (e.g. timestamps) from the price series,
        carried through so a fold can be reported and plotted in real time. Note
        ``*_end_date`` are *inclusive* labels of the last bar in each half-open
        range, which is what a human wants to read.
    best_strategy_name:
        ``name`` of the candidate selected as best in-sample on this fold.
    best_index:
        Index of that candidate within the ``candidates`` sequence.
    in_sample_score:
        The selected candidate's score *on the in-sample window* (the number the
        selection optimised, hence optimistic).
    out_of_sample_score:
        The same candidate's score on the out-of-sample window (the honest,
        unfitted number).
    """

    split: WindowSplit
    train_start_date: object
    train_end_date: object
    test_start_date: object
    test_end_date: object
    best_strategy_name: str
    best_index: int
    in_sample_score: float
    out_of_sample_score: float


@dataclass(frozen=True)
class WalkForwardResult:
    """Aggregated walk-forward analysis across all folds.

    Attributes
    ----------
    windows:
        Per-fold :class:`WalkForwardWindow` results, in chronological order.
    mode:
        The windowing scheme used.
    in_sample_scores, out_of_sample_scores:
        Convenience arrays of the per-fold scores, aligned to ``windows``.
    """

    windows: list[WalkForwardWindow]
    mode: WindowMode
    in_sample_scores: np.ndarray
    out_of_sample_scores: np.ndarray

    @property
    def mean_in_sample(self) -> float:
        """Average in-sample score across folds (ignoring undefined folds)."""
        return (
            float(np.nanmean(self.in_sample_scores)) if self.windows else float("nan")
        )

    @property
    def mean_out_of_sample(self) -> float:
        """Average out-of-sample score across folds (ignoring undefined folds)."""
        return (
            float(np.nanmean(self.out_of_sample_scores))
            if self.windows
            else float("nan")
        )

    @property
    def degradation(self) -> float:
        """In-sample minus out-of-sample mean score -- the over-fitting gap.

        A large positive value is the signature of over-fitting: the strategy
        looked good only on the data it was tuned on. A value near zero means the
        in-sample edge survived contact with unseen data.
        """
        return self.mean_in_sample - self.mean_out_of_sample

    @property
    def efficiency(self) -> float:
        """Out-of-sample mean as a fraction of in-sample mean (the WFE ratio).

        The walk-forward efficiency: ``1.0`` means out-of-sample fully matched
        in-sample; values well below ``1.0`` quantify decay. ``nan`` when the
        in-sample mean is ~zero (the ratio is then undefined).
        """
        denom = self.mean_in_sample
        if not np.isfinite(denom) or abs(denom) < _EFFICIENCY_FLOOR:
            return float("nan")
        return self.mean_out_of_sample / denom


# An in-sample mean Sharpe smaller in magnitude than this is treated as "zero"
# for the efficiency ratio: dividing by a near-zero baseline yields a wild,
# meaningless number, so we return nan instead of a spurious ratio.
_EFFICIENCY_FLOOR: float = 1e-9


def _score_returns(returns: pd.Series, score: Score) -> float:
    """Apply ``score`` to a return series, mapping an undefined result to nan."""
    value = score(returns)
    return float(value)


def run_walkforward(
    prices: pd.Series | pd.DataFrame,
    candidates: Sequence[_SignalStrategy],
    *,
    train_size: int,
    test_size: int,
    mode: WindowMode = "rolling",
    step: int | None = None,
    score: Score = _default_score,
    config: BacktestConfig | None = None,
) -> WalkForwardResult:
    """Run a walk-forward analysis selecting among ``candidates`` each fold.

    For every fold produced by :func:`generate_windows` the procedure is:

    1. **Select in-sample.** Each candidate is backtested on the in-sample slice
       alone and scored; the highest-scoring candidate is chosen. Only
       in-sample data is touched here, so the choice cannot peek at the future.
    2. **Evaluate out-of-sample.** The chosen candidate is backtested over the
       in-sample-through-test span and scored *only on the out-of-sample tail*.
       Running through the in-sample span lets indicators warm up on data that
       was genuinely available by then (legitimately *past* information), while
       the score reflects exclusively the unseen out-of-sample bars.

    The difference between the in-sample and out-of-sample scores, aggregated
    over folds, is the over-fitting measure exposed on
    :class:`WalkForwardResult`.

    Parameters
    ----------
    prices:
        Close prices as a :class:`~pandas.Series` or a ``close``-bearing
        :class:`~pandas.DataFrame`, exactly as :func:`run_backtest` accepts.
    candidates:
        The parameter grid expressed as concrete strategy objects (each with a
        ``name`` and ``generate_signals``). The best is selected per fold.
    train_size, test_size, mode, step:
        Forwarded to :func:`generate_windows`.
    score:
        Ranking/evaluation score on a return series; defaults to Sharpe.
    config:
        Execution frictions/sizing passed to every backtest; defaults to the
        frictionless :class:`~src.backtest.engine.BacktestConfig`.

    Returns
    -------
    WalkForwardResult
        Per-fold selections and scores plus aggregate over-fitting statistics.

    Raises
    ------
    ValueError
        If ``candidates`` is empty or the data is too short to form any fold.

    Notes
    -----
    A candidate whose warm-up exceeds a slice produces an all-flat (zero) return
    series and hence an undefined score; such a candidate simply cannot win that
    fold's selection. If *every* candidate is undefined in-sample on a fold, that
    fold's scores are ``nan`` and are skipped by the aggregate means.
    """
    if len(candidates) == 0:
        raise ValueError("need at least one candidate strategy")

    config = config or BacktestConfig()
    close = _as_close_series(prices)
    n = len(close)

    splits = generate_windows(
        n, train_size=train_size, test_size=test_size, mode=mode, step=step
    )
    if not splits:
        raise ValueError(
            "price series too short for any walk-forward fold "
            f"(n={n}, train_size={train_size}, test_size={test_size})"
        )

    windows: list[WalkForwardWindow] = []
    in_scores: list[float] = []
    out_scores: list[float] = []

    for split in splits:
        # --- 1. In-sample selection (sees only the training slice) -----------
        train_slice = close.iloc[split.train_start : split.train_end]
        best_index = -1
        best_in_score = -np.inf
        for j, candidate in enumerate(candidates):
            result = run_backtest(train_slice, candidate, config)
            candidate_score = _score_returns(result.returns, score)
            # NaN never compares true here, so an undefined candidate is never
            # selected -- exactly the desired behaviour.
            if candidate_score > best_in_score:
                best_in_score = candidate_score
                best_index = j

        if best_index < 0:
            # Every candidate was undefined in-sample on this fold: record nan
            # and move on rather than fabricate a selection.
            in_sample_score = float("nan")
            out_of_sample_score = float("nan")
            chosen_index = 0
            chosen = candidates[0]
        else:
            chosen_index = best_index
            chosen = candidates[best_index]
            in_sample_score = float(best_in_score)

            # --- 2. Out-of-sample evaluation (params already fixed) ----------
            # Run across train_start..test_end so indicators warm up on data the
            # strategy legitimately knew by then, but score ONLY the test tail.
            eval_slice = close.iloc[split.train_start : split.test_end]
            eval_result = run_backtest(eval_slice, chosen, config)
            oos_len = split.test_end - split.test_start
            oos_returns = eval_result.returns.iloc[-oos_len:]
            out_of_sample_score = _score_returns(oos_returns, score)

        windows.append(
            WalkForwardWindow(
                split=split,
                train_start_date=close.index[split.train_start],
                train_end_date=close.index[split.train_end - 1],
                test_start_date=close.index[split.test_start],
                test_end_date=close.index[split.test_end - 1],
                best_strategy_name=getattr(chosen, "name", repr(chosen)),
                best_index=chosen_index,
                in_sample_score=in_sample_score,
                out_of_sample_score=out_of_sample_score,
            )
        )
        in_scores.append(in_sample_score)
        out_scores.append(out_of_sample_score)

    return WalkForwardResult(
        windows=windows,
        mode=mode,
        in_sample_scores=np.array(in_scores, dtype=float),
        out_of_sample_scores=np.array(out_scores, dtype=float),
    )


def _as_close_series(prices: pd.Series | pd.DataFrame) -> pd.Series:
    """Coerce price input to a close Series, reusing the engine's extractor.

    Kept as a thin wrapper so the walk-forward layer accepts exactly the same
    inputs as :func:`run_backtest` without duplicating the extraction logic.
    """
    from src.backtest.engine import _extract_close

    return _extract_close(prices)
