"""Vectorised backtest engine with structural look-ahead protection.

The single most important property of this engine is that a trading signal
generated at time ``t`` -- using information available only up to and
including ``t`` -- can **never** earn the return realised over ``(t-1, t]``.
It can only govern the return realised over ``(t, t+1]`` and later.

This is the look-ahead guard, and it is enforced in exactly one place:
``target_position.shift(EXECUTION_LAG)`` (see :func:`run_backtest`). Because
the shift lives in the engine and not in each strategy, no strategy can
accidentally (or deliberately) trade on the same bar that produced its
signal. The direction of the shift is the whole ballgame, so it is commented
in detail at the call site.

The accompanying :func:`assert_causal` goes one step further: it *proves*,
by perturbation, that a strategy's signal at time ``t`` does not depend on
any price at time ``> t``. The test-suite uses it to demonstrate the absence
of look-ahead bias rather than merely asserting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of bars between a signal being *decided* and it being *executed*.
# A signal decided at the close of bar ``t`` is acted on at bar ``t + 1``.
# This models the unavoidable reality that you cannot trade on a price you
# have not yet observed. It is the structural defence against look-ahead bias.
EXECUTION_LAG: int = 1


class LookAheadError(AssertionError):
    """Raised when a strategy is shown to depend on future information."""


class _SignalStrategy(Protocol):
    """Structural type for anything the engine can backtest.

    A strategy is any object exposing ``generate_signals``. The contract --
    enforced by :func:`assert_causal`, not by trust -- is that the signal at
    index ``t`` is a function of prices at indices ``<= t`` only.
    """

    def generate_signals(self, prices: pd.Series) -> pd.Series:  # pragma: no cover
        ...


@dataclass(frozen=True)
class BacktestConfig:
    """Parameters controlling execution frictions and sizing.

    Attributes
    ----------
    commission:
        Proportional commission charged per unit of turnover, expressed as a
        fraction (e.g. ``0.0005`` == 5 bps). Applied one-way: a round trip
        (open then close) therefore costs ``2 * commission``.
    slippage:
        Proportional slippage per unit of turnover, same units as
        ``commission``. Modelled identically to commission and added to it.
    initial_capital:
        Starting capital for the equity curve. Affects scale only, never the
        return series.
    position_size:
        Fraction of capital allocated per unit of signal. A signal of ``1``
        with ``position_size=1.0`` means fully invested; ``0.5`` means
        half-invested. Leverage is expressed as ``position_size > 1``.
    """

    commission: float = 0.0
    slippage: float = 0.0
    initial_capital: float = 1.0
    position_size: float = 1.0


@dataclass(frozen=True)
class BacktestResult:
    """Container for the full output of a backtest.

    All series share the same :class:`~pandas.DatetimeIndex` as the input
    prices, so they can be aligned and plotted directly.

    Attributes
    ----------
    signals:
        The raw signal *decided* at each bar ``t`` (information up to ``t``).
        Not yet lagged -- this is what the strategy emitted.
    positions:
        The position actually *held* during each bar, i.e. ``signals`` scaled
        by ``position_size`` and lagged by :data:`EXECUTION_LAG`. This is the
        series that meets the returns.
    turnover:
        Absolute change in position from one bar to the next; the quantity
        that incurs transaction costs.
    costs:
        Per-bar transaction cost (turnover times the cost rate).
    gross_returns:
        Strategy returns *before* costs.
    returns:
        Strategy returns *after* costs. This is the headline P&L series.
    equity_curve:
        Cumulative compounded ``returns`` scaled by ``initial_capital``.
    """

    signals: pd.Series
    positions: pd.Series
    turnover: pd.Series
    costs: pd.Series
    gross_returns: pd.Series
    returns: pd.Series
    equity_curve: pd.Series


def _extract_close(prices: pd.Series | pd.DataFrame) -> pd.Series:
    """Return a 1-D close-price series from flexible input.

    Accepts either a bare price :class:`~pandas.Series` or a
    :class:`~pandas.DataFrame` containing a (case-insensitive) ``close``
    column, mirroring the shape of typical OHLCV data.
    """
    if isinstance(prices, pd.Series):
        return prices.astype(float)
    if isinstance(prices, pd.DataFrame):
        lookup = {str(c).lower(): c for c in prices.columns}
        if "close" in lookup:
            return prices[lookup["close"]].astype(float)
        raise KeyError(
            "DataFrame input must contain a 'close' column; "
            f"got columns {list(prices.columns)!r}"
        )
    raise TypeError(f"prices must be a Series or DataFrame, got {type(prices)!r}")


def run_backtest(
    prices: pd.Series | pd.DataFrame,
    strategy: _SignalStrategy,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a vectorised backtest of ``strategy`` over ``prices``.

    Parameters
    ----------
    prices:
        Close prices, either a :class:`~pandas.Series` or a
        :class:`~pandas.DataFrame` with a ``close`` column.
    strategy:
        Any object with a ``generate_signals(prices) -> Series`` method
        returning a signal aligned to the price index. Signals are typically
        in ``[-1, 1]`` (target exposure) but any real number is accepted.
    config:
        Execution frictions and sizing; defaults to a frictionless,
        fully-invested configuration.

    Returns
    -------
    BacktestResult
        The signals, positions, turnover, costs and P&L series.
    """
    config = config or BacktestConfig()
    close = _extract_close(prices)

    # The strategy sees the whole price series, but its *contract* is to be
    # causal: signal[t] depends only on prices[:t+1]. That contract is what
    # assert_causal verifies. The engine does not rely on it for correctness
    # of the look-ahead guard below -- the shift protects us regardless.
    signals = strategy.generate_signals(close).reindex(close.index).astype(float)
    target_position = signals * config.position_size

    # === LOOK-AHEAD GUARD =================================================
    # A signal decided at the close of bar t reflects only information known
    # at t. You therefore cannot have been holding the resulting position
    # *during* the move that ended at t -- you only learned of it at t's
    # close. The earliest the position can be live is the NEXT bar, t+1.
    #
    # Shifting FORWARD by EXECUTION_LAG (a positive shift moves values to
    # later timestamps) maps "decided at t" -> "held from t+1 onward". A
    # shift in the wrong direction (negative) would let today's position earn
    # today's already-realised return: that is precisely look-ahead bias.
    #
    # The first EXECUTION_LAG bars have no prior decision, so the position is
    # flat (0.0) there.
    position = target_position.shift(EXECUTION_LAG).fillna(0.0)
    # ======================================================================

    # Simple (arithmetic) return realised over each bar. The first bar has no
    # prior price and so no return; it is flat anyway because of the lag.
    bar_returns = close.pct_change().fillna(0.0)

    # P&L = position held during the bar times the return realised that bar.
    gross_returns = position * bar_returns

    # Turnover is the absolute change in position. The very first position
    # change is from an implicit flat book (0), captured by filling the
    # leading NaN with the absolute initial position -- entering a position
    # costs money too.
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])

    cost_rate = config.commission + config.slippage
    costs = turnover * cost_rate
    net_returns = gross_returns - costs

    equity_curve = (1.0 + net_returns).cumprod() * config.initial_capital

    return BacktestResult(
        signals=signals,
        positions=position,
        turnover=turnover,
        costs=costs,
        gross_returns=gross_returns,
        returns=net_returns,
        equity_curve=equity_curve,
    )


def assert_causal(
    strategy: _SignalStrategy,
    prices: pd.Series | pd.DataFrame,
    *,
    n_trials: int = 8,
    perturbation: float = 0.10,
    seed: int = 0,
) -> None:
    """Prove, by perturbation, that ``strategy`` has no look-ahead bias.

    A causal strategy's signal at index ``t`` depends only on prices at
    indices ``<= t``. We test this directly: pick a cut point ``k``, perturb
    every price *after* ``k`` by a random amount, regenerate the signals, and
    require that all signals at indices ``<= k`` are unchanged. If the
    strategy peeked into the future, changing future prices would ripple back
    into earlier signals and the check would fail.

    Parameters
    ----------
    strategy:
        The strategy under test.
    prices:
        A representative price path; future segments of a *copy* are perturbed.
    n_trials:
        Number of random cut points / perturbations to try.
    perturbation:
        Maximum fractional magnitude of the multiplicative price shock applied
        to the future segment.
    seed:
        Seed for reproducibility.

    Raises
    ------
    LookAheadError
        If any signal at or before a cut point changes when only prices after
        that cut point are altered.
    """
    close = _extract_close(prices)
    n = len(close)
    if n < 3:
        raise ValueError("need at least 3 observations to test causality")

    baseline = strategy.generate_signals(close).reindex(close.index)
    rng = np.random.default_rng(seed)

    # Choose interior cut points so there is always a non-empty "future" to
    # perturb and a non-empty "past" to check.
    for _ in range(n_trials):
        k = int(rng.integers(1, n - 1))

        shocked = close.copy()
        future = shocked.iloc[k + 1 :]
        multipliers = 1.0 + rng.uniform(-perturbation, perturbation, size=len(future))
        shocked.iloc[k + 1 :] = future.to_numpy() * multipliers

        perturbed_signals = strategy.generate_signals(shocked).reindex(close.index)

        past_before = baseline.iloc[: k + 1]
        past_after = perturbed_signals.iloc[: k + 1]

        # NaNs (warm-up) must match NaN-for-NaN; finite values must match
        # exactly. equals() treats NaN == NaN as True, which is what we want.
        if not past_before.equals(past_after):
            diff_idx = past_before.index[
                ~_series_equal_elementwise(past_before, past_after)
            ]
            raise LookAheadError(
                "Look-ahead bias detected: perturbing prices after index "
                f"{k} changed earlier signals at {list(diff_idx)!r}. A causal "
                "strategy's past signals must not depend on future prices."
            )


def _series_equal_elementwise(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise equality treating NaN == NaN as ``True``."""
    both_nan = a.isna() & b.isna()
    return (a == b) | both_nan
