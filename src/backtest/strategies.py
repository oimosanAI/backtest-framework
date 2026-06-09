"""Strategy definitions, abstracted as causal signal generators.

A *strategy* here is anything that turns a price series into a signal series
of target exposures. The abstraction is deliberately thin -- a single
``generate_signals`` method -- so strategies are trivially swappable in the
engine and, crucially, so the causality contract can be verified uniformly by
:func:`src.backtest.engine.assert_causal`.

The contract every strategy must honour:

    The signal at index ``t`` is a function of prices at indices ``<= t``
    only. It must never look into the future.

The moving-average cross below honours this because ``Series.rolling`` only
ever aggregates the current and *past* observations within its window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

# Signal values. Kept as named constants so the meaning of the magic numbers
# is explicit at every use site.
LONG: float = 1.0
FLAT: float = 0.0
SHORT: float = -1.0


@runtime_checkable
class Strategy(Protocol):
    """Structural type for a backtestable strategy."""

    name: str

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        """Return target-exposure signals aligned to ``prices.index``."""
        ...


@dataclass(frozen=True)
class MovingAverageCrossStrategy:
    """Classic dual moving-average crossover.

    Go long when the fast moving average is above the slow moving average,
    and (optionally) short when it is below. During the warm-up period, before
    the slow average is defined, the strategy is flat.

    Parameters
    ----------
    fast_window:
        Look-back length of the fast simple moving average, in bars.
    slow_window:
        Look-back length of the slow simple moving average, in bars. Must be
        strictly greater than ``fast_window`` for the crossover to be
        meaningful.
    long_only:
        If ``True``, the down-cross produces a flat position instead of a
        short. Useful for cash-equity universes where shorting is restricted.
    """

    fast_window: int
    slow_window: int
    long_only: bool = False

    def __post_init__(self) -> None:
        if self.fast_window < 1 or self.slow_window < 1:
            raise ValueError("moving-average windows must be >= 1")
        if self.fast_window >= self.slow_window:
            raise ValueError(
                "fast_window must be strictly less than slow_window "
                f"(got fast={self.fast_window}, slow={self.slow_window})"
            )

    @property
    def name(self) -> str:
        """Human-readable identifier, e.g. ``MA(20,50)``."""
        suffix = ",long-only" if self.long_only else ""
        return f"MA({self.fast_window},{self.slow_window}{suffix})"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        """Produce crossover signals.

        Why this is causal: ``rolling(window).mean()`` at index ``t`` averages
        observations ``t-window+1 .. t`` -- strictly the present and the past.
        No future price can influence the average at ``t``, so no future price
        can influence the signal at ``t``.
        """
        prices = prices.astype(float)
        # min_periods defaults to the window length, so the averages are NaN
        # until enough history exists. That NaN warm-up is what keeps us flat
        # before the strategy has a defined opinion.
        fast = prices.rolling(self.fast_window).mean()
        slow = prices.rolling(self.slow_window).mean()

        signal = pd.Series(FLAT, index=prices.index, dtype=float)
        signal[fast > slow] = LONG
        signal[fast < slow] = SHORT if not self.long_only else FLAT

        # Any bar where either average is undefined (warm-up) is forced flat,
        # overriding the comparisons above (NaN comparisons are False anyway,
        # but we make the intent explicit).
        warmup = fast.isna() | slow.isna()
        signal[warmup] = FLAT
        return signal
