"""A backtesting framework focused on the *rigour* of validation.

The guiding question of this project is not "is this strategy impressive?"
but "can this backtest be trusted?". Every module is designed so that the
classic ways a backtest lies to you -- look-ahead bias above all -- are
avoided *structurally* rather than by convention.

Public API
----------
- :class:`~src.backtest.engine.BacktestConfig`
- :class:`~src.backtest.engine.BacktestResult`
- :func:`~src.backtest.engine.run_backtest`
- :func:`~src.backtest.engine.assert_causal`
- :class:`~src.backtest.strategies.MovingAverageCrossStrategy`
- :mod:`~src.backtest.metrics`
- :mod:`~src.backtest.data`
"""

from src.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    LookAheadError,
    assert_causal,
    run_backtest,
)
from src.backtest.strategies import MovingAverageCrossStrategy, Strategy

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "LookAheadError",
    "assert_causal",
    "run_backtest",
    "MovingAverageCrossStrategy",
    "Strategy",
]
