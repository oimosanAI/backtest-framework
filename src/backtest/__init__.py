"""A backtesting framework focused on the *rigour* of validation.

The guiding question of this project is not "is this strategy impressive?"
but "can this backtest be trusted?". Every module is designed so that the
classic ways a backtest lies to you -- look-ahead bias above all -- are
avoided *structurally* rather than by convention.

The three pillars of a *believable* backtest, each defending against a
specific lie a backtest can tell:

1. **Look-ahead bias** -- defused structurally in :mod:`~src.backtest.engine`
   and *proven* absent by :func:`~src.backtest.engine.assert_causal`.
2. **Statistical significance** -- :mod:`~src.backtest.significance` asks whether
   an edge could be mere luck (permutation test), how precise the numbers are
   (bootstrap), and whether trying many strategies fooled us (multiple testing).
3. **Over-fitting** -- :mod:`~src.backtest.walkforward` separates in-sample fit
   from out-of-sample reality and measures the gap between them.

Public API
----------
- :class:`~src.backtest.engine.BacktestConfig`
- :class:`~src.backtest.engine.BacktestResult`
- :func:`~src.backtest.engine.run_backtest`
- :func:`~src.backtest.engine.assert_causal`
- :class:`~src.backtest.strategies.MovingAverageCrossStrategy`
- :func:`~src.backtest.significance.bootstrap_statistic`
- :func:`~src.backtest.significance.permutation_test`
- :func:`~src.backtest.significance.best_strategy_permutation_test`
- :func:`~src.backtest.walkforward.run_walkforward`
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
from src.backtest.significance import (
    BootstrapResult,
    MultipleTestingResult,
    PermutationResult,
    best_strategy_permutation_test,
    bonferroni_adjust,
    bootstrap_statistic,
    permutation_test,
    sidak_adjust,
)
from src.backtest.strategies import MovingAverageCrossStrategy, Strategy
from src.backtest.walkforward import (
    WalkForwardResult,
    WalkForwardWindow,
    WindowSplit,
    generate_windows,
    run_walkforward,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "LookAheadError",
    "assert_causal",
    "run_backtest",
    "MovingAverageCrossStrategy",
    "Strategy",
    "BootstrapResult",
    "PermutationResult",
    "MultipleTestingResult",
    "bootstrap_statistic",
    "permutation_test",
    "best_strategy_permutation_test",
    "bonferroni_adjust",
    "sidak_adjust",
    "WindowSplit",
    "WalkForwardWindow",
    "WalkForwardResult",
    "generate_windows",
    "run_walkforward",
]
