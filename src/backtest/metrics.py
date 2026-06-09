"""Performance and risk metrics.

Every annualisation factor is defined as a named constant with the reasoning
spelled out, because a silently-wrong annualisation convention is one of the
quietest ways a backtest misleads. All functions operate on a series of
*periodic* (typically daily) simple returns unless stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Annualisation constants
# ---------------------------------------------------------------------------

# US equity markets trade on roughly 252 days per year (365.25 calendar days
# minus weekends and ~9 public holidays). 252 is the long-standing industry
# convention for annualising daily statistics; using 365 here would overstate
# annualised volatility by ~sqrt(365/252) ~ 1.20x.
TRADING_DAYS_PER_YEAR: int = 252

# Volatility scales with the square root of time under the i.i.d. assumption
# (variance is additive, so standard deviation grows as sqrt(n)). Hence daily
# volatility is annualised by multiplying by sqrt(252), not by 252.
VOL_ANNUALISATION_FACTOR: float = float(np.sqrt(TRADING_DAYS_PER_YEAR))

# Sample standard deviation uses ddof=1 (Bessel's correction) because a
# backtest return series is a *sample*, not the full population, of outcomes.
_DDOF: int = 1

# Standard deviations at or below this absolute floor are treated as zero. A
# genuinely constant series should have std exactly 0, but finite-precision
# arithmetic leaves ~1e-18 of noise; no real daily-return series has a
# standard deviation anywhere near 1e-12, so this cleanly separates "constant"
# (undefined Sharpe) from "real" without risking false positives.
_STD_FLOOR: float = 1e-12


@dataclass(frozen=True)
class DrawdownInfo:
    """Result of a maximum-drawdown calculation.

    Attributes
    ----------
    max_drawdown:
        The largest peak-to-trough decline, as a negative fraction (e.g.
        ``-0.25`` for a 25% drawdown). Zero if the equity never declines.
    peak_date:
        Index label of the peak preceding the worst trough.
    trough_date:
        Index label of the worst trough.
    duration:
        Number of bars from the peak to the trough (the time spent underwater
        on the way down to the worst point).
    """

    max_drawdown: float
    peak_date: object
    trough_date: object
    duration: int


def _to_returns(returns: pd.Series) -> pd.Series:
    """Coerce to a float return series, dropping a possible leading NaN."""
    return returns.astype(float).dropna()


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Compound ``returns`` into a wealth index starting at ``initial``."""
    return (1.0 + _to_returns(returns)).cumprod() * initial


def annualized_return(returns: pd.Series) -> float:
    """Compound annual growth rate (CAGR) implied by the return series.

    Computed from the *total* compounded return over the sample, annualised by
    the number of years the sample spans (``n_bars / 252``). This geometric
    definition is the rate that, compounded annually, reproduces the realised
    end-to-end wealth -- unlike a naive ``mean * 252`` which ignores
    compounding.
    """
    r = _to_returns(returns)
    if r.empty:
        return float("nan")
    total_growth = float((1.0 + r).prod())
    years = len(r) / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return float("nan")
    # A wiped-out account (total_growth <= 0) is a -100% annual return.
    if total_growth <= 0:
        return -1.0
    return total_growth ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series) -> float:
    """Annualised standard deviation of returns.

    Daily volatility times ``sqrt(252)`` (see
    :data:`VOL_ANNUALISATION_FACTOR`). Returns ``nan`` for fewer than two
    observations, where sample variance is undefined.
    """
    r = _to_returns(returns)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=_DDOF)) * VOL_ANNUALISATION_FACTOR


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sharpe ratio.

    Parameters
    ----------
    returns:
        Periodic simple returns.
    risk_free_rate:
        *Annual* risk-free rate (e.g. ``0.04`` for 4%). It is de-annualised to
        a per-period rate before being subtracted from each return, so the
        excess-return series is consistent with the periodicity of the data.
    periods_per_year:
        Number of periods per year for annualisation; defaults to 252 for
        daily data.

    Notes
    -----
    The ratio is ``mean(excess) / std(excess) * sqrt(periods_per_year)``. The
    ``sqrt`` annualisation follows from the same i.i.d. scaling that underlies
    :data:`VOL_ANNUALISATION_FACTOR`. Returns ``nan`` when the excess-return
    standard deviation is zero (a risk-free series has an undefined Sharpe).
    """
    r = _to_returns(returns)
    if len(r) < 2:
        return float("nan")
    per_period_rf = risk_free_rate / periods_per_year
    excess = r - per_period_rf
    std = excess.std(ddof=_DDOF)
    if std <= _STD_FLOOR:
        return float("nan")
    return float(excess.mean() / std) * float(np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> DrawdownInfo:
    """Maximum peak-to-trough decline of the compounded equity curve.

    Returns a :class:`DrawdownInfo` with the depth, the peak/trough dates and
    the peak-to-trough duration in bars. For a non-decreasing equity curve the
    drawdown is ``0.0``.
    """
    r = _to_returns(returns)
    if r.empty:
        return DrawdownInfo(0.0, None, None, 0)

    wealth = (1.0 + r).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0

    trough_pos = int(drawdown.to_numpy().argmin())
    trough_date = drawdown.index[trough_pos]
    max_dd = float(drawdown.iloc[trough_pos])

    if max_dd == 0.0:
        return DrawdownInfo(0.0, wealth.index[0], wealth.index[0], 0)

    # The relevant peak is the highest equity at or before the trough.
    peak_pos = int(wealth.iloc[: trough_pos + 1].to_numpy().argmax())
    peak_date = wealth.index[peak_pos]
    duration = trough_pos - peak_pos

    return DrawdownInfo(max_dd, peak_date, trough_date, duration)


def win_rate(returns: pd.Series) -> float:
    """Fraction of non-zero return periods that are positive.

    Flat periods (exactly zero return -- typically while out of the market) are
    excluded so the statistic measures the quality of *active* bets rather than
    being diluted by time spent flat. Returns ``nan`` if there are no non-zero
    periods.
    """
    r = _to_returns(returns)
    active = r[r != 0.0]
    if active.empty:
        return float("nan")
    return float((active > 0).mean())


def profit_loss_ratio(returns: pd.Series) -> float:
    """Average winning return divided by the absolute average losing return.

    Also known as the payoff ratio. A value of ``2.0`` means the average win
    is twice the size of the average loss. Returns ``nan`` if there are no
    losers (the ratio would be infinite) or no winners.
    """
    r = _to_returns(returns)
    wins = r[r > 0.0]
    losses = r[r < 0.0]
    if wins.empty or losses.empty:
        return float("nan")
    return float(wins.mean() / abs(losses.mean()))


def summary(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Compute the full metric suite as a flat dictionary.

    Convenient for building summary tables. The drawdown depth is unpacked to a
    scalar under ``max_drawdown``; its dates/duration remain available via
    :func:`max_drawdown` directly.
    """
    dd = max_drawdown(returns)
    return {
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": dd.max_drawdown,
        "max_drawdown_duration": float(dd.duration),
        "win_rate": win_rate(returns),
        "profit_loss_ratio": profit_loss_ratio(returns),
    }
