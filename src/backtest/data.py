"""Price-data acquisition and sanity checking via yfinance.

Data handling is where a backtest quietly inherits other people's mistakes:
unadjusted splits, silent gaps, duplicated timestamps. This module fetches
adjusted close prices and runs explicit, fail-loud checks rather than trusting
the feed.
"""

from __future__ import annotations

import pandas as pd

# Default column we treat as "the price". We request auto-adjusted data from
# yfinance, so the 'Close' column is already split- and dividend-adjusted;
# this is the series a long-horizon backtest should compound.
CLOSE_COLUMN: str = "Close"


class DataError(RuntimeError):
    """Raised when price data cannot be fetched or fails validation."""


def load_prices(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    *,
    auto_adjust: bool = True,
) -> pd.Series:
    """Download adjusted close prices for ``ticker`` from Yahoo! Finance.

    Parameters
    ----------
    ticker:
        Symbol to fetch, e.g. ``"SPY"``.
    start, end:
        ISO date strings (``"2015-01-01"``) bounding the request. ``None``
        lets yfinance choose its defaults (max available / today).
    auto_adjust:
        If ``True`` (default) prices are adjusted for splits and dividends, so
        the close column is directly compoundable. Disabling this is almost
        always a mistake for a total-return backtest and is exposed only for
        completeness.

    Returns
    -------
    pandas.Series
        A clean, ascending, gap-free (no NaN) close-price series named after
        the ticker.

    Raises
    ------
    DataError
        If the download fails, returns nothing, or fails validation.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise DataError(
            "yfinance is required for live data downloads; install it with "
            "`pip install yfinance`."
        ) from exc

    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
            progress=False,
            # One ticker at a time keeps the column layout flat and avoids the
            # MultiIndex surprises of multi-ticker downloads.
            multi_level_index=False,
        )
    except Exception as exc:  # network/parse errors surface from deep inside
        raise DataError(f"failed to download data for {ticker!r}: {exc}") from exc

    if raw is None or raw.empty:
        raise DataError(
            f"no data returned for {ticker!r} in range {start}..{end}; "
            "check the symbol and date range."
        )
    if CLOSE_COLUMN not in raw.columns:
        raise DataError(
            f"expected a {CLOSE_COLUMN!r} column for {ticker!r}; "
            f"got {list(raw.columns)!r}"
        )

    close = raw[CLOSE_COLUMN]
    return clean_prices(close, name=ticker)


def clean_prices(prices: pd.Series, name: str | None = None) -> pd.Series:
    """Validate and normalise a raw close-price series.

    The checks are intentionally strict and fail loudly:

    - sorts the index ascending (yfinance is usually sorted, but never trust),
    - drops duplicated timestamps, keeping the last observation,
    - removes leading/trailing NaNs and asserts no interior NaNs remain,
    - asserts all prices are strictly positive (a non-positive price breaks
      ``pct_change`` and signals corrupt data).

    Parameters
    ----------
    prices:
        Raw close-price series.
    name:
        Optional name to assign to the returned series.

    Raises
    ------
    DataError
        If the series has no valid observations, interior gaps remain, or any
        price is non-positive.
    """
    cleaned = prices.astype(float).sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]

    # An empty or all-NaN series has no valid prices at all; report that
    # directly rather than falling through to the "interior gap" message.
    if cleaned.first_valid_index() is None:
        raise DataError("price series contains no valid (non-NaN) observations.")

    # Trim NaNs at the ends (common when a ticker's history starts mid-range),
    # then any remaining NaN is an interior gap we refuse to silently fill.
    cleaned = cleaned.loc[cleaned.first_valid_index() : cleaned.last_valid_index()]
    if cleaned.isna().any():
        n_gaps = int(cleaned.isna().sum())
        raise DataError(
            f"{n_gaps} interior gap(s) (NaN) found in price series; refusing to "
            "silently fill -- inspect the source data."
        )
    if (cleaned <= 0).any():
        raise DataError("non-positive prices found; data is corrupt.")

    if name is not None:
        cleaned.name = name
    return cleaned
