"""Tests for data validation and the download wrapper (no network access).

The live download path is exercised against a fake ``yfinance`` injected via
``monkeypatch`` so the suite stays deterministic and offline.
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from src.backtest import data


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2023-01-01", periods=n, freq="B")


class TestCleanPrices:
    def test_sorts_and_dedupes(self) -> None:
        idx = pd.to_datetime(["2023-01-03", "2023-01-01", "2023-01-02", "2023-01-02"])
        raw = pd.Series([3.0, 1.0, 2.0, 99.0], index=idx)
        cleaned = data.clean_prices(raw)
        # Ascending order, last duplicate kept (99.0 for 2023-01-02).
        assert list(cleaned.index) == sorted(set(idx))
        assert cleaned.loc["2023-01-02"] == 99.0

    def test_trims_edge_nans(self) -> None:
        raw = pd.Series([float("nan"), 1.0, 2.0, float("nan")], index=_index(4))
        cleaned = data.clean_prices(raw)
        assert cleaned.tolist() == [1.0, 2.0]

    def test_rejects_interior_gap(self) -> None:
        raw = pd.Series([1.0, float("nan"), 3.0], index=_index(3))
        with pytest.raises(data.DataError):
            data.clean_prices(raw)

    def test_rejects_non_positive_prices(self) -> None:
        raw = pd.Series([1.0, 0.0, 3.0], index=_index(3))
        with pytest.raises(data.DataError):
            data.clean_prices(raw)

    def test_all_nan_series_reports_no_valid_data(self) -> None:
        """An all-NaN series must say "no valid observations", not the
        misleading "interior gap" message."""
        raw = pd.Series([float("nan")] * 3, index=_index(3))
        with pytest.raises(data.DataError, match="no valid"):
            data.clean_prices(raw)

    def test_empty_series_reports_no_valid_data(self) -> None:
        raw = pd.Series([], dtype=float)
        with pytest.raises(data.DataError, match="no valid"):
            data.clean_prices(raw)


class TestLoadPrices:
    def _install_fake_yfinance(self, monkeypatch, frame: pd.DataFrame) -> None:
        """Inject a stub ``yfinance`` module whose ``download`` returns ``frame``."""
        fake = types.ModuleType("yfinance")

        def _download(ticker, **kwargs):  # noqa: ANN001, ANN003 - test stub
            return frame

        fake.download = _download
        monkeypatch.setitem(sys.modules, "yfinance", fake)

    def test_returns_clean_close_series(self, monkeypatch) -> None:
        frame = pd.DataFrame(
            {"Open": [10.0, 11.0], "Close": [10.5, 11.5]}, index=_index(2)
        )
        self._install_fake_yfinance(monkeypatch, frame)
        series = data.load_prices("TEST")
        assert series.name == "TEST"
        assert series.tolist() == [10.5, 11.5]

    def test_empty_download_raises(self, monkeypatch) -> None:
        self._install_fake_yfinance(monkeypatch, pd.DataFrame())
        with pytest.raises(data.DataError):
            data.load_prices("EMPTY")

    def test_missing_close_column_raises(self, monkeypatch) -> None:
        frame = pd.DataFrame({"Open": [10.0, 11.0]}, index=_index(2))
        self._install_fake_yfinance(monkeypatch, frame)
        with pytest.raises(data.DataError):
            data.load_prices("NOCLOSE")
