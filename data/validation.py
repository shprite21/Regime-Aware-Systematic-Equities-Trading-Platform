"""Professional market data quality diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataQualityConfig:
    """Data quality threshold configuration."""

    stale_days: int = 5
    outlier_zscore: float = 8.0
    split_ratio_threshold: float = 0.35
    max_missing_fraction: float = 0.10


@dataclass
class DataQualityReport:
    """Market data quality report."""

    missing_data: pd.DataFrame
    stale_prices: pd.DataFrame
    outliers: pd.DataFrame
    adjustment_warnings: pd.DataFrame
    duplicate_timestamps: pd.Series
    survivorship_warnings: list[str] = field(default_factory=list)

    def has_issues(self) -> bool:
        """Return whether any material issue was detected."""

        return (
            not self.missing_data.empty
            or not self.stale_prices.empty
            or not self.outliers.empty
            or not self.adjustment_warnings.empty
            or bool(self.duplicate_timestamps.sum())
            or bool(self.survivorship_warnings)
        )


class MarketDataQualityValidator:
    """Validate OHLCV panels for research-grade data quality."""

    def __init__(self, config: DataQualityConfig | None = None) -> None:
        self.config = config or DataQualityConfig()

    def validate(self, ohlcv: dict[str, pd.DataFrame]) -> DataQualityReport:
        """Run missing, stale, outlier, adjustment, duplicate, and survivorship checks."""

        missing = []
        stale = []
        outliers = []
        adjustments = []
        duplicates = {}
        for symbol, frame in ohlcv.items():
            duplicates[symbol] = int(frame.index.duplicated().sum())
            missing_fraction = frame.isna().mean()
            for column, fraction in missing_fraction.items():
                if fraction > self.config.max_missing_fraction:
                    missing.append({"symbol": symbol, "field": column, "missing_fraction": float(fraction)})
            close = _close(frame)
            stale_dates = self._stale_dates(close)
            stale.extend({"symbol": symbol, "date": date, "close": float(close.loc[date])} for date in stale_dates)
            outlier_dates = self._outlier_dates(close.pct_change())
            outliers.extend({"symbol": symbol, "date": date, "return": float(close.pct_change().loc[date])} for date in outlier_dates)
            adjustment_warnings = self._adjustment_warnings(frame, symbol)
            adjustments.extend(adjustment_warnings)
        return DataQualityReport(
            missing_data=pd.DataFrame(missing),
            stale_prices=pd.DataFrame(stale),
            outliers=pd.DataFrame(outliers),
            adjustment_warnings=pd.DataFrame(adjustments),
            duplicate_timestamps=pd.Series(duplicates, name="duplicate_timestamps"),
            survivorship_warnings=self._survivorship_warnings(ohlcv),
        )

    def _stale_dates(self, close: pd.Series) -> list[pd.Timestamp]:
        unchanged = close.diff().abs() < 1e-12
        rolling_stale = unchanged.rolling(self.config.stale_days).sum() >= self.config.stale_days
        return list(close.index[rolling_stale.fillna(False)])

    def _outlier_dates(self, returns: pd.Series) -> list[pd.Timestamp]:
        rolling_mean = returns.rolling(63, min_periods=20).mean()
        rolling_std = returns.rolling(63, min_periods=20).std().replace(0.0, np.nan)
        zscore = ((returns - rolling_mean) / rolling_std).replace([np.inf, -np.inf], np.nan)
        return list(returns.index[zscore.abs() > self.config.outlier_zscore])

    def _adjustment_warnings(self, frame: pd.DataFrame, symbol: str) -> list[dict[str, object]]:
        warnings = []
        if "Adj Close" in frame.columns and "Close" in frame.columns:
            ratio = frame["Adj Close"] / frame["Close"].replace(0.0, np.nan)
            ratio_change = ratio.pct_change().abs()
            flagged = ratio_change > self.config.split_ratio_threshold
            for date in ratio.index[flagged.fillna(False)]:
                warnings.append({"symbol": symbol, "date": date, "warning": "large adjustment ratio change"})
        else:
            close = _close(frame)
            large_gap = close.pct_change().abs() > self.config.split_ratio_threshold
            for date in close.index[large_gap.fillna(False)]:
                warnings.append({"symbol": symbol, "date": date, "warning": "possible split/dividend adjustment issue"})
        return warnings

    @staticmethod
    def _survivorship_warnings(ohlcv: dict[str, pd.DataFrame]) -> list[str]:
        if not ohlcv:
            return ["No OHLCV data supplied"]
        starts = {symbol: frame.index.min() for symbol, frame in ohlcv.items()}
        if len(set(starts.values())) > 1:
            return ["Universe has staggered start dates; review survivorship and listing-date bias."]
        return []


def _close(frame: pd.DataFrame) -> pd.Series:
    if "Adj Close" in frame.columns:
        return frame["Adj Close"]
    return frame["Close"]

