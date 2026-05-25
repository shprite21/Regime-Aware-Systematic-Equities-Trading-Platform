"""Deterministic regime classifiers based on volatility, trend, and breadth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_detection.base import BaseRegimeDetector, RegimeResult


@dataclass
class VolatilityRegimeClassifier(BaseRegimeDetector):
    """Classify low, medium, and high volatility regimes using rolling quantiles."""

    lookback: int = 21
    low_quantile: float = 0.33
    high_quantile: float = 0.67
    annualization_factor: int = 252
    name: str = "volatility"

    def fit(self, features: pd.DataFrame) -> "VolatilityRegimeClassifier":
        return self

    def predict(self, features: pd.DataFrame) -> RegimeResult:
        returns = self._as_return_series(features)
        realized_vol = returns.rolling(self.lookback).std() * np.sqrt(self.annualization_factor)
        low_threshold = realized_vol.rolling(self.lookback * 6, min_periods=self.lookback).quantile(
            self.low_quantile
        )
        high_threshold = realized_vol.rolling(self.lookback * 6, min_periods=self.lookback).quantile(
            self.high_quantile
        )

        labels = pd.Series(1, index=realized_vol.index, name="volatility_regime")
        labels = labels.mask(realized_vol <= low_threshold, 0)
        labels = labels.mask(realized_vol >= high_threshold, 2)
        probabilities = pd.get_dummies(labels).reindex(columns=[0, 1, 2], fill_value=0).astype(float)
        return RegimeResult(
            labels=labels.astype(int),
            probabilities=probabilities,
            diagnostics={"realized_volatility": realized_vol},
        )

    @staticmethod
    def _as_return_series(features: pd.DataFrame) -> pd.Series:
        if isinstance(features, pd.Series):
            return features
        if "returns" in features.columns:
            return features["returns"]
        if "returns_1d" in features.columns:
            candidate = features["returns_1d"]
            if isinstance(candidate, pd.Series):
                return candidate
        return features.select_dtypes(include=[np.number]).mean(axis=1)


@dataclass
class TrendFilter(BaseRegimeDetector):
    """Classify bearish, neutral, and bullish trend regimes from moving averages."""

    short_window: int = 50
    long_window: int = 200
    slope_window: int = 20
    name: str = "trend"

    def fit(self, features: pd.DataFrame) -> "TrendFilter":
        return self

    def predict(self, features: pd.DataFrame) -> RegimeResult:
        prices = self._as_price_series(features)
        short_ma = prices.rolling(self.short_window).mean()
        long_ma = prices.rolling(self.long_window).mean()
        long_slope = long_ma.diff(self.slope_window)

        labels = pd.Series(1, index=prices.index, name="trend_regime")
        labels = labels.mask((short_ma > long_ma) & (long_slope > 0), 2)
        labels = labels.mask((short_ma < long_ma) & (long_slope < 0), 0)
        probabilities = pd.get_dummies(labels).reindex(columns=[0, 1, 2], fill_value=0).astype(float)
        return RegimeResult(
            labels=labels.astype(int),
            probabilities=probabilities,
            diagnostics={"short_ma": short_ma, "long_ma": long_ma, "long_slope": long_slope},
        )

    @staticmethod
    def _as_price_series(features: pd.DataFrame) -> pd.Series:
        if isinstance(features, pd.Series):
            return features
        if "Close" in features.columns:
            return features["Close"]
        if "close" in features.columns:
            return features["close"]
        return features.select_dtypes(include=[np.number]).mean(axis=1)


@dataclass
class MarketBreadthRegimeDetector(BaseRegimeDetector):
    """Classify regimes from the percentage of stocks above a moving average."""

    moving_average_window: int = 200
    bearish_threshold: float = 0.4
    bullish_threshold: float = 0.6
    name: str = "market_breadth"

    def fit(self, features: pd.DataFrame) -> "MarketBreadthRegimeDetector":
        return self

    def predict(self, features: pd.DataFrame) -> RegimeResult:
        prices = features.select_dtypes(include=[np.number]).sort_index()
        if prices.empty:
            raise ValueError("MarketBreadthRegimeDetector requires a numeric price panel")

        moving_average = prices.rolling(self.moving_average_window).mean()
        breadth = (prices > moving_average).sum(axis=1) / prices.count(axis=1).replace(0, np.nan)
        advancing = (prices.pct_change() > 0).sum(axis=1) / prices.count(axis=1).replace(0, np.nan)

        labels = pd.Series(1, index=prices.index, name="breadth_regime")
        labels = labels.mask(breadth <= self.bearish_threshold, 0)
        labels = labels.mask(breadth >= self.bullish_threshold, 2)
        probabilities = pd.get_dummies(labels).reindex(columns=[0, 1, 2], fill_value=0).astype(float)
        return RegimeResult(
            labels=labels.astype(int),
            probabilities=probabilities,
            diagnostics={"breadth": breadth, "advancing_fraction": advancing},
        )

