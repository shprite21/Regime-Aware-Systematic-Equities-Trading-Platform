"""Feature engineering for equities research."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.models import MarketDataBundle


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for rolling feature construction."""

    return_windows: list[int] = field(default_factory=lambda: [1, 5, 21, 63, 126, 252])
    volatility_windows: list[int] = field(default_factory=lambda: [21, 63])
    momentum_windows: list[int] = field(default_factory=lambda: [63, 126, 252])
    annualization_factor: int = 252
    zscore_window: int = 63


class FeatureEngineer:
    """Build reusable feature panels from a market data bundle."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    def build(self, bundle: MarketDataBundle) -> dict[str, pd.DataFrame]:
        """Build a dictionary of aligned feature panels."""

        close = bundle.close()
        volume = bundle.volume()
        returns = close.pct_change()
        features: dict[str, pd.DataFrame] = {"returns_1d": returns}

        for window in self.config.return_windows:
            features[f"returns_{window}d"] = close.pct_change(window)

        for window in self.config.volatility_windows:
            features[f"volatility_{window}d"] = (
                returns.rolling(window).std() * np.sqrt(self.config.annualization_factor)
            )

        for window in self.config.momentum_windows:
            features[f"momentum_{window}d"] = close.pct_change(window)

        dollar_volume = close * volume
        features["dollar_volume"] = dollar_volume
        features["liquidity_rank"] = dollar_volume.rank(axis=1, pct=True)
        features["drawdown"] = close / close.cummax() - 1.0
        features["cross_sectional_return_rank"] = returns.rank(axis=1, pct=True)
        features["return_zscore"] = self._rolling_zscore(returns, self.config.zscore_window)

        if bundle.benchmark is not None:
            benchmark_close = self._close_from_frame(bundle.benchmark).reindex(close.index).ffill()
            benchmark_returns = benchmark_close.pct_change()
            features["beta_63d"] = self._rolling_beta(returns, benchmark_returns, 63)
            features["relative_strength_vs_benchmark"] = close.pct_change(63).sub(
                benchmark_close.pct_change(63), axis=0
            )

        if bundle.sectors:
            sector_close = {
                symbol: self._close_from_frame(frame)
                for symbol, frame in bundle.sectors.items()
            }
            sector_panel = pd.DataFrame(sector_close).reindex(close.index).ffill()
            features["sector_momentum_126d"] = sector_panel.pct_change(126)

        if bundle.volatility:
            volatility_close = {
                symbol: self._close_from_frame(frame)
                for symbol, frame in bundle.volatility.items()
            }
            features["volatility_index"] = pd.DataFrame(volatility_close).reindex(close.index).ffill()

        if bundle.macro is not None:
            features["macro"] = bundle.macro.reindex(close.index).ffill()

        return bundle.with_features(features).features

    @staticmethod
    def _close_from_frame(frame: pd.DataFrame) -> pd.Series:
        column = "Adj Close" if "Adj Close" in frame.columns else "Close"
        return frame[column]

    @staticmethod
    def _rolling_zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
        mean = frame.rolling(window).mean()
        std = frame.rolling(window).std().replace(0, np.nan)
        return (frame - mean) / std

    @staticmethod
    def _rolling_beta(asset_returns: pd.DataFrame, benchmark_returns: pd.Series, window: int) -> pd.DataFrame:
        benchmark_returns = benchmark_returns.reindex(asset_returns.index)
        variance = benchmark_returns.rolling(window).var()
        betas = {}
        for column in asset_returns:
            covariance = asset_returns[column].rolling(window).cov(benchmark_returns)
            betas[column] = covariance / variance
        return pd.DataFrame(betas)

