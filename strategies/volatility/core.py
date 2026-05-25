"""Volatility strategy family."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.models import MarketDataBundle
from strategies.base import BaseStrategy, StrategyConfig, StrategySignal, prices_from_data
from strategies.breakout import VolatilityBreakoutStrategy
from strategies.utils import normalize_weights


class ATRBreakoutStrategy(BaseStrategy):
    """ATR-based trend breakout strategy."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(
            config
            or StrategyConfig(
                name="atr_breakout",
                lookback=20,
                params={"atr_window": 14, "atr_multiplier": 1.5},
            )
        )

    def generate(self, data: MarketDataBundle | pd.DataFrame) -> StrategySignal:
        prices = self._select_universe(prices_from_data(data))
        high, low, close = _high_low_close(data, prices)
        atr_window = int(self.config.params.get("atr_window", 14))
        atr_multiplier = float(self.config.params.get("atr_multiplier", 1.5))
        atr = _average_true_range(high, low, close, atr_window).replace(0.0, np.nan)
        midpoint = close.rolling(self.config.lookback).mean().shift(1)

        raw = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        raw = raw.mask(close > midpoint + atr_multiplier * atr, 1.0)
        if self.config.long_short:
            raw = raw.mask(close < midpoint - atr_multiplier * atr, -1.0)

        scores = (close - midpoint).div(atr, axis=0)
        weights = normalize_weights(raw.div(atr, axis=0), self.config.gross_leverage)
        return StrategySignal(self.config.name, weights=weights, scores=scores)


class VolatilityCompressionStrategy(BaseStrategy):
    """Trade breakouts after realized volatility compresses below its history."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(
            config
            or StrategyConfig(
                name="volatility_compression",
                lookback=63,
                params={"short_window": 10, "breakout_window": 20, "compression_quantile": 0.25},
            )
        )

    def generate(self, data: MarketDataBundle | pd.DataFrame) -> StrategySignal:
        prices = self._select_universe(prices_from_data(data))
        returns = prices.pct_change()
        short_window = int(self.config.params.get("short_window", 10))
        breakout_window = int(self.config.params.get("breakout_window", 20))
        compression_quantile = float(self.config.params.get("compression_quantile", 0.25))

        realized_volatility = returns.rolling(short_window).std()
        compression_threshold = realized_volatility.rolling(self.config.lookback).quantile(compression_quantile)
        compressed = realized_volatility < compression_threshold
        breakout_high = prices.rolling(breakout_window).max().shift(1)
        breakout_low = prices.rolling(breakout_window).min().shift(1)

        raw = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        raw = raw.mask(compressed.shift(1) & (prices > breakout_high), 1.0)
        if self.config.long_short:
            raw = raw.mask(compressed.shift(1) & (prices < breakout_low), -1.0)

        scores = realized_volatility / compression_threshold.replace(0.0, np.nan)
        weights = normalize_weights(raw, self.config.gross_leverage)
        return StrategySignal(self.config.name, weights=weights, scores=-scores)


def _high_low_close(
    data: MarketDataBundle | pd.DataFrame,
    fallback_prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if isinstance(data, MarketDataBundle):
        high = data.field_panel("High").reindex(columns=fallback_prices.columns)
        low = data.field_panel("Low").reindex(columns=fallback_prices.columns)
        close = data.close().reindex(columns=fallback_prices.columns)
        return high, low, close
    return fallback_prices, fallback_prices, fallback_prices


def _average_true_range(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    true_range = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for symbol in close.columns:
        components = pd.concat(
            [
                high[symbol] - low[symbol],
                (high[symbol] - close[symbol].shift(1)).abs(),
                (low[symbol] - close[symbol].shift(1)).abs(),
            ],
            axis=1,
        )
        true_range[symbol] = components.max(axis=1)
    return true_range.rolling(window).mean()
