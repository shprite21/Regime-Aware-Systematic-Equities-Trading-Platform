"""Mean-reversion strategy family."""

from __future__ import annotations

import pandas as pd

from data.models import MarketDataBundle
from strategies.base import BaseStrategy, StrategyConfig, StrategySignal, prices_from_data
from strategies.utils import normalize_weights, rolling_zscore


class MeanReversionStrategy(BaseStrategy):
    """Trade reversals from rolling return z-scores."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(
            config
            or StrategyConfig(
                name="mean_reversion",
                lookback=20,
                params={"z_entry": 1.5},
            )
        )

    def generate(self, data: MarketDataBundle | pd.DataFrame) -> StrategySignal:
        prices = self._select_universe(prices_from_data(data))
        returns = prices.pct_change()
        zscores = rolling_zscore(returns, self.config.lookback).shift(1)
        z_entry = float(self.config.params.get("z_entry", 1.5))

        raw = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        raw = raw.mask(zscores <= -z_entry, 1.0)
        if self.config.long_short:
            raw = raw.mask(zscores >= z_entry, -1.0)

        weights = normalize_weights(raw, self.config.gross_leverage)
        return StrategySignal(self.config.name, weights=weights, scores=-zscores)


class BollingerBandStrategy(BaseStrategy):
    """Bollinger Band reversal system."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(
            config
            or StrategyConfig(
                name="bollinger_band",
                lookback=20,
                params={"num_std": 2.0},
            )
        )

    def generate(self, data: MarketDataBundle | pd.DataFrame) -> StrategySignal:
        prices = self._select_universe(prices_from_data(data))
        mean = prices.rolling(self.config.lookback).mean()
        std = prices.rolling(self.config.lookback).std()
        num_std = float(self.config.params.get("num_std", 2.0))
        upper = mean + num_std * std
        lower = mean - num_std * std

        raw = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        raw = raw.mask(prices.shift(1) < lower.shift(1), 1.0)
        if self.config.long_short:
            raw = raw.mask(prices.shift(1) > upper.shift(1), -1.0)

        scores = (prices - mean).div(std.replace(0.0, pd.NA))
        weights = normalize_weights(raw, self.config.gross_leverage)
        return StrategySignal(self.config.name, weights=weights, scores=-scores)


class RSIReversalStrategy(BaseStrategy):
    """RSI reversal strategy."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(
            config
            or StrategyConfig(
                name="rsi_reversal",
                lookback=14,
                params={"lower": 30, "upper": 70},
            )
        )

    def generate(self, data: MarketDataBundle | pd.DataFrame) -> StrategySignal:
        prices = self._select_universe(prices_from_data(data))
        rsi = self._rsi(prices, self.config.lookback).shift(1)
        lower = float(self.config.params.get("lower", 30))
        upper = float(self.config.params.get("upper", 70))

        raw = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        raw = raw.mask(rsi < lower, 1.0)
        if self.config.long_short:
            raw = raw.mask(rsi > upper, -1.0)

        scores = 50.0 - rsi
        weights = normalize_weights(raw, self.config.gross_leverage)
        return StrategySignal(self.config.name, weights=weights, scores=scores)

    @staticmethod
    def _rsi(prices: pd.DataFrame, window: int) -> pd.DataFrame:
        delta = prices.diff()
        gains = delta.clip(lower=0.0).rolling(window).mean()
        losses = (-delta.clip(upper=0.0)).rolling(window).mean()
        relative_strength = gains / losses.replace(0.0, pd.NA)
        return 100.0 - (100.0 / (1.0 + relative_strength))


class ShortTermReversalStrategy(MeanReversionStrategy):
    """Alias-style short-term reversal system with a shorter default lookback."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(
            config
            or StrategyConfig(
                name="short_term_reversal",
                lookback=5,
                params={"z_entry": 1.0},
            )
        )
