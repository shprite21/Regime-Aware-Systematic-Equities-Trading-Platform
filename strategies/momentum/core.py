"""Momentum strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.models import MarketDataBundle
from strategies.base import BaseStrategy, StrategyConfig, StrategySignal, prices_from_data
from strategies.utils import normalize_weights


class CrossSectionalMomentumStrategy(BaseStrategy):
    """Long recent winners and short recent losers across the equity universe."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(
            config
            or StrategyConfig(
                name="cross_sectional_momentum",
                lookback=126,
                params={"top_quantile": 0.2, "bottom_quantile": 0.2},
            )
        )

    def generate(self, data: MarketDataBundle | pd.DataFrame) -> StrategySignal:
        prices = self._select_universe(prices_from_data(data))
        scores = prices.pct_change(self.config.lookback).shift(1)
        top_quantile = float(self.config.params.get("top_quantile", 0.2))
        bottom_quantile = float(self.config.params.get("bottom_quantile", 0.2))

        ranks = scores.rank(axis=1, pct=True)
        raw = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        raw = raw.mask(ranks >= 1.0 - top_quantile, 1.0)
        if self.config.long_short:
            raw = raw.mask(ranks <= bottom_quantile, -1.0)

        weights = normalize_weights(raw, self.config.gross_leverage)
        return StrategySignal(self.config.name, weights=weights, scores=scores)


class TimeSeriesMomentumStrategy(BaseStrategy):
    """Go long assets with positive own momentum and short negative momentum."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(
            config
            or StrategyConfig(
                name="time_series_momentum",
                lookback=126,
                params={"volatility_window": 21},
            )
        )

    def generate(self, data: MarketDataBundle | pd.DataFrame) -> StrategySignal:
        prices = self._select_universe(prices_from_data(data))
        returns = prices.pct_change()
        scores = prices.pct_change(self.config.lookback).shift(1)
        volatility_window = int(self.config.params.get("volatility_window", 21))
        vol = returns.rolling(volatility_window).std().replace(0.0, np.nan)

        raw = np.sign(scores).where(scores.notna(), 0.0)
        if not self.config.long_short:
            raw = raw.clip(lower=0.0)
        risk_scaled = raw.div(vol, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        weights = normalize_weights(risk_scaled, self.config.gross_leverage)
        return StrategySignal(self.config.name, weights=weights, scores=scores)


class MovingAverageTrendStrategy(BaseStrategy):
    """Moving-average trend-following strategy."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(
            config
            or StrategyConfig(
                name="moving_average_trend",
                lookback=200,
                params={"short_window": 50, "long_window": 200, "volatility_window": 21},
            )
        )

    def generate(self, data: MarketDataBundle | pd.DataFrame) -> StrategySignal:
        prices = self._select_universe(prices_from_data(data))
        short_window = int(self.config.params.get("short_window", 50))
        long_window = int(self.config.params.get("long_window", self.config.lookback))
        volatility_window = int(self.config.params.get("volatility_window", 21))

        short_average = prices.rolling(short_window).mean()
        long_average = prices.rolling(long_window).mean()
        trend_strength = (short_average / long_average - 1.0).shift(1)
        raw = np.sign(trend_strength).where(trend_strength.notna(), 0.0)
        if not self.config.long_short:
            raw = raw.clip(lower=0.0)

        volatility = prices.pct_change().rolling(volatility_window).std().replace(0.0, np.nan)
        weights = normalize_weights(raw.div(volatility, axis=0), self.config.gross_leverage)
        return StrategySignal(self.config.name, weights=weights, scores=trend_strength)
