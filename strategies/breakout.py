"""Breakout strategies."""

from __future__ import annotations

import pandas as pd

from data.models import MarketDataBundle
from strategies.base import BaseStrategy, StrategyConfig, StrategySignal, prices_from_data
from strategies.utils import normalize_weights


class VolatilityBreakoutStrategy(BaseStrategy):
    """Trade breakouts above rolling highs or below rolling lows."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(
            config
            or StrategyConfig(
                name="volatility_breakout",
                lookback=55,
                params={"volatility_window": 21},
            )
        )

    def generate(self, data: MarketDataBundle | pd.DataFrame) -> StrategySignal:
        prices = self._select_universe(prices_from_data(data))
        rolling_high = prices.rolling(self.config.lookback).max().shift(1)
        rolling_low = prices.rolling(self.config.lookback).min().shift(1)
        returns = prices.pct_change()
        volatility_window = int(self.config.params.get("volatility_window", 21))
        volatility = returns.rolling(volatility_window).std().replace(0.0, pd.NA)

        raw = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        raw = raw.mask(prices > rolling_high, 1.0)
        if self.config.long_short:
            raw = raw.mask(prices < rolling_low, -1.0)

        scores = (prices - rolling_high) / volatility
        weights = normalize_weights(raw.div(volatility, axis=0), self.config.gross_leverage)
        return StrategySignal(self.config.name, weights=weights, scores=scores)

