"""Sector rotation strategy."""

from __future__ import annotations

import pandas as pd

from data.models import MarketDataBundle
from strategies.base import BaseStrategy, StrategyConfig, StrategySignal, prices_from_data
from strategies.utils import normalize_weights


class SectorRotationStrategy(BaseStrategy):
    """Allocate to the strongest sector ETFs by medium-term momentum."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(
            config
            or StrategyConfig(
                name="sector_rotation",
                lookback=126,
                long_short=False,
                params={"top_n": 3},
            )
        )

    def generate(self, data: MarketDataBundle | pd.DataFrame) -> StrategySignal:
        if isinstance(data, MarketDataBundle) and data.sectors:
            prices = pd.DataFrame(
                {
                    symbol: frame["Adj Close"] if "Adj Close" in frame.columns else frame["Close"]
                    for symbol, frame in data.sectors.items()
                }
            ).sort_index()
        else:
            prices = prices_from_data(data)
        prices = self._select_universe(prices)
        scores = prices.pct_change(self.config.lookback).shift(1)
        top_n = int(self.config.params.get("top_n", 3))

        ranks = scores.rank(axis=1, ascending=False, method="first")
        raw = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        raw = raw.mask(ranks <= top_n, 1.0)
        weights = normalize_weights(raw, self.config.gross_leverage)
        return StrategySignal(self.config.name, weights=weights, scores=scores)

