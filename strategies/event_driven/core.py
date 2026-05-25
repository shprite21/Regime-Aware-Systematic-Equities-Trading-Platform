"""Event-driven strategy family."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.models import MarketDataBundle
from strategies.base import BaseStrategy, StrategyConfig, StrategySignal, prices_from_data
from strategies.events import PEADStrategy
from strategies.utils import normalize_weights


class GapContinuationReversalStrategy(BaseStrategy):
    """Trade opening gaps as continuation or reversal signals."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(
            config
            or StrategyConfig(
                name="gap_continuation_reversal",
                lookback=20,
                params={"gap_z": 1.5, "mode": "continuation"},
            )
        )

    def generate(self, data: MarketDataBundle | pd.DataFrame) -> StrategySignal:
        prices = self._select_universe(prices_from_data(data))
        opens = _open_panel(data, prices)
        gap = opens / prices.shift(1) - 1.0
        gap_z = (gap - gap.rolling(self.config.lookback).mean()) / gap.rolling(self.config.lookback).std()
        gap_z = gap_z.shift(1)
        threshold = float(self.config.params.get("gap_z", 1.5))
        mode = str(self.config.params.get("mode", "continuation"))
        direction = np.sign(gap_z)
        if mode == "reversal":
            direction = -direction

        raw = direction.where(gap_z.abs() >= threshold, 0.0)
        if not self.config.long_short:
            raw = raw.clip(lower=0.0)
        weights = normalize_weights(raw, self.config.gross_leverage)
        return StrategySignal(self.config.name, weights=weights, scores=gap_z)


class EarningsMomentumStrategy(BaseStrategy):
    """Rank assets by earnings surprise momentum."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(
            config
            or StrategyConfig(
                name="earnings_momentum",
                lookback=63,
                params={"top_quantile": 0.3, "bottom_quantile": 0.3},
            )
        )

    def generate(self, data: MarketDataBundle | pd.DataFrame) -> StrategySignal:
        prices = self._select_universe(prices_from_data(data))
        surprises = None
        if isinstance(data, MarketDataBundle):
            feature = data.features.get("earnings_surprise")
            if isinstance(feature, pd.DataFrame):
                surprises = feature.reindex(prices.index).reindex(columns=prices.columns).fillna(0.0)
        if surprises is None:
            surprises = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        scores = surprises.rolling(self.config.lookback, min_periods=1).sum().shift(1)
        ranks = scores.rank(axis=1, pct=True)
        top_quantile = float(self.config.params.get("top_quantile", 0.3))
        bottom_quantile = float(self.config.params.get("bottom_quantile", 0.3))
        raw = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        raw = raw.mask(ranks >= 1.0 - top_quantile, 1.0)
        if self.config.long_short:
            raw = raw.mask(ranks <= bottom_quantile, -1.0)
        weights = normalize_weights(raw, self.config.gross_leverage)
        return StrategySignal(self.config.name, weights=weights, scores=scores)


def _open_panel(data: MarketDataBundle | pd.DataFrame, fallback_prices: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, MarketDataBundle):
        return data.field_panel("Open").reindex(index=fallback_prices.index, columns=fallback_prices.columns)
    return fallback_prices

