"""Event-driven alpha strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.models import MarketDataBundle
from strategies.base import BaseStrategy, StrategyConfig, StrategySignal, prices_from_data
from strategies.utils import normalize_weights


class PEADStrategy(BaseStrategy):
    """Post Earnings Announcement Drift strategy.

    The strategy expects an earnings surprise panel where rows are dates and
    columns are tickers. Positive surprises create long drift signals and
    negative surprises create short drift signals for ``drift_days``.
    """

    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(
            config
            or StrategyConfig(
                name="pead",
                lookback=1,
                params={"drift_days": 20, "min_abs_surprise": 0.0},
            )
        )

    def generate(self, data: MarketDataBundle | pd.DataFrame) -> StrategySignal:
        prices = self._select_universe(prices_from_data(data))
        surprises = self._get_surprises(data, prices)
        if surprises is None:
            weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            return StrategySignal(
                self.config.name,
                weights=weights,
                scores=weights.copy(),
                metadata={"warning": "No earnings_surprise panel supplied"},
            )

        surprises = surprises.reindex(prices.index).reindex(columns=prices.columns).fillna(0.0)
        min_abs_surprise = float(self.config.params.get("min_abs_surprise", 0.0))
        drift_days = int(self.config.params.get("drift_days", 20))
        event_direction = np.sign(surprises.where(surprises.abs() >= min_abs_surprise, 0.0))
        raw = event_direction.replace(0.0, np.nan).ffill(limit=drift_days).fillna(0.0)
        if not self.config.long_short:
            raw = raw.clip(lower=0.0)
        weights = normalize_weights(raw, self.config.gross_leverage)
        return StrategySignal(self.config.name, weights=weights, scores=surprises)

    def _get_surprises(
        self,
        data: MarketDataBundle | pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame | None:
        configured = self.config.params.get("earnings_surprise")
        if isinstance(configured, pd.DataFrame):
            return configured
        if isinstance(data, MarketDataBundle):
            feature = data.features.get("earnings_surprise")
            if isinstance(feature, pd.DataFrame):
                return feature
        return None

