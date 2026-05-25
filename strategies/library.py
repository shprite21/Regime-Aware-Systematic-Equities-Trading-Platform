"""Utilities for composing strategy research pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from data.models import MarketDataBundle
from strategies.base import BaseStrategy, StrategySignal
from strategies.utils import normalize_weights


@dataclass
class StrategyPipeline:
    """Run a collection of independent strategies on a shared data bundle."""

    strategies: Iterable[BaseStrategy]

    def generate(self, data: MarketDataBundle | pd.DataFrame) -> dict[str, StrategySignal]:
        """Generate signals for every configured strategy."""

        return {strategy.config.name: strategy.generate(data) for strategy in self.strategies}

    @staticmethod
    def equal_weight_blend(signals: dict[str, StrategySignal], gross_leverage: float = 1.0) -> StrategySignal:
        """Blend strategy target weights into one portfolio signal."""

        if not signals:
            raise ValueError("At least one strategy signal is required")
        index = next(iter(signals.values())).weights.index
        columns = sorted(set().union(*(signal.weights.columns for signal in signals.values())))
        blended = pd.DataFrame(0.0, index=index, columns=columns)
        for signal in signals.values():
            blended = blended.add(signal.weights.reindex(index=index, columns=columns).fillna(0.0), fill_value=0.0)
        blended = blended / len(signals)
        return StrategySignal(
            name="equal_weight_strategy_blend",
            weights=normalize_weights(blended, gross_leverage),
            metadata={"components": list(signals)},
        )

