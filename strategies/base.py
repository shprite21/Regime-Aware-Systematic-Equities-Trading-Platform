"""Base strategy contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from data.models import MarketDataBundle


@dataclass(frozen=True)
class StrategyConfig:
    """Shared strategy configuration."""

    name: str
    universe: list[str] | None = None
    lookback: int = 63
    holding_period: int = 1
    long_short: bool = True
    gross_leverage: float = 1.0
    params: dict[str, object] = field(default_factory=dict)


@dataclass
class StrategySignal:
    """Standardized output from any alpha strategy."""

    name: str
    weights: pd.DataFrame
    scores: pd.DataFrame | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def shifted(self, periods: int = 1) -> "StrategySignal":
        """Return signal with weights shifted for execution lag."""

        return StrategySignal(
            name=self.name,
            weights=self.weights.shift(periods).fillna(0.0),
            scores=self.scores,
            metadata={**self.metadata, "execution_lag": periods},
        )


class BaseStrategy(ABC):
    """Interface for research alpha strategies."""

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(self, data: MarketDataBundle | pd.DataFrame) -> StrategySignal:
        """Generate target weights and optional alpha scores."""

    def _select_universe(self, prices: pd.DataFrame) -> pd.DataFrame:
        if self.config.universe is None:
            return prices
        selected = [symbol for symbol in self.config.universe if symbol in prices.columns]
        if not selected:
            raise ValueError(f"No configured symbols are present for {self.config.name}")
        return prices[selected]


def prices_from_data(data: MarketDataBundle | pd.DataFrame) -> pd.DataFrame:
    """Extract close prices from a market bundle or accept an existing price panel."""

    if isinstance(data, MarketDataBundle):
        return data.close()
    return data.sort_index()

