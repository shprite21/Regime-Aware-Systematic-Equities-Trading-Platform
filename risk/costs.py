"""Transaction cost and slippage models."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TransactionCostModel:
    """Linear transaction cost model based on portfolio turnover."""

    cost_bps: float = 2.0
    minimum_cost: float = 0.0

    def estimate(self, turnover: pd.Series | pd.DataFrame, capital: float = 1.0) -> pd.Series:
        """Estimate transaction costs in currency units."""

        turnover_series = turnover.sum(axis=1) if isinstance(turnover, pd.DataFrame) else turnover
        cost = turnover_series.abs() * (self.cost_bps / 10_000.0) * capital
        return cost.clip(lower=self.minimum_cost if self.minimum_cost > 0 else 0.0)

    def as_return_drag(self, turnover: pd.Series | pd.DataFrame) -> pd.Series:
        """Estimate costs as a return drag."""

        turnover_series = turnover.sum(axis=1) if isinstance(turnover, pd.DataFrame) else turnover
        return turnover_series.abs() * (self.cost_bps / 10_000.0)


@dataclass(frozen=True)
class SlippageModel:
    """Slippage model combining fixed and volatility-linked execution impact."""

    slippage_bps: float = 1.0
    volatility_multiplier: float = 0.0

    def estimate(self, turnover: pd.Series, realized_volatility: pd.Series | None = None) -> pd.Series:
        """Estimate slippage as a return drag."""

        fixed = turnover.abs() * (self.slippage_bps / 10_000.0)
        if realized_volatility is None or self.volatility_multiplier == 0:
            return fixed
        variable = turnover.abs() * realized_volatility.reindex(turnover.index).fillna(0.0)
        return fixed + self.volatility_multiplier * variable

