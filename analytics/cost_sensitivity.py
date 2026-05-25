"""Transaction cost, capacity, and implementation shortfall sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from analytics.metrics import aggregate_metrics
from backtesting import BacktestConfig, VectorizedBacktester


@dataclass(frozen=True)
class CostSensitivityConfig:
    """Cost sensitivity sweep settings."""

    slippage_bps: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0)
    commission_bps: tuple[float, ...] = (0.0, 1.0, 2.0)
    spread_bps: tuple[float, ...] = (0.0, 2.0)
    participation_rates: tuple[float, ...] = (0.05, 0.10, 0.20)
    initial_capital: float = 1_000_000.0


class CostSensitivityAnalyzer:
    """Stress test strategy performance under execution cost assumptions."""

    def __init__(self, config: CostSensitivityConfig | None = None) -> None:
        self.config = config or CostSensitivityConfig()

    def sweep(self, prices: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
        """Run cost sweeps and return performance metrics per cost scenario."""

        rows = []
        for slippage, commission, spread in product(
            self.config.slippage_bps,
            self.config.commission_bps,
            self.config.spread_bps,
        ):
            total_slippage = slippage + spread / 2.0
            result = VectorizedBacktester(
                BacktestConfig(
                    initial_capital=self.config.initial_capital,
                    transaction_cost_bps=commission,
                    slippage_bps=total_slippage,
                )
            ).run(prices, weights)
            metrics = aggregate_metrics(result.returns, result.equity_curve)
            rows.append(
                {
                    "slippage_bps": slippage,
                    "commission_bps": commission,
                    "spread_bps": spread,
                    "cost_adjusted_sharpe": metrics["sharpe"],
                    "total_return": metrics["total_return"],
                    "max_drawdown": metrics["max_drawdown"],
                    "average_turnover": float(result.turnover.mean()),
                    "implementation_shortfall": float((result.gross_returns - result.returns).sum()),
                }
            )
        return pd.DataFrame(rows)

    def capacity_analysis(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        dollar_volume: pd.DataFrame,
    ) -> pd.DataFrame:
        """Estimate capacity by participation rate from dollar-volume constraints."""

        turnover_dollars = weights.diff().abs().mul(prices).fillna(0.0)
        rows = []
        for participation_rate in self.config.participation_rates:
            daily_capacity = dollar_volume.reindex_like(prices).fillna(0.0) * participation_rate
            utilization = turnover_dollars.div(daily_capacity.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
            rows.append(
                {
                    "participation_rate": participation_rate,
                    "median_utilization": float(utilization.stack().median()),
                    "p95_utilization": float(utilization.stack().quantile(0.95)),
                    "capacity_warning_days": float((utilization.max(axis=1) > 1.0).mean()),
                }
            )
        return pd.DataFrame(rows)


def implementation_shortfall(gross_returns: pd.Series, net_returns: pd.Series) -> pd.Series:
    """Return cumulative implementation shortfall from execution costs."""

    return (gross_returns - net_returns).cumsum().rename("implementation_shortfall")

