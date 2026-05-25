"""Performance report assembly."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from analytics.metrics import aggregate_metrics, drawdown_series, regime_performance, rolling_sharpe, turnover
from backtesting.engine import BacktestResult


@dataclass
class PerformanceReport:
    """Research report object for a completed backtest."""

    metrics: dict[str, float]
    equity_curve: pd.Series
    returns: pd.Series
    drawdowns: pd.Series
    rolling_sharpe: pd.Series
    turnover: pd.Series
    regime_breakdown: pd.DataFrame | None = None
    attribution: pd.DataFrame | None = None
    benchmark_metrics: pd.DataFrame | None = None
    factor_exposures: pd.DataFrame | None = None
    regime_transition_matrix: pd.DataFrame | None = None
    portfolio_diagnostics: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_backtest(
        cls,
        result: BacktestResult,
        regimes: pd.Series | None = None,
        rolling_window: int = 63,
    ) -> "PerformanceReport":
        """Create a report from a backtest result."""

        regime_breakdown = None
        if regimes is not None:
            regime_breakdown = regime_performance(result.returns, regimes)

        return cls(
            metrics=aggregate_metrics(result.returns, result.equity_curve),
            equity_curve=result.equity_curve,
            returns=result.returns,
            drawdowns=drawdown_series(result.equity_curve),
            rolling_sharpe=rolling_sharpe(result.returns, rolling_window),
            turnover=turnover(result.weights),
            regime_breakdown=regime_breakdown,
            attribution=result.attribution,
            metadata=result.metadata,
        )

    def metrics_frame(self) -> pd.DataFrame:
        """Return metrics as a one-column DataFrame."""

        return pd.DataFrame.from_dict(self.metrics, orient="index", columns=["value"])
