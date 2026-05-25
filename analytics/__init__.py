"""Performance analytics and reporting."""

from analytics.metrics import (
    aggregate_metrics,
    calmar_ratio,
    drawdown_series,
    exposure_analysis,
    factor_decomposition,
    max_drawdown,
    regime_performance,
    rolling_correlations,
    rolling_returns,
    rolling_sharpe,
    sharpe_ratio,
    sortino_ratio,
    turnover,
)
from analytics.tearsheet import PerformanceReport

__all__ = [
    "PerformanceReport",
    "aggregate_metrics",
    "calmar_ratio",
    "drawdown_series",
    "exposure_analysis",
    "factor_decomposition",
    "max_drawdown",
    "regime_performance",
    "rolling_correlations",
    "rolling_returns",
    "rolling_sharpe",
    "sharpe_ratio",
    "sortino_ratio",
    "turnover",
]
