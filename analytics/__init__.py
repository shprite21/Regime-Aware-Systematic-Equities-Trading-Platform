"""Performance analytics and reporting."""

from analytics.attribution import AttributionAnalyzer
from analytics.benchmarking import (
    BenchmarkComparison,
    BenchmarkConfig,
    BenchmarkEngine,
    alpha_beta,
    information_ratio,
    tracking_error,
)
from analytics.correlation import CorrelationAnalysisResult, CorrelationConfig, StrategyCorrelationAnalyzer
from analytics.cost_sensitivity import CostSensitivityAnalyzer, CostSensitivityConfig
from analytics.factor_exposure import (
    FactorExposureAnalyzer,
    FactorExposureConfig,
    FactorExposureResult,
    build_equity_style_factors,
    sector_exposures,
)
from analytics.html_report import AdvancedHTMLReportBuilder, HTMLReportConfig, HTMLReportInputs
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
from analytics.portfolio_diagnostics import PortfolioDiagnostics, PortfolioDiagnosticsConfig
from analytics.regime_analysis import RegimeAnalytics, RegimeAnalyticsResult
from analytics.tearsheet import PerformanceReport

__all__ = [
    "AdvancedHTMLReportBuilder",
    "AttributionAnalyzer",
    "BenchmarkComparison",
    "BenchmarkConfig",
    "BenchmarkEngine",
    "CorrelationAnalysisResult",
    "CorrelationConfig",
    "CostSensitivityAnalyzer",
    "CostSensitivityConfig",
    "FactorExposureAnalyzer",
    "FactorExposureConfig",
    "FactorExposureResult",
    "HTMLReportConfig",
    "HTMLReportInputs",
    "PerformanceReport",
    "PortfolioDiagnostics",
    "PortfolioDiagnosticsConfig",
    "RegimeAnalytics",
    "RegimeAnalyticsResult",
    "StrategyCorrelationAnalyzer",
    "aggregate_metrics",
    "alpha_beta",
    "build_equity_style_factors",
    "calmar_ratio",
    "drawdown_series",
    "exposure_analysis",
    "factor_decomposition",
    "information_ratio",
    "max_drawdown",
    "regime_performance",
    "rolling_correlations",
    "rolling_returns",
    "rolling_sharpe",
    "sector_exposures",
    "sharpe_ratio",
    "sortino_ratio",
    "tracking_error",
    "turnover",
]
