"""Run a synthetic end-to-end research experiment.

This script intentionally uses generated data so the platform can be exercised
offline. Swap the synthetic bundle for ``OHLCVIngestor`` to run against live or
cached data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analytics import (
    AdvancedHTMLReportBuilder,
    BenchmarkEngine,
    CostSensitivityAnalyzer,
    CostSensitivityConfig,
    FactorExposureAnalyzer,
    HTMLReportInputs,
    PortfolioDiagnostics,
    RegimeAnalytics,
    StrategyCorrelationAnalyzer,
    build_equity_style_factors,
)
from analytics.tearsheet import PerformanceReport
from backtesting import WalkForwardConfig, WalkForwardValidator
from backtesting.engine import BacktestConfig, PerformanceAttributor, VectorizedBacktester
from data.features import FeatureEngineer
from data.models import MarketDataBundle
from data.validation import MarketDataQualityValidator
from portfolio.allocation import AllocationConfig, RegimeConditionedAllocator
from regime_detection import MarketBreadthRegimeDetector, RegimeEnsemble, VolatilityRegimeClassifier
from research import ExperimentTracker, ParameterRobustnessTester, RobustnessConfig
from risk.management import ExposureConstraint, RiskConfig, VolatilityTargeter
from strategies import (
    ATRBreakoutStrategy,
    BollingerBandStrategy,
    CointegrationPairsStrategy,
    CrossSectionalMomentumStrategy,
    EarningsMomentumStrategy,
    GapContinuationReversalStrategy,
    MeanReversionStrategy,
    MovingAverageTrendStrategy,
    PairsTradingStrategy,
    RSIReversalStrategy,
    TimeSeriesMomentumStrategy,
    VolatilityBreakoutStrategy,
    VolatilityCompressionStrategy,
)
from strategies.base import StrategyConfig
from strategies.events import PEADStrategy
from strategies.library import StrategyPipeline

REPORT_DIR = PROJECT_ROOT / "reports"


def make_synthetic_bundle(seed: int = 7) -> MarketDataBundle:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=900)
    symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "JPM", "XOM", "UNH"]
    regimes = np.repeat([0, 1, 2, 1, 0], repeats=[150, 220, 180, 220, 130])[: len(dates)]
    drift = np.choose(regimes, [-0.0002, 0.0003, 0.0006])
    volatility = np.choose(regimes, [0.018, 0.011, 0.014])

    ohlcv = {}
    for index, symbol in enumerate(symbols):
        asset_noise = rng.normal(0, volatility + index * 0.0004)
        returns = drift + asset_noise + rng.normal(0, 0.003, len(dates))
        close = 100 * (1 + pd.Series(returns, index=dates)).cumprod()
        high = close * (1 + rng.uniform(0.001, 0.02, len(dates)))
        low = close * (1 - rng.uniform(0.001, 0.02, len(dates)))
        open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, len(dates)))
        volume = rng.integers(1_000_000, 7_500_000, len(dates))
        ohlcv[symbol] = pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Adj Close": close,
                "Volume": volume,
            },
            index=dates,
        )

    benchmark = pd.DataFrame({"Close": pd.DataFrame({k: v["Close"] for k, v in ohlcv.items()}).mean(axis=1)})
    bundle = MarketDataBundle(ohlcv=ohlcv, benchmark=benchmark)
    surprises = pd.DataFrame(0.0, index=dates, columns=symbols)
    event_dates = dates[::63]
    surprises.loc[event_dates] = rng.normal(0, 1, (len(event_dates), len(symbols)))
    bundle.features["earnings_surprise"] = surprises
    return bundle


def write_html_report(
    report: PerformanceReport,
    regimes: pd.Series,
    contribution: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a self-contained HTML research report."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_table = pd.Series(report.metrics).round(4).to_frame("value").to_html(classes="table")
    regime_table = (
        report.regime_breakdown.round(4).to_html(classes="table")
        if report.regime_breakdown is not None
        else "<p>No regime breakdown available.</p>"
    )
    contribution_table = contribution.cumsum().tail(10).round(4).to_html(classes="table")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Regime-Aware Equities Research Report</title>
  <style>
    body {{
      font-family: Arial, Helvetica, sans-serif;
      margin: 0;
      background: #f6f7f9;
      color: #1f2933;
    }}
    header {{
      background: #0f172a;
      color: white;
      padding: 28px 40px;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px;
    }}
    section {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 22px;
      margin-bottom: 20px;
    }}
    h1, h2 {{
      margin-top: 0;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 20px;
    }}
    .table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 14px;
    }}
    .table th, .table td {{
      border-bottom: 1px solid #e5e7eb;
      padding: 8px 10px;
      text-align: right;
    }}
    .table th:first-child, .table td:first-child {{
      text-align: left;
    }}
    svg {{
      width: 100%;
      height: 260px;
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
    }}
    .subtitle {{
      color: #cbd5e1;
      margin-bottom: 0;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Regime-Aware Systematic Equities Trading Platform</h1>
    <p class="subtitle">Synthetic end-to-end research workflow report</p>
  </header>
  <main>
    <section>
      <h2>Performance Metrics</h2>
      {metrics_table}
    </section>
    <section class="grid">
      <div>
        <h2>Equity Curve</h2>
        {_line_svg(report.equity_curve, "#2563eb")}
      </div>
      <div>
        <h2>Drawdown</h2>
        {_line_svg(report.drawdowns, "#dc2626")}
      </div>
      <div>
        <h2>Rolling Sharpe</h2>
        {_line_svg(report.rolling_sharpe.dropna(), "#059669")}
      </div>
      <div>
        <h2>Regime Timeline</h2>
        {_line_svg(regimes.reindex(report.equity_curve.index).ffill(), "#7c3aed")}
      </div>
    </section>
    <section>
      <h2>Regime-Wise Performance</h2>
      {regime_table}
    </section>
    <section>
      <h2>Strategy Contribution Tail</h2>
      {contribution_table}
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def _line_svg(series: pd.Series, color: str) -> str:
    clean = series.dropna()
    if clean.empty:
        return "<svg viewBox='0 0 640 260'><text x='20' y='40'>No data</text></svg>"
    values = clean.astype(float).to_numpy()
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    spread = maximum - minimum if maximum != minimum else 1.0
    width = 640
    height = 260
    margin = 28
    x_scale = (width - 2 * margin) / max(len(values) - 1, 1)
    y_scale = (height - 2 * margin) / spread
    points = []
    for index, value in enumerate(values):
        x = margin + index * x_scale
        y = height - margin - (value - minimum) * y_scale
        points.append(f"{x:.2f},{y:.2f}")
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img'>"
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='#d1d5db'/>"
        f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='#d1d5db'/>"
        f"<polyline fill='none' stroke='{color}' stroke-width='2.2' points='{' '.join(points)}'/>"
        f"<text x='{margin}' y='18' font-size='12' fill='#6b7280'>max {maximum:.4f}</text>"
        f"<text x='{margin}' y='{height - 6}' font-size='12' fill='#6b7280'>min {minimum:.4f}</text>"
        "</svg>"
    )


def main() -> None:
    bundle = make_synthetic_bundle()
    features = FeatureEngineer().build(bundle)
    prices = bundle.close()
    returns = prices.pct_change().fillna(0.0)

    volatility_regime = VolatilityRegimeClassifier().fit_predict(returns.mean(axis=1).to_frame("returns"))
    breadth_regime = MarketBreadthRegimeDetector(moving_average_window=100).fit_predict(prices)
    regimes = RegimeEnsemble(weights={"volatility": 0.6, "breadth": 0.4}).combine(
        {"volatility": volatility_regime, "breadth": breadth_regime}
    )

    pipeline = StrategyPipeline(
        [
            CrossSectionalMomentumStrategy(),
            TimeSeriesMomentumStrategy(),
            MovingAverageTrendStrategy(),
            MeanReversionStrategy(),
            BollingerBandStrategy(),
            RSIReversalStrategy(),
            VolatilityBreakoutStrategy(),
            ATRBreakoutStrategy(),
            VolatilityCompressionStrategy(),
            PEADStrategy(),
            GapContinuationReversalStrategy(),
            EarningsMomentumStrategy(),
            PairsTradingStrategy(),
            CointegrationPairsStrategy(),
        ]
    )
    signals = pipeline.generate(bundle)
    blended = StrategyPipeline.equal_weight_blend(signals)

    allocation = RegimeConditionedAllocator(
        AllocationConfig(gross_leverage=1.0, min_weight=-0.20, max_weight=0.20)
    ).allocate(blended.weights, regimes.labels)
    risk_config = RiskConfig(max_asset_weight=0.20, max_gross_exposure=1.0, volatility_target=0.12)
    constrained = ExposureConstraint(risk_config).apply(allocation.weights)
    targeted = VolatilityTargeter(risk_config).scale_weights(constrained, returns)

    result = VectorizedBacktester(BacktestConfig(rebalance_frequency="W-FRI")).run(prices, targeted)
    contribution = PerformanceAttributor.strategy_contribution(prices, signals)
    report = PerformanceReport.from_backtest(result, regimes.labels)

    benchmark = BenchmarkEngine().compare(result.returns, prices)
    correlation = StrategyCorrelationAnalyzer().analyze(contribution)
    factors = build_equity_style_factors(prices, bundle.volume())
    factor_result = FactorExposureAnalyzer().analyze(result.returns, factors, regimes.labels)
    regime_analytics = RegimeAnalytics().analyze(regimes.labels, result.returns, contribution, result.weights)
    diagnostics = PortfolioDiagnostics().analyze(result.weights, returns)
    cost_sensitivity = CostSensitivityAnalyzer(
        CostSensitivityConfig(slippage_bps=(0.0, 1.0, 3.0), commission_bps=(0.0, 2.0), spread_bps=(0.0, 2.0))
    ).sweep(prices, targeted)
    robustness = ParameterRobustnessTester(RobustnessConfig(metric="sharpe")).grid_search(
        prices,
        {"lookback": [63, 126], "top_quantile": [0.2, 0.3]},
        lambda params: CrossSectionalMomentumStrategy(
            StrategyConfig(
                name="robustness_cross_sectional_momentum",
                lookback=int(params["lookback"]),
                params={"top_quantile": params["top_quantile"], "bottom_quantile": params["top_quantile"]},
            )
        ).generate(bundle),
    )
    walk_forward = WalkForwardValidator(
        WalkForwardConfig(train_window=252, test_window=126, mode="rolling")
    ).evaluate(
        prices,
        lambda frame, params: CrossSectionalMomentumStrategy(
            StrategyConfig(
                name="walk_forward_cross_sectional_momentum",
                lookback=int(params.get("lookback", 126)) if params else 126,
            )
        ).generate(frame),
        regimes=regimes.labels,
        parameter_refitter=lambda train_prices: {"lookback": 126 if len(train_prices) > 300 else 63},
    )
    quality_report = MarketDataQualityValidator().validate(bundle.ohlcv)
    data_quality_summary = pd.DataFrame(
        {
            "count": {
                "missing_data_issues": len(quality_report.missing_data),
                "stale_price_issues": len(quality_report.stale_prices),
                "outlier_issues": len(quality_report.outliers),
                "adjustment_warnings": len(quality_report.adjustment_warnings),
                "duplicate_timestamps": int(quality_report.duplicate_timestamps.sum()),
                "survivorship_warnings": len(quality_report.survivorship_warnings),
            }
        }
    )

    report.benchmark_metrics = benchmark.metrics
    report.factor_exposures = factor_result.static_betas.to_frame("beta")
    report.regime_transition_matrix = regime_analytics.transition_probabilities
    report.portfolio_diagnostics = diagnostics

    report_path = REPORT_DIR / "research_report.html"
    AdvancedHTMLReportBuilder().write(
        report,
        report_path,
        HTMLReportInputs(
            regimes=regimes.labels,
            strategy_contribution=contribution,
            benchmark_equity=benchmark.benchmark_equity,
            benchmark_metrics=benchmark.metrics,
            walk_forward_metrics=walk_forward.window_metrics,
            correlation_matrix=correlation.static_correlation,
            factor_betas=factor_result.static_betas,
            regime_transition_matrix=regime_analytics.transition_probabilities,
            robustness_results=robustness.results,
            cost_sensitivity=cost_sensitivity,
            portfolio_diagnostics={
                "turnover": diagnostics["turnover"],
                "exposure_concentration": diagnostics["exposure_concentration"],
                "leverage_utilization": diagnostics["leverage_utilization"],
                "expected_shortfall": diagnostics["expected_shortfall"],
                "cvar_decomposition": diagnostics["cvar_decomposition"],
            },
            data_quality_summary=data_quality_summary,
        ),
    )
    ExperimentTracker(PROJECT_ROOT / "outputs" / "experiments").start_run(
        name="synthetic_regime_aware_research",
        config={"source": "synthetic", "symbols": prices.columns.tolist()},
        metrics=report.metrics,
        parameters={"strategy_count": len(signals), "feature_panel_count": len(features)},
        artifacts={"html_report": str(report_path)},
    )

    print("Performance metrics")
    print(pd.Series(report.metrics).round(4).to_string())
    print("\nRegime breakdown")
    print(report.regime_breakdown.round(4).to_string())
    print("\nStrategy contribution tail")
    print(contribution.cumsum().tail().round(4).to_string())
    print("\nBenchmark comparison")
    print(benchmark.metrics.round(4).to_string())
    print("\nWalk-forward summary")
    print(pd.Series(walk_forward.summary()).round(4).to_string())
    print(f"\nFeature panels generated: {len(features)}")
    print(f"HTML report written to: {report_path}")


if __name__ == "__main__":
    main()
