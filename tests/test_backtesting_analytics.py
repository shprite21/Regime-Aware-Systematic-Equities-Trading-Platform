from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.metrics import (
    aggregate_metrics,
    exposure_analysis,
    factor_decomposition,
    regime_performance,
    rolling_correlations,
)
from analytics.tearsheet import PerformanceReport
from backtesting import BacktestConfig, EventDrivenBacktester, RollingRetrainingEvaluator, VectorizedBacktester
from strategies import CrossSectionalMomentumStrategy


def test_backtester_and_report_produce_finite_metrics(synthetic_prices):
    weights = pd.DataFrame(
        1.0 / synthetic_prices.shape[1],
        index=synthetic_prices.index,
        columns=synthetic_prices.columns,
    )
    result = VectorizedBacktester(
        BacktestConfig(rebalance_frequency="W-FRI", transaction_cost_bps=1.0, slippage_bps=1.0)
    ).run(synthetic_prices, weights)
    regimes = pd.Series(1, index=synthetic_prices.index)
    report = PerformanceReport.from_backtest(result, regimes)

    assert result.equity_curve.gt(0).all()
    assert np.isfinite(result.returns.to_numpy()).all()
    assert np.isfinite(list(report.metrics.values())).all()
    assert report.regime_breakdown is not None
    assert not report.regime_breakdown.empty


def test_aggregate_and_regime_metrics_are_stable(synthetic_prices):
    returns = synthetic_prices.pct_change().mean(axis=1).fillna(0.0)
    equity = (1.0 + returns).cumprod()
    regimes = pd.Series([0, 1, 2, 1] * (len(returns) // 4) + [1] * (len(returns) % 4), index=returns.index)

    metrics = aggregate_metrics(returns, equity)
    by_regime = regime_performance(returns, regimes)

    assert "sharpe" in metrics
    assert np.isfinite(list(metrics.values())).all()
    assert set(by_regime.index).issubset({0, 1, 2})


def test_readme_aligned_analytics_and_backtest_helpers(synthetic_prices):
    returns = synthetic_prices.pct_change().fillna(0.0)
    weights = pd.DataFrame(
        1.0 / synthetic_prices.shape[1],
        index=synthetic_prices.index[::21],
        columns=synthetic_prices.columns,
    )
    event_result = EventDrivenBacktester().run_events(synthetic_prices, weights)
    exposures = exposure_analysis(event_result.weights)
    correlations = rolling_correlations(returns, window=21)
    factors = pd.DataFrame(
        {
            "market": returns.mean(axis=1),
            "size": returns.iloc[:, :3].mean(axis=1) - returns.iloc[:, 3:].mean(axis=1),
        }
    )
    factor_loadings = factor_decomposition(event_result.returns, factors)
    walk_forward = RollingRetrainingEvaluator().evaluate(
        synthetic_prices,
        lambda frame: CrossSectionalMomentumStrategy().generate(frame),
        train_window=126,
        test_window=42,
    )

    assert event_result.equity_curve.gt(0).all()
    assert {"long_exposure", "gross_exposure", "net_exposure"}.issubset(exposures.columns)
    assert "average_pairwise_correlation" in correlations.columns
    assert "alpha" in factor_loadings.index
    assert walk_forward
