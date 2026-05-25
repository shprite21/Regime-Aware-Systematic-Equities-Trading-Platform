from __future__ import annotations

import numpy as np
import pandas as pd

from analytics import (
    AttributionAnalyzer,
    BenchmarkEngine,
    CorrelationConfig,
    CostSensitivityAnalyzer,
    CostSensitivityConfig,
    FactorExposureAnalyzer,
    FactorExposureConfig,
    PortfolioDiagnostics,
    RegimeAnalytics,
    StrategyCorrelationAnalyzer,
    build_equity_style_factors,
)
from backtesting import BacktestConfig, VectorizedBacktester
from strategies import CrossSectionalMomentumStrategy, MeanReversionStrategy
from strategies.library import StrategyPipeline


def test_benchmarking_factor_correlation_and_regime_analytics(synthetic_bundle):
    prices = synthetic_bundle.close()
    returns = prices.pct_change().fillna(0.0)
    signal = CrossSectionalMomentumStrategy().generate(synthetic_bundle)
    result = VectorizedBacktester(BacktestConfig(rebalance_frequency="W-FRI")).run(prices, signal.weights)

    benchmark = BenchmarkEngine().compare(result.returns, prices)
    assert {"equal_weight", "buy_and_hold", "momentum_only", "risk_parity"}.issubset(
        set(benchmark.benchmark_returns.columns)
    )
    assert "information_ratio" in benchmark.metrics.columns

    signals = StrategyPipeline([CrossSectionalMomentumStrategy(), MeanReversionStrategy()]).generate(synthetic_bundle)
    strategy_returns = pd.DataFrame(
        {
            name: (strategy_signal.weights.shift(1).fillna(0.0) * returns).sum(axis=1)
            for name, strategy_signal in signals.items()
        }
    )
    correlation = StrategyCorrelationAnalyzer(CorrelationConfig(rolling_window=21)).analyze(strategy_returns)
    assert correlation.static_correlation.shape[0] == strategy_returns.shape[1]
    assert 0 <= correlation.diversification_score <= 1

    factors = build_equity_style_factors(prices, synthetic_bundle.volume())
    factor_result = FactorExposureAnalyzer(FactorExposureConfig(rolling_window=40)).analyze(
        result.returns,
        factors,
    )
    assert "market" in factor_result.static_betas.index
    assert not factor_result.neutrality_diagnostics.empty

    regimes = pd.Series([0, 1, 2, 1] * (len(result.returns) // 4), index=result.returns.index[: (len(result.returns) // 4) * 4])
    regimes = regimes.reindex(result.returns.index).ffill().fillna(1)
    regime_result = RegimeAnalytics().analyze(regimes, result.returns, strategy_returns)
    assert not regime_result.transition_probabilities.empty
    assert not regime_result.conditional_performance.empty


def test_cost_sensitivity_portfolio_diagnostics_and_attribution(synthetic_bundle):
    prices = synthetic_bundle.close()
    returns = prices.pct_change().fillna(0.0)
    weights = pd.DataFrame(1.0 / prices.shape[1], index=prices.index, columns=prices.columns)
    result = VectorizedBacktester().run(prices, weights)

    cost = CostSensitivityAnalyzer(
        CostSensitivityConfig(slippage_bps=(0.0, 1.0), commission_bps=(0.0, 2.0), spread_bps=(0.0,))
    ).sweep(prices, weights)
    assert {"cost_adjusted_sharpe", "implementation_shortfall"}.issubset(cost.columns)

    diagnostics = PortfolioDiagnostics().analyze(result.weights, returns)
    assert "expected_shortfall" in diagnostics
    assert np.isfinite(diagnostics["expected_shortfall"])

    attribution = AttributionAnalyzer()
    asset_contribution = attribution.asset_attribution(result.weights, returns)
    regime_contribution = attribution.regime_attribution(result.returns, pd.Series(1, index=result.returns.index))
    assert asset_contribution.shape == returns.shape
    assert not regime_contribution.empty

