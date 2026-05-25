"""Run focused advanced diagnostics on the synthetic research dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analytics import (
    BenchmarkEngine,
    FactorExposureAnalyzer,
    RegimeAnalytics,
    StrategyCorrelationAnalyzer,
    build_equity_style_factors,
)
from backtesting import VectorizedBacktester
from scripts.run_research import make_synthetic_bundle
from strategies import CrossSectionalMomentumStrategy, MeanReversionStrategy, TimeSeriesMomentumStrategy
from strategies.library import StrategyPipeline


def main() -> None:
    bundle = make_synthetic_bundle()
    prices = bundle.close()
    returns = prices.pct_change().fillna(0.0)
    signals = StrategyPipeline(
        [CrossSectionalMomentumStrategy(), TimeSeriesMomentumStrategy(), MeanReversionStrategy()]
    ).generate(bundle)
    strategy_returns = pd.DataFrame(
        {
            name: (signal.weights.shift(1).fillna(0.0) * returns).sum(axis=1)
            for name, signal in signals.items()
        }
    )
    blended = StrategyPipeline.equal_weight_blend(signals)
    result = VectorizedBacktester().run(prices, blended.weights)
    benchmarks = BenchmarkEngine().compare(result.returns, prices)
    correlations = StrategyCorrelationAnalyzer().analyze(strategy_returns)
    factors = FactorExposureAnalyzer().analyze(result.returns, build_equity_style_factors(prices, bundle.volume()))
    regimes = pd.Series(1, index=result.returns.index)
    regime_analytics = RegimeAnalytics().analyze(regimes, result.returns, strategy_returns)

    print("Benchmark metrics")
    print(benchmarks.metrics.round(4).to_string())
    print("\nStrategy correlation matrix")
    print(correlations.static_correlation.round(4).to_string())
    print("\nStatic factor betas")
    print(factors.static_betas.round(4).to_string())
    print("\nRegime transition probabilities")
    print(regime_analytics.transition_probabilities.round(4).to_string())


if __name__ == "__main__":
    main()

