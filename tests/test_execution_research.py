from __future__ import annotations

import pandas as pd

from execution import ExecutionConfig, ExecutionSimulator, Order
from research import ResearchPipeline
from strategies import CrossSectionalMomentumStrategy, MeanReversionStrategy


def test_execution_simulator_creates_cost_aware_fills(synthetic_prices):
    simulator = ExecutionSimulator(ExecutionConfig(slippage_bps=2.0, commission_bps=1.0))
    timestamp = synthetic_prices.index[10]
    orders = [Order(timestamp=timestamp, symbol="AAA", quantity=100.0, side="buy")]
    volumes = pd.DataFrame(10_000.0, index=synthetic_prices.index, columns=synthetic_prices.columns)

    fills = simulator.simulate_orders(orders, synthetic_prices, volumes)

    assert len(fills) == 1
    assert fills[0].price > synthetic_prices.loc[timestamp, "AAA"]
    assert fills[0].commission > 0


def test_research_pipeline_runs_end_to_end(synthetic_bundle):
    pipeline = ResearchPipeline([CrossSectionalMomentumStrategy(), MeanReversionStrategy()])
    result = pipeline.run(synthetic_bundle)

    assert result.features
    assert not result.regimes.empty
    assert result.signals
    assert result.backtest.equity_curve.gt(0).all()
    assert "sharpe" in result.report.metrics
