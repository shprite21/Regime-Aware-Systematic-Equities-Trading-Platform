from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio import (
    AllocationConfig,
    ConstrainedOptimizer,
    InverseVolatilityAllocator,
    RegimeConditionedAllocator,
    RollingSharpeAllocator,
)
from regime_detection import MarketBreadthRegimeDetector, RegimeEnsemble, VolatilityRegimeClassifier


def test_regime_detectors_and_ensemble_align(synthetic_prices):
    returns = synthetic_prices.pct_change().fillna(0.0)
    volatility = VolatilityRegimeClassifier(lookback=10).fit_predict(returns.mean(axis=1).to_frame("returns"))
    breadth = MarketBreadthRegimeDetector(moving_average_window=30).fit_predict(synthetic_prices)
    combined = RegimeEnsemble().combine({"volatility": volatility, "breadth": breadth})

    assert combined.labels.index.equals(synthetic_prices.index)
    assert set(combined.labels.dropna().unique()).issubset({0, 1, 2})
    assert combined.probabilities is not None


def test_allocators_return_sane_weights(synthetic_prices):
    returns = synthetic_prices.pct_change().fillna(0.0)
    inverse_vol = InverseVolatilityAllocator(
        AllocationConfig(lookback=20, long_only=True, min_weight=0.0, max_weight=0.5)
    ).allocate(returns)
    rolling_sharpe = RollingSharpeAllocator(
        AllocationConfig(lookback=20, long_only=True, min_weight=0.0, max_weight=0.7)
    ).allocate(returns)
    regimes = pd.Series(1, index=synthetic_prices.index)
    regime_weights = RegimeConditionedAllocator(
        AllocationConfig(gross_leverage=1.0, min_weight=0.0, max_weight=0.5)
    ).allocate(inverse_vol.weights, regimes)

    for result in [inverse_vol, rolling_sharpe, regime_weights]:
        assert result.weights.index.equals(synthetic_prices.index)
        assert np.isfinite(result.weights.to_numpy()).all()
        assert (result.weights.abs().sum(axis=1) <= 1.0 + 1e-8).all()


def test_constrained_optimizer_respects_long_only_budget(synthetic_prices):
    returns = synthetic_prices.pct_change().dropna()
    optimizer = ConstrainedOptimizer(
        AllocationConfig(long_only=True, min_weight=0.0, max_weight=0.5, gross_leverage=1.0)
    )
    weights = optimizer.allocate(returns.mean(), returns.cov())

    assert np.isfinite(weights.to_numpy()).all()
    assert abs(weights.sum() - 1.0) < 1e-4
    assert (weights >= -1e-8).all()
    assert (weights <= 0.5 + 1e-8).all()

