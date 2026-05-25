from __future__ import annotations

import numpy as np

from strategies import (
    ATRBreakoutStrategy,
    BollingerBandStrategy,
    CointegrationPairsStrategy,
    CrossSectionalMomentumStrategy,
    EarningsMomentumStrategy,
    GapContinuationReversalStrategy,
    MeanReversionStrategy,
    MovingAverageTrendStrategy,
    PEADStrategy,
    PairsTradingStrategy,
    RSIReversalStrategy,
    SectorRotationStrategy,
    ShortTermReversalStrategy,
    TimeSeriesMomentumStrategy,
    VolatilityBreakoutStrategy,
    VolatilityCompressionStrategy,
)


def test_strategy_signal_contracts(synthetic_bundle):
    strategies = [
        CrossSectionalMomentumStrategy(),
        TimeSeriesMomentumStrategy(),
        MovingAverageTrendStrategy(),
        MeanReversionStrategy(),
        ShortTermReversalStrategy(),
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
        SectorRotationStrategy(),
    ]

    prices = synthetic_bundle.close()
    for strategy in strategies:
        signal = strategy.generate(synthetic_bundle)
        assert signal.weights.index.equals(prices.index)
        assert set(signal.weights.columns).issubset(set(prices.columns))
        assert np.isfinite(signal.weights.to_numpy()).all()
        assert (signal.weights.abs().sum(axis=1) <= strategy.config.gross_leverage + 1e-9).all()


def test_cross_sectional_momentum_generates_nonzero_weights(synthetic_bundle):
    signal = CrossSectionalMomentumStrategy().generate(synthetic_bundle)
    assert signal.weights.abs().sum(axis=1).iloc[150:].gt(0).any()
