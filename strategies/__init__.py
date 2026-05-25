"""Strategy library for systematic equities research."""

from strategies.base import BaseStrategy, StrategyConfig, StrategySignal
from strategies.event_driven import EarningsMomentumStrategy, GapContinuationReversalStrategy, PEADStrategy
from strategies.mean_reversion import (
    BollingerBandStrategy,
    MeanReversionStrategy,
    RSIReversalStrategy,
    ShortTermReversalStrategy,
)
from strategies.momentum import (
    CrossSectionalMomentumStrategy,
    MovingAverageTrendStrategy,
    TimeSeriesMomentumStrategy,
)
from strategies.sector_rotation import SectorRotationStrategy
from strategies.stat_arb import CointegrationPairsStrategy, PairsTradingStrategy
from strategies.volatility import (
    ATRBreakoutStrategy,
    VolatilityBreakoutStrategy,
    VolatilityCompressionStrategy,
)

__all__ = [
    "ATRBreakoutStrategy",
    "BaseStrategy",
    "BollingerBandStrategy",
    "CointegrationPairsStrategy",
    "CrossSectionalMomentumStrategy",
    "EarningsMomentumStrategy",
    "GapContinuationReversalStrategy",
    "MeanReversionStrategy",
    "MovingAverageTrendStrategy",
    "PEADStrategy",
    "PairsTradingStrategy",
    "RSIReversalStrategy",
    "SectorRotationStrategy",
    "ShortTermReversalStrategy",
    "StrategyConfig",
    "StrategySignal",
    "TimeSeriesMomentumStrategy",
    "VolatilityBreakoutStrategy",
    "VolatilityCompressionStrategy",
]
