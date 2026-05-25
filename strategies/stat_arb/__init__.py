"""Statistical arbitrage strategy family."""

from strategies.pairs import PairsTradingStrategy
from strategies.stat_arb.core import CointegrationPairsStrategy

__all__ = [
    "CointegrationPairsStrategy",
    "PairsTradingStrategy",
]

