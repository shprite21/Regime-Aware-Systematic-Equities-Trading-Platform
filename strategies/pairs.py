"""Pairs trading strategy."""

from __future__ import annotations

import itertools

import pandas as pd

from data.models import MarketDataBundle
from strategies.base import BaseStrategy, StrategyConfig, StrategySignal, prices_from_data
from strategies.utils import normalize_weights


class PairsTradingStrategy(BaseStrategy):
    """Mean-reverting spread strategy for configured equity pairs."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(
            config
            or StrategyConfig(
                name="pairs_trading",
                lookback=60,
                params={"pairs": None, "z_entry": 2.0, "z_exit": 0.5},
            )
        )

    def generate(self, data: MarketDataBundle | pd.DataFrame) -> StrategySignal:
        prices = self._select_universe(prices_from_data(data))
        pairs = self._resolve_pairs(prices)
        z_entry = float(self.config.params.get("z_entry", 2.0))

        raw = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        score_frames = {}
        for left, right in pairs:
            beta = self._rolling_beta(prices[left], prices[right], self.config.lookback)
            spread = prices[left] - beta * prices[right]
            zscore = (spread - spread.rolling(self.config.lookback).mean()) / spread.rolling(
                self.config.lookback
            ).std()
            zscore = zscore.shift(1)
            score_frames[f"{left}_{right}"] = zscore

            short_left = zscore > z_entry
            long_left = zscore < -z_entry
            raw.loc[short_left, left] -= 0.5
            raw.loc[short_left, right] += 0.5
            raw.loc[long_left, left] += 0.5
            raw.loc[long_left, right] -= 0.5

        if not self.config.long_short:
            raw = raw.clip(lower=0.0)

        weights = normalize_weights(raw, self.config.gross_leverage)
        scores = pd.DataFrame(score_frames, index=prices.index)
        return StrategySignal(self.config.name, weights=weights, scores=scores)

    def _resolve_pairs(self, prices: pd.DataFrame) -> list[tuple[str, str]]:
        configured = self.config.params.get("pairs")
        if configured:
            return [tuple(pair) for pair in configured]  # type: ignore[arg-type]
        columns = list(prices.columns)
        if len(columns) < 2:
            return []
        returns = prices.pct_change().dropna(how="all")
        correlations = returns.corr().where(lambda frame: frame < 1.0)
        stacked = correlations.abs().stack().sort_values(ascending=False)
        pairs: list[tuple[str, str]] = []
        used = set()
        for left, right in stacked.index:
            key = tuple(sorted((left, right)))
            if key not in used:
                pairs.append((left, right))
                used.add(key)
            if len(pairs) >= max(1, len(columns) // 4):
                break
        return pairs or list(itertools.combinations(columns[:2], 2))

    @staticmethod
    def _rolling_beta(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
        covariance = left.rolling(window).cov(right)
        variance = right.rolling(window).var().replace(0.0, pd.NA)
        return covariance / variance

