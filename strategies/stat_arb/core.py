"""Statistical arbitrage strategy family."""

from __future__ import annotations

import itertools

import pandas as pd

from data.models import MarketDataBundle
from strategies.base import StrategyConfig, StrategySignal, prices_from_data
from strategies.pairs import PairsTradingStrategy


class CointegrationPairsStrategy(PairsTradingStrategy):
    """Pairs strategy that selects pairs using cointegration tests when available."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(
            config
            or StrategyConfig(
                name="cointegration_pairs",
                lookback=90,
                params={"pairs": None, "z_entry": 2.0, "max_pvalue": 0.10, "max_pairs": 5},
            )
        )

    def generate(self, data: MarketDataBundle | pd.DataFrame) -> StrategySignal:
        prices = self._select_universe(prices_from_data(data))
        selected_pairs = self._cointegrated_pairs(prices)
        original_pairs = self.config.params.get("pairs")
        self.config.params["pairs"] = selected_pairs
        try:
            return super().generate(prices)
        finally:
            self.config.params["pairs"] = original_pairs

    def _cointegrated_pairs(self, prices: pd.DataFrame) -> list[tuple[str, str]]:
        configured = self.config.params.get("pairs")
        if configured:
            return [tuple(pair) for pair in configured]  # type: ignore[arg-type]

        max_pvalue = float(self.config.params.get("max_pvalue", 0.10))
        max_pairs = int(self.config.params.get("max_pairs", 5))
        candidates: list[tuple[float, tuple[str, str]]] = []
        for left, right in itertools.combinations(prices.columns, 2):
            pair_prices = prices[[left, right]].dropna()
            if len(pair_prices) < self.config.lookback:
                continue
            pvalue = _cointegration_pvalue(pair_prices[left], pair_prices[right])
            if pvalue <= max_pvalue:
                candidates.append((pvalue, (left, right)))
        candidates.sort(key=lambda item: item[0])
        if candidates:
            return [pair for _, pair in candidates[:max_pairs]]
        return self._resolve_pairs(prices)[:max_pairs]


def _cointegration_pvalue(left: pd.Series, right: pd.Series) -> float:
    try:
        from statsmodels.tsa.stattools import coint

        return float(coint(left, right)[1])
    except Exception:
        correlation = left.pct_change().corr(right.pct_change())
        if pd.isna(correlation):
            return 1.0
        return float(max(0.0, 1.0 - abs(correlation)))

