"""Benchmark comparison and active performance analytics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analytics.metrics import aggregate_metrics
from portfolio.allocation import AllocationConfig, RiskParityAllocator
from strategies import CrossSectionalMomentumStrategy


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for benchmark construction and comparison."""

    initial_capital: float = 1_000_000.0
    annualization_factor: int = 252
    momentum_lookback: int = 126
    risk_parity_lookback: int = 63
    include_spy_proxy: bool = True


@dataclass
class BenchmarkComparison:
    """Benchmark comparison outputs."""

    benchmark_returns: pd.DataFrame
    benchmark_equity: pd.DataFrame
    relative_returns: pd.DataFrame
    excess_returns: pd.DataFrame
    metrics: pd.DataFrame
    attribution: pd.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)


class BenchmarkEngine:
    """Build and compare systematic equity benchmarks."""

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        self.config = config or BenchmarkConfig()

    def build_benchmarks(
        self,
        prices: pd.DataFrame,
        spy_prices: pd.Series | pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Create benchmark return streams from a price panel."""

        prices = prices.sort_index()
        returns = prices.pct_change().fillna(0.0)
        benchmarks: dict[str, pd.Series] = {}

        if spy_prices is not None:
            spy_series = _as_series(spy_prices, "SPY").reindex(prices.index).ffill()
            benchmarks["SPY"] = spy_series.pct_change().fillna(0.0)
        elif self.config.include_spy_proxy:
            benchmarks["SPY_proxy"] = returns.mean(axis=1)

        benchmarks["equal_weight"] = returns.mean(axis=1)
        benchmarks["buy_and_hold"] = self._buy_and_hold_returns(prices)
        benchmarks["momentum_only"] = self._momentum_returns(prices)
        benchmarks["risk_parity"] = self._risk_parity_returns(returns)
        return pd.DataFrame(benchmarks).reindex(prices.index).fillna(0.0)

    def compare(
        self,
        strategy_returns: pd.Series,
        prices: pd.DataFrame,
        spy_prices: pd.Series | pd.DataFrame | None = None,
    ) -> BenchmarkComparison:
        """Compare a strategy return stream against benchmark return streams."""

        strategy_returns = strategy_returns.rename("strategy").sort_index().fillna(0.0)
        benchmark_returns = self.build_benchmarks(prices, spy_prices).reindex(strategy_returns.index).fillna(0.0)
        strategy_equity = (1.0 + strategy_returns).cumprod() * self.config.initial_capital
        benchmark_equity = (1.0 + benchmark_returns).cumprod() * self.config.initial_capital
        relative_returns = benchmark_returns.apply(lambda series: strategy_returns - series)
        excess_returns = relative_returns.copy()
        metrics = self._comparison_metrics(strategy_returns, benchmark_returns, strategy_equity)
        attribution = pd.DataFrame(
            {
                "strategy_return": strategy_returns,
                "average_benchmark_return": benchmark_returns.mean(axis=1),
                "active_return": strategy_returns - benchmark_returns.mean(axis=1),
            }
        )
        return BenchmarkComparison(
            benchmark_returns=benchmark_returns,
            benchmark_equity=benchmark_equity,
            relative_returns=relative_returns,
            excess_returns=excess_returns,
            metrics=metrics,
            attribution=attribution,
            metadata={"strategy_equity_final": float(strategy_equity.iloc[-1])},
        )

    def _buy_and_hold_returns(self, prices: pd.DataFrame) -> pd.Series:
        initial_weights = pd.Series(1.0 / prices.shape[1], index=prices.columns)
        shares = initial_weights / prices.iloc[0]
        portfolio_value = prices.mul(shares, axis=1).sum(axis=1)
        return portfolio_value.pct_change().fillna(0.0).rename("buy_and_hold")

    def _momentum_returns(self, prices: pd.DataFrame) -> pd.Series:
        signal = CrossSectionalMomentumStrategy().generate(prices)
        returns = prices.pct_change().fillna(0.0)
        return (signal.weights.shift(1).fillna(0.0) * returns).sum(axis=1).rename("momentum_only")

    def _risk_parity_returns(self, returns: pd.DataFrame) -> pd.Series:
        config = AllocationConfig(
            lookback=self.config.risk_parity_lookback,
            long_only=True,
            min_weight=0.0,
            max_weight=1.0,
            gross_leverage=1.0,
        )
        weights = RiskParityAllocator(config).allocate(returns).weights
        return (weights.shift(1).fillna(0.0) * returns).sum(axis=1).rename("risk_parity")

    def _comparison_metrics(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.DataFrame,
        strategy_equity: pd.Series,
    ) -> pd.DataFrame:
        rows = []
        for benchmark_name, benchmark in benchmark_returns.items():
            aligned = pd.concat([strategy_returns, benchmark], axis=1, keys=["strategy", "benchmark"]).dropna()
            if aligned.empty:
                continue
            excess = aligned["strategy"] - aligned["benchmark"]
            tracking_error = excess.std() * np.sqrt(self.config.annualization_factor)
            information_ratio = 0.0
            if tracking_error and not np.isnan(tracking_error):
                information_ratio = excess.mean() * self.config.annualization_factor / tracking_error
            alpha, beta = alpha_beta(aligned["strategy"], aligned["benchmark"], self.config.annualization_factor)
            benchmark_equity = (1.0 + aligned["benchmark"]).cumprod() * self.config.initial_capital
            benchmark_metrics = aggregate_metrics(aligned["benchmark"], benchmark_equity)
            strategy_metrics = aggregate_metrics(aligned["strategy"], strategy_equity.reindex(aligned.index))
            rows.append(
                {
                    "benchmark": benchmark_name,
                    "tracking_error": tracking_error,
                    "information_ratio": information_ratio,
                    "alpha": alpha,
                    "beta": beta,
                    "correlation": aligned["strategy"].corr(aligned["benchmark"]),
                    "strategy_sharpe": strategy_metrics["sharpe"],
                    "benchmark_sharpe": benchmark_metrics["sharpe"],
                    "excess_total_return": strategy_metrics["total_return"] - benchmark_metrics["total_return"],
                }
            )
        return pd.DataFrame(rows).set_index("benchmark") if rows else pd.DataFrame()


def alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    annualization_factor: int = 252,
) -> tuple[float, float]:
    """Estimate annualized alpha and beta against a benchmark."""

    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, keys=["strategy", "benchmark"]).dropna()
    if aligned.empty or aligned["benchmark"].var() == 0:
        return 0.0, 0.0
    beta = float(aligned["strategy"].cov(aligned["benchmark"]) / aligned["benchmark"].var())
    alpha = float((aligned["strategy"].mean() - beta * aligned["benchmark"].mean()) * annualization_factor)
    return alpha, beta


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    annualization_factor: int = 252,
) -> float:
    """Annualized information ratio."""

    excess = (strategy_returns - benchmark_returns).dropna()
    tracking_error = excess.std() * np.sqrt(annualization_factor)
    if tracking_error == 0 or np.isnan(tracking_error):
        return 0.0
    return float(excess.mean() * annualization_factor / tracking_error)


def tracking_error(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    annualization_factor: int = 252,
) -> float:
    """Annualized tracking error."""

    return float((strategy_returns - benchmark_returns).dropna().std() * np.sqrt(annualization_factor))


def _as_series(frame_or_series: pd.Series | pd.DataFrame, name: str) -> pd.Series:
    if isinstance(frame_or_series, pd.Series):
        return frame_or_series.rename(name)
    if "Adj Close" in frame_or_series.columns:
        return frame_or_series["Adj Close"].rename(name)
    if "Close" in frame_or_series.columns:
        return frame_or_series["Close"].rename(name)
    return frame_or_series.iloc[:, 0].rename(name)

