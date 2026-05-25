"""Vectorized backtesting and attribution engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from risk.costs import SlippageModel, TransactionCostModel
from strategies.base import StrategySignal


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest execution assumptions."""

    initial_capital: float = 1_000_000.0
    rebalance_frequency: str | None = "W-FRI"
    execution_lag: int = 1
    transaction_cost_bps: float = 2.0
    slippage_bps: float = 1.0
    allow_short: bool = True


@dataclass
class BacktestResult:
    """Portfolio-level backtest result."""

    equity_curve: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    slippage: pd.Series
    attribution: pd.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)

    def summary(self) -> dict[str, float]:
        """Return a compact summary without importing analytics at module import time."""

        from analytics.metrics import aggregate_metrics

        return aggregate_metrics(self.returns, self.equity_curve)


class VectorizedBacktester:
    """Vectorized portfolio backtester for target-weight strategies."""

    def __init__(
        self,
        config: BacktestConfig | None = None,
        cost_model: TransactionCostModel | None = None,
        slippage_model: SlippageModel | None = None,
    ) -> None:
        self.config = config or BacktestConfig()
        self.cost_model = cost_model or TransactionCostModel(self.config.transaction_cost_bps)
        self.slippage_model = slippage_model or SlippageModel(self.config.slippage_bps)

    def run(self, prices: pd.DataFrame, target_weights: pd.DataFrame) -> BacktestResult:
        """Run a vectorized portfolio backtest."""

        prices = prices.sort_index()
        asset_returns = prices.pct_change().fillna(0.0)
        target_weights = target_weights.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
        if not self.config.allow_short:
            target_weights = target_weights.clip(lower=0.0)

        rebalanced_targets = self._apply_rebalance_frequency(target_weights)
        executed_weights = rebalanced_targets.shift(self.config.execution_lag).fillna(0.0)
        turnover = executed_weights.diff().abs().sum(axis=1).fillna(executed_weights.abs().sum(axis=1))

        attribution = executed_weights * asset_returns
        gross_returns = attribution.sum(axis=1)
        realized_volatility = gross_returns.rolling(21).std().fillna(0.0)
        costs = self.cost_model.as_return_drag(turnover)
        slippage = self.slippage_model.estimate(turnover, realized_volatility)
        net_returns = gross_returns - costs - slippage
        equity_curve = (1.0 + net_returns).cumprod() * self.config.initial_capital
        equity_curve.name = "equity"
        net_returns.name = "returns"

        return BacktestResult(
            equity_curve=equity_curve,
            returns=net_returns,
            gross_returns=gross_returns.rename("gross_returns"),
            weights=executed_weights,
            turnover=turnover.rename("turnover"),
            costs=costs.rename("transaction_costs"),
            slippage=slippage.rename("slippage"),
            attribution=attribution,
            metadata={
                "initial_capital": self.config.initial_capital,
                "rebalance_frequency": self.config.rebalance_frequency,
                "execution_lag": self.config.execution_lag,
            },
        )

    def _apply_rebalance_frequency(self, target_weights: pd.DataFrame) -> pd.DataFrame:
        if self.config.rebalance_frequency is None:
            return target_weights
        if not isinstance(target_weights.index, pd.DatetimeIndex):
            return target_weights
        rebalanced = target_weights.resample(self.config.rebalance_frequency).last()
        return rebalanced.reindex(target_weights.index).ffill().fillna(0.0)


class WalkForwardEvaluator:
    """Rolling train/test evaluation helper."""

    def __init__(self, backtester: VectorizedBacktester | None = None) -> None:
        self.backtester = backtester or VectorizedBacktester()

    def evaluate(
        self,
        prices: pd.DataFrame,
        strategy_factory: Callable[[pd.DataFrame], StrategySignal],
        train_window: int = 252,
        test_window: int = 63,
    ) -> list[BacktestResult]:
        """Evaluate a strategy over rolling out-of-sample windows."""

        results: list[BacktestResult] = []
        start = train_window
        while start < len(prices):
            stop = min(start + test_window, len(prices))
            research_slice = prices.iloc[:start]
            test_slice = prices.iloc[start:stop]
            if len(test_slice) < 2:
                break
            signal = strategy_factory(research_slice)
            test_weights = signal.weights.reindex(prices.index).ffill().iloc[start:stop].fillna(0.0)
            results.append(self.backtester.run(test_slice, test_weights))
            start = stop
        return results


class RollingRetrainingEvaluator(WalkForwardEvaluator):
    """Walk-forward evaluator that explicitly refits a strategy factory each window."""

    def evaluate(
        self,
        prices: pd.DataFrame,
        retrain_factory: Callable[[pd.DataFrame], StrategySignal],
        train_window: int = 252,
        test_window: int = 63,
    ) -> list[BacktestResult]:
        """Run rolling retraining and out-of-sample testing."""

        return super().evaluate(prices, retrain_factory, train_window, test_window)


class EventDrivenBacktester:
    """Event-driven wrapper for sparse rebalance or signal event dates."""

    def __init__(self, backtester: VectorizedBacktester | None = None) -> None:
        self.backtester = backtester or VectorizedBacktester(BacktestConfig(rebalance_frequency=None))

    def run_events(
        self,
        prices: pd.DataFrame,
        event_weights: pd.DataFrame,
    ) -> BacktestResult:
        """Forward-fill sparse event weights and run a portfolio backtest."""

        target_weights = event_weights.reindex(prices.index).ffill().fillna(0.0)
        return self.backtester.run(prices, target_weights)


class PerformanceAttributor:
    """Portfolio performance attribution utilities."""

    @staticmethod
    def asset_contribution(result: BacktestResult) -> pd.DataFrame:
        """Return asset-level return contribution."""

        return result.attribution

    @staticmethod
    def strategy_contribution(
        prices: pd.DataFrame,
        signals: dict[str, StrategySignal],
        execution_lag: int = 1,
    ) -> pd.DataFrame:
        """Estimate return contribution for each strategy signal independently."""

        returns = prices.pct_change().fillna(0.0)
        contributions = {}
        for name, signal in signals.items():
            weights = signal.weights.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
            contributions[name] = (weights.shift(execution_lag).fillna(0.0) * returns).sum(axis=1)
        return pd.DataFrame(contributions)
