"""Institutional-style walk-forward validation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import pandas as pd

from analytics.metrics import aggregate_metrics
from backtesting.engine import BacktestConfig, BacktestResult, VectorizedBacktester
from strategies.base import StrategySignal


@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward validation configuration."""

    train_window: int = 252
    test_window: int = 63
    step_size: int | None = None
    mode: str = "rolling"
    min_train_window: int = 252
    rebalance_frequency: str | None = "W-FRI"
    execution_lag: int = 1


@dataclass
class WalkForwardWindowResult:
    """Single train/test walk-forward window result."""

    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    test_result: BacktestResult
    train_regime_counts: dict[object, int] = field(default_factory=dict)
    test_regime_counts: dict[object, int] = field(default_factory=dict)
    parameters: dict[str, object] = field(default_factory=dict)


@dataclass
class WalkForwardValidationResult:
    """Aggregated walk-forward validation outputs."""

    windows: list[WalkForwardWindowResult]
    oos_returns: pd.Series
    oos_equity_curve: pd.Series
    window_metrics: pd.DataFrame
    stability_metrics: dict[str, float]

    def summary(self) -> dict[str, float]:
        """Return aggregate out-of-sample metrics."""

        metrics = aggregate_metrics(self.oos_returns, self.oos_equity_curve)
        metrics.update(self.stability_metrics)
        return metrics


class WalkForwardValidator:
    """Run expanding or rolling walk-forward strategy validation."""

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig()

    def evaluate(
        self,
        prices: pd.DataFrame,
        strategy_factory: Callable[[pd.DataFrame, dict[str, object] | None], StrategySignal],
        regimes: pd.Series | None = None,
        parameter_refitter: Callable[[pd.DataFrame], dict[str, object]] | None = None,
    ) -> WalkForwardValidationResult:
        """Evaluate a strategy factory across train/test windows."""

        prices = prices.sort_index()
        windows: list[WalkForwardWindowResult] = []
        oos_returns = []
        backtester = VectorizedBacktester(
            BacktestConfig(
                rebalance_frequency=self.config.rebalance_frequency,
                execution_lag=self.config.execution_lag,
            )
        )
        for window_id, (train_slice, test_slice) in enumerate(self._window_slices(prices), start=1):
            train_prices = prices.iloc[train_slice]
            test_prices = prices.iloc[test_slice]
            parameters = parameter_refitter(train_prices) if parameter_refitter else {}
            train_signal = strategy_factory(train_prices, parameters)
            train_result = backtester.run(train_prices, train_signal.weights)
            full_signal = strategy_factory(prices.iloc[: test_slice.stop], parameters)
            test_weights = full_signal.weights.reindex(prices.index).ffill().iloc[test_slice].fillna(0.0)
            test_result = backtester.run(test_prices, test_weights)
            train_regime_counts, test_regime_counts = _regime_counts(regimes, train_prices.index, test_prices.index)
            windows.append(
                WalkForwardWindowResult(
                    window_id=window_id,
                    train_start=train_prices.index[0],
                    train_end=train_prices.index[-1],
                    test_start=test_prices.index[0],
                    test_end=test_prices.index[-1],
                    train_metrics=train_result.summary(),
                    test_metrics=test_result.summary(),
                    test_result=test_result,
                    train_regime_counts=train_regime_counts,
                    test_regime_counts=test_regime_counts,
                    parameters=parameters,
                )
            )
            oos_returns.append(test_result.returns)
        combined_returns = pd.concat(oos_returns).sort_index() if oos_returns else pd.Series(dtype=float)
        combined_returns = combined_returns[~combined_returns.index.duplicated(keep="first")]
        combined_equity = (1.0 + combined_returns).cumprod() * backtester.config.initial_capital
        window_metrics = self._window_metrics_frame(windows)
        return WalkForwardValidationResult(
            windows=windows,
            oos_returns=combined_returns,
            oos_equity_curve=combined_equity,
            window_metrics=window_metrics,
            stability_metrics=self._stability_metrics(window_metrics),
        )

    def _window_slices(self, prices: pd.DataFrame) -> Iterable[tuple[slice, slice]]:
        step = self.config.step_size or self.config.test_window
        start = self.config.min_train_window
        while start < len(prices):
            test_stop = min(start + self.config.test_window, len(prices))
            if self.config.mode == "expanding":
                train_start = 0
            elif self.config.mode == "rolling":
                train_start = max(0, start - self.config.train_window)
            else:
                raise ValueError("WalkForwardConfig.mode must be 'rolling' or 'expanding'")
            if start - train_start < self.config.min_train_window or test_stop - start < 2:
                break
            yield slice(train_start, start), slice(start, test_stop)
            start += step

    @staticmethod
    def _window_metrics_frame(windows: list[WalkForwardWindowResult]) -> pd.DataFrame:
        rows = []
        for window in windows:
            row = {
                "window_id": window.window_id,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
            }
            row.update({f"train_{key}": value for key, value in window.train_metrics.items()})
            row.update({f"test_{key}": value for key, value in window.test_metrics.items()})
            rows.append(row)
        return pd.DataFrame(rows).set_index("window_id") if rows else pd.DataFrame()

    @staticmethod
    def _stability_metrics(window_metrics: pd.DataFrame) -> dict[str, float]:
        if window_metrics.empty:
            return {"oos_sharpe_stability": 0.0, "oos_drawdown_stability": 0.0}
        sharpe = window_metrics.get("test_sharpe", pd.Series(dtype=float))
        drawdown = window_metrics.get("test_max_drawdown", pd.Series(dtype=float))
        sharpe_mean = float(sharpe.mean()) if not sharpe.empty else 0.0
        sharpe_std = float(sharpe.std()) if not sharpe.empty else 0.0
        return {
            "oos_sharpe_mean": sharpe_mean,
            "oos_sharpe_stability": sharpe_mean / sharpe_std if sharpe_std and pd.notna(sharpe_std) else 0.0,
            "oos_drawdown_mean": float(drawdown.mean()) if not drawdown.empty else 0.0,
            "oos_drawdown_stability": float(drawdown.std()) if not drawdown.empty else 0.0,
        }


def _regime_counts(
    regimes: pd.Series | None,
    train_index: pd.Index,
    test_index: pd.Index,
) -> tuple[dict[object, int], dict[object, int]]:
    if regimes is None:
        return {}, {}
    aligned = regimes.sort_index()
    train_counts = aligned.reindex(train_index).ffill().value_counts().to_dict()
    test_counts = aligned.reindex(test_index).ffill().value_counts().to_dict()
    return train_counts, test_counts
