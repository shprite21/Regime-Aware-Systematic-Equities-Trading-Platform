"""Parameter robustness and sensitivity testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import pandas as pd

from analytics.metrics import aggregate_metrics
from backtesting import BacktestConfig, VectorizedBacktester
from strategies.base import StrategySignal


@dataclass(frozen=True)
class RobustnessConfig:
    """Parameter sweep configuration."""

    metric: str = "sharpe"
    rebalance_frequency: str | None = "W-FRI"
    execution_lag: int = 1


@dataclass
class RobustnessResult:
    """Parameter robustness outputs."""

    results: pd.DataFrame
    heatmaps: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict)
    stability_summary: dict[str, float] = field(default_factory=dict)

    def top(self, n: int = 10, metric: str | None = None) -> pd.DataFrame:
        """Return top parameter sets by metric."""

        selected_metric = metric or self.stability_summary.get("primary_metric", "sharpe")
        if selected_metric not in self.results:
            selected_metric = "sharpe"
        return self.results.sort_values(selected_metric, ascending=False).head(n)


class ParameterRobustnessTester:
    """Run grid search and parameter sensitivity diagnostics."""

    def __init__(self, config: RobustnessConfig | None = None) -> None:
        self.config = config or RobustnessConfig()

    def grid_search(
        self,
        prices: pd.DataFrame,
        parameter_grid: dict[str, list[object]],
        strategy_factory: Callable[[dict[str, object]], StrategySignal],
    ) -> RobustnessResult:
        """Run a parameter grid search for a strategy factory."""

        rows = []
        backtester = VectorizedBacktester(
            BacktestConfig(
                rebalance_frequency=self.config.rebalance_frequency,
                execution_lag=self.config.execution_lag,
            )
        )
        for parameters in _parameter_product(parameter_grid):
            signal = strategy_factory(parameters)
            result = backtester.run(prices, signal.weights)
            metrics = aggregate_metrics(result.returns, result.equity_curve)
            rows.append(
                {
                    **parameters,
                    **metrics,
                    "average_turnover": float(result.turnover.mean()),
                }
            )
        results = pd.DataFrame(rows)
        heatmaps = self._build_heatmaps(results, list(parameter_grid))
        stability = self._stability(results)
        stability["primary_metric"] = self.config.metric
        return RobustnessResult(results=results, heatmaps=heatmaps, stability_summary=stability)

    def _build_heatmaps(self, results: pd.DataFrame, parameter_names: list[str]) -> dict[tuple[str, str], pd.DataFrame]:
        heatmaps = {}
        for left_index, left in enumerate(parameter_names):
            for right in parameter_names[left_index + 1 :]:
                if left in results and right in results and self.config.metric in results:
                    heatmaps[(left, right)] = results.pivot_table(
                        index=left,
                        columns=right,
                        values=self.config.metric,
                        aggfunc="mean",
                    )
        return heatmaps

    def _stability(self, results: pd.DataFrame) -> dict[str, float]:
        if results.empty or self.config.metric not in results:
            return {}
        metric = results[self.config.metric]
        best = metric.max()
        median = metric.median()
        degradation = float((best - median) / abs(best)) if best else 0.0
        return {
            "metric_mean": float(metric.mean()),
            "metric_median": float(median),
            "metric_std": float(metric.std()),
            "performance_degradation": degradation,
            "stable_fraction": float((metric >= median).mean()),
        }


def _parameter_product(parameter_grid: dict[str, list[object]]) -> list[dict[str, object]]:
    keys = list(parameter_grid)
    values = [parameter_grid[key] for key in keys]
    return [dict(zip(keys, combination)) for combination in product(*values)]

