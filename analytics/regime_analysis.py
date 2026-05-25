"""Regime transition and conditional performance analytics."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from analytics.metrics import aggregate_metrics


@dataclass
class RegimeAnalyticsResult:
    """Regime transition analytics outputs."""

    transition_counts: pd.DataFrame
    transition_probabilities: pd.DataFrame
    persistence: pd.Series
    duration_stats: pd.DataFrame
    conditional_performance: pd.DataFrame
    strategy_effectiveness: pd.DataFrame
    allocation_diagnostics: pd.DataFrame | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class RegimeAnalytics:
    """Analyze regime transitions, persistence, and conditional strategy performance."""

    def analyze(
        self,
        regimes: pd.Series,
        portfolio_returns: pd.Series,
        strategy_returns: pd.DataFrame | None = None,
        weights: pd.DataFrame | None = None,
    ) -> RegimeAnalyticsResult:
        """Run regime transition and conditional performance analysis."""

        regimes = regimes.dropna()
        transition_counts = transition_matrix(regimes, normalize=False)
        transition_probabilities = transition_matrix(regimes, normalize=True)
        durations = regime_durations(regimes)
        duration_stats = durations.groupby("regime")["duration"].describe() if not durations.empty else pd.DataFrame()
        persistence = pd.Series(
            {
                regime: transition_probabilities.loc[regime, regime]
                for regime in transition_probabilities.index
                if regime in transition_probabilities.columns
            },
            name="persistence_probability",
        )
        conditional = conditional_performance(portfolio_returns, regimes)
        effectiveness = (
            strategy_effectiveness_by_regime(strategy_returns, regimes)
            if strategy_returns is not None
            else pd.DataFrame()
        )
        allocation_diagnostics = (
            transition_allocation_diagnostics(weights, regimes) if weights is not None else None
        )
        return RegimeAnalyticsResult(
            transition_counts=transition_counts,
            transition_probabilities=transition_probabilities,
            persistence=persistence,
            duration_stats=duration_stats,
            conditional_performance=conditional,
            strategy_effectiveness=effectiveness,
            allocation_diagnostics=allocation_diagnostics,
        )


def transition_matrix(regimes: pd.Series, normalize: bool = True) -> pd.DataFrame:
    """Create a regime transition matrix."""

    current = regimes.shift(1).dropna()
    next_regime = regimes.reindex(current.index)
    matrix = pd.crosstab(current, next_regime)
    if normalize:
        return matrix.div(matrix.sum(axis=1).replace(0, pd.NA), axis=0).fillna(0.0)
    return matrix


def regime_durations(regimes: pd.Series) -> pd.DataFrame:
    """Return contiguous regime duration observations."""

    if regimes.empty:
        return pd.DataFrame(columns=["regime", "start", "end", "duration"])
    groups = (regimes != regimes.shift()).cumsum()
    rows = []
    for _, group in regimes.groupby(groups):
        rows.append(
            {
                "regime": group.iloc[0],
                "start": group.index[0],
                "end": group.index[-1],
                "duration": len(group),
            }
        )
    return pd.DataFrame(rows)


def average_regime_duration(regimes: pd.Series) -> pd.Series:
    """Average contiguous duration by regime."""

    durations = regime_durations(regimes)
    if durations.empty:
        return pd.Series(dtype=float)
    return durations.groupby("regime")["duration"].mean()


def conditional_performance(returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    """Portfolio performance metrics by regime."""

    aligned = pd.DataFrame({"returns": returns, "regime": regimes.reindex(returns.index).ffill()}).dropna()
    rows = []
    for regime, group in aligned.groupby("regime"):
        equity = (1.0 + group["returns"]).cumprod()
        metrics = aggregate_metrics(group["returns"], equity)
        metrics["regime"] = regime
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("regime") if rows else pd.DataFrame()


def strategy_effectiveness_by_regime(strategy_returns: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    """Compute strategy Sharpe by regime."""

    rows = []
    aligned = strategy_returns.join(regimes.rename("regime"), how="inner").dropna()
    for regime, group in aligned.groupby("regime"):
        for strategy in strategy_returns.columns:
            metrics = aggregate_metrics(group[strategy], (1.0 + group[strategy]).cumprod())
            rows.append({"regime": regime, "strategy": strategy, "sharpe": metrics["sharpe"], "total_return": metrics["total_return"]})
    return pd.DataFrame(rows)


def transition_allocation_diagnostics(weights: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    """Measure allocation changes around regime transitions."""

    aligned_regimes = regimes.reindex(weights.index).ffill()
    transitions = aligned_regimes != aligned_regimes.shift(1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    return pd.DataFrame(
        {
            "regime": aligned_regimes,
            "transition": transitions,
            "turnover": turnover,
            "gross_exposure": weights.abs().sum(axis=1),
            "net_exposure": weights.sum(axis=1),
        }
    )

