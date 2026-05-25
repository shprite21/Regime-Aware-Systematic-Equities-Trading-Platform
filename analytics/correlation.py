"""Strategy correlation, clustering, and redundancy diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorrelationConfig:
    """Configuration for strategy dependency analysis."""

    rolling_window: int = 63
    redundancy_threshold: float = 0.80
    covariance_shrinkage: float = 0.10


@dataclass
class CorrelationAnalysisResult:
    """Outputs from strategy correlation analytics."""

    static_correlation: pd.DataFrame
    covariance: pd.DataFrame
    rolling_average_correlation: pd.Series
    cluster_labels: pd.Series
    diversification_score: float
    redundant_pairs: pd.DataFrame
    linkage_matrix: np.ndarray | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


class StrategyCorrelationAnalyzer:
    """Analyze cross-strategy dependency and diversification quality."""

    def __init__(self, config: CorrelationConfig | None = None) -> None:
        self.config = config or CorrelationConfig()

    def analyze(self, strategy_returns: pd.DataFrame) -> CorrelationAnalysisResult:
        """Run static, rolling, covariance, clustering, and redundancy diagnostics."""

        returns = strategy_returns.dropna(how="all").fillna(0.0)
        static_correlation = returns.corr().fillna(0.0)
        covariance = self.covariance_estimate(returns)
        rolling_average = self.rolling_average_correlation(returns)
        cluster_labels, linkage_matrix = self.hierarchical_clusters(static_correlation)
        redundant_pairs = self.redundant_pairs(static_correlation)
        score = diversification_score(static_correlation)
        return CorrelationAnalysisResult(
            static_correlation=static_correlation,
            covariance=covariance,
            rolling_average_correlation=rolling_average,
            cluster_labels=cluster_labels,
            diversification_score=score,
            redundant_pairs=redundant_pairs,
            linkage_matrix=linkage_matrix,
        )

    def covariance_estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Shrink sample covariance toward a diagonal covariance matrix."""

        sample = returns.cov().fillna(0.0)
        diagonal = pd.DataFrame(np.diag(np.diag(sample)), index=sample.index, columns=sample.columns)
        shrinkage = self.config.covariance_shrinkage
        return (1.0 - shrinkage) * sample + shrinkage * diagonal

    def rolling_correlation_matrix(self, returns: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
        """Return rolling correlation matrices keyed by window end date."""

        matrices: dict[pd.Timestamp, pd.DataFrame] = {}
        for index_position in range(self.config.rolling_window, len(returns) + 1):
            window = returns.iloc[index_position - self.config.rolling_window : index_position]
            matrices[returns.index[index_position - 1]] = window.corr().fillna(0.0)
        return matrices

    def rolling_average_correlation(self, returns: pd.DataFrame) -> pd.Series:
        """Return rolling average pairwise strategy correlation."""

        values = {}
        for date, matrix in self.rolling_correlation_matrix(returns).items():
            values[date] = _average_off_diagonal(matrix)
        return pd.Series(values, name="rolling_average_correlation")

    def hierarchical_clusters(self, correlation: pd.DataFrame) -> tuple[pd.Series, np.ndarray | None]:
        """Cluster strategies using scipy when available, otherwise threshold groups."""

        if len(correlation) == 0:
            return pd.Series(dtype=int), None
        try:
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import squareform

            distance = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - correlation.clip(-1, 1))))
            np.fill_diagonal(distance.values, 0.0)
            linkage_matrix = linkage(squareform(distance.values), method="average")
            labels = fcluster(linkage_matrix, t=0.7, criterion="distance")
            return pd.Series(labels, index=correlation.index, name="cluster"), linkage_matrix
        except Exception:
            labels = _threshold_clusters(correlation, self.config.redundancy_threshold)
            return pd.Series(labels, index=correlation.index, name="cluster"), None

    def redundant_pairs(self, correlation: pd.DataFrame) -> pd.DataFrame:
        """Identify highly correlated strategy pairs."""

        rows = []
        columns = list(correlation.columns)
        for left_index, left in enumerate(columns):
            for right in columns[left_index + 1 :]:
                value = float(correlation.loc[left, right])
                if abs(value) >= self.config.redundancy_threshold:
                    rows.append({"left": left, "right": right, "correlation": value})
        return pd.DataFrame(rows).sort_values("correlation", ascending=False) if rows else pd.DataFrame()


def diversification_score(correlation: pd.DataFrame) -> float:
    """Score diversification from 0 to 1 using average absolute off-diagonal correlation."""

    if correlation.shape[0] <= 1:
        return 1.0
    average = abs(_average_off_diagonal(correlation.abs()))
    return float(max(0.0, min(1.0, 1.0 - average)))


def _average_off_diagonal(matrix: pd.DataFrame) -> float:
    if matrix.shape[0] <= 1:
        return 0.0
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(matrix.to_numpy()[mask].mean())


def _threshold_clusters(correlation: pd.DataFrame, threshold: float) -> list[int]:
    labels: dict[str, int] = {}
    cluster_id = 0
    for column in correlation.columns:
        if column in labels:
            continue
        cluster_id += 1
        labels[column] = cluster_id
        related = correlation.index[correlation[column].abs() >= threshold].tolist()
        for member in related:
            labels.setdefault(member, cluster_id)
    return [labels[column] for column in correlation.columns]

