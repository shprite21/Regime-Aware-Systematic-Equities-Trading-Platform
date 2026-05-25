"""Base contracts for regime detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RegimeResult:
    """Standardized output from any regime detector."""

    labels: pd.Series
    probabilities: pd.DataFrame | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)

    def align_to(self, index: pd.Index, method: str = "ffill") -> "RegimeResult":
        """Align labels and probabilities to a target index."""

        labels = self.labels.reindex(index)
        if method:
            labels = getattr(labels, method)()

        probabilities = None
        if self.probabilities is not None:
            probabilities = self.probabilities.reindex(index)
            if method:
                probabilities = getattr(probabilities, method)()

        return RegimeResult(labels=labels, probabilities=probabilities, diagnostics=self.diagnostics)

    def counts(self) -> pd.Series:
        """Return regime counts."""

        return self.labels.value_counts().sort_index()


class BaseRegimeDetector(ABC):
    """Interface for all regime detectors."""

    name: str = "base_regime_detector"

    @abstractmethod
    def fit(self, features: pd.DataFrame) -> "BaseRegimeDetector":
        """Fit the regime detector."""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> RegimeResult:
        """Predict regimes for a feature matrix."""

    def fit_predict(self, features: pd.DataFrame) -> RegimeResult:
        """Fit and predict in one call."""

        return self.fit(features).predict(features)
