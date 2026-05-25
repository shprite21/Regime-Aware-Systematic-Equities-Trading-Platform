"""Regime ensemble utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from regime_detection.base import RegimeResult


@dataclass
class RegimeEnsemble:
    """Combine multiple regime detectors with a weighted vote."""

    weights: dict[str, float] = field(default_factory=dict)
    name: str = "regime_ensemble"

    def combine(self, results: dict[str, RegimeResult]) -> RegimeResult:
        """Combine detector labels into one ordinal regime.

        Regime convention is 0 = defensive, 1 = neutral, 2 = constructive.
        """

        if not results:
            raise ValueError("At least one regime result is required")

        index = next(iter(results.values())).labels.index
        score = pd.Series(0.0, index=index)
        total_weight = 0.0
        aligned_labels = {}
        for name, result in results.items():
            weight = self.weights.get(name, 1.0)
            labels = result.align_to(index).labels.astype(float)
            aligned_labels[name] = labels
            score = score.add(labels * weight, fill_value=0.0)
            total_weight += weight

        score = score / total_weight
        labels = pd.Series(1, index=index, name="ensemble_regime")
        labels = labels.mask(score < 0.75, 0)
        labels = labels.mask(score > 1.25, 2)

        probabilities = pd.get_dummies(labels).reindex(columns=[0, 1, 2], fill_value=0).astype(float)
        return RegimeResult(
            labels=labels.astype(int),
            probabilities=probabilities,
            diagnostics={
                "score": score,
                "component_labels": pd.DataFrame(aligned_labels),
                "weights": self.weights,
            },
        )

