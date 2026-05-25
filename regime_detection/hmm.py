"""Hidden Markov Model regime detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:  # pragma: no cover - depends on installed research stack
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover - lean runtime fallback
    GaussianMixture = None
    StandardScaler = None

from regime_detection.base import BaseRegimeDetector, RegimeResult


@dataclass
class HMMRegimeDetector(BaseRegimeDetector):
    """Gaussian HMM detector for latent market regimes.

    If ``hmmlearn`` is unavailable, the detector falls back to a Gaussian
    mixture model with the same standardized feature interface. This keeps the
    research pipeline usable in lean environments while preserving the HMM path
    for full installations.
    """

    n_states: int = 3
    covariance_type: str = "diag"
    random_state: int = 42
    n_iter: int = 250

    name: str = "hmm"

    def __post_init__(self) -> None:
        self.scaler_: object | None = None
        self.model_: object | None = None
        self.feature_columns_: list[str] | None = None
        self.state_order_: dict[int, int] | None = None
        self.backend_: str | None = None

    def fit(self, features: pd.DataFrame) -> "HMMRegimeDetector":
        matrix = self._prepare_features(features)
        self.feature_columns_ = list(matrix.columns)
        self.scaler_ = StandardScaler() if StandardScaler is not None else _SimpleStandardScaler()
        scaled = self.scaler_.fit_transform(matrix)

        try:
            from hmmlearn.hmm import GaussianHMM

            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            self.backend_ = "hmmlearn.GaussianHMM"
        except ImportError:  # pragma: no cover - depends on optional environment
            if GaussianMixture is not None:
                model = GaussianMixture(
                    n_components=self.n_states,
                    covariance_type="diag",
                    random_state=self.random_state,
                    max_iter=self.n_iter,
                )
                self.backend_ = "sklearn.GaussianMixture"
            else:
                model = _QuantileStateModel(n_states=self.n_states)
                self.backend_ = "quantile_fallback"

        model.fit(scaled)
        self.model_ = model
        raw_labels = self._predict_raw(scaled)
        self.state_order_ = self._order_states_by_volatility(matrix, raw_labels)
        return self

    def predict(self, features: pd.DataFrame) -> RegimeResult:
        if self.model_ is None or self.scaler_ is None or self.feature_columns_ is None:
            raise RuntimeError("HMMRegimeDetector must be fit before predict")

        matrix = self._prepare_features(features)[self.feature_columns_]
        scaled = self.scaler_.transform(matrix)
        raw_labels = self._predict_raw(scaled)
        labels = pd.Series(
            [self.state_order_.get(int(label), int(label)) for label in raw_labels],
            index=matrix.index,
            name="hmm_regime",
        )
        probabilities = self._predict_probabilities(scaled, matrix.index)
        if probabilities is not None and self.state_order_ is not None:
            probabilities = probabilities.rename(columns=self.state_order_)
            probabilities = probabilities.reindex(sorted(probabilities.columns), axis=1)
        return RegimeResult(
            labels=labels,
            probabilities=probabilities,
            diagnostics={
                "backend": self.backend_,
                "n_states": self.n_states,
                "state_order": self.state_order_,
            },
        )

    @staticmethod
    def _prepare_features(features: pd.DataFrame) -> pd.DataFrame:
        matrix = features.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        matrix = matrix.dropna(how="any")
        if matrix.empty:
            raise ValueError("Regime features are empty after dropping missing values")
        return matrix

    def _predict_raw(self, scaled: np.ndarray) -> np.ndarray:
        if hasattr(self.model_, "predict"):
            return self.model_.predict(scaled)
        raise RuntimeError("Underlying regime model does not support predict")

    def _predict_probabilities(self, scaled: np.ndarray, index: pd.Index) -> pd.DataFrame | None:
        if hasattr(self.model_, "predict_proba"):
            probabilities = self.model_.predict_proba(scaled)
            return pd.DataFrame(probabilities, index=index, columns=range(probabilities.shape[1]))
        if hasattr(self.model_, "score_samples"):
            result = self.model_.score_samples(scaled)
            if isinstance(result, tuple) and len(result) == 2:
                probabilities = result[1]
                return pd.DataFrame(probabilities, index=index, columns=range(probabilities.shape[1]))
        return None

    def _order_states_by_volatility(self, matrix: pd.DataFrame, raw_labels: np.ndarray) -> dict[int, int]:
        volatility_proxy = matrix.abs().mean(axis=1)
        state_volatility = (
            pd.DataFrame({"state": raw_labels, "volatility": volatility_proxy})
            .groupby("state")["volatility"]
            .mean()
            .sort_values()
        )
        return {int(raw_state): rank for rank, raw_state in enumerate(state_volatility.index)}


class _SimpleStandardScaler:
    """Small fallback compatible with the scikit-learn scaler API used here."""

    def fit_transform(self, matrix: pd.DataFrame) -> np.ndarray:
        values = matrix.to_numpy(dtype=float)
        self.mean_ = np.nanmean(values, axis=0)
        self.scale_ = np.nanstd(values, axis=0)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self.transform(matrix)

    def transform(self, matrix: pd.DataFrame) -> np.ndarray:
        values = matrix.to_numpy(dtype=float)
        return (values - self.mean_) / self.scale_


class _QuantileStateModel:
    """Deterministic fallback that buckets observations by feature magnitude."""

    def __init__(self, n_states: int) -> None:
        self.n_states = n_states
        self.thresholds_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "_QuantileStateModel":
        score = np.nanmean(np.abs(values), axis=1)
        quantiles = np.linspace(0, 1, self.n_states + 1)[1:-1]
        self.thresholds_ = np.quantile(score, quantiles)
        return self

    def predict(self, values: np.ndarray) -> np.ndarray:
        if self.thresholds_ is None:
            raise RuntimeError("Quantile fallback model must be fit before predict")
        score = np.nanmean(np.abs(values), axis=1)
        return np.digitize(score, self.thresholds_)

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        labels = self.predict(values)
        probabilities = np.zeros((len(labels), self.n_states))
        probabilities[np.arange(len(labels)), labels] = 1.0
        return probabilities
