"""Macro and risk-on/risk-off regime detection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from regime_detection.base import BaseRegimeDetector, RegimeResult


@dataclass
class MacroRiskRegimeDetector(BaseRegimeDetector):
    """Classify regimes from macro and risk sentiment indicators.

    Higher values in ``risk_off_columns`` increase defensive regime pressure.
    Higher values in ``risk_on_columns`` increase constructive regime pressure.
    """

    lookback: int = 252
    risk_off_columns: list[str] = field(default_factory=lambda: ["VIX", "credit_spread", "UNRATE"])
    risk_on_columns: list[str] = field(default_factory=lambda: ["growth", "breadth", "momentum"])
    defensive_threshold: float = 0.5
    constructive_threshold: float = -0.5
    name: str = "macro_risk"

    def fit(self, features: pd.DataFrame) -> "MacroRiskRegimeDetector":
        return self

    def predict(self, features: pd.DataFrame) -> RegimeResult:
        macro = features.select_dtypes(include=[np.number]).sort_index()
        if macro.empty:
            raise ValueError("MacroRiskRegimeDetector requires numeric macro features")

        zscores = (macro - macro.rolling(self.lookback, min_periods=20).mean()) / macro.rolling(
            self.lookback,
            min_periods=20,
        ).std()
        risk_score = pd.Series(0.0, index=macro.index)
        for column in macro.columns:
            if column in self.risk_off_columns:
                risk_score = risk_score.add(zscores[column], fill_value=0.0)
            elif column in self.risk_on_columns:
                risk_score = risk_score.sub(zscores[column], fill_value=0.0)
        risk_score = risk_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        labels = pd.Series(1, index=macro.index, name="macro_risk_regime")
        labels = labels.mask(risk_score >= self.defensive_threshold, 0)
        labels = labels.mask(risk_score <= self.constructive_threshold, 2)
        probabilities = pd.get_dummies(labels).reindex(columns=[0, 1, 2], fill_value=0).astype(float)
        return RegimeResult(
            labels=labels.astype(int),
            probabilities=probabilities,
            diagnostics={"risk_score": risk_score, "zscores": zscores},
        )

