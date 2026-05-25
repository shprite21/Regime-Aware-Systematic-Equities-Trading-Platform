"""Shared strategy utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_weights(raw: pd.DataFrame, gross_leverage: float = 1.0) -> pd.DataFrame:
    """Normalize weights to a target gross exposure row by row."""

    cleaned = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = cleaned.abs().sum(axis=1).replace(0.0, np.nan)
    return cleaned.div(gross, axis=0).fillna(0.0) * gross_leverage


def long_only_from_scores(scores: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    """Convert cross-sectional scores into long-only normalized weights."""

    if top_n is None:
        raw = scores.clip(lower=0.0)
    else:
        ranks = scores.rank(axis=1, ascending=False, method="first")
        raw = scores.where(ranks <= top_n, 0.0).clip(lower=0.0)
    row_sum = raw.sum(axis=1).replace(0.0, np.nan)
    return raw.div(row_sum, axis=0).fillna(0.0)


def rolling_zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute rolling z-scores."""

    mean = frame.rolling(window).mean()
    std = frame.rolling(window).std().replace(0.0, np.nan)
    return (frame - mean) / std

