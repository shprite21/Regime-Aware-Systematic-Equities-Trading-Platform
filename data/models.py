"""Core market data containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class DataSourceConfig:
    """Configuration for market and auxiliary data sources."""

    symbols: list[str]
    start: str | None = None
    end: str | None = None
    interval: str = "1d"
    benchmark: str | None = "SPY"
    sector_etfs: list[str] = field(default_factory=list)
    volatility_indices: list[str] = field(default_factory=lambda: ["^VIX"])
    macro_indicators: list[str] = field(default_factory=list)
    cache_dir: Path | None = None


@dataclass
class MarketDataBundle:
    """Container for aligned market, sector, volatility, macro, and feature data."""

    ohlcv: dict[str, pd.DataFrame]
    benchmark: pd.DataFrame | None = None
    sectors: dict[str, pd.DataFrame] = field(default_factory=dict)
    volatility: dict[str, pd.DataFrame] = field(default_factory=dict)
    macro: pd.DataFrame | None = None
    features: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def symbols(self) -> list[str]:
        """Return the sorted tradable symbol universe."""

        return sorted(self.ohlcv)

    def field_panel(self, field: str = "Close") -> pd.DataFrame:
        """Return a wide panel for one OHLCV field."""

        series = {}
        for symbol, frame in self.ohlcv.items():
            if field not in frame.columns:
                raise KeyError(f"{symbol} is missing required field {field!r}")
            series[symbol] = frame[field]
        return pd.DataFrame(series).sort_index()

    def close(self) -> pd.DataFrame:
        """Return adjusted close when available, otherwise close prices."""

        preferred = "Adj Close"
        if all(preferred in frame.columns for frame in self.ohlcv.values()):
            return self.field_panel(preferred)
        return self.field_panel("Close")

    def volume(self) -> pd.DataFrame:
        """Return volume panel."""

        return self.field_panel("Volume")

    def returns(self, periods: int = 1) -> pd.DataFrame:
        """Return percentage returns from the close panel."""

        return self.close().pct_change(periods)

    def with_features(self, features: Mapping[str, pd.DataFrame]) -> "MarketDataBundle":
        """Attach feature panels and return this bundle for fluent pipelines."""

        self.features.update(dict(features))
        return self

    def aligned_index(self) -> pd.DatetimeIndex:
        """Return the common index across all tradable OHLCV frames."""

        if not self.ohlcv:
            return pd.DatetimeIndex([])
        index = None
        for frame in self.ohlcv.values():
            index = frame.index if index is None else index.intersection(frame.index)
        return pd.DatetimeIndex(index).sort_values()


def align_frames(frames: Mapping[str, pd.DataFrame], method: str = "ffill") -> dict[str, pd.DataFrame]:
    """Align frames on the union of dates using a conservative fill policy."""

    if not frames:
        return {}
    union_index = pd.Index([])
    for frame in frames.values():
        union_index = union_index.union(frame.index)

    aligned = {}
    for key, frame in frames.items():
        sorted_frame = frame.sort_index().reindex(union_index)
        if method:
            sorted_frame = getattr(sorted_frame, method)()
        aligned[key] = sorted_frame
    return aligned
