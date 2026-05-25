"""Market data ingestion and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from data.models import DataSourceConfig, MarketDataBundle, align_frames


OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


@dataclass(frozen=True)
class DataValidator:
    """Validate OHLCV frames before they enter the research stack."""

    required_columns: tuple[str, ...] = OHLCV_COLUMNS

    def validate_ohlcv(self, frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Return a validated and sorted OHLCV frame."""

        missing = [column for column in self.required_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{symbol} is missing required OHLCV columns: {missing}")
        if not isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.copy()
            frame.index = pd.to_datetime(frame.index)
        frame = frame.sort_index()
        numeric_columns = [column for column in frame.columns if column in self.required_columns]
        frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
        if (frame["Close"] <= 0).any():
            raise ValueError(f"{symbol} contains non-positive close prices")
        return frame


class OHLCVIngestor:
    """Ingest OHLCV, sector ETF, volatility index, and macro proxy data."""

    def __init__(
        self,
        config: DataSourceConfig,
        validator: DataValidator | None = None,
    ) -> None:
        self.config = config
        self.validator = validator or DataValidator()

    def from_csv_directory(self, directory: str | Path, symbols: Iterable[str] | None = None) -> dict[str, pd.DataFrame]:
        """Load one CSV per symbol from a directory.

        Filenames are expected to be either ``SYMBOL.csv`` or ``symbol.csv``.
        """

        directory = Path(directory)
        selected = list(symbols or self.config.symbols)
        frames: dict[str, pd.DataFrame] = {}
        for symbol in selected:
            path = directory / f"{symbol}.csv"
            if not path.exists():
                path = directory / f"{symbol.lower()}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Could not find CSV for {symbol} in {directory}")
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
            frames[symbol] = self.validator.validate_ohlcv(frame, symbol)
        return align_frames(frames)

    def from_yfinance(self, symbols: Iterable[str]) -> dict[str, pd.DataFrame]:
        """Download market data from yfinance.

        The import is optional so unit tests and offline research can use CSV or
        synthetic data without requiring network access.
        """

        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - depends on optional environment
            raise ImportError("Install yfinance to use OHLCVIngestor.from_yfinance") from exc

        frames: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            frame = yf.download(
                symbol,
                start=self.config.start,
                end=self.config.end,
                interval=self.config.interval,
                auto_adjust=False,
                progress=False,
            )
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            if frame.empty:
                raise ValueError(f"yfinance returned no rows for {symbol}")
            frames[symbol] = self.validator.validate_ohlcv(frame, symbol)
        return align_frames(frames)

    def build_bundle(
        self,
        ohlcv: dict[str, pd.DataFrame] | None = None,
        benchmark: pd.DataFrame | None = None,
        sectors: dict[str, pd.DataFrame] | None = None,
        volatility: dict[str, pd.DataFrame] | None = None,
        macro: pd.DataFrame | None = None,
    ) -> MarketDataBundle:
        """Create a validated market data bundle from supplied or downloaded data."""

        if ohlcv is None:
            ohlcv = self.from_yfinance(self.config.symbols)
        else:
            ohlcv = {
                symbol: self.validator.validate_ohlcv(frame, symbol)
                for symbol, frame in ohlcv.items()
            }

        if benchmark is None and self.config.benchmark:
            benchmark_data = self.from_yfinance([self.config.benchmark])
            benchmark = benchmark_data[self.config.benchmark]
        if sectors is None and self.config.sector_etfs:
            sectors = self.from_yfinance(self.config.sector_etfs)
        if volatility is None and self.config.volatility_indices:
            volatility = self.from_yfinance(self.config.volatility_indices)

        return MarketDataBundle(
            ohlcv=align_frames(ohlcv),
            benchmark=benchmark,
            sectors=align_frames(sectors or {}),
            volatility=align_frames(volatility or {}),
            macro=macro.sort_index() if macro is not None else None,
            metadata={
                "symbols": list(ohlcv),
                "start": self.config.start,
                "end": self.config.end,
                "interval": self.config.interval,
            },
        )

