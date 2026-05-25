from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.models import MarketDataBundle


@pytest.fixture()
def synthetic_prices() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=320)
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    returns = rng.normal(0.0003, 0.012, size=(len(dates), len(symbols)))
    returns[:, 0] += 0.0004
    returns[:, 1] -= 0.0002
    return pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=dates, columns=symbols)


@pytest.fixture()
def synthetic_bundle(synthetic_prices: pd.DataFrame) -> MarketDataBundle:
    rng = np.random.default_rng(123)
    ohlcv = {}
    for symbol in synthetic_prices:
        close = synthetic_prices[symbol]
        ohlcv[symbol] = pd.DataFrame(
            {
                "Open": close.shift(1).fillna(close.iloc[0]),
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Adj Close": close,
                "Volume": rng.integers(1_000_000, 2_000_000, len(close)),
            },
            index=synthetic_prices.index,
        )
    bundle = MarketDataBundle(ohlcv=ohlcv)
    surprises = pd.DataFrame(0.0, index=synthetic_prices.index, columns=synthetic_prices.columns)
    surprises.iloc[::50] = rng.normal(0, 1, size=(len(surprises.iloc[::50]), len(surprises.columns)))
    bundle.features["earnings_surprise"] = surprises
    return bundle

