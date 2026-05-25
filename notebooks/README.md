# Notebooks

Use this directory for exploratory research notebooks. A typical workflow:

1. Pull or load OHLCV, sector ETF, volatility index, and macro data.
2. Build feature panels with `data.features.FeatureEngineer`.
3. Fit regime detectors from `regime_detection`.
4. Generate strategy signals from `strategies`.
5. Allocate capital with `portfolio`.
6. Backtest and analyze with `backtesting` and `analytics`.

