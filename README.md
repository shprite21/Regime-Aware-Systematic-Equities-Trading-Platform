# Regime-Aware Systematic Equities Trading Platform

A research-grade quantitative trading and portfolio management framework focused on adaptive multi-strategy systematic equities trading across changing market regimes.

This platform combines regime detection, alpha generation, portfolio optimization, risk management, realistic backtesting, analytics, and visualization into a unified research-to-execution pipeline.

The system is designed to resemble institutional systematic equities research infrastructure while remaining readable enough for research portfolios, MFE applications, and quant developer interviews.

## Overview

Modern systematic equity strategies rarely rely on a single signal.

Professional quantitative platforms combine multiple weak-but-persistent alpha sources and dynamically allocate capital depending on the prevailing market environment. This project implements a modular regime-aware architecture capable of:

- detecting market regimes
- generating diversified alpha from independent strategy families
- dynamically weighting strategies and assets
- managing portfolio risk through execution-aware controls
- performing realistic portfolio-level backtests
- analyzing performance robustness across regimes
- producing professional research visualizations

The core research motivation is simple:

> Markets are non-stationary. Different strategies perform differently across different environments.

Instead of relying on a static trading system, this platform adapts exposure and allocation based on market structure, volatility, breadth, trend, and strategy performance.

## Core Features

### Regime Detection Engine

Detects market states using statistical and machine learning methods:

- Hidden Markov Models
- volatility regime classification
- trend filters
- market breadth analysis
- macro risk regime detection
- ensemble regime voting
- risk-on/risk-off style regime scoring

Supported regime concepts include:

- trending bull markets
- trending bear markets
- high-volatility markets
- low-volatility markets
- sideways or choppy markets
- crisis-style defensive regimes

### Multi-Strategy Alpha Library

The platform combines structurally different strategies designed to perform under different market environments.

Trend and momentum:

- cross-sectional momentum
- time-series momentum
- volatility breakout systems
- moving-average-style trend filters

Mean reversion:

- RSI reversal
- Bollinger Band reversal
- short-term z-score mean reversion
- short-term reversal systems

Volatility systems:

- volatility expansion breakouts
- ATR-based breakout systems
- volatility compression setups
- realized-volatility scaled signals

Event-driven:

- post-earnings announcement drift
- gap continuation and reversal
- earnings momentum
- configurable earnings surprise panel support
- event signal persistence windows

Relative strength and rotation:

- sector rotation
- relative strength ranking
- breadth-aware allocation inputs

Statistical arbitrage:

- pairs trading
- cointegration pair selection
- rolling beta spread construction
- z-score mean reversion

Each strategy produces a standardized `StrategySignal` with target weights, optional scores, and metadata. This keeps research components composable and easy to evaluate independently.

### Adaptive Allocation Engine

Capital can be dynamically allocated using:

- inverse volatility weighting
- risk parity
- regime-conditioned weights
- rolling Sharpe weighting
- constrained mean-variance optimization
- target gross exposure and per-asset limits

Allocation can adapt based on:

- detected regime
- rolling strategy performance
- volatility targets
- drawdown constraints
- correlation and covariance structure
- confidence scores from strategy outputs

### Risk Management

Professional risk controls are integrated throughout the framework:

- volatility targeting
- stop-loss systems
- trailing exits
- asset exposure caps
- gross and net leverage constraints
- drawdown monitoring
- transaction cost modeling
- slippage simulation
- execution-lag handling

### Data Pipeline

The platform supports:

- OHLCV market data
- yfinance ingestion
- CSV-based local data ingestion
- sector ETF data
- volatility index data
- macro indicator panels
- feature engineering for returns, momentum, volatility, liquidity, beta, drawdown, relative strength, and cross-sectional ranks

### Backtesting Engine

Supports:

- vectorized backtesting
- event-driven simulation through sparse signal events
- portfolio-level testing
- multi-strategy evaluation
- walk-forward testing
- rolling retraining
- expanding and rolling train/test windows
- train/test regime separation diagnostics
- out-of-sample stability metrics
- realistic execution assumptions
- transaction costs and slippage
- performance attribution

### Research Analytics

Includes:

- benchmark comparison against SPY proxy, equal-weight, buy-and-hold, momentum-only, and risk parity benchmarks
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- max drawdown
- rolling Sharpe
- rolling returns
- turnover analysis
- exposure analysis
- factor decomposition
- factor exposure drift diagnostics
- strategy correlation and clustering
- parameter robustness testing
- transaction cost sensitivity analysis
- regime transition analytics
- hit rate
- skew and kurtosis
- regime-wise performance breakdown
- strategy and asset contribution analysis

### Visualization Suite

Generates professional quantitative research plots:

- equity curves
- drawdown charts
- rolling Sharpe ratio
- regime overlays
- regime timelines
- allocation heatmaps
- strategy contribution charts
- rolling correlations
- factor exposure plots
- volatility regime maps
- advanced self-contained HTML research reports

## Tech Stack

Primary language:

- Python

Core libraries:

- pandas
- numpy
- scipy
- scikit-learn
- statsmodels
- hmmlearn
- matplotlib
- plotly
- yfinance
- cvxpy
- backtesting.py
- PyYAML

Planned extensions:

- C++ performance modules
- `backtesting.py` adapters
- distributed backtesting
- GPU acceleration

## Repository Structure

```text
regime-aware-systematic-equities/
|
+-- data/               Data ingestion, validation, market data containers, feature engineering
+-- strategies/         Modular alpha strategy library
+-- regime_detection/   HMM, volatility, trend, breadth, and ensemble regime detectors
+-- portfolio/          Allocation, risk parity, Sharpe weighting, constrained optimization
+-- risk/               Risk controls, transaction costs, slippage, drawdown monitors
+-- execution/          Order, fill, commission, slippage, and participation simulation
+-- backtesting/        Vectorized backtester, walk-forward evaluation, attribution
+-- analytics/          Performance metrics and report assembly
+-- visualization/      Plotly and Matplotlib research plots
+-- research/           End-to-end research pipeline orchestration
+-- configs/            Config-driven experiment templates
+-- notebooks/          Exploratory research workspace
+-- scripts/            Runnable research examples
+-- tests/              Unit tests for core contracts
+-- pyproject.toml      Package metadata and dependencies
+-- README.md
```

## Quick Start

Install the project with development dependencies:

```bash
pip install -e ".[dev]"
```

Run the synthetic end-to-end research experiment:

```bash
python scripts/run_research.py
```

The script writes an institutional-style HTML report to:

```text
reports/research_report.html
```

Run tests:

```bash
python -m pytest
```

## Example Workflow

```python
from analytics.tearsheet import PerformanceReport
from backtesting import VectorizedBacktester
from data.features import FeatureEngineer
from portfolio import RegimeConditionedAllocator
from regime_detection import RegimeEnsemble, VolatilityRegimeClassifier
from strategies.library import StrategyPipeline

# 1. Load or ingest market data
bundle = load_market_data()

# 2. Engineer features
features = FeatureEngineer().build(bundle)

# 3. Detect market regimes
volatility_regime = VolatilityRegimeClassifier().fit_predict(bundle.returns())
regimes = RegimeEnsemble().combine({"volatility": volatility_regime})

# 4. Generate independent strategy signals
signals = StrategyPipeline(strategies).generate(bundle)
blended_signal = StrategyPipeline.equal_weight_blend(signals)

# 5. Allocate capital dynamically
weights = RegimeConditionedAllocator().allocate(blended_signal.weights, regimes.labels)

# 6. Run portfolio backtest
result = VectorizedBacktester().run(bundle.close(), weights.weights)

# 7. Analyze performance
report = PerformanceReport.from_backtest(result, regimes.labels)
print(report.metrics_frame())
```

## Current Development Status

Implemented:

- core data containers and ingestion interfaces
- feature engineering pipeline
- HMM-style regime detection with fallback support
- volatility, trend, breadth, macro risk, and ensemble regime detectors
- professional data quality validation
- cross-sectional momentum
- time-series momentum
- moving average trend following
- mean reversion
- short-term reversal
- Bollinger Band reversal
- RSI reversal
- volatility breakout
- ATR breakout
- volatility compression
- PEAD signal support
- gap continuation and reversal
- earnings momentum
- sector rotation
- pairs trading
- cointegration pairs
- inverse volatility allocation
- risk parity
- rolling Sharpe allocation
- regime-conditioned allocation
- constrained optimization
- volatility targeting
- stop-loss and trailing-stop systems
- transaction cost and slippage models
- execution simulator with orders and fills
- vectorized backtesting
- event-driven backtesting wrapper
- walk-forward evaluation
- institutional walk-forward validator with train/test windows
- rolling retraining evaluator
- benchmark comparison engine
- strategy correlation and clustering analytics
- factor exposure analytics
- regime transition analytics
- transaction cost sensitivity analysis
- parameter robustness testing
- experiment tracking
- performance attribution
- expanded attribution by strategy, asset, sector, factor, and regime
- exposure analysis
- factor decomposition
- rolling correlation analytics
- analytics tear sheet object
- advanced HTML research report builder
- visualization utilities
- research pipeline orchestration
- synthetic end-to-end research script
- unit tests for core contracts

## Roadmap

Phase 1:

- core backtesting engine
- momentum strategies
- mean reversion strategies
- regime detection pipeline
- config-driven synthetic research workflow

Phase 2:

- adaptive portfolio allocation
- PEAD research dataset integration
- advanced analytics dashboard
- sector neutrality constraints

Phase 3:

- reinforcement learning allocation overlays
- factor modeling
- Bayesian optimization
- intraday execution simulation
- live paper trading integration
- distributed backtesting
- GPU acceleration

## Applications

This project is designed for:

- quantitative research
- systematic equities trading
- portfolio construction research
- MFE applications
- quant developer roles
- quant research internships
- systematic trading experimentation
- GitHub showcase portfolios

## Design Principles

The codebase is intentionally modular.

Every strategy produces a `StrategySignal`, regime detectors return a `RegimeResult`, allocators expose target portfolio weights, and the backtesting engine consumes price panels plus target weights. This keeps experimentation reproducible while allowing research components to be swapped independently.

The goal is not to create a single monolithic trading strategy. The goal is to create reusable research infrastructure for testing how strategy families behave across market regimes.

## Disclaimer

This project is for educational and research purposes only.

It is not financial advice and should not be used for live trading without extensive validation, independent review, market impact analysis, operational monitoring, and robust risk controls.

## References And Inspiration

Inspired by concepts from:

- systematic equities trading
- statistical arbitrage
- adaptive asset allocation
- factor investing
- quantitative portfolio management
- regime-switching models
- volatility targeting frameworks
- institutional portfolio construction

## Author

Arnaav Raj  
Information Science Undergraduate | Quantitative Finance Research | Systematic Equities and Regime-Aware Trading Systems
