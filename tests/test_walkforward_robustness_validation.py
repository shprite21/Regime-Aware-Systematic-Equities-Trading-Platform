from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtesting import WalkForwardConfig, WalkForwardValidator
from data import MarketDataQualityValidator
from research import ExperimentTracker, ParameterRobustnessTester, RobustnessConfig
from strategies import CrossSectionalMomentumStrategy
from strategies.base import StrategyConfig


def test_walk_forward_validator_runs_oos_windows(synthetic_prices):
    validator = WalkForwardValidator(
        WalkForwardConfig(train_window=100, min_train_window=100, test_window=50, mode="rolling")
    )
    result = validator.evaluate(
        synthetic_prices,
        lambda frame, params: CrossSectionalMomentumStrategy(
            StrategyConfig(name="wf_momentum", lookback=40)
        ).generate(frame),
    )

    assert result.windows
    assert not result.oos_returns.empty
    assert "test_sharpe" in result.window_metrics.columns


def test_parameter_robustness_and_experiment_tracking(tmp_path: Path, synthetic_bundle):
    prices = synthetic_bundle.close()
    tester = ParameterRobustnessTester(RobustnessConfig(metric="sharpe"))
    robustness = tester.grid_search(
        prices,
        {"lookback": [20, 40], "top_quantile": [0.2, 0.3]},
        lambda params: CrossSectionalMomentumStrategy(
            StrategyConfig(
                name="robustness_momentum",
                lookback=int(params["lookback"]),
                params={"top_quantile": params["top_quantile"], "bottom_quantile": params["top_quantile"]},
            )
        ).generate(synthetic_bundle),
    )
    assert len(robustness.results) == 4
    assert robustness.heatmaps

    tracker = ExperimentTracker(tmp_path)
    record = tracker.start_run(
        "unit_test",
        config={"seed": 1},
        metrics={"sharpe": 1.0},
        parameters={"lookback": 20},
    )
    loaded = tracker.load(record.experiment_id)
    assert loaded.metrics["sharpe"] == 1.0
    assert tracker.compare()


def test_market_data_quality_validator_detects_issues(synthetic_bundle):
    report = MarketDataQualityValidator().validate(synthetic_bundle.ohlcv)

    assert isinstance(report.duplicate_timestamps, pd.Series)
    assert report.missing_data.empty

