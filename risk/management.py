"""Portfolio risk management controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskConfig:
    """Shared risk management settings."""

    annualization_factor: int = 252
    volatility_target: float = 0.10
    volatility_lookback: int = 21
    max_leverage_multiplier: float = 2.0
    max_asset_weight: float = 0.15
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 0.25
    stop_loss: float = 0.08
    trailing_stop: float = 0.12
    drawdown_alert: float = 0.15


class VolatilityTargeter:
    """Scale target weights to hit a portfolio volatility target."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def scale_weights(self, weights: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.DataFrame:
        """Scale weights by trailing realized portfolio volatility."""

        aligned_returns = asset_returns.reindex(index=weights.index, columns=weights.columns).fillna(0.0)
        raw_portfolio_returns = (weights.shift(1).fillna(0.0) * aligned_returns).sum(axis=1)
        realized_vol = raw_portfolio_returns.rolling(self.config.volatility_lookback).std()
        realized_vol = realized_vol * np.sqrt(self.config.annualization_factor)
        scale = self.config.volatility_target / realized_vol.replace(0.0, np.nan)
        scale = scale.shift(1).clip(upper=self.config.max_leverage_multiplier).fillna(1.0)
        return weights.mul(scale, axis=0)


class ExposureConstraint:
    """Apply asset, gross, and net exposure constraints to target weights."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def apply(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Clip and rescale weights to configured exposure limits."""

        constrained = weights.clip(
            lower=-self.config.max_asset_weight,
            upper=self.config.max_asset_weight,
        )
        gross = constrained.abs().sum(axis=1)
        gross_scale = (self.config.max_gross_exposure / gross.replace(0.0, np.nan)).clip(upper=1.0)
        constrained = constrained.mul(gross_scale.fillna(1.0), axis=0)

        net = constrained.sum(axis=1)
        excess_net = (net.abs() - self.config.max_net_exposure).clip(lower=0.0)
        adjustment = excess_net / constrained.shape[1]
        constrained = constrained.sub(np.sign(net) * adjustment, axis=0)
        return constrained.clip(
            lower=-self.config.max_asset_weight,
            upper=self.config.max_asset_weight,
        ).fillna(0.0)


class StopLossSystem:
    """Zero asset weights after asset-level drawdowns exceed a stop threshold."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def apply(self, prices: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
        """Apply a simple peak-to-trough stop-loss filter."""

        aligned_prices = prices.reindex(index=weights.index, columns=weights.columns).ffill()
        asset_drawdown = aligned_prices / aligned_prices.cummax() - 1.0
        stopped = asset_drawdown <= -abs(self.config.stop_loss)
        return weights.mask(stopped, 0.0).fillna(0.0)


class TrailingStopSystem:
    """Apply trailing stops based on rolling asset peaks."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def apply(self, prices: pd.DataFrame, weights: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
        """Zero weights when prices breach a rolling trailing stop."""

        aligned_prices = prices.reindex(index=weights.index, columns=weights.columns).ffill()
        trailing_peak = aligned_prices.rolling(lookback, min_periods=1).max()
        trailing_drawdown = aligned_prices / trailing_peak - 1.0
        stopped = trailing_drawdown <= -abs(self.config.trailing_stop)
        return weights.mask(stopped, 0.0).fillna(0.0)


class DrawdownMonitor:
    """Monitor portfolio drawdowns and alert zones."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def compute(self, equity_curve: pd.Series) -> pd.DataFrame:
        """Return drawdown diagnostics."""

        running_max = equity_curve.cummax()
        drawdown = equity_curve / running_max - 1.0
        return pd.DataFrame(
            {
                "equity": equity_curve,
                "running_max": running_max,
                "drawdown": drawdown,
                "alert": drawdown <= -abs(self.config.drawdown_alert),
            }
        )

