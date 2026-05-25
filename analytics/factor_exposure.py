"""Factor exposure regression and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorExposureConfig:
    """Configuration for factor exposure analytics."""

    rolling_window: int = 126
    annualization_factor: int = 252
    beta_drift_threshold: float = 0.35


@dataclass
class FactorExposureResult:
    """Factor exposure outputs."""

    static_betas: pd.Series
    rolling_betas: pd.DataFrame
    factor_contributions: pd.DataFrame
    neutrality_diagnostics: pd.DataFrame
    beta_drift: pd.DataFrame
    regime_exposures: pd.DataFrame | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class FactorExposureAnalyzer:
    """Estimate factor exposures and hidden beta drift."""

    def __init__(self, config: FactorExposureConfig | None = None) -> None:
        self.config = config or FactorExposureConfig()

    def analyze(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        regimes: pd.Series | None = None,
    ) -> FactorExposureResult:
        """Run static, rolling, contribution, drift, and regime-conditioned factor analysis."""

        aligned = _align(portfolio_returns, factor_returns)
        if aligned.empty:
            empty = pd.DataFrame()
            return FactorExposureResult(pd.Series(dtype=float), empty, empty, empty, empty)
        y = aligned["portfolio"]
        x = aligned.drop(columns="portfolio")
        static_betas = _regression(y, x)
        rolling_betas = self.rolling_factor_regression(y, x)
        contributions = x.mul(static_betas.drop("alpha", errors="ignore"), axis=1)
        neutrality = self.factor_neutrality_diagnostics(static_betas)
        beta_drift = self.hidden_beta_drift(rolling_betas)
        regime_exposures = None
        if regimes is not None:
            regime_exposures = self.regime_conditioned_exposures(y, x, regimes)
        return FactorExposureResult(
            static_betas=static_betas,
            rolling_betas=rolling_betas,
            factor_contributions=contributions,
            neutrality_diagnostics=neutrality,
            beta_drift=beta_drift,
            regime_exposures=regime_exposures,
        )

    def rolling_factor_regression(self, portfolio_returns: pd.Series, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """Estimate rolling factor betas."""

        aligned = _align(portfolio_returns, factor_returns)
        rows = []
        for index_position in range(self.config.rolling_window, len(aligned) + 1):
            window = aligned.iloc[index_position - self.config.rolling_window : index_position]
            betas = _regression(window["portfolio"], window.drop(columns="portfolio"))
            betas.name = aligned.index[index_position - 1]
            rows.append(betas)
        return pd.DataFrame(rows)

    def hidden_beta_drift(self, rolling_betas: pd.DataFrame) -> pd.DataFrame:
        """Flag rolling factor beta moves above a configured threshold."""

        if rolling_betas.empty:
            return pd.DataFrame()
        beta_columns = [column for column in rolling_betas.columns if column != "alpha"]
        drift = rolling_betas[beta_columns].diff().abs()
        return drift.gt(self.config.beta_drift_threshold).astype(int)

    def factor_neutrality_diagnostics(self, betas: pd.Series) -> pd.DataFrame:
        """Summarize factor neutrality from absolute static exposures."""

        rows = []
        for factor, beta in betas.drop("alpha", errors="ignore").items():
            rows.append(
                {
                    "factor": factor,
                    "beta": float(beta),
                    "absolute_beta": abs(float(beta)),
                    "neutrality_score": max(0.0, 1.0 - abs(float(beta))),
                }
            )
        return pd.DataFrame(rows).set_index("factor") if rows else pd.DataFrame()

    def regime_conditioned_exposures(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        regimes: pd.Series,
    ) -> pd.DataFrame:
        """Estimate factor exposures separately by regime."""

        aligned = _align(portfolio_returns, factor_returns)
        aligned["regime"] = regimes.reindex(aligned.index).ffill()
        rows = []
        for regime, group in aligned.dropna().groupby("regime"):
            if len(group) < max(10, factor_returns.shape[1] + 2):
                continue
            betas = _regression(group["portfolio"], group.drop(columns=["portfolio", "regime"]))
            betas["regime"] = regime
            rows.append(betas)
        return pd.DataFrame(rows).set_index("regime") if rows else pd.DataFrame()


def build_equity_style_factors(prices: pd.DataFrame, volumes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Construct simple market, momentum, size, value, quality, and low-vol factors."""

    returns = prices.pct_change().fillna(0.0)
    market = returns.mean(axis=1)
    momentum_scores = prices.pct_change(126)
    high_momentum = returns.where(momentum_scores.rank(axis=1, pct=True) >= 0.7).mean(axis=1)
    low_momentum = returns.where(momentum_scores.rank(axis=1, pct=True) <= 0.3).mean(axis=1)
    momentum = (high_momentum - low_momentum).fillna(0.0)
    if volumes is not None:
        size_proxy = prices.mul(volumes.reindex_like(prices).ffill()).rank(axis=1, pct=True)
    else:
        size_proxy = prices.rank(axis=1, pct=True)
    small = returns.where(size_proxy <= 0.3).mean(axis=1)
    large = returns.where(size_proxy >= 0.7).mean(axis=1)
    size = (small - large).fillna(0.0)
    value = (-prices.pct_change(252).rank(axis=1, pct=True)).mul(returns).mean(axis=1).fillna(0.0)
    quality = returns.rolling(63).mean().rank(axis=1, pct=True).mul(returns).mean(axis=1).fillna(0.0)
    volatility_rank = returns.rolling(63).std().rank(axis=1, pct=True)
    low_volatility = returns.where(volatility_rank <= 0.3).mean(axis=1).fillna(0.0) - market
    return pd.DataFrame(
        {
            "market": market,
            "momentum": momentum,
            "size": size,
            "value": value,
            "quality": quality,
            "low_volatility": low_volatility,
        }
    )


def sector_exposures(weights: pd.DataFrame, sector_map: dict[str, str]) -> pd.DataFrame:
    """Aggregate asset weights into sector exposure time series."""

    sectors = sorted(set(sector_map.values()))
    exposures = pd.DataFrame(index=weights.index)
    for sector in sectors:
        members = [symbol for symbol, mapped in sector_map.items() if mapped == sector and symbol in weights.columns]
        if members:
            exposures[sector] = weights[members].sum(axis=1)
    return exposures


def _align(portfolio_returns: pd.Series, factor_returns: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([portfolio_returns.rename("portfolio"), factor_returns], axis=1, join="inner").dropna()


def _regression(y: pd.Series, x: pd.DataFrame) -> pd.Series:
    if len(y) == 0 or x.empty:
        return pd.Series(dtype=float)
    x_values = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
    coefficients = np.linalg.lstsq(x_values, y.to_numpy(dtype=float), rcond=None)[0]
    return pd.Series(coefficients, index=["alpha", *x.columns])

