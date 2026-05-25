"""Advanced portfolio diagnostics and risk decomposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioDiagnosticsConfig:
    """Configuration for portfolio diagnostics."""

    rolling_window: int = 63
    cvar_alpha: float = 0.05


class PortfolioDiagnostics:
    """Institutional portfolio analytics for systematic portfolios."""

    def __init__(self, config: PortfolioDiagnosticsConfig | None = None) -> None:
        self.config = config or PortfolioDiagnosticsConfig()

    def analyze(self, weights: pd.DataFrame, returns: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series | float]:
        """Run turnover, holding period, concentration, leverage, risk contribution, and CVaR diagnostics."""

        aligned_returns = returns.reindex(index=weights.index, columns=weights.columns).fillna(0.0)
        portfolio_returns = (weights.shift(1).fillna(0.0) * aligned_returns).sum(axis=1)
        covariance = aligned_returns.rolling(self.config.rolling_window).cov()
        return {
            "turnover": turnover_analysis(weights),
            "holding_period_distribution": holding_period_distribution(weights),
            "exposure_concentration": exposure_concentration(weights),
            "leverage_utilization": leverage_utilization(weights),
            "rolling_volatility_contribution": rolling_volatility_contribution(weights, aligned_returns, self.config.rolling_window),
            "marginal_risk_contribution": marginal_risk_contribution(weights, aligned_returns),
            "expected_shortfall": expected_shortfall(portfolio_returns, self.config.cvar_alpha),
            "cvar_decomposition": cvar_decomposition(weights, aligned_returns, self.config.cvar_alpha),
            "rolling_covariance": covariance,
        }


def turnover_analysis(weights: pd.DataFrame) -> pd.Series:
    """Daily portfolio turnover."""

    return weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1)).rename("turnover")


def holding_period_distribution(weights: pd.DataFrame) -> pd.Series:
    """Estimate holding period lengths from non-zero position runs."""

    durations: list[int] = []
    active = weights.abs() > 1e-12
    for column in active.columns:
        groups = (active[column] != active[column].shift()).cumsum()
        for _, group in active[column].groupby(groups):
            if bool(group.iloc[0]):
                durations.append(len(group))
    return pd.Series(durations, name="holding_period_days")


def exposure_concentration(weights: pd.DataFrame) -> pd.Series:
    """Herfindahl concentration of absolute exposures."""

    gross = weights.abs().sum(axis=1).replace(0.0, np.nan)
    shares = weights.abs().div(gross, axis=0).fillna(0.0)
    return (shares**2).sum(axis=1).rename("exposure_concentration")


def leverage_utilization(weights: pd.DataFrame) -> pd.DataFrame:
    """Gross, net, long, and short exposure tracking."""

    return pd.DataFrame(
        {
            "gross_exposure": weights.abs().sum(axis=1),
            "net_exposure": weights.sum(axis=1),
            "long_exposure": weights.clip(lower=0.0).sum(axis=1),
            "short_exposure": weights.clip(upper=0.0).sum(axis=1),
        }
    )


def rolling_volatility_contribution(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """Rolling volatility contribution by asset."""

    contributions = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
    for index_position in range(window, len(weights)):
        date = weights.index[index_position]
        covariance = returns.iloc[index_position - window : index_position].cov().fillna(0.0)
        weight = weights.iloc[index_position].reindex(covariance.columns).fillna(0.0)
        variance = float(weight.T @ covariance @ weight)
        if variance <= 0:
            continue
        marginal = covariance @ weight
        contributions.loc[date, covariance.columns] = weight * marginal / variance
    return contributions


def marginal_risk_contribution(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Marginal risk contribution from full-sample covariance."""

    covariance = returns.cov().fillna(0.0)
    result = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
    for date, weight in weights.iterrows():
        aligned_weight = weight.reindex(covariance.columns).fillna(0.0)
        portfolio_volatility = np.sqrt(float(aligned_weight.T @ covariance @ aligned_weight))
        if portfolio_volatility > 0:
            result.loc[date, covariance.columns] = (covariance @ aligned_weight) / portfolio_volatility
    return result


def expected_shortfall(returns: pd.Series, alpha: float = 0.05) -> float:
    """Expected shortfall / CVaR of portfolio returns."""

    clean = returns.dropna()
    if clean.empty:
        return 0.0
    threshold = clean.quantile(alpha)
    return float(clean[clean <= threshold].mean())


def cvar_decomposition(weights: pd.DataFrame, returns: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
    """Approximate CVaR contribution by asset on tail days."""

    portfolio_returns = (weights.shift(1).fillna(0.0) * returns).sum(axis=1)
    threshold = portfolio_returns.quantile(alpha)
    tail_days = portfolio_returns <= threshold
    if not tail_days.any():
        return pd.Series(0.0, index=weights.columns, name="cvar_contribution")
    contributions = (weights.shift(1).fillna(0.0) * returns).loc[tail_days].mean()
    return contributions.rename("cvar_contribution")

