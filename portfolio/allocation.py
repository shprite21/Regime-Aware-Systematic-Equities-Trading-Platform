"""Portfolio allocation and constrained optimization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:  # pragma: no cover - exercised only when scipy is installed
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover - lean runtime fallback
    minimize = None


@dataclass(frozen=True)
class AllocationConfig:
    """Shared portfolio allocation settings."""

    lookback: int = 63
    gross_leverage: float = 1.0
    long_only: bool = False
    min_weight: float = -0.15
    max_weight: float = 0.15
    target_volatility: float | None = None
    annualization_factor: int = 252
    risk_aversion: float = 5.0
    regime_multipliers: dict[int | str, float] = field(
        default_factory=lambda: {0: 0.5, 1: 1.0, 2: 1.2}
    )


@dataclass
class AllocationResult:
    """Standardized allocator output."""

    weights: pd.DataFrame
    diagnostics: dict[str, object] = field(default_factory=dict)


class InverseVolatilityAllocator:
    """Allocate capital inversely to realized volatility."""

    def __init__(self, config: AllocationConfig | None = None) -> None:
        self.config = config or AllocationConfig(long_only=True, min_weight=0.0)

    def allocate(self, returns: pd.DataFrame) -> AllocationResult:
        vol = returns.rolling(self.config.lookback).std().replace(0.0, np.nan)
        inverse_vol = 1.0 / vol
        if self.config.long_only:
            inverse_vol = inverse_vol.clip(lower=0.0)
        weights = _normalize_frame(inverse_vol, self.config.gross_leverage)
        weights = _apply_bounds(weights, self.config.min_weight, self.config.max_weight)
        weights = _normalize_frame(weights, self.config.gross_leverage)
        return AllocationResult(weights=weights.fillna(0.0), diagnostics={"volatility": vol})


class RiskParityAllocator:
    """Rolling equal risk contribution allocator."""

    def __init__(self, config: AllocationConfig | None = None) -> None:
        self.config = config or AllocationConfig(long_only=True, min_weight=0.0)

    def allocate(self, returns: pd.DataFrame) -> AllocationResult:
        weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
        for index_position in range(self.config.lookback, len(returns)):
            window = returns.iloc[index_position - self.config.lookback : index_position].dropna(how="all")
            covariance = window.cov().fillna(0.0).values
            if np.allclose(covariance, 0.0):
                continue
            weights.iloc[index_position] = self._solve(covariance, list(returns.columns))
        return AllocationResult(weights=weights, diagnostics={"method": "equal_risk_contribution"})

    def _solve(self, covariance: np.ndarray, columns: list[str]) -> pd.Series:
        n_assets = len(columns)
        initial = np.repeat(1.0 / n_assets, n_assets)
        if minimize is None:
            diagonal = np.diag(covariance)
            inverse_vol = 1.0 / np.sqrt(np.where(diagonal > 0, diagonal, np.nan))
            inverse_vol = np.nan_to_num(inverse_vol, nan=0.0)
            if inverse_vol.sum() == 0:
                return pd.Series(initial * self.config.gross_leverage, index=columns)
            return pd.Series(inverse_vol / inverse_vol.sum() * self.config.gross_leverage, index=columns)
        bounds = [(max(0.0, self.config.min_weight), self.config.max_weight)] * n_assets

        def objective(weights: np.ndarray) -> float:
            portfolio_variance = float(weights.T @ covariance @ weights)
            if portfolio_variance <= 0:
                return 1e6
            marginal_risk = covariance @ weights
            risk_contribution = weights * marginal_risk / np.sqrt(portfolio_variance)
            target = np.repeat(risk_contribution.sum() / n_assets, n_assets)
            return float(((risk_contribution - target) ** 2).sum())

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=({"type": "eq", "fun": lambda weights: weights.sum() - self.config.gross_leverage},),
            options={"maxiter": 200, "ftol": 1e-10},
        )
        if not result.success:
            return pd.Series(initial * self.config.gross_leverage, index=columns)
        return pd.Series(result.x, index=columns)


class RollingSharpeAllocator:
    """Allocate among strategy return streams using rolling Sharpe scores."""

    def __init__(self, config: AllocationConfig | None = None) -> None:
        self.config = config or AllocationConfig(long_only=True, min_weight=0.0)

    def allocate(self, strategy_returns: pd.DataFrame) -> AllocationResult:
        mean = strategy_returns.rolling(self.config.lookback).mean()
        volatility = strategy_returns.rolling(self.config.lookback).std().replace(0.0, np.nan)
        sharpe = mean / volatility * np.sqrt(self.config.annualization_factor)
        scores = sharpe.clip(lower=0.0) if self.config.long_only else sharpe
        weights = _normalize_frame(scores, self.config.gross_leverage)
        weights = _apply_bounds(weights, self.config.min_weight, self.config.max_weight)
        weights = _normalize_frame(weights, self.config.gross_leverage)
        return AllocationResult(weights=weights.fillna(0.0), diagnostics={"rolling_sharpe": sharpe})


class RegimeConditionedAllocator:
    """Scale base portfolio weights according to detected market regimes."""

    def __init__(self, config: AllocationConfig | None = None) -> None:
        self.config = config or AllocationConfig()

    def allocate(self, base_weights: pd.DataFrame, regimes: pd.Series) -> AllocationResult:
        regimes = regimes.reindex(base_weights.index).ffill()
        multipliers = regimes.map(self.config.regime_multipliers).fillna(1.0)
        scaled = base_weights.mul(multipliers, axis=0)
        scaled = _apply_bounds(scaled, self.config.min_weight, self.config.max_weight)
        target_gross = self.config.gross_leverage * multipliers
        weights = _normalize_frame(scaled, target_gross)
        return AllocationResult(
            weights=weights.fillna(0.0),
            diagnostics={"regime_multipliers": multipliers},
        )


class ConstrainedOptimizer:
    """Mean-variance optimizer with exposure and box constraints."""

    def __init__(self, config: AllocationConfig | None = None) -> None:
        self.config = config or AllocationConfig(long_only=True, min_weight=0.0)

    def allocate(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
    ) -> pd.Series:
        expected_returns = expected_returns.dropna()
        covariance = covariance.reindex(index=expected_returns.index, columns=expected_returns.index).fillna(0.0)
        if expected_returns.empty:
            return pd.Series(dtype=float)

        try:
            return self._allocate_cvxpy(expected_returns, covariance)
        except Exception:
            return self._allocate_scipy(expected_returns, covariance)

    def allocate_rolling(self, returns: pd.DataFrame) -> AllocationResult:
        weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
        for index_position in range(self.config.lookback, len(returns)):
            window = returns.iloc[index_position - self.config.lookback : index_position].dropna(how="all")
            if window.empty:
                continue
            expected_returns = window.mean()
            covariance = window.cov()
            weights.iloc[index_position] = self.allocate(expected_returns, covariance).reindex(returns.columns).fillna(0.0)
        return AllocationResult(weights=weights, diagnostics={"method": "constrained_mean_variance"})

    def _allocate_cvxpy(self, expected_returns: pd.Series, covariance: pd.DataFrame) -> pd.Series:
        import cvxpy as cp

        n_assets = len(expected_returns)
        weights = cp.Variable(n_assets)
        covariance_matrix = covariance.values + np.eye(n_assets) * 1e-8
        objective = cp.Maximize(
            expected_returns.values @ weights
            - self.config.risk_aversion * cp.quad_form(weights, covariance_matrix)
        )
        constraints = [
            weights >= self.config.min_weight,
            weights <= self.config.max_weight,
            cp.norm1(weights) <= self.config.gross_leverage,
        ]
        if self.config.long_only:
            constraints.append(cp.sum(weights) == self.config.gross_leverage)
        else:
            constraints.append(cp.abs(cp.sum(weights)) <= self.config.gross_leverage)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)
        if weights.value is None:
            raise RuntimeError("cvxpy failed to produce optimizer weights")
        return pd.Series(np.asarray(weights.value).ravel(), index=expected_returns.index)

    def _allocate_scipy(self, expected_returns: pd.Series, covariance: pd.DataFrame) -> pd.Series:
        n_assets = len(expected_returns)
        if minimize is None:
            return self._allocate_heuristic(expected_returns)
        if self.config.long_only:
            initial = np.repeat(self.config.gross_leverage / n_assets, n_assets)
            constraints = ({"type": "eq", "fun": lambda weights: weights.sum() - self.config.gross_leverage},)
        else:
            initial = np.repeat(0.0, n_assets)
            constraints = ({"type": "ineq", "fun": lambda weights: self.config.gross_leverage - np.abs(weights).sum()},)

        bounds = [(self.config.min_weight, self.config.max_weight)] * n_assets
        covariance_matrix = covariance.values + np.eye(n_assets) * 1e-8

        def objective(weights: np.ndarray) -> float:
            utility = expected_returns.values @ weights
            penalty = self.config.risk_aversion * float(weights.T @ covariance_matrix @ weights)
            return float(-(utility - penalty))

        result = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints)
        if not result.success:
            return pd.Series(initial, index=expected_returns.index)
        return pd.Series(result.x, index=expected_returns.index)

    def _allocate_heuristic(self, expected_returns: pd.Series) -> pd.Series:
        if self.config.long_only:
            scores = expected_returns.clip(lower=0.0)
            if scores.sum() == 0:
                scores = pd.Series(1.0, index=expected_returns.index)
            weights = scores / scores.sum() * self.config.gross_leverage
        else:
            scores = expected_returns - expected_returns.mean()
            weights = scores.clip(lower=self.config.min_weight, upper=self.config.max_weight)
            gross = weights.abs().sum()
            if gross > 0:
                weights = weights / gross * self.config.gross_leverage
        return weights.clip(lower=self.config.min_weight, upper=self.config.max_weight)


def _normalize_frame(frame: pd.DataFrame, gross_leverage: float | pd.Series) -> pd.DataFrame:
    gross = frame.abs().sum(axis=1).replace(0.0, np.nan)
    normalized = frame.div(gross, axis=0).fillna(0.0)
    if isinstance(gross_leverage, pd.Series):
        return normalized.mul(gross_leverage.reindex(frame.index).fillna(0.0), axis=0)
    return normalized * gross_leverage


def _apply_bounds(frame: pd.DataFrame, minimum: float, maximum: float) -> pd.DataFrame:
    return frame.clip(lower=minimum, upper=maximum)
