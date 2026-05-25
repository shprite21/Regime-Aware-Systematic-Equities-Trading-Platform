"""Expanded performance attribution analytics."""

from __future__ import annotations

import pandas as pd


class AttributionAnalyzer:
    """Compute strategy, asset, sector, factor, and regime attribution."""

    def strategy_attribution(self, strategy_returns: pd.DataFrame) -> pd.DataFrame:
        """Cumulative contribution by strategy."""

        return strategy_returns.fillna(0.0).cumsum()

    def asset_attribution(self, weights: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """Asset-level return contribution."""

        return weights.shift(1).reindex_like(returns).fillna(0.0) * returns.fillna(0.0)

    def sector_attribution(
        self,
        asset_contribution: pd.DataFrame,
        sector_map: dict[str, str],
    ) -> pd.DataFrame:
        """Aggregate asset contribution to sector contribution."""

        sectors = sorted(set(sector_map.values()))
        output = pd.DataFrame(index=asset_contribution.index)
        for sector in sectors:
            members = [
                symbol
                for symbol, mapped_sector in sector_map.items()
                if mapped_sector == sector and symbol in asset_contribution.columns
            ]
            if members:
                output[sector] = asset_contribution[members].sum(axis=1)
        return output

    def factor_attribution(
        self,
        factor_returns: pd.DataFrame,
        factor_betas: pd.Series | pd.DataFrame,
    ) -> pd.DataFrame:
        """Estimate factor contribution from factor returns and betas."""

        if isinstance(factor_betas, pd.Series):
            betas = factor_betas.drop("alpha", errors="ignore")
            return factor_returns.reindex(columns=betas.index).mul(betas, axis=1)
        aligned = factor_betas.reindex(factor_returns.index).ffill()
        common = [column for column in factor_returns.columns if column in aligned.columns]
        return factor_returns[common] * aligned[common]

    def regime_attribution(self, returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
        """Return contribution aggregated by regime."""

        aligned = pd.DataFrame({"returns": returns, "regime": regimes.reindex(returns.index).ffill()}).dropna()
        rows = []
        for regime, group in aligned.groupby("regime"):
            rows.append(
                {
                    "regime": regime,
                    "total_contribution": float(group["returns"].sum()),
                    "average_return": float(group["returns"].mean()),
                    "observations": float(len(group)),
                }
            )
        return pd.DataFrame(rows).set_index("regime") if rows else pd.DataFrame()

    def interaction_effects(
        self,
        strategy_returns: pd.DataFrame,
        regimes: pd.Series,
    ) -> pd.DataFrame:
        """Estimate strategy-regime interaction effects."""

        aligned = strategy_returns.join(regimes.rename("regime"), how="inner").dropna()
        rows = []
        unconditional = strategy_returns.mean()
        for regime, group in aligned.groupby("regime"):
            for strategy in strategy_returns.columns:
                rows.append(
                    {
                        "regime": regime,
                        "strategy": strategy,
                        "conditional_mean": float(group[strategy].mean()),
                        "interaction_effect": float(group[strategy].mean() - unconditional[strategy]),
                    }
                )
        return pd.DataFrame(rows)

