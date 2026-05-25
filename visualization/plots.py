"""Visualization utilities for quant research workflows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from analytics.metrics import drawdown_series, rolling_sharpe


@dataclass(frozen=True)
class PlotConfig:
    """Plotting preferences."""

    interactive: bool = True
    template: str = "plotly_white"
    title_prefix: str = "Regime-Aware Equities"


class QuantPlotter:
    """Create common research visualizations with Plotly or Matplotlib."""

    def __init__(self, config: PlotConfig | None = None) -> None:
        self.config = config or PlotConfig()

    def equity_curve(self, equity_curve: pd.Series):
        """Plot portfolio equity curve."""

        if self.config.interactive:
            go = self._plotly()
            figure = go.Figure()
            figure.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, name="Equity"))
            figure.update_layout(title=f"{self.config.title_prefix}: Equity Curve", template=self.config.template)
            return figure
        axis = self._matplotlib_axis(f"{self.config.title_prefix}: Equity Curve")
        equity_curve.plot(ax=axis)
        return axis.figure

    def drawdowns(self, equity_curve: pd.Series):
        """Plot drawdown curve."""

        drawdowns = drawdown_series(equity_curve)
        if self.config.interactive:
            go = self._plotly()
            figure = go.Figure()
            figure.add_trace(go.Scatter(x=drawdowns.index, y=drawdowns, fill="tozeroy", name="Drawdown"))
            figure.update_layout(title=f"{self.config.title_prefix}: Drawdowns", template=self.config.template)
            return figure
        axis = self._matplotlib_axis(f"{self.config.title_prefix}: Drawdowns")
        drawdowns.plot(ax=axis)
        return axis.figure

    def rolling_sharpe(self, returns: pd.Series, window: int = 63):
        """Plot rolling Sharpe ratio."""

        sharpe = rolling_sharpe(returns, window)
        if self.config.interactive:
            go = self._plotly()
            figure = go.Figure()
            figure.add_trace(go.Scatter(x=sharpe.index, y=sharpe, name="Rolling Sharpe"))
            figure.update_layout(title=f"{self.config.title_prefix}: Rolling Sharpe", template=self.config.template)
            return figure
        axis = self._matplotlib_axis(f"{self.config.title_prefix}: Rolling Sharpe")
        sharpe.plot(ax=axis)
        return axis.figure

    def regime_overlay(self, equity_curve: pd.Series, regimes: pd.Series):
        """Plot equity curve with regime labels overlaid."""

        aligned_regimes = regimes.reindex(equity_curve.index).ffill()
        if self.config.interactive:
            go = self._plotly()
            figure = go.Figure()
            figure.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, name="Equity"))
            scaled_regime = aligned_regimes / max(aligned_regimes.max(), 1) * equity_curve.max()
            figure.add_trace(
                go.Scatter(
                    x=scaled_regime.index,
                    y=scaled_regime,
                    name="Regime",
                    opacity=0.35,
                    line={"dash": "dot"},
                )
            )
            figure.update_layout(title=f"{self.config.title_prefix}: Regime Overlay", template=self.config.template)
            return figure
        axis = self._matplotlib_axis(f"{self.config.title_prefix}: Regime Overlay")
        equity_curve.plot(ax=axis)
        axis2 = axis.twinx()
        aligned_regimes.plot(ax=axis2, alpha=0.35, style="--")
        return axis.figure

    def regime_timeline(self, regimes: pd.Series):
        """Plot a standalone market regime timeline."""

        regimes = regimes.sort_index()
        if self.config.interactive:
            go = self._plotly()
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=regimes.index,
                    y=regimes,
                    mode="lines",
                    line={"shape": "hv"},
                    name="Regime",
                )
            )
            figure.update_layout(title=f"{self.config.title_prefix}: Regime Timeline", template=self.config.template)
            return figure
        axis = self._matplotlib_axis(f"{self.config.title_prefix}: Regime Timeline")
        regimes.plot(ax=axis, drawstyle="steps-post")
        return axis.figure

    def allocation_heatmap(self, weights: pd.DataFrame):
        """Plot target or executed allocation heatmap."""

        if self.config.interactive:
            go = self._plotly()
            figure = go.Figure(
                data=go.Heatmap(
                    z=weights.T.values,
                    x=weights.index,
                    y=weights.columns,
                    colorscale="RdBu",
                    zmid=0,
                )
            )
            figure.update_layout(title=f"{self.config.title_prefix}: Allocation Heatmap", template=self.config.template)
            return figure
        axis = self._matplotlib_axis(f"{self.config.title_prefix}: Allocation Heatmap")
        axis.imshow(weights.T.values, aspect="auto", cmap="RdBu")
        axis.set_yticks(range(len(weights.columns)))
        axis.set_yticklabels(weights.columns)
        return axis.figure

    def heatmap(self, matrix: pd.DataFrame, title: str = "Heatmap"):
        """Plot a generic matrix heatmap."""

        if self.config.interactive:
            go = self._plotly()
            figure = go.Figure(
                data=go.Heatmap(
                    z=matrix.values,
                    x=matrix.columns,
                    y=matrix.index,
                    colorscale="RdBu",
                    zmid=0 if matrix.min().min() < 0 else None,
                )
            )
            figure.update_layout(title=f"{self.config.title_prefix}: {title}", template=self.config.template)
            return figure
        axis = self._matplotlib_axis(f"{self.config.title_prefix}: {title}")
        axis.imshow(matrix.values, aspect="auto", cmap="RdBu")
        axis.set_xticks(range(len(matrix.columns)))
        axis.set_xticklabels(matrix.columns, rotation=45, ha="right")
        axis.set_yticks(range(len(matrix.index)))
        axis.set_yticklabels(matrix.index)
        return axis.figure

    def dendrogram(self, linkage_matrix, labels: list[str]):
        """Plot hierarchical clustering dendrogram when scipy is installed."""

        axis = self._matplotlib_axis(f"{self.config.title_prefix}: Strategy Clusters")
        try:
            from scipy.cluster.hierarchy import dendrogram

            dendrogram(linkage_matrix, labels=labels, ax=axis)
        except Exception:
            axis.text(0.05, 0.5, "Dendrogram requires scipy linkage output.", transform=axis.transAxes)
        return axis.figure

    def strategy_contribution(self, contributions: pd.DataFrame):
        """Plot cumulative strategy return contributions."""

        cumulative = contributions.cumsum()
        if self.config.interactive:
            go = self._plotly()
            figure = go.Figure()
            for column in cumulative:
                figure.add_trace(go.Scatter(x=cumulative.index, y=cumulative[column], name=column))
            figure.update_layout(
                title=f"{self.config.title_prefix}: Strategy Contribution",
                template=self.config.template,
            )
            return figure
        axis = self._matplotlib_axis(f"{self.config.title_prefix}: Strategy Contribution")
        cumulative.plot(ax=axis)
        return axis.figure

    def rolling_correlations(self, correlations: pd.DataFrame):
        """Plot rolling average pairwise correlations."""

        if self.config.interactive:
            go = self._plotly()
            figure = go.Figure()
            for column in correlations:
                figure.add_trace(go.Scatter(x=correlations.index, y=correlations[column], name=column))
            figure.update_layout(
                title=f"{self.config.title_prefix}: Rolling Correlations",
                template=self.config.template,
            )
            return figure
        axis = self._matplotlib_axis(f"{self.config.title_prefix}: Rolling Correlations")
        correlations.plot(ax=axis)
        return axis.figure

    def factor_exposures(self, exposures: pd.Series | pd.DataFrame):
        """Plot factor exposure bars or time series."""

        if self.config.interactive:
            go = self._plotly()
            figure = go.Figure()
            if isinstance(exposures, pd.Series):
                figure.add_trace(go.Bar(x=exposures.index, y=exposures.values, name="Exposure"))
            else:
                for column in exposures:
                    figure.add_trace(go.Scatter(x=exposures.index, y=exposures[column], name=column))
            figure.update_layout(title=f"{self.config.title_prefix}: Factor Exposures", template=self.config.template)
            return figure
        axis = self._matplotlib_axis(f"{self.config.title_prefix}: Factor Exposures")
        exposures.plot(ax=axis, kind="bar" if isinstance(exposures, pd.Series) else "line")
        return axis.figure

    def volatility_regime_map(self, realized_volatility: pd.Series, regimes: pd.Series):
        """Plot realized volatility colored by regime."""

        aligned = pd.DataFrame(
            {
                "realized_volatility": realized_volatility,
                "regime": regimes.reindex(realized_volatility.index).ffill(),
            }
        ).dropna()
        if self.config.interactive:
            go = self._plotly()
            figure = go.Figure()
            for regime, group in aligned.groupby("regime"):
                figure.add_trace(
                    go.Scatter(
                        x=group.index,
                        y=group["realized_volatility"],
                        mode="markers",
                        name=f"Regime {regime}",
                    )
                )
            figure.update_layout(
                title=f"{self.config.title_prefix}: Volatility Regime Map",
                template=self.config.template,
            )
            return figure
        axis = self._matplotlib_axis(f"{self.config.title_prefix}: Volatility Regime Map")
        axis.scatter(aligned.index, aligned["realized_volatility"], c=aligned["regime"])
        return axis.figure

    def walk_forward_equity(self, equity_curve: pd.Series, window_metrics: pd.DataFrame | None = None):
        """Plot out-of-sample walk-forward equity and optional test segments."""

        if self.config.interactive:
            go = self._plotly()
            figure = go.Figure()
            figure.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, name="OOS Equity"))
            if window_metrics is not None and not window_metrics.empty:
                for _, row in window_metrics.iterrows():
                    figure.add_vrect(
                        x0=row["test_start"],
                        x1=row["test_end"],
                        fillcolor="#dbeafe",
                        opacity=0.18,
                        line_width=0,
                    )
            figure.update_layout(title=f"{self.config.title_prefix}: Walk-Forward Equity", template=self.config.template)
            return figure
        axis = self._matplotlib_axis(f"{self.config.title_prefix}: Walk-Forward Equity")
        equity_curve.plot(ax=axis)
        return axis.figure

    def robustness_heatmap(self, heatmap: pd.DataFrame, metric: str = "Sharpe"):
        """Plot parameter robustness heatmap."""

        return self.heatmap(heatmap, title=f"Parameter Robustness: {metric}")

    @staticmethod
    def _plotly():
        try:
            import plotly.graph_objects as go
        except ImportError as exc:  # pragma: no cover - depends on optional environment
            raise ImportError("Install plotly to use interactive QuantPlotter outputs") from exc
        return go

    @staticmethod
    def _matplotlib_axis(title: str):
        import matplotlib.pyplot as plt

        _, axis = plt.subplots(figsize=(12, 6))
        axis.set_title(title)
        return axis
