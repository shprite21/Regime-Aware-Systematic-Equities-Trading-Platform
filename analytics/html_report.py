"""Reusable HTML research report generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.tearsheet import PerformanceReport


@dataclass(frozen=True)
class HTMLReportConfig:
    """HTML report configuration."""

    title: str = "Regime-Aware Systematic Equities Research Report"
    subtitle: str = "Institutional-style systematic equities diagnostics"
    include_plotly: bool = True
    theme_color: str = "#0f172a"


@dataclass
class HTMLReportInputs:
    """Optional sections for the advanced report."""

    regimes: pd.Series | None = None
    strategy_contribution: pd.DataFrame | None = None
    benchmark_equity: pd.DataFrame | None = None
    benchmark_metrics: pd.DataFrame | None = None
    walk_forward_metrics: pd.DataFrame | None = None
    correlation_matrix: pd.DataFrame | None = None
    factor_betas: pd.Series | pd.DataFrame | None = None
    regime_transition_matrix: pd.DataFrame | None = None
    robustness_results: pd.DataFrame | None = None
    cost_sensitivity: pd.DataFrame | None = None
    portfolio_diagnostics: dict[str, object] = field(default_factory=dict)
    data_quality_summary: pd.DataFrame | None = None


class AdvancedHTMLReportBuilder:
    """Build self-contained institutional-style HTML research reports."""

    def __init__(self, config: HTMLReportConfig | None = None) -> None:
        self.config = config or HTMLReportConfig()

    def write(
        self,
        report: PerformanceReport,
        output_path: str | Path,
        inputs: HTMLReportInputs | None = None,
    ) -> Path:
        """Write an HTML report and return the output path."""

        inputs = inputs or HTMLReportInputs()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.render(report, inputs), encoding="utf-8")
        return output_path

    def render(self, report: PerformanceReport, inputs: HTMLReportInputs) -> str:
        """Render the full report to an HTML string."""

        kpis = _kpi_cards(report.metrics)
        sections = [
            _section("Summary Dashboard", kpis + _table(pd.Series(report.metrics).round(4).to_frame("value"))),
            _section(
                "Core Performance Charts",
                _chart_grid(
                    [
                        ("Equity Curve", _line_svg(report.equity_curve, "#2563eb")),
                        ("Drawdown", _line_svg(report.drawdowns, "#dc2626")),
                        ("Rolling Sharpe", _line_svg(report.rolling_sharpe.dropna(), "#059669")),
                        (
                            "Regime Timeline",
                            _line_svg(inputs.regimes.reindex(report.equity_curve.index).ffill(), "#7c3aed")
                            if inputs.regimes is not None
                            else "<p>No regime series supplied.</p>",
                        ),
                    ]
                ),
            ),
            _section("Benchmark Comparison", _benchmark_section(inputs)),
            _section("Regime Analytics", _regime_section(report, inputs)),
            _section("Strategy Contribution", _strategy_section(inputs)),
            _section("Correlation And Clustering", _table(inputs.correlation_matrix)),
            _section("Factor Exposure Diagnostics", _factor_section(inputs)),
            _section("Walk-Forward Validation", _table(inputs.walk_forward_metrics)),
            _section("Robustness Testing", _table(inputs.robustness_results)),
            _section("Transaction Cost Sensitivity", _table(inputs.cost_sensitivity)),
            _section("Portfolio Diagnostics", _portfolio_diagnostics_section(inputs.portfolio_diagnostics)),
            _section("Data Quality", _table(inputs.data_quality_summary)),
        ]
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{self.config.title}</title>
  <style>{_css(self.config.theme_color)}</style>
</head>
<body>
  <header>
    <h1>{self.config.title}</h1>
    <p>{self.config.subtitle}</p>
  </header>
  <nav>
    <a href="#summary-dashboard">Summary</a>
    <a href="#benchmark-comparison">Benchmarks</a>
    <a href="#regime-analytics">Regimes</a>
    <a href="#factor-exposure-diagnostics">Factors</a>
    <a href="#walk-forward-validation">Walk-Forward</a>
    <a href="#robustness-testing">Robustness</a>
  </nav>
  <main>
    {''.join(sections)}
  </main>
</body>
</html>
"""


def _section(title: str, body: str) -> str:
    anchor = title.lower().replace(" ", "-")
    return f"<details id='{anchor}' open><summary>{title}</summary><div class='section-body'>{body}</div></details>"


def _benchmark_section(inputs: HTMLReportInputs) -> str:
    charts = ""
    if inputs.benchmark_equity is not None and not inputs.benchmark_equity.empty:
        charts = _chart_grid(
            [(column, _line_svg(inputs.benchmark_equity[column], "#334155")) for column in inputs.benchmark_equity.columns[:6]]
        )
    return charts + _table(inputs.benchmark_metrics)


def _regime_section(report: PerformanceReport, inputs: HTMLReportInputs) -> str:
    return (
        "<h3>Regime-Wise Performance</h3>"
        + _table(report.regime_breakdown)
        + "<h3>Transition Probabilities</h3>"
        + _table(inputs.regime_transition_matrix)
    )


def _strategy_section(inputs: HTMLReportInputs) -> str:
    if inputs.strategy_contribution is None:
        return "<p>No strategy attribution supplied.</p>"
    cumulative = inputs.strategy_contribution.cumsum()
    charts = _chart_grid([(column, _line_svg(cumulative[column], "#0ea5e9")) for column in cumulative.columns[:6]])
    return charts + _table(cumulative.tail(10).round(4))


def _factor_section(inputs: HTMLReportInputs) -> str:
    if inputs.factor_betas is None:
        return "<p>No factor exposure diagnostics supplied.</p>"
    if isinstance(inputs.factor_betas, pd.Series):
        return _table(inputs.factor_betas.round(4).to_frame("beta"))
    return _table(inputs.factor_betas.round(4))


def _portfolio_diagnostics_section(diagnostics: dict[str, object]) -> str:
    if not diagnostics:
        return "<p>No portfolio diagnostics supplied.</p>"
    parts = []
    for name, value in diagnostics.items():
        if isinstance(value, pd.DataFrame):
            parts.append(f"<h3>{name.replace('_', ' ').title()}</h3>{_table(value.tail(10).round(4))}")
        elif isinstance(value, pd.Series):
            parts.append(f"<h3>{name.replace('_', ' ').title()}</h3>{_table(value.tail(10).round(4).to_frame(name))}")
        else:
            parts.append(f"<p><strong>{name}:</strong> {value}</p>")
    return "".join(parts)


def _kpi_cards(metrics: dict[str, float]) -> str:
    selected = ["total_return", "annualized_return", "annualized_volatility", "sharpe", "max_drawdown"]
    cards = []
    for key in selected:
        value = metrics.get(key)
        if value is not None:
            cards.append(f"<div class='kpi'><span>{key.replace('_', ' ').title()}</span><strong>{value:.4f}</strong></div>")
    return f"<div class='kpi-grid'>{''.join(cards)}</div>"


def _chart_grid(items: list[tuple[str, str]]) -> str:
    cards = [f"<div class='chart-card'><h3>{title}</h3>{chart}</div>" for title, chart in items]
    return f"<div class='chart-grid'>{''.join(cards)}</div>"


def _table(frame: pd.DataFrame | pd.Series | None) -> str:
    if frame is None:
        return "<p>No data supplied.</p>"
    if isinstance(frame, pd.Series):
        frame = frame.to_frame("value")
    if frame.empty:
        return "<p>No data available.</p>"
    return frame.to_html(classes="table", border=0)


def _line_svg(series: pd.Series, color: str) -> str:
    clean = series.dropna()
    if clean.empty:
        return "<svg viewBox='0 0 640 260'><text x='20' y='40'>No data</text></svg>"
    values = clean.astype(float).to_numpy()
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    spread = maximum - minimum if maximum != minimum else 1.0
    width = 640
    height = 260
    margin = 30
    x_scale = (width - 2 * margin) / max(len(values) - 1, 1)
    y_scale = (height - 2 * margin) / spread
    points = []
    for index, value in enumerate(values):
        x = margin + index * x_scale
        y = height - margin - (value - minimum) * y_scale
        points.append(f"{x:.2f},{y:.2f}")
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img'>"
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='#d1d5db'/>"
        f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='#d1d5db'/>"
        f"<polyline fill='none' stroke='{color}' stroke-width='2.2' points='{' '.join(points)}'/>"
        f"<text x='{margin}' y='18' font-size='12' fill='#64748b'>max {maximum:.4f}</text>"
        f"<text x='{margin}' y='{height - 6}' font-size='12' fill='#64748b'>min {minimum:.4f}</text>"
        "</svg>"
    )


def _css(theme_color: str) -> str:
    return f"""
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; background: #f8fafc; color: #172033; }}
    header {{ background: {theme_color}; color: white; padding: 30px 42px; }}
    header h1 {{ margin: 0 0 8px 0; }}
    header p {{ margin: 0; color: #cbd5e1; }}
    nav {{ position: sticky; top: 0; background: #ffffff; border-bottom: 1px solid #e5e7eb; padding: 12px 42px; z-index: 10; }}
    nav a {{ color: #334155; text-decoration: none; margin-right: 20px; font-size: 14px; }}
    main {{ max-width: 1260px; margin: 0 auto; padding: 26px; }}
    details {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; margin-bottom: 18px; }}
    summary {{ cursor: pointer; padding: 18px 22px; font-size: 19px; font-weight: 700; }}
    .section-body {{ padding: 0 22px 22px 22px; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin-bottom: 22px; }}
    .kpi {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; background: #f8fafc; }}
    .kpi span {{ display: block; color: #64748b; font-size: 13px; margin-bottom: 8px; }}
    .kpi strong {{ font-size: 24px; color: #0f172a; }}
    .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
    .chart-card h3 {{ margin-bottom: 8px; }}
    svg {{ width: 100%; height: 260px; border: 1px solid #e5e7eb; border-radius: 6px; background: #fff; }}
    .table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 10px; }}
    .table th, .table td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: right; }}
    .table th:first-child, .table td:first-child {{ text-align: left; }}
    """

