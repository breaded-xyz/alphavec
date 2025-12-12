"""
Tearsheet rendering utilities for alphavec simulations.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import metrics as _metrics_mod

TEARSHEET_NOTES = _metrics_mod.TEARSHEET_NOTES


def tearsheet(
    *,
    metrics: pd.DataFrame,
    returns: pd.Series,
    output_path: str | Path | None = None,
) -> str:
    """
    Render a self-contained HTML tearsheet (Plotly charts) from metrics and returns.
    """

    import plotly.graph_objects as go
    import plotly.io as pio

    benchmark_asset = None
    if "Benchmark Asset" in metrics.index:
        benchmark_asset = metrics.loc["Benchmark Asset", "Value"]

    init_cash = float(metrics.attrs.get("init_cash", 1.0))
    equity: pd.Series
    if isinstance(metrics.attrs.get("equity"), pd.Series):
        equity = metrics.attrs["equity"]
    else:
        equity = (1.0 + returns.fillna(0.0)).cumprod()
        equity.name = "equity"

    benchmark_equity = metrics.attrs.get("benchmark_equity")
    if not isinstance(benchmark_equity, pd.Series):
        benchmark_equity = None

    equity_pct = equity / float(init_cash) - 1.0
    equity_pct = (equity_pct * 100.0).rename("Portfolio %")

    benchmark_equity_pct: pd.Series | None = None
    if benchmark_equity is not None and len(benchmark_equity) == len(equity):
        benchmark_equity_pct = benchmark_equity / float(init_cash) - 1.0
        label = (
            f"Benchmark ({benchmark_asset}) %"
            if isinstance(benchmark_asset, str) and benchmark_asset
            else "Benchmark %"
        )
        benchmark_equity_pct = (benchmark_equity_pct * 100.0).rename(label)

    dd = equity / equity.cummax() - 1.0
    dd_pct = (dd * 100.0).rename("Drawdown %")

    equity_fig = go.Figure()
    equity_fig.add_trace(go.Scatter(x=equity_pct.index, y=equity_pct, name=equity_pct.name))
    if benchmark_equity_pct is not None:
        equity_fig.add_trace(
            go.Scatter(
                x=benchmark_equity_pct.index,
                y=benchmark_equity_pct,
                name=benchmark_equity_pct.name,
            )
        )
    equity_fig.update_layout(
        title="Equity Curve (Cumulative Return %)",
        xaxis_title="Date",
        yaxis_title="Cumulative return (%)",
        legend_title="Series",
        template="plotly_white",
    )

    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(x=dd_pct.index, y=dd_pct, name=dd_pct.name, fill="tozeroy"))
    dd_fig.update_layout(
        title="Drawdown (%)",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        showlegend=False,
    )

    returns_pct = (returns.fillna(0.0) * 100.0).rename("Returns (%)")
    dist_fig = go.Figure()
    dist_fig.add_trace(go.Histogram(x=returns_pct, nbinsx=60, name=returns_pct.name))
    dist_fig.update_layout(
        title="Returns Distribution (%)",
        xaxis_title="Return (%)",
        yaxis_title="Count",
        template="plotly_white",
        showlegend=False,
    )

    table_html = metrics.copy()
    table_html.index.name = "Metric"
    metrics_table = table_html.to_html(classes="metrics", escape=True)

    plots_html = []
    plots_html.append(pio.to_html(equity_fig, include_plotlyjs=True, full_html=False))
    plots_html.append(pio.to_html(dd_fig, include_plotlyjs=False, full_html=False))
    plots_html.append(pio.to_html(dist_fig, include_plotlyjs=False, full_html=False))

    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Tearsheet</title>
    <style>
      body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
      h1 {{ margin: 0 0 16px 0; }}
      h2 {{ margin: 28px 0 12px 0; }}
      table.metrics {{ border-collapse: collapse; width: 100%; background-color: #ffffff; color: #000000; }}
      table.metrics th, table.metrics td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
      table.metrics th {{ background: #f6f6f6; text-align: left; }}
      .plot {{ margin-top: 12px; }}
    </style>
  </head>
  <body>
    <h1>Tearsheet</h1>
    {metrics_table}
    <h2>Equity</h2>
    <div class="plot">{plots_html[0]}</div>
    <h2>Drawdown</h2>
    <div class="plot">{plots_html[1]}</div>
    <h2>Returns Distribution</h2>
    <div class="plot">{plots_html[2]}</div>
  </body>
</html>"""

    if output_path is not None:
        Path(output_path).write_text(html, encoding="utf-8")

    return html
