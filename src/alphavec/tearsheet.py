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
    signal_smooth_window: int = 30,
) -> str:
    """
    Render a self-contained HTML tearsheet (Plotly charts) from metrics and returns.

    Args:
        metrics: Metrics DataFrame produced by `simulate()`.
        returns: Portfolio period returns.
        output_path: Optional path to write the HTML.
        signal_smooth_window: Rolling window (in periods) used to smooth Signal time-series plots.
    """

    import plotly.graph_objects as go
    import plotly.io as pio

    def _roll_mean(s: pd.Series, window: int) -> pd.Series:
        if window <= 1:
            return s
        min_periods = max(3, window // 5)
        return s.rolling(window=window, min_periods=min_periods).mean()

    def _zero_line(fig: go.Figure) -> None:
        fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color="#999999")

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

    wf = metrics.attrs.get("weight_forward")
    wf_deciles = metrics.attrs.get("weight_forward_deciles")

    wf_ic_fig = None
    wf_spread_fig = None
    wf_attrib_fig = None
    wf_deciles_fig = None
    wf_ic_dist_fig = None
    wf_spread_dist_fig = None

    if isinstance(wf, pd.DataFrame) and len(wf.index) > 0:
        smooth_window = max(1, int(signal_smooth_window))

        wf_ic_fig = go.Figure()
        if "ic" in wf.columns:
            ic = wf["ic"]
            wf_ic_fig.add_trace(
                go.Scatter(
                    x=ic.index,
                    y=ic,
                    name="IC (raw)",
                    mode="lines",
                    line=dict(width=1),
                    opacity=0.25,
                    visible="legendonly",
                )
            )
            wf_ic_fig.add_trace(
                go.Scatter(
                    x=ic.index,
                    y=_roll_mean(ic, smooth_window),
                    name=f"IC ({smooth_window}p mean)",
                    mode="lines",
                    line=dict(width=2),
                )
            )
        if "rank_ic" in wf.columns:
            ric = wf["rank_ic"]
            wf_ic_fig.add_trace(
                go.Scatter(
                    x=ric.index,
                    y=ric,
                    name="Rank IC (raw)",
                    mode="lines",
                    line=dict(width=1),
                    opacity=0.25,
                    visible="legendonly",
                )
            )
            wf_ic_fig.add_trace(
                go.Scatter(
                    x=ric.index,
                    y=_roll_mean(ric, smooth_window),
                    name=f"Rank IC ({smooth_window}p mean)",
                    mode="lines",
                    line=dict(width=2),
                )
            )
        _zero_line(wf_ic_fig)
        wf_ic_fig.update_layout(
            title="Signal: IC",
            xaxis_title="Date",
            yaxis_title="Correlation",
            template="plotly_white",
            legend_title="Series",
        )

        if "top_bottom_spread" in wf.columns:
            spread_pct = (wf["top_bottom_spread"] * 100.0).rename("Top - Bottom (deciles) %")
            wf_spread_fig = go.Figure()
            wf_spread_fig.add_trace(
                go.Scatter(
                    x=spread_pct.index,
                    y=spread_pct,
                    name=f"{spread_pct.name} (raw)",
                    mode="lines",
                    line=dict(width=1),
                    opacity=0.25,
                    visible="legendonly",
                )
            )
            wf_spread_fig.add_trace(
                go.Scatter(
                    x=spread_pct.index,
                    y=_roll_mean(spread_pct, smooth_window),
                    name=f"{spread_pct.name} ({smooth_window}p mean)",
                    mode="lines",
                    line=dict(width=2),
                )
            )
            _zero_line(wf_spread_fig)
            wf_spread_fig.update_layout(
                title="Signal: Top-Bottom Decile Spread (%)",
                xaxis_title="Date",
                yaxis_title="Spread (%)",
                template="plotly_white",
                legend_title="Series",
            )

        if (
            "forward_return_selection_per_gross" in wf.columns
            or "forward_return_directional_per_gross" in wf.columns
        ):
            wf_attrib_fig = go.Figure()
            if "forward_return_selection_per_gross" in wf.columns:
                sel = (wf["forward_return_selection_per_gross"] * 100.0).rename("Selection (per gross) %")
                wf_attrib_fig.add_trace(
                    go.Scatter(
                        x=sel.index,
                        y=sel,
                        name=f"{sel.name} (raw)",
                        mode="lines",
                        line=dict(width=1),
                        opacity=0.25,
                        visible="legendonly",
                    )
                )
                wf_attrib_fig.add_trace(
                    go.Scatter(
                        x=sel.index,
                        y=_roll_mean(sel, smooth_window),
                        name=f"{sel.name} ({smooth_window}p mean)",
                        mode="lines",
                        line=dict(width=2),
                    )
                )
            if "forward_return_directional_per_gross" in wf.columns:
                direc = (wf["forward_return_directional_per_gross"] * 100.0).rename(
                    "Directional (per gross) %"
                )
                wf_attrib_fig.add_trace(
                    go.Scatter(
                        x=direc.index,
                        y=direc,
                        name=f"{direc.name} (raw)",
                        mode="lines",
                        line=dict(width=1),
                        opacity=0.25,
                        visible="legendonly",
                    )
                )
                wf_attrib_fig.add_trace(
                    go.Scatter(
                        x=direc.index,
                        y=_roll_mean(direc, smooth_window),
                        name=f"{direc.name} ({smooth_window}p mean)",
                        mode="lines",
                        line=dict(width=2),
                    )
                )
            _zero_line(wf_attrib_fig)
            wf_attrib_fig.update_layout(
                title="Signal: Forward Return Attribution (per gross, next)",
                xaxis_title="Date",
                yaxis_title="Return (%)",
                template="plotly_white",
                legend_title="Series",
            )

        if "ic" in wf.columns or "rank_ic" in wf.columns:
            wf_ic_dist_fig = go.Figure()
            if "ic" in wf.columns:
                wf_ic_dist_fig.add_trace(go.Histogram(x=wf["ic"].dropna(), nbinsx=60, name="IC"))
            if "rank_ic" in wf.columns:
                wf_ic_dist_fig.add_trace(
                    go.Histogram(x=wf["rank_ic"].dropna(), nbinsx=60, name="Rank IC")
                )
            wf_ic_dist_fig.update_layout(
                title="Signal: IC Distribution",
                xaxis_title="Correlation",
                yaxis_title="Count",
                template="plotly_white",
                barmode="overlay",
            )
            wf_ic_dist_fig.update_traces(opacity=0.6)

        if "top_bottom_spread" in wf.columns:
            wf_spread_dist_fig = go.Figure()
            wf_spread_dist_fig.add_trace(
                go.Histogram(x=(wf["top_bottom_spread"] * 100.0).dropna(), nbinsx=60, name="Spread (%)")
            )
            wf_spread_dist_fig.update_layout(
                title="Signal: Decile Spread Distribution (%)",
                xaxis_title="Spread (%)",
                yaxis_title="Count",
                template="plotly_white",
                showlegend=False,
            )

    if isinstance(wf_deciles, pd.Series) and len(wf_deciles.index) > 0:
        wf_deciles_fig = go.Figure()
        wf_deciles_fig.add_trace(
            go.Bar(x=wf_deciles.index.astype(str), y=(wf_deciles * 100.0), name="Mean next return (%)")
        )
        wf_deciles_fig.update_layout(
            title="Signal: Mean Next Return by Weight Decile",
            xaxis_title="Weight decile (active universe)",
            yaxis_title="Mean next return (%)",
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

    weights_section_html = ""
    if (
        wf_ic_fig is not None
        or wf_spread_fig is not None
        or wf_attrib_fig is not None
        or wf_deciles_fig is not None
        or wf_ic_dist_fig is not None
        or wf_spread_dist_fig is not None
    ):
        wf_plots = []
        if wf_ic_fig is not None:
            wf_plots.append(pio.to_html(wf_ic_fig, include_plotlyjs=False, full_html=False))
        if wf_spread_fig is not None:
            wf_plots.append(pio.to_html(wf_spread_fig, include_plotlyjs=False, full_html=False))
        if wf_attrib_fig is not None:
            wf_plots.append(pio.to_html(wf_attrib_fig, include_plotlyjs=False, full_html=False))
        if wf_deciles_fig is not None:
            wf_plots.append(pio.to_html(wf_deciles_fig, include_plotlyjs=False, full_html=False))
        if wf_ic_dist_fig is not None:
            wf_plots.append(pio.to_html(wf_ic_dist_fig, include_plotlyjs=False, full_html=False))
        if wf_spread_dist_fig is not None:
            wf_plots.append(pio.to_html(wf_spread_dist_fig, include_plotlyjs=False, full_html=False))

        weights_section_html = (
            "<h2>Signal</h2>"
            + "".join(f'<div class="plot">{p}</div>' for p in wf_plots)
        )

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
      table.metrics tr, table.metrics th, table.metrics td {{ text-align: left !important; }}
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
    {weights_section_html}
  </body>
</html>"""

    if output_path is not None:
        Path(output_path).write_text(html, encoding="utf-8")

    return html
