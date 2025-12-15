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
    rolling_sharpe_window: int = 30,
) -> str:
    """
    Render a self-contained HTML tearsheet (Plotly charts) from metrics and returns.

    Args:
        metrics: Metrics DataFrame produced by `simulate()`.
        returns: Portfolio period returns.
        output_path: Optional path to write the HTML.
        signal_smooth_window: Rolling window (in periods) used to smooth Signal time-series plots.
        rolling_sharpe_window: Rolling window (in periods) used to compute Rolling Sharpe.
    """

    import plotly.graph_objects as go
    import plotly.io as pio
    import numpy as np

    def _note_block(definition: str, interpretation: str) -> str:
        return (
            '<div class="note">'
            f"<strong>Definition:</strong> {definition}<br>"
            f"<strong>Interpretation:</strong> {interpretation}"
            "</div>"
        )

    def _roll_mean(s: pd.Series, window: int) -> pd.Series:
        if window <= 1:
            return s
        min_periods = max(3, window // 5)
        return s.rolling(window=window, min_periods=min_periods).mean()

    def _zero_line(fig: go.Figure) -> None:
        fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color="#999999")

    def _metric_value(metric: str, default: object) -> object:
        try:
            v = metrics.loc[metric, "Value"]
        except Exception:
            return default
        return default if pd.isna(v) else v

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

    try:
        rolling_window = max(2, int(rolling_sharpe_window))
    except Exception:
        rolling_window = 30

    freq_rule = str(_metric_value("Period frequency", "1D"))
    trading_days_year = int(_metric_value("Trading Days Year", 365))
    risk_free_rate = float(_metric_value("Risk Free Rate", 0.0))
    annual_factor = float(_metrics_mod._annualization_factor(freq_rule, trading_days_year))
    annual_factor = annual_factor if np.isfinite(annual_factor) and annual_factor > 0 else 1.0
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / annual_factor) - 1.0

    excess = returns.fillna(0.0) - rf_per_period
    min_periods = max(3, rolling_window // 5)
    roll_mean = excess.rolling(window=rolling_window, min_periods=min_periods).mean()
    roll_std = excess.rolling(window=rolling_window, min_periods=min_periods).std(ddof=1)
    rolling_sharpe = (roll_mean / roll_std) * np.sqrt(annual_factor)
    rolling_sharpe = rolling_sharpe.rename(f"Rolling Sharpe ({rolling_window}p)")

    rolling_sharpe_fig = go.Figure()
    rolling_sharpe_fig.add_trace(
        go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, name=rolling_sharpe.name)
    )
    _zero_line(rolling_sharpe_fig)
    rolling_sharpe_fig.update_layout(
        title=f"Rolling Sharpe ({rolling_window} periods)",
        xaxis_title="Date",
        yaxis_title="Sharpe",
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
    wf_deciles_median = metrics.attrs.get("weight_forward_deciles_median")
    wf_deciles_std = metrics.attrs.get("weight_forward_deciles_std")
    wf_deciles_count = metrics.attrs.get("weight_forward_deciles_count")
    wf_deciles_contrib = metrics.attrs.get("weight_forward_deciles_contrib")
    wf_deciles_contrib_long = metrics.attrs.get("weight_forward_deciles_contrib_long")
    wf_deciles_contrib_short = metrics.attrs.get("weight_forward_deciles_contrib_short")

    wf_ic_fig = None
    wf_spread_fig = None
    wf_attrib_fig = None
    wf_deciles_fig = None
    wf_deciles_median_fig = None
    wf_deciles_sharpe_fig = None
    wf_deciles_contrib_fig = None
    wf_deciles_contrib_ls_fig = None
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
        counts: pd.Series | None = None
        if isinstance(wf_deciles_count, pd.Series) and len(wf_deciles_count.index) > 0:
            counts = wf_deciles_count.reindex(wf_deciles.index)
        x = wf_deciles.index.astype(int).to_numpy()
        ticktext = None
        customdata = None
        if counts is not None:
            counts_int = counts.fillna(0.0).astype(int).to_numpy()
            ticktext = [f"{d}<br>n={n}" for d, n in zip(x, counts_int)]
            customdata = counts_int

        wf_deciles_fig = go.Figure()
        wf_deciles_fig.add_trace(
            go.Bar(
                x=x,
                y=(wf_deciles * 100.0),
                name="Mean next return (%)",
                customdata=customdata,
                hovertemplate=(
                    "Decile=%{x}<br>Mean next return=%{y:.4f}%<br>n=%{customdata}<extra></extra>"
                    if customdata is not None
                    else "Decile=%{x}<br>Mean next return=%{y:.4f}%<extra></extra>"
                ),
            )
        )
        if ticktext is not None:
            wf_deciles_fig.update_xaxes(tickmode="array", tickvals=x, ticktext=ticktext)
        wf_deciles_fig.update_layout(
            title="Signal: Mean Next Return by Weight Decile",
            xaxis_title="Weight decile (active universe)",
            yaxis_title="Mean next return (%)",
            template="plotly_white",
            showlegend=False,
        )

    if isinstance(wf_deciles_median, pd.Series) and len(wf_deciles_median.index) > 0:
        counts: pd.Series | None = None
        if isinstance(wf_deciles_count, pd.Series) and len(wf_deciles_count.index) > 0:
            counts = wf_deciles_count.reindex(wf_deciles_median.index)
        x = wf_deciles_median.index.astype(int).to_numpy()
        ticktext = None
        customdata = None
        if counts is not None:
            counts_int = counts.fillna(0.0).astype(int).to_numpy()
            ticktext = [f"{d}<br>n={n}" for d, n in zip(x, counts_int)]
            customdata = counts_int

        wf_deciles_median_fig = go.Figure()
        wf_deciles_median_fig.add_trace(
            go.Bar(
                x=x,
                y=(wf_deciles_median * 100.0),
                name="Median next return (%)",
                customdata=customdata,
                hovertemplate=(
                    "Decile=%{x}<br>Median next return=%{y:.4f}%<br>n=%{customdata}<extra></extra>"
                    if customdata is not None
                    else "Decile=%{x}<br>Median next return=%{y:.4f}%<extra></extra>"
                ),
            )
        )
        if ticktext is not None:
            wf_deciles_median_fig.update_xaxes(tickmode="array", tickvals=x, ticktext=ticktext)
        wf_deciles_median_fig.update_layout(
            title="Signal: Median Next Return by Weight Decile",
            xaxis_title="Weight decile (active universe)",
            yaxis_title="Median next return (%)",
            template="plotly_white",
            showlegend=False,
        )

    if (
        isinstance(wf_deciles, pd.Series)
        and isinstance(wf_deciles_std, pd.Series)
        and len(wf_deciles.index) > 0
        and len(wf_deciles_std.index) > 0
    ):
        std = wf_deciles_std.reindex(wf_deciles.index)
        mean_excess = wf_deciles - rf_per_period
        sharpe = (mean_excess / std) * np.sqrt(annual_factor)
        sharpe = sharpe.replace([np.inf, -np.inf], np.nan).rename("Sharpe (annualized)")

        counts: pd.Series | None = None
        if isinstance(wf_deciles_count, pd.Series) and len(wf_deciles_count.index) > 0:
            counts = wf_deciles_count.reindex(sharpe.index)
        x = sharpe.index.astype(int).to_numpy()
        ticktext = None
        customdata = None
        if counts is not None:
            counts_int = counts.fillna(0.0).astype(int).to_numpy()
            ticktext = [f"{d}<br>n={n}" for d, n in zip(x, counts_int)]
            customdata = counts_int

        wf_deciles_sharpe_fig = go.Figure()
        wf_deciles_sharpe_fig.add_trace(
            go.Bar(
                x=x,
                y=sharpe,
                name=sharpe.name,
                customdata=customdata,
                hovertemplate=(
                    "Decile=%{x}<br>Sharpe=%{y:.4f}<br>n=%{customdata}<extra></extra>"
                    if customdata is not None
                    else "Decile=%{x}<br>Sharpe=%{y:.4f}<extra></extra>"
                ),
            )
        )
        _zero_line(wf_deciles_sharpe_fig)
        if ticktext is not None:
            wf_deciles_sharpe_fig.update_xaxes(tickmode="array", tickvals=x, ticktext=ticktext)
        wf_deciles_sharpe_fig.update_layout(
            title="Signal: Sharpe by Weight Decile (Annualized)",
            xaxis_title="Weight decile (active universe)",
            yaxis_title="Sharpe",
            template="plotly_white",
            showlegend=False,
        )

    if isinstance(wf_deciles_contrib, pd.Series) and len(wf_deciles_contrib.index) > 0:
        counts: pd.Series | None = None
        if isinstance(wf_deciles_count, pd.Series) and len(wf_deciles_count.index) > 0:
            counts = wf_deciles_count.reindex(wf_deciles_contrib.index)
        x = wf_deciles_contrib.index.astype(int).to_numpy()
        ticktext = None
        customdata = None
        if counts is not None:
            counts_int = counts.fillna(0.0).astype(int).to_numpy()
            ticktext = [f"{d}<br>n={n}" for d, n in zip(x, counts_int)]
            customdata = counts_int

        wf_deciles_contrib_fig = go.Figure()
        wf_deciles_contrib_fig.add_trace(
            go.Bar(
                x=x,
                y=(wf_deciles_contrib * 100.0),
                name="Mean contribution (%)",
                customdata=customdata,
                hovertemplate=(
                    "Decile=%{x}<br>Mean contribution=%{y:.6f}%<br>n=%{customdata}<extra></extra>"
                    if customdata is not None
                    else "Decile=%{x}<br>Mean contribution=%{y:.6f}%<extra></extra>"
                ),
            )
        )
        _zero_line(wf_deciles_contrib_fig)
        if ticktext is not None:
            wf_deciles_contrib_fig.update_xaxes(tickmode="array", tickvals=x, ticktext=ticktext)
        wf_deciles_contrib_fig.update_layout(
            title="Signal: Mean Next Return Contribution by Weight Decile",
            xaxis_title="Weight decile (active universe)",
            yaxis_title="Mean contribution to next return (%)",
            template="plotly_white",
            showlegend=False,
        )

    if (
        isinstance(wf_deciles_contrib_long, pd.Series)
        and isinstance(wf_deciles_contrib_short, pd.Series)
        and len(wf_deciles_contrib_long.index) > 0
        and len(wf_deciles_contrib_short.index) > 0
    ):
        long_c = wf_deciles_contrib_long.reindex(wf_deciles_contrib_short.index)
        short_c = wf_deciles_contrib_short

        counts: pd.Series | None = None
        if isinstance(wf_deciles_count, pd.Series) and len(wf_deciles_count.index) > 0:
            counts = wf_deciles_count.reindex(short_c.index)
        x = short_c.index.astype(int).to_numpy()
        ticktext = None
        customdata = None
        if counts is not None:
            counts_int = counts.fillna(0.0).astype(int).to_numpy()
            ticktext = [f"{d}<br>n={n}" for d, n in zip(x, counts_int)]
            customdata = counts_int

        wf_deciles_contrib_ls_fig = go.Figure()
        wf_deciles_contrib_ls_fig.add_trace(
            go.Bar(
                x=x,
                y=(long_c * 100.0),
                name="Long contribution (%)",
                customdata=customdata,
                hovertemplate=(
                    "Decile=%{x}<br>Long contrib=%{y:.6f}%<br>n=%{customdata}<extra></extra>"
                    if customdata is not None
                    else "Decile=%{x}<br>Long contrib=%{y:.6f}%<extra></extra>"
                ),
            )
        )
        wf_deciles_contrib_ls_fig.add_trace(
            go.Bar(
                x=x,
                y=(short_c * 100.0),
                name="Short contribution (%)",
                customdata=customdata,
                hovertemplate=(
                    "Decile=%{x}<br>Short contrib=%{y:.6f}%<br>n=%{customdata}<extra></extra>"
                    if customdata is not None
                    else "Decile=%{x}<br>Short contrib=%{y:.6f}%<extra></extra>"
                ),
            )
        )
        _zero_line(wf_deciles_contrib_ls_fig)
        if ticktext is not None:
            wf_deciles_contrib_ls_fig.update_xaxes(tickmode="array", tickvals=x, ticktext=ticktext)
        wf_deciles_contrib_ls_fig.update_layout(
            title="Signal: Mean Contribution by Weight Decile (Long and Short)",
            xaxis_title="Weight decile (active universe)",
            yaxis_title="Mean contribution to next return (%)",
            template="plotly_white",
            barmode="relative",
            legend_title="Series",
        )

    table_html = metrics.copy()
    table_html.index.name = "Metric"
    metrics_table = table_html.to_html(classes="metrics", escape=True)
    metrics_table = metrics_table.replace("\\n", "<br>")

    plots_html = []
    plots_html.append(
        pio.to_html(equity_fig, include_plotlyjs=True, full_html=False)
        + _note_block(
            "Portfolio cumulative return (%) over time, optionally compared to a benchmark.",
            "Upward sloping is good; compare vs benchmark for relative performance and watch for regime changes.",
        )
    )
    plots_html.append(
        pio.to_html(dd_fig, include_plotlyjs=False, full_html=False)
        + _note_block(
            "Percent drawdown from the running peak of the equity curve.",
            "More negative and longer-lasting drawdowns indicate higher risk; evaluate depth and recovery speed.",
        )
    )
    plots_html.append(
        pio.to_html(rolling_sharpe_fig, include_plotlyjs=False, full_html=False)
        + _note_block(
            f"Rolling {rolling_window}-period Sharpe ratio: rolling mean(excess return) / rolling std(excess return), annualized.",
            "Higher and stable is better; sustained negative values indicate persistent underperformance vs the risk-free rate.",
        )
    )
    plots_html.append(
        pio.to_html(dist_fig, include_plotlyjs=False, full_html=False)
        + _note_block(
            "Histogram of period returns (%).",
            "Skew and fat tails matter: a positive mean with a negative median suggests outlier-driven performance.",
        )
    )

    weights_section_html = ""
    if (
        wf_ic_fig is not None
        or wf_spread_fig is not None
        or wf_attrib_fig is not None
        or wf_deciles_fig is not None
        or wf_deciles_median_fig is not None
        or wf_deciles_sharpe_fig is not None
        or wf_deciles_contrib_fig is not None
        or wf_deciles_contrib_ls_fig is not None
        or wf_ic_dist_fig is not None
        or wf_spread_dist_fig is not None
    ):
        wf_plots = []
        if wf_ic_fig is not None:
            wf_plots.append(
                pio.to_html(wf_ic_fig, include_plotlyjs=False, full_html=False)
                + _note_block(
                    "Per-period cross-sectional correlation between weights and next-period returns (IC), with an optional rolling mean overlay.",
                    "Positive IC means higher weights tend to predict higher next returns; noisy IC suggests weak or unstable signal.",
                )
            )
        if wf_spread_fig is not None:
            wf_plots.append(
                pio.to_html(wf_spread_fig, include_plotlyjs=False, full_html=False)
                + _note_block(
                    "Per-period difference between mean next returns of the top weight decile and bottom weight decile.",
                    "A consistently positive spread indicates a monotonic signal; negative/flat spread suggests weak ranking power.",
                )
            )
        if wf_attrib_fig is not None:
            wf_plots.append(
                pio.to_html(wf_attrib_fig, include_plotlyjs=False, full_html=False)
                + _note_block(
                    "Decomposition of next-period portfolio return into selection vs directional components (per gross exposure).",
                    "Selection reflects cross-sectional ranking skill; directional reflects net bias. Large directional can imply unintended market exposure.",
                )
            )
        if wf_deciles_fig is not None:
            wf_plots.append(
                pio.to_html(wf_deciles_fig, include_plotlyjs=False, full_html=False)
                + _note_block(
                    "Mean next-period return of assets in each weight decile (bucketed each period by sorting non-zero weights).",
                    "Should be increasing with decile for a good long signal; does not account for portfolio sizing—use contribution charts for PnL impact.",
                )
            )
        if wf_deciles_median_fig is not None:
            wf_plots.append(
                pio.to_html(wf_deciles_median_fig, include_plotlyjs=False, full_html=False)
                + _note_block(
                    "Median next-period return of assets in each weight decile.",
                    "More robust than the mean; big gaps vs the mean indicate skew/outliers driving results.",
                )
            )
        if wf_deciles_sharpe_fig is not None:
            wf_plots.append(
                pio.to_html(wf_deciles_sharpe_fig, include_plotlyjs=False, full_html=False)
                + _note_block(
                    "Annualized Sharpe computed across all decile observations: mean(excess next return) / std(next return).",
                    "Higher is better risk-adjusted ranking; low `n` makes estimates noisy, and near-zero std can distort results.",
                )
            )
        if wf_deciles_contrib_fig is not None:
            wf_plots.append(
                pio.to_html(wf_deciles_contrib_fig, include_plotlyjs=False, full_html=False)
                + _note_block(
                    "Mean next-period portfolio contribution per decile: average of (weight × next return) across observations.",
                    "Directly answers which buckets help/hurt PnL; negative bars identify deciles where your sizing and direction are misaligned with next returns.",
                )
            )
        if wf_deciles_contrib_ls_fig is not None:
            wf_plots.append(
                pio.to_html(wf_deciles_contrib_ls_fig, include_plotlyjs=False, full_html=False)
                + _note_block(
                    "Same as contribution-by-decile, split into contributions from long weights and short weights within each decile.",
                    "Shows whether long/short legs are working; a negative short contribution means shorts tend to rise (hurting performance), and vice versa.",
                )
            )
        if wf_ic_dist_fig is not None:
            wf_plots.append(
                pio.to_html(wf_ic_dist_fig, include_plotlyjs=False, full_html=False)
                + _note_block(
                    "Histogram of per-period IC / Rank IC values.",
                    "A distribution shifted above 0 indicates consistent predictive power; wide distributions indicate instability.",
                )
            )
        if wf_spread_dist_fig is not None:
            wf_plots.append(
                pio.to_html(wf_spread_dist_fig, include_plotlyjs=False, full_html=False)
                + _note_block(
                    "Histogram of per-period top-minus-bottom decile spreads.",
                    "A distribution centered above 0 supports a useful ranking signal; frequent negative spreads indicate inversion or regime sensitivity.",
                )
            )

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
      table.metrics td:nth-child(4) {{ white-space: pre-line; }}
      .plot {{ margin-top: 12px; }}
      .note {{ margin-top: 8px; background-color: #ffffff; color: #444444; font-size: 0.95rem; line-height: 1.35; }}
      .note strong {{ background-color: #ffffff; color: #111111; font-weight: 600; }}
    </style>
  </head>
  <body>
    <h1>Tearsheet</h1>
    {metrics_table}
    <h2>Equity</h2>
    <div class="plot">{plots_html[0]}</div>
    <h2>Drawdown</h2>
    <div class="plot">{plots_html[1]}</div>
    <h2>Rolling Sharpe</h2>
    <div class="plot">{plots_html[2]}</div>
    <h2>Returns Distribution</h2>
    <div class="plot">{plots_html[3]}</div>
    {weights_section_html}
  </body>
</html>"""

    if output_path is not None:
        Path(output_path).write_text(html, encoding="utf-8")

    return html
