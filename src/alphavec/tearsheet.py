"""
Tearsheet rendering utilities for alphavec simulations.

Produces a single self-contained HTML document with static (non-JS) charts rendered via
Matplotlib/Seaborn and embedded as base64 PNGs.
"""

from __future__ import annotations

from base64 import b64encode
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from . import metrics as _metrics_mod

TEARSHEET_NOTES = _metrics_mod.TEARSHEET_NOTES

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .search import GridSearchResults
    from .sim import SimulationResult


def _fig_to_base64_png(fig: "Figure", *, dpi: int = 160) -> str:
    buf = BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
    )
    return b64encode(buf.getvalue()).decode("ascii")


def _note_block(definition: str, interpretation: str) -> str:
    return (
        '<div class="note">'
        f"<strong>Definition:</strong> {definition}<br>"
        f"<strong>Interpretation:</strong> {interpretation}"
        "</div>"
    )


def _plot_block(*, title: str, fig: "Figure", note: tuple[str, str] | None = None) -> str:
    import matplotlib.pyplot as plt

    try:
        png_b64 = _fig_to_base64_png(fig)
    finally:
        plt.close(fig)
    note_html = "" if note is None else _note_block(note[0], note[1])
    return (
        '<div class="plot">'
        f"<h3>{title}</h3>"
        f'<img alt="{title}" src="data:image/png;base64,{png_b64}"/>'
        f"{note_html}"
        "</div>"
    )


def tearsheet(
    *,
    sim_result: "SimulationResult | None" = None,
    grid_results: "GridSearchResults | None" = None,
    output_path: str | Path | None = None,
    smooth_periods: int = 0,
) -> str:
    """
    Render a self-contained HTML tearsheet (no JS) from metrics and returns.

    The report includes performance charts and (when available) signal diagnostics such as
    signal-vs-return scatters, decile return/contribution charts, and alpha decay by lag vs
    next-period returns.

    Args:
        sim_result: Optional `SimulationResult` from `simulate()`. If omitted and `grid_results` is
            provided, the tearsheet renders from `grid_results.best.result`.
        grid_results: Optional results from `grid_search()` to render heatmaps (and
            optionally supply `best` as the simulation source).
        output_path: Optional path to write the HTML.
        smooth_periods: Rolling window (in periods) to smooth all time-series charts.
            Default is 0 (no smoothing).

    Returns:
        The rendered HTML document as a string. If `output_path` is provided, the same HTML is also
        written to disk as UTF-8.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )

    if sim_result is not None:
        pass
    elif grid_results is not None and grid_results.best is not None:
        sim_result = grid_results.best.result
    else:
        raise TypeError("Missing required argument: `sim_result` (or pass `grid_results` with a `best`).")

    metrics = sim_result.metrics
    returns = sim_result.returns

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
    if isinstance(metrics.attrs.get("equity"), pd.Series):
        equity = metrics.attrs["equity"]
    else:
        equity = (1.0 + returns.fillna(0.0)).cumprod()
        equity.name = "equity"

    benchmark_equity = metrics.attrs.get("benchmark_equity")
    if not isinstance(benchmark_equity, pd.Series):
        benchmark_equity = None

    equity_pct = (equity / float(init_cash) - 1.0) * 100.0
    equity_pct = equity_pct.rename("Portfolio %")

    benchmark_equity_pct: pd.Series | None = None
    if benchmark_equity is not None and len(benchmark_equity) == len(equity):
        benchmark_equity_pct = (benchmark_equity / float(init_cash) - 1.0) * 100.0
        label = (
            f"Benchmark ({benchmark_asset}) %"
            if isinstance(benchmark_asset, str) and benchmark_asset
            else "Benchmark %"
        )
        benchmark_equity_pct = benchmark_equity_pct.rename(label)

    dd_pct = (equity / equity.cummax() - 1.0) * 100.0

    # Helper function for smoothing time series
    def _smooth(s: pd.Series, window: int) -> pd.Series:
        if window <= 1:
            return s
        min_periods = max(3, window // 5)
        return s.rolling(window=window, min_periods=min_periods).mean()

    # Use smooth_periods, with a minimum of 2 for rolling sharpe to be meaningful
    smooth_window = max(0, int(smooth_periods))
    rolling_sharpe_window = max(2, smooth_window) if smooth_window > 0 else 30

    freq_rule = str(_metric_value("Period frequency", "1D"))
    trading_days_year = int(_metric_value("Trading Days Year", 365))
    risk_free_rate = float(_metric_value("Risk Free Rate", 0.0))
    annual_factor = float(_metrics_mod._annualization_factor(freq_rule, trading_days_year))
    annual_factor = annual_factor if np.isfinite(annual_factor) and annual_factor > 0 else 1.0
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / annual_factor) - 1.0

    excess = returns.fillna(0.0) - rf_per_period
    min_periods = min(max(2, rolling_sharpe_window // 5), rolling_sharpe_window)
    roll_mean = excess.rolling(window=rolling_sharpe_window, min_periods=min_periods).mean()
    roll_std = excess.rolling(window=rolling_sharpe_window, min_periods=min_periods).std(ddof=1)
    rolling_sharpe = (roll_mean / roll_std) * np.sqrt(annual_factor)

    # Metrics table
    table_html = metrics.copy()
    table_html.index.name = "Metric"
    metrics_table = table_html.to_html(classes="metrics", escape=True)
    metrics_table = metrics_table.replace("\\n", "<br>")

    plots_html: list[str] = []

    # Equity
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity_pct.index, equity_pct.values, label=equity_pct.name)
    if benchmark_equity_pct is not None:
        ax.plot(benchmark_equity_pct.index, benchmark_equity_pct.values, label=benchmark_equity_pct.name)
    ax.set_title("Equity Curve (Cumulative Return %)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (%)")
    ax.legend(loc="best")
    plots_html.append(
        _plot_block(
            title="Equity Curve (Cumulative Return %)",
            fig=fig,
            note=(
                "Portfolio cumulative return (%) over time, optionally compared to a benchmark.",
                "Upward sloping is good; compare vs benchmark for relative performance and watch for regime changes.",
            ),
        )
    )

    # Drawdown
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(dd_pct.index, dd_pct.values, color="#d62728")
    ax.fill_between(dd_pct.index, dd_pct.values, 0.0, alpha=0.25, color="#d62728")
    ax.set_title("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    plots_html.append(
        _plot_block(
            title="Drawdown (%)",
            fig=fig,
            note=(
                "Percent drawdown from the running peak of the equity curve.",
                "More negative and longer-lasting drawdowns indicate higher risk; evaluate depth and recovery speed.",
            ),
        )
    )

    # Rolling Sharpe
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color="#1f77b4")
    ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1)
    ax.set_title(f"Rolling Sharpe ({rolling_sharpe_window} periods, annualized)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe (annualized)")
    plots_html.append(
        _plot_block(
            title=f"Rolling Sharpe ({rolling_sharpe_window} periods, annualized)",
            fig=fig,
            note=(
                f"Rolling {rolling_sharpe_window}-period Sharpe ratio (annualized), computed from per-period excess returns.",
                "Higher and stable is better; sustained negative values indicate persistent underperformance vs the risk-free rate.",
            ),
        )
    )

    # Return distribution with skew/kurtosis
    returns_pct = returns.fillna(0.0) * 100.0
    returns_clean = returns_pct[np.isfinite(returns_pct)]

    from scipy import stats
    skew_val = stats.skew(returns_clean) if len(returns_clean) > 2 else np.nan
    kurt_val = stats.kurtosis(returns_clean) if len(returns_clean) > 3 else np.nan

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.hist(returns_clean.values, bins=60, color="#1f77b4", alpha=0.8)
    ax.set_title(f"Returns Distribution (Per-Period, %) | Skew: {skew_val:.2f}, Kurtosis: {kurt_val:.2f}")
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Count")
    plots_html.append(
        _plot_block(
            title="Returns Distribution (Per-Period, %)",
            fig=fig,
            note=(
                "Histogram of period returns (%) with skewness and excess kurtosis. Skew > 0 indicates a right tail; kurtosis > 0 indicates fat tails.",
                "Positive skew with high kurtosis suggests occasional large wins; negative skew indicates crash risk.",
            ),
        )
    )

    # Q-Q plot
    fig, ax = plt.subplots(figsize=(10, 3.5))
    stats.probplot(returns_clean, dist="norm", plot=ax)
    ax.set_title("Returns Q-Q Plot (Normal, per-period)")
    ax.grid(True, alpha=0.3)
    plots_html.append(
        _plot_block(
            title="Returns Q-Q Plot",
            fig=fig,
            note=(
                "Quantile-quantile plot comparing return distribution to a normal distribution.",
                "Points on the red line indicate normality; deviations in tails show fat tails or skew, which affect risk management.",
            ),
        )
    )

    # Trading Activity & Costs section
    trading_blocks: list[str] = []

    # Transaction costs
    transaction_costs = metrics.attrs.get("transaction_costs")
    if isinstance(transaction_costs, pd.Series) and len(transaction_costs) > 0:
        costs_pct = transaction_costs * 100.0
        costs_plot = _smooth(costs_pct, smooth_window)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(costs_plot.index, costs_plot.values, color="#d62728", linewidth=2)
        title = (
            f"Transaction Costs (% of Portfolio) - {smooth_window}p smooth"
            if smooth_window > 0
            else "Transaction Costs (% of Portfolio)"
        )
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Transaction costs (% of portfolio)")
        trading_blocks.append(
            _plot_block(
                title="Transaction Costs (% of Portfolio)",
                fig=fig,
                note=(
                    "Transaction costs per period as a percentage of portfolio value.",
                    "Cumulative transaction costs directly reduce net returns; compare to gross returns to assess impact on alpha.",
                ),
            )
        )

    # Number of positions
    n_positions = metrics.attrs.get("n_positions")
    if isinstance(n_positions, pd.Series) and len(n_positions) > 0:
        n_positions_plot = _smooth(n_positions, smooth_window)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(n_positions_plot.index, n_positions_plot.values, color="#9467bd", linewidth=2)
        title = (
            f"Active Positions (Count) - {smooth_window}p smooth"
            if smooth_window > 0
            else "Active Positions (Count)"
        )
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Count")
        trading_blocks.append(
            _plot_block(
                title="Active Positions (Count)",
                fig=fig,
                note=(
                    "Number of positions with non-zero weight over time.",
                    "Declining position counts may indicate increasing concentration or signal degradation; stable counts suggest consistent strategy execution.",
                ),
            )
        )

    trading_section_html = ""
    if len(trading_blocks) > 0:
        trading_section_html = "<h2>Trading Activity &amp; Costs</h2>" + "".join(trading_blocks)

    # Exposure & Risk Management section
    exposure_blocks: list[str] = []

    # Net and Gross Exposure
    net_exposure = metrics.attrs.get("net_exposure")
    gross_exposure = metrics.attrs.get("gross_exposure")
    if (isinstance(net_exposure, pd.Series) and len(net_exposure) > 0) or (
        isinstance(gross_exposure, pd.Series) and len(gross_exposure) > 0
    ):
        fig, ax = plt.subplots(figsize=(10, 3.5))
        if isinstance(gross_exposure, pd.Series) and len(gross_exposure) > 0:
            gross_pct = gross_exposure * 100.0
            gross_plot = _smooth(gross_pct, smooth_window)
            ax.plot(gross_plot.index, gross_plot.values, label="Gross Exposure", linewidth=2, color="#1f77b4")
        if isinstance(net_exposure, pd.Series) and len(net_exposure) > 0:
            net_pct = net_exposure * 100.0
            net_plot = _smooth(net_pct, smooth_window)
            ax.plot(net_plot.index, net_plot.values, label="Net Exposure", linewidth=2, color="#ff7f0e")
        ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1)
        title = (
            f"Net and Gross Exposure (% of Portfolio) - {smooth_window}p smooth"
            if smooth_window > 0
            else "Net and Gross Exposure (% of Portfolio)"
        )
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Exposure (% of portfolio)")
        ax.legend(loc="best")
        exposure_blocks.append(
            _plot_block(
                title="Net and Gross Exposure (% of Portfolio)",
                fig=fig,
                note=(
                    "Net exposure (long − short) and gross exposure (|long| + |short|) as % of portfolio value.",
                    "Net shows directional bias; gross shows leverage. Stable exposure suggests consistent positioning; shifts may indicate regime changes.",
                ),
            )
        )

    # Concentration (Herfindahl index if available)
    concentration = metrics.attrs.get("concentration")
    if isinstance(concentration, pd.Series) and len(concentration) > 0:
        concentration_plot = _smooth(concentration, smooth_window)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(concentration_plot.index, concentration_plot.values, color="#8c564b", linewidth=2)
        title = (
            f"Position Concentration (Herfindahl Index) - {smooth_window}p smooth"
            if smooth_window > 0
            else "Position Concentration (Herfindahl Index)"
        )
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Herfindahl index")
        exposure_blocks.append(
            _plot_block(
                title="Position Concentration (Herfindahl Index)",
                fig=fig,
                note=(
                    "Herfindahl index of position weights (sum of squared weights); ranges from 1/N (equal-weighted) to 1 (single position).",
                    "Higher values indicate concentrated portfolios with idiosyncratic risk; lower values suggest better diversification.",
                ),
            )
        )

    exposure_section_html = ""
    if len(exposure_blocks) > 0:
        exposure_section_html = "<h2>Exposure &amp; Risk Management</h2>" + "".join(exposure_blocks)

    # Signal diagnostics
    signal_blocks: list[str] = []

    wf = metrics.attrs.get("weight_forward")
    if isinstance(wf, pd.DataFrame) and len(wf.index) > 0:
        if ("directionality" in wf.columns) and ("forward_return_per_gross" in wf.columns):
            x = pd.to_numeric(wf["directionality"], errors="coerce")
            y = pd.to_numeric(wf["forward_return_per_gross"], errors="coerce") * 100.0
            mask = np.isfinite(x) & np.isfinite(y)
            if int(mask.sum()) > 2:
                x_vals = x[mask].to_numpy(dtype=float)
                y_vals = y[mask].to_numpy(dtype=float)
                fig, ax = plt.subplots(figsize=(10, 3.5))
                ax.scatter(x_vals, y_vals, s=18, alpha=0.6, color="#1f77b4")
                if np.unique(x_vals).size > 1:
                    slope, intercept = np.polyfit(x_vals, y_vals, 1)
                    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                    ax.plot(
                        x_line,
                        slope * x_line + intercept,
                        color="#d62728",
                        linewidth=2,
                        label=f"Fit: y = {slope:.3f}x + {intercept:.3f}",
                    )
                    ax.legend(loc="best")
                ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1)
                ax.set_title("Signal Directionality vs Next-Period Return (Per Gross)")
                ax.set_xlabel("Directionality (net / gross)")
                ax.set_ylabel("Next-period return per gross (%)")
                signal_blocks.append(
                    _plot_block(
                        title="Signal Directionality vs Next-Period Return (Per Gross)",
                        fig=fig,
                        note=(
                            "Scatter of per-period directionality (net/gross) versus next-period return per gross (%).",
                            "An upward slope suggests directional tilt aligns with next-period returns; a flat fit suggests limited directional edge.",
                        ),
                    )
                )

        ic_col = None
        ic_label = ""
        if "ic" in wf.columns:
            ic_col = "ic"
            ic_label = "IC"
        elif "rank_ic" in wf.columns:
            ic_col = "rank_ic"
            ic_label = "Rank IC"
        if ic_col is not None and "forward_return_per_gross" in wf.columns:
            x = pd.to_numeric(wf[ic_col], errors="coerce")
            y = pd.to_numeric(wf["forward_return_per_gross"], errors="coerce") * 100.0
            mask = np.isfinite(x) & np.isfinite(y)
            if int(mask.sum()) > 2:
                x_vals = x[mask].to_numpy(dtype=float)
                y_vals = y[mask].to_numpy(dtype=float)
                fig, ax = plt.subplots(figsize=(10, 3.5))
                ax.scatter(x_vals, y_vals, s=18, alpha=0.6, color="#1f77b4")
                if np.unique(x_vals).size > 1:
                    slope, intercept = np.polyfit(x_vals, y_vals, 1)
                    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                    ax.plot(
                        x_line,
                        slope * x_line + intercept,
                        color="#d62728",
                        linewidth=2,
                        label=f"Fit: y = {slope:.3f}x + {intercept:.3f}",
                    )
                    ax.legend(loc="best")
                ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1)
                ax.set_title(f"Signal {ic_label} vs Next-Period Return (Per Gross)")
                ax.set_xlabel(f"Signal {ic_label}")
                ax.set_ylabel("Next-period return per gross (%)")
                signal_blocks.append(
                    _plot_block(
                        title=f"Signal {ic_label} vs Next-Period Return (Per Gross)",
                        fig=fig,
                        note=(
                            f"Scatter of per-period {ic_label} versus next-period return per gross (%).",
                            "Useful for market-neutral strategies where directionality is near zero; higher IC should align with higher returns.",
                        ),
                    )
                )

    # Alpha decay by lag (next-period return types)
    decay = metrics.attrs.get("alpha_decay_next_return_by_type")
    if isinstance(decay, pd.DataFrame) and len(decay.index) > 0:
        decay_num = decay.apply(pd.to_numeric, errors="coerce")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        x = decay_num.index.to_numpy(dtype=int)

        series_specs = [
            ("total_per_gross_mean", "Total"),
            ("selection_per_gross_mean", "Selection"),
            ("directional_per_gross_mean", "Directional"),
        ]
        for col, label in series_specs:
            if col not in decay_num.columns:
                continue
            y = (decay_num[col] * 100.0).to_numpy(dtype=float)
            ax.plot(x, y, marker="o", linewidth=2, label=label)

        ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1)
        ax.set_title("Signal: Alpha Decay by Lag (Next-Period Return per Gross)")
        ax.set_xlabel("Lag (periods)")
        ax.set_ylabel("Mean next-period return per gross (%)")
        ax.legend(loc="best")
        signal_blocks.append(
            _plot_block(
                title="Signal: Alpha Decay by Lag (Next-Period Return per Gross)",
                fig=fig,
                note=(
                    "Mean next-period return per unit gross weight when acting on the signal with a delay (lag), split into total / selection / directional components.",
                    "Faster decay implies a shorter-lived edge; selection decay suggests cross-sectional ranking stability issues, while directional decay points to net bias timing.",
                ),
            )
        )

    # Deciles
    wf_deciles = metrics.attrs.get("weight_forward_deciles")
    wf_deciles_median = metrics.attrs.get("weight_forward_deciles_median")
    wf_deciles_std = metrics.attrs.get("weight_forward_deciles_std")
    wf_deciles_contrib_long = metrics.attrs.get("weight_forward_deciles_contrib_long")
    wf_deciles_contrib_short = metrics.attrs.get("weight_forward_deciles_contrib_short")
    def _decile_ticks(base_index: pd.Index) -> tuple[np.ndarray, list[str]]:
        idx = base_index.astype(int)
        labels = [str(i) for i in idx]
        return idx.to_numpy(), labels
    mean_deciles = wf_deciles if isinstance(wf_deciles, pd.Series) and len(wf_deciles.index) > 0 else None
    median_deciles = (
        wf_deciles_median
        if isinstance(wf_deciles_median, pd.Series) and len(wf_deciles_median.index) > 0
        else None
    )
    if mean_deciles is not None or median_deciles is not None:
        base = mean_deciles if mean_deciles is not None else median_deciles
        x, labels = _decile_ticks(base.index)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        if mean_deciles is not None:
            mean_y = (mean_deciles.reindex(base.index) * 100.0).to_numpy(dtype=float)
            ax.bar(x, mean_y, color="#1f77b4", alpha=0.85, label="Mean")
        if median_deciles is not None:
            median_y = (median_deciles.reindex(base.index) * 100.0).to_numpy(dtype=float)
            ax.scatter(x, median_y, color="#ff7f0e", s=40, label="Median", zorder=3)
        ax.set_title("Signal: Next-Period Return by Weight Decile (Mean/Median)")
        ax.set_xlabel("Weight decile (active universe)")
        ax.set_ylabel("Next-period return (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc="best")
        signal_blocks.append(
            _plot_block(
                title="Signal: Next-Period Return by Weight Decile (Mean/Median)",
                fig=fig,
                note=(
                    "Mean (bars) and median (dots) next-period return (%) by weight decile, computed over the active universe (non-zero weights).",
                    "Monotonic separation suggests signal strength; mean >> median implies outlier-driven performance.",
                ),
            )
        )

    if (
        isinstance(wf_deciles_contrib_long, pd.Series)
        and isinstance(wf_deciles_contrib_short, pd.Series)
        and len(wf_deciles_contrib_long.index) > 0
        and len(wf_deciles_contrib_short.index) > 0
    ):
        long_c = wf_deciles_contrib_long
        short_c = wf_deciles_contrib_short.reindex(long_c.index)
        x, labels = _decile_ticks(long_c.index)
        long_y = (long_c * 100.0).to_numpy(dtype=float)
        short_y = (short_c * 100.0).to_numpy(dtype=float)
        width = 0.36
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.bar(x - width / 2, long_y, width=width, label="Long contribution", color="#1f77b4", alpha=0.9)
        ax.bar(x + width / 2, short_y, width=width, label="Short contribution", color="#ff7f0e", alpha=0.9)
        ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1)
        ax.set_title("Signal: Next-Period Return Contribution by Weight Decile (Long/Short)")
        ax.set_xlabel("Weight decile (active universe)")
        ax.set_ylabel("Mean next-period return contribution (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc="best")
        signal_blocks.append(
            _plot_block(
                title="Signal: Next-Period Return Contribution by Weight Decile (Long/Short)",
                fig=fig,
                note=(
                    "Mean next-period return contribution (%) by decile, split into long vs short sides.",
                    "Highlights which side drives PnL; large asymmetry can indicate a side-specific edge or unintended bias.",
                ),
            )
        )

    signal_section_html = ""
    if len(signal_blocks) > 0:
        signal_section_html = "<h2>Signal</h2>" + "".join(signal_blocks)

    # Parameter search heatmaps
    search_section_html = ""
    if grid_results is not None and len(grid_results.param_grids) > 0:
        heatmaps: list[str] = []
        for i, grid in enumerate(grid_results.param_grids):
            z = grid_results.pivot(grid_index=i)
            z_num = z.apply(pd.to_numeric, errors="coerce")
            fig, ax = plt.subplots(
                figsize=(max(6.0, 0.6 * z_num.shape[1] + 2.0), max(4.0, 0.55 * z_num.shape[0] + 2.0))
            )
            sns.heatmap(
                z_num,
                ax=ax,
                cmap="RdBu_r",
                center=0.0,
                cbar_kws={"label": grid_results.objective_metric},
            )
            ax.set_title(f"{grid_results.objective_metric} heatmap ({grid.label()})")
            ax.set_xlabel(grid.param2_name)
            ax.set_ylabel(grid.param1_name)
            fig.tight_layout()
            heatmaps.append(
                _plot_block(
                    title=f"{grid_results.objective_metric} heatmap ({grid.label()})",
                    fig=fig,
                    note=(
                        f"Objective value ({grid_results.objective_metric}) across the 2D parameter grid.",
                        "Look for stable regions (broad plateaus) rather than isolated spikes to reduce overfitting risk.",
                    ),
                )
            )
        search_section_html = "<h2>Parameter Search</h2>" + "".join(heatmaps)

    plots_section = (
        "<h2>Performance</h2>"
        + "".join(plots_html)
        + trading_section_html
        + exposure_section_html
        + signal_section_html
        + search_section_html
    )

    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Tearsheet</title>
    <style>
      .alphavec-tearsheet {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; padding: 24px; background: #ffffff; color: #000000; }}
      .alphavec-tearsheet h1 {{ margin: 0 0 16px 0; }}
      .alphavec-tearsheet h2 {{ margin: 28px 0 12px 0; }}
      .alphavec-tearsheet h3 {{ margin: 18px 0 10px 0; font-size: 1.05rem; }}
      .alphavec-tearsheet table.metrics {{ border-collapse: collapse; width: 100%; background-color: #ffffff; color: #000000; }}
      .alphavec-tearsheet table.metrics th, .alphavec-tearsheet table.metrics td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
      .alphavec-tearsheet table.metrics th {{ background: #f6f6f6; text-align: left; }}
      .alphavec-tearsheet table.metrics tr, .alphavec-tearsheet table.metrics th, .alphavec-tearsheet table.metrics td {{ text-align: left !important; }}
      .alphavec-tearsheet table.metrics td:nth-child(4) {{ white-space: pre-line; }}
      .alphavec-tearsheet .plot {{ margin-top: 12px; }}
      .alphavec-tearsheet .plot img {{ max-width: 100%; height: auto; border: 1px solid #eee; background: #fff; }}
      .alphavec-tearsheet .note {{ margin-top: 8px; background: #ffffff; color: #444444; font-size: 0.95rem; line-height: 1.35; }}
      .alphavec-tearsheet .note strong {{ color: #111111; font-weight: 600; }}
    </style>
  </head>
  <body>
    <div class="alphavec-tearsheet">
      <h1>Tearsheet</h1>
      {metrics_table}
      {plots_section}
    </div>
  </body>
</html>
"""

    if output_path is not None:
        Path(output_path).write_text(html, encoding="utf-8")

    return html
