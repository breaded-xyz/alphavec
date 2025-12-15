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
    signal_smooth_window: int = 30,
    rolling_sharpe_window: int = 30,
) -> str:
    """
    Render a self-contained HTML tearsheet (no JS) from metrics and returns.

    Args:
        sim_result: Optional `SimulationResult` from `simulate()`. If omitted and `grid_results` is
            provided, the tearsheet renders from `grid_results.best.result`.
        grid_results: Optional results from `grid_search()` to render heatmaps (and
            optionally supply `best` as the simulation source).
        output_path: Optional path to write the HTML.
        signal_smooth_window: Rolling window (in periods) used to smooth Signal time-series plots.
        rolling_sharpe_window: Rolling window (in periods) used to compute Rolling Sharpe.

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
    ax.plot(dd_pct.index, dd_pct.values, color="#1f77b4")
    ax.fill_between(dd_pct.index, dd_pct.values, 0.0, alpha=0.25, color="#1f77b4")
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
    ax.set_title(f"Rolling Sharpe ({rolling_window} periods)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe")
    plots_html.append(
        _plot_block(
            title=f"Rolling Sharpe ({rolling_window} periods)",
            fig=fig,
            note=(
                f"Rolling {rolling_window}-period Sharpe ratio (annualized), computed from per-period excess returns.",
                "Higher and stable is better; sustained negative values indicate persistent underperformance vs the risk-free rate.",
            ),
        )
    )

    # Return distribution
    returns_pct = returns.fillna(0.0) * 100.0
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.hist(returns_pct.values, bins=60, color="#1f77b4", alpha=0.8)
    ax.set_title("Returns Distribution (%)")
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Count")
    plots_html.append(
        _plot_block(
            title="Returns Distribution (%)",
            fig=fig,
            note=(
                "Histogram of period returns (%).",
                "Skew and fat tails matter: a positive mean with a negative median suggests outlier-driven performance.",
            ),
        )
    )

    # Signal diagnostics
    def _roll_mean(s: pd.Series, window: int) -> pd.Series:
        if window <= 1:
            return s
        min_periods = max(3, window // 5)
        return s.rolling(window=window, min_periods=min_periods).mean()

    try:
        smooth_window = max(1, int(signal_smooth_window))
    except Exception:
        smooth_window = 30

    signal_blocks: list[str] = []

    wf = metrics.attrs.get("weight_forward")
    if isinstance(wf, pd.DataFrame) and len(wf.index) > 0:
        # IC / Rank IC
        if ("ic" in wf.columns) or ("rank_ic" in wf.columns):
            fig, ax = plt.subplots(figsize=(10, 3.5))
            if "ic" in wf.columns:
                ic = pd.to_numeric(wf["ic"], errors="coerce")
                if smooth_window > 1:
                    ax.plot(ic.index, ic.values, label="IC (raw)", linewidth=1, alpha=0.25)
                ax.plot(
                    ic.index,
                    _roll_mean(ic, smooth_window).values,
                    label=f"IC ({smooth_window}p mean)",
                    linewidth=2,
                )
            if "rank_ic" in wf.columns:
                ric = pd.to_numeric(wf["rank_ic"], errors="coerce")
                if smooth_window > 1:
                    ax.plot(ric.index, ric.values, label="Rank IC (raw)", linewidth=1, alpha=0.25)
                ax.plot(
                    ric.index,
                    _roll_mean(ric, smooth_window).values,
                    label=f"Rank IC ({smooth_window}p mean)",
                    linewidth=2,
                )
            ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1)
            ax.set_title("Signal: IC (Smoothed)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Correlation")
            ax.legend(loc="best")
            signal_blocks.append(
                _plot_block(
                    title="Signal: IC (Smoothed)",
                    fig=fig,
                    note=(
                        "Per-period cross-sectional correlation between weights at t and next-period returns (IC), with a rolling mean overlay.",
                        "Sustained positive values suggest predictive alignment; noisy or near-zero values suggest a weak or unstable signal.",
                    ),
                )
            )

        # Top-bottom spread
        if "top_bottom_spread" in wf.columns:
            spread = pd.to_numeric(wf["top_bottom_spread"], errors="coerce") * 100.0
            fig, ax = plt.subplots(figsize=(10, 3.5))
            if smooth_window > 1:
                ax.plot(spread.index, spread.values, label="Spread (raw)", linewidth=1, alpha=0.25)
            ax.plot(
                spread.index,
                _roll_mean(spread, smooth_window).values,
                label=f"Spread ({smooth_window}p mean)",
                linewidth=2,
            )
            ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1)
            ax.set_title("Signal: Top-Bottom Decile Spread (%)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Spread (%)")
            ax.legend(loc="best")
            signal_blocks.append(
                _plot_block(
                    title="Signal: Top-Bottom Decile Spread (%)",
                    fig=fig,
                    note=(
                        "Next-period return spread (%) between the top and bottom weight deciles each period, with a rolling mean overlay.",
                        "Positive and stable spread indicates separation between high- and low-weighted assets; instability often indicates regime dependence.",
                    ),
                )
            )

        # Selection vs directional (per gross)
        sel_col = "forward_return_selection_per_gross"
        dir_col = "forward_return_directional_per_gross"
        if (sel_col in wf.columns) or (dir_col in wf.columns):
            fig, ax = plt.subplots(figsize=(10, 3.5))
            if sel_col in wf.columns:
                sel = pd.to_numeric(wf[sel_col], errors="coerce") * 100.0
                ax.plot(
                    sel.index,
                    _roll_mean(sel, smooth_window).values,
                    label=f"Selection ({smooth_window}p mean)",
                    linewidth=2,
                )
            if dir_col in wf.columns:
                dirn = pd.to_numeric(wf[dir_col], errors="coerce") * 100.0
                ax.plot(
                    dirn.index,
                    _roll_mean(dirn, smooth_window).values,
                    label=f"Directional ({smooth_window}p mean)",
                    linewidth=2,
                )
            ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1)
            ax.set_title("Signal: Attribution (Per Gross, %)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Contribution (%)")
            ax.legend(loc="best")
            signal_blocks.append(
                _plot_block(
                    title="Signal: Attribution (Per Gross, %)",
                    fig=fig,
                    note=(
                        "Rolling decomposition of forward return per unit gross weight into selection vs directional components.",
                        "Selection reflects cross-sectional picking skill; directional reflects net bias interacting with the average universe return.",
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
        ax.set_title("Signal: Alpha Decay by Lag (Next Return, Per Gross)")
        ax.set_xlabel("Lag (periods)")
        ax.set_ylabel("Mean next return per gross (%)")
        ax.legend(loc="best")
        signal_blocks.append(
            _plot_block(
                title="Signal: Alpha Decay by Lag (Next Return, Per Gross)",
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
    wf_deciles_count = metrics.attrs.get("weight_forward_deciles_count")
    wf_deciles_contrib = metrics.attrs.get("weight_forward_deciles_contrib")
    wf_deciles_contrib_long = metrics.attrs.get("weight_forward_deciles_contrib_long")
    wf_deciles_contrib_short = metrics.attrs.get("weight_forward_deciles_contrib_short")
    if isinstance(wf_deciles, pd.Series) and len(wf_deciles.index) > 0:
        counts = None
        if isinstance(wf_deciles_count, pd.Series) and len(wf_deciles_count.index) > 0:
            counts = wf_deciles_count.reindex(wf_deciles.index)
        x = wf_deciles.index.astype(int).to_numpy()
        y = (wf_deciles * 100.0).to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.bar(x, y, color="#1f77b4", alpha=0.9)
        ax.set_title("Signal: Mean Next Return by Weight Decile")
        ax.set_xlabel("Weight decile (active universe)")
        ax.set_ylabel("Mean next return (%)")
        if counts is not None:
            ax.set_xticks(x, [f"{d}\n(n={int(n)})" for d, n in zip(x, counts.fillna(0).astype(int))])
        signal_blocks.append(
            _plot_block(
                title="Signal: Mean Next Return by Weight Decile",
                fig=fig,
                note=(
                    "Mean next-period return (%) by weight decile, computed over the active universe (non-zero weights).",
                    "A healthy signal typically shows monotonic separation (top deciles outperform bottom), and reasonable per-decile sample sizes (n).",
                ),
            )
        )

    if isinstance(wf_deciles_median, pd.Series) and len(wf_deciles_median.index) > 0:
        counts = None
        if isinstance(wf_deciles_count, pd.Series) and len(wf_deciles_count.index) > 0:
            counts = wf_deciles_count.reindex(wf_deciles_median.index)
        x = wf_deciles_median.index.astype(int).to_numpy()
        y = (wf_deciles_median * 100.0).to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.bar(x, y, color="#1f77b4", alpha=0.9)
        ax.set_title("Signal: Median Next Return by Weight Decile")
        ax.set_xlabel("Weight decile (active universe)")
        ax.set_ylabel("Median next return (%)")
        if counts is not None:
            ax.set_xticks(x, [f"{d}\n(n={int(n)})" for d, n in zip(x, counts.fillna(0).astype(int))])
        signal_blocks.append(
            _plot_block(
                title="Signal: Median Next Return by Weight Decile",
                fig=fig,
                note=(
                    "Median next-period return (%) by weight decile, computed over the active universe (non-zero weights).",
                    "Less sensitive to outliers than the mean; if mean is strong but median is weak, performance may be outlier-driven.",
                ),
            )
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
        sharpe = sharpe.replace([np.inf, -np.inf], np.nan)
        counts = None
        if isinstance(wf_deciles_count, pd.Series) and len(wf_deciles_count.index) > 0:
            counts = wf_deciles_count.reindex(sharpe.index)
        x = sharpe.index.astype(int).to_numpy()
        y = sharpe.to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.bar(x, y, color="#1f77b4", alpha=0.9)
        ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1)
        ax.set_title("Signal: Sharpe by Weight Decile (Annualized)")
        ax.set_xlabel("Weight decile (active universe)")
        ax.set_ylabel("Sharpe (annualized)")
        if counts is not None:
            ax.set_xticks(x, [f"{d}\n(n={int(n)})" for d, n in zip(x, counts.fillna(0).astype(int))])
        signal_blocks.append(
            _plot_block(
                title="Signal: Sharpe by Weight Decile (Annualized)",
                fig=fig,
                note=(
                    "Annualized Sharpe by decile: mean(next return − risk-free per period) / std(next return), annualized.",
                    "Higher is better; instability or extreme values often indicate small n or a very noisy decile return series.",
                ),
            )
        )

    if isinstance(wf_deciles_contrib, pd.Series) and len(wf_deciles_contrib.index) > 0:
        counts = None
        if isinstance(wf_deciles_count, pd.Series) and len(wf_deciles_count.index) > 0:
            counts = wf_deciles_count.reindex(wf_deciles_contrib.index)
        x = wf_deciles_contrib.index.astype(int).to_numpy()
        y = (wf_deciles_contrib * 100.0).to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.bar(x, y, color="#1f77b4", alpha=0.9)
        ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1)
        ax.set_title("Signal: Mean Next Return Contribution by Weight Decile")
        ax.set_xlabel("Weight decile (active universe)")
        ax.set_ylabel("Mean contribution to next return (%)")
        if counts is not None:
            ax.set_xticks(x, [f"{d}\n(n={int(n)})" for d, n in zip(x, counts.fillna(0).astype(int))])
        signal_blocks.append(
            _plot_block(
                title="Signal: Mean Next Return Contribution by Weight Decile",
                fig=fig,
                note=(
                    "Mean contribution to next-period portfolio return (%) by weight decile.",
                    "Shows which deciles drive portfolio PnL; ideally contribution aligns with intended signal shape (e.g., top deciles contribute positively).",
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
        counts = None
        if isinstance(wf_deciles_count, pd.Series) and len(wf_deciles_count.index) > 0:
            counts = wf_deciles_count.reindex(long_c.index)
        x = long_c.index.astype(int).to_numpy()
        long_y = (long_c * 100.0).to_numpy(dtype=float)
        short_y = (short_c * 100.0).to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.bar(x, long_y, label="Long contribution (%)", color="#1f77b4", alpha=0.9)
        ax.bar(x, short_y, bottom=long_y, label="Short contribution (%)", color="#ff7f0e", alpha=0.9)
        ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1)
        ax.set_title("Signal: Mean Contribution by Weight Decile (Long and Short)")
        ax.set_xlabel("Weight decile (active universe)")
        ax.set_ylabel("Mean contribution to next return (%)")
        ax.legend(loc="best")
        if counts is not None:
            ax.set_xticks(x, [f"{d}\n(n={int(n)})" for d, n in zip(x, counts.fillna(0).astype(int))])
        signal_blocks.append(
            _plot_block(
                title="Signal: Mean Contribution by Weight Decile (Long and Short)",
                fig=fig,
                note=(
                    "Mean next-period return contribution (%) by decile, split into long vs short sides.",
                    "Helps distinguish selection vs directional effects; large asymmetry can indicate a side-specific edge or unintended bias.",
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

    plots_section = "<h2>Performance</h2>" + "".join(plots_html) + signal_section_html + search_section_html

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
