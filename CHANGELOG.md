# Changelog

All notable changes to alphavec will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-12-14

### BREAKING CHANGES

#### Statistics Calculations Now Use Sample Statistics (ddof=1)

All standard deviation and variance calculations now use **sample statistics** (Bessel's correction, `ddof=1`) instead of population statistics (`ddof=0`), aligning alphavec with industry standards and academic literature.

**Why this change?**
Backtests represent samples from a larger population of possible market outcomes. Sample statistics provide unbiased estimators and are the standard in:
- Academic finance literature (Sharpe 1994, Lo 2002)
- Industry-standard Python libraries (quantstats, empyrical, pyfolio, ffn)
- Professional portfolio analytics platforms

**Affected Metrics:**
- **Annualized volatility**: ~0.2% higher for typical backtest lengths (250+ periods)
- **Annualized Sharpe**: ~0.2% lower (due to higher volatility denominator)
- **Tracking error**: ~0.2% higher
- **Information ratio**: ~0.2% lower (due to higher tracking error)
- **Beta**: More accurate (uses sample variance and covariance)
- **Alpha**: Slightly different (depends on beta calculation)

**Impact:**
- For a backtest with 252 periods, volatility increases by approximately sqrt(252/251) ≈ 1.002x
- Longer backtests have proportionally smaller changes
- These changes make alphavec metrics directly comparable to industry benchmarks

**Migration:**
If you need to compare v0.2.0 metrics with v0.1.x metrics, expect small differences (~0.2% for metrics based on standard deviation). The new values are statistically more accurate and industry-standard.

### Added

#### New Risk Metrics

Added 7 additional industry-standard risk metrics in a new "Risk" category:

1. **Annualized Sortino Ratio**
   - Excess return per unit of downside risk
   - More appropriate than Sharpe for asymmetric return distributions
   - Only penalizes downside volatility, not upside

2. **Downside Deviation**
   - Standard deviation of negative returns (annualized)
   - Used in Sortino ratio calculation
   - Focuses on harmful volatility only

3. **VaR 95% (Value at Risk)**
   - 5th percentile of returns distribution
   - Worst expected single-period loss in 19 out of 20 periods
   - Standard risk measure in institutional finance

4. **CVaR 95% (Conditional Value at Risk)**
   - Mean return when VaR is exceeded
   - Also known as Expected Shortfall
   - Measures average loss in worst 5% of cases

5. **Omega Ratio**
   - Probability-weighted ratio of gains vs losses (threshold = 0)
   - Values >1 indicate gains outweigh losses
   - Captures all moments of the return distribution

6. **Gain-to-Pain Ratio**
   - Total return divided by sum of absolute returns
   - Measures return efficiency relative to total volatility
   - Simpler alternative to Sharpe/Sortino

7. **Ulcer Index**
   - Root mean square of drawdowns (annualized)
   - Alternative to max drawdown that accounts for duration
   - Penalizes both depth and duration of drawdowns

#### Category Column in Metrics DataFrame

All metrics now include a `Category` column for better organization:
- **Meta**: Simulation metadata (dates, frequency, benchmark)
- **Performance**: Return, volatility, Sharpe, drawdowns
- **Costs & Trading**: Fees, funding, turnover, order counts
- **Exposure**: Gross/net exposure metrics
- **Benchmark**: Alpha, beta, tracking error, information ratio
- **Distribution**: Skewness, kurtosis, win/loss statistics
- **Portfolio**: Holding periods, weights, cost ratios
- **Risk**: New downside and tail risk metrics

### Improved

#### Documentation Enhancements

- **Clarified statistical methodology**: All metric notes now explicitly state when sample statistics (ddof=1) are used
- **Kurtosis**: Now explicitly labeled as "excess kurtosis" (normal = 0) rather than ambiguous "kurtosis"
- **Active Return**: Clarified that arithmetic mean (not geometric) is used to match tracking error calculation
- **Beta/Alpha**: Updated to note use of sample covariance and variance per CAPM standards

#### More Accurate CAPM Calculations

- Beta calculation now uses `np.cov(x, y, ddof=1)` for sample covariance instead of manual population covariance
- More consistent with academic CAPM methodology
- Aligns with professional financial analysis tools

### Technical Details

#### Statistical Methodology

Alphavec now follows the industry-standard approach:

```python
# Sample statistics (ddof=1) - unbiased estimators
volatility = returns.std(ddof=1) * sqrt(annual_factor)
covariance = np.cov(x, y, ddof=1)
variance = np.var(x, ddof=1)
```

**Rationale:**
- Backtests are finite samples, not complete populations
- Sample variance: E[s²] = σ² (unbiased)
- Population variance: E[s²] = ((n-1)/n)σ² (biased, underestimates true variance)
- Division by (n-1) instead of n provides Bessel's correction

**When to use population statistics:**
- Only when analyzing a complete, finite population
- Not appropriate for time series backtests

#### Total Metrics

- v0.1.x: 46 metrics across 6 categories
- v0.2.0: 53 metrics across 8 categories (+7 risk metrics, +1 Category column)

---

## [0.1.3] - 2024-XX-XX

### Changed
- Updated Python version requirement in README
- Adjusted Python requirements in pyproject.toml
- Removed pyarrow from production requirements

### Fixed
- Typos and improved README content

---

## [0.1.2] - 2024-XX-XX

### Fixed
- Various bug fixes and improvements

---

## Earlier Versions

See git history for changes in versions 0.1.1 and earlier.

---

## Upgrade Guide

### From 0.1.x to 0.2.0

**Q: Will my code break?**
A: No, the API is unchanged. Your code will run without modifications.

**Q: Will my metric values change?**
A: Yes, slightly. Metrics based on standard deviation will change by ~0.2% for typical backtest lengths (250+ periods). Shorter backtests will have larger percentage changes.

**Q: Should I recalculate my strategies?**
A: No need to recalculate. The new values are more statistically accurate. If you're comparing strategies, use consistent versions of alphavec.

**Q: Can I get the old behavior?**
A: No configuration option is provided. The old behavior was statistically incorrect for backtesting applications. If you need exact reproduction of v0.1.x values for verification purposes, pin to `alphavec==0.1.3`.

**Q: Which new metrics should I use?**
A:
- **Sortino Ratio**: If your returns are asymmetric (more upside or downside)
- **VaR/CVaR**: If you need regulatory compliance or tail risk measures
- **Omega Ratio**: If you want a comprehensive risk measure
- **Ulcer Index**: If you care about drawdown duration, not just depth

**Q: Will the Category column break my code?**
A: Only if you iterate over DataFrame columns expecting exactly 2 columns. The DataFrame now has 3 columns: `["Category", "Value", "Note"]`. Update any hardcoded column indexes or use column names.

---

[0.2.0]: https://github.com/breaded-xyz/alphavec/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/breaded-xyz/alphavec/releases/tag/v0.1.3
[0.1.2]: https://github.com/breaded-xyz/alphavec/releases/tag/v0.1.2
