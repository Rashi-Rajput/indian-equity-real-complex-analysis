"""Return, risk, monotonicity, and bounded-variation measures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def risk_measures(returns: pd.Series, confidence: float) -> dict[str, float]:
    """Return positive loss VaR/ES estimates for the left tail of simple returns."""
    values = returns.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if values.size < 2:
        raise ValueError("At least two finite returns are required for risk measures.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")

    alpha = 1.0 - confidence
    quantile = float(np.quantile(values, alpha))
    tail = values[values <= quantile]
    mu = float(np.mean(values))
    sigma = float(np.std(values, ddof=1))
    z = float(stats.norm.ppf(alpha))
    return {
        "Historical_VaR": -quantile,
        "Historical_ES": -float(np.mean(tail)),
        "Parametric_VaR": -(mu + sigma * z),
        "Parametric_ES": -float(mu - sigma * stats.norm.pdf(z) / alpha),
    }


def monotonic_runs(series: pd.Series) -> pd.DataFrame:
    """Return contiguous strictly increasing/decreasing runs in sampled prices."""
    values = series.dropna()
    changes = np.sign(np.diff(values.to_numpy(dtype=float)))
    records = []
    start = 0
    while start < len(changes):
        direction = changes[start]
        stop = start + 1
        while stop < len(changes) and changes[stop] == direction:
            stop += 1
        if direction != 0:
            records.append({
                "Direction": "increasing" if direction > 0 else "decreasing",
                "Start": values.index[start],
                "End": values.index[stop],
                "Price_Changes": stop - start,
            })
        start = stop
    return pd.DataFrame(records, columns=["Direction", "Start", "End", "Price_Changes"])


def movement_summary(prices: pd.DataFrame) -> pd.DataFrame:
    """Quantify consecutive upward and downward sampled price movements."""
    rows = []
    for ticker in prices.columns:
        changes = np.diff(prices[ticker].to_numpy(dtype=float))
        up = changes[changes > 0]
        down = changes[changes < 0]
        rows.append({
            "Instrument": ticker,
            "Up_Days": int(up.size),
            "Down_Days": int(down.size),
            "Flat_Days": int(np.count_nonzero(changes == 0)),
            "Gross_Upward_Price_Change": float(up.sum()),
            "Gross_Downward_Price_Change": float(-down.sum()),
            "Net_Price_Change": float(changes.sum()),
        })
    return pd.DataFrame(rows).set_index("Instrument")


def variation_measures(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute sampled total variation and scale-adjusted path measures."""
    rows = []
    for ticker in prices.columns:
        path = prices[ticker].to_numpy(dtype=float)
        total_variation = float(np.abs(np.diff(path)).sum())
        rows.append({
            "Instrument": ticker,
            "Total_Variation": total_variation,
            "Normalized_Variation": total_variation / path[0],
            "Log_Price_Variation": float(np.abs(np.diff(np.log(path))).sum()),
            "Path_Efficiency": (
                abs(path[-1] - path[0]) / total_variation
                if total_variation > 0 else np.nan
            ),
        })
    return pd.DataFrame(rows).set_index("Instrument")


def comparative_summary(
    simple_returns: pd.DataFrame,
    log_returns: pd.DataFrame,
    confidence_levels: tuple[float, ...],
    trading_days: int,
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize returns and risk; VaR/ES are positive losses on simple returns.

    Historical ES averages observations at or below the empirical lower-tail
    quantile. Parametric VaR/ES assume normally distributed daily simple returns.
    """
    rows = []
    for ticker in simple_returns.columns:
        simple = simple_returns[ticker].dropna()
        log = log_returns[ticker].dropna()
        daily_volatility = float(log.std(ddof=1))
        row = {
            "Instrument": ticker,
            "Mean_Simple_Daily": float(simple.mean()),
            "Mean_Log_Daily": float(log.mean()),
            "Mean_Absolute_Log_Return": float(log.abs().mean()),
            "Cumulative_Simple": float(np.prod(1.0 + simple.to_numpy()) - 1.0),
            "Cumulative_Log": float(log.sum()),
            "Daily_Volatility": daily_volatility,
            "Annualized_Volatility": daily_volatility * np.sqrt(trading_days),
        }
        if prices is not None:
            path = prices[ticker].dropna().to_numpy(dtype=float)
            total_variation = float(np.abs(np.diff(path)).sum())
            row.update({
                "Total_Variation": total_variation,
                "Log_Price_Total_Variation": float(np.abs(np.diff(np.log(path))).sum()),
                "Normalized_Total_Variation": total_variation / path[0],
                "Path_Efficiency": (
                    abs(path[-1] - path[0]) / total_variation
                    if total_variation > 0 else np.nan
                ),
            })
        for confidence in confidence_levels:
            suffix = str(int(confidence * 100))
            for name, value in risk_measures(simple, confidence).items():
                row[f"{name}_{suffix}"] = value
        rows.append(row)
    return pd.DataFrame(rows).set_index("Instrument")


def distribution_summary(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Describe observed log-return distributions; kurtosis is Fisher excess."""
    rows = []
    for ticker in log_returns.columns:
        values = log_returns[ticker].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) < 4:
            raise ValueError(f"Too few finite log returns to summarize {ticker}.")
        rows.append({
            "Instrument": ticker,
            "Observations": len(values),
            "Skewness": float(stats.skew(values, bias=False)),
            "Excess_Kurtosis": float(stats.kurtosis(values, fisher=True, bias=False)),
            "Minimum": float(values.min()),
            "Maximum": float(values.max()),
            "Empirical_1pct": float(values.quantile(0.01)),
            "Empirical_99pct": float(values.quantile(0.99)),
            "Normal_1pct": float(values.mean() + values.std(ddof=1) * stats.norm.ppf(0.01)),
            "Normal_99pct": float(values.mean() + values.std(ddof=1) * stats.norm.ppf(0.99)),
        })
    return pd.DataFrame(rows).set_index("Instrument")
