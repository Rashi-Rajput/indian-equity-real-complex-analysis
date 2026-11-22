"""Visualization helpers for the market and option analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from .quantitative_finance import black_scholes


def _save_show(fig: Figure, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=220, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_market_data(
    prices: pd.DataFrame,
    log_returns: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: str | Path = "outputs/plots",
) -> None:
    """Plot normalized prices, returns, volatility, and risk measures."""
    output_path = Path(output_dir)
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ticker in prices.columns:
        axes[0].plot(prices.index, prices[ticker] / prices[ticker].iloc[0], label=ticker)
    axes[0].set_title("Adjusted close normalized to the first observation")
    axes[0].set_ylabel("Normalized price")
    axes[0].legend(ncol=3)

    for ticker in log_returns.columns:
        axes[1].plot(
            log_returns.index, log_returns[ticker], linewidth=0.65, alpha=0.7,
            label=ticker,
        )
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_title("Daily log returns")
    axes[1].set_ylabel("Log return")
    axes[1].legend(ncol=3)
    fig.tight_layout()
    _save_show(fig, output_path, "normalized_prices_and_log_returns.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    comparison["Annualized_Volatility"].plot(kind="bar", ax=ax)
    ax.set_title("Annualized historical volatility")
    ax.set_ylabel("Annualized volatility")
    ax.set_xlabel("Instrument")
    fig.tight_layout()
    _save_show(fig, output_path, "annualized_volatility.png")

    fig, ax = plt.subplots(figsize=(11, 6))
    comparison[[
        "Historical_VaR_95", "Historical_ES_95",
        "Parametric_VaR_95", "Parametric_ES_95"
    ]].plot(kind="bar", ax=ax)
    ax.set_title("Daily 95% loss VaR and Expected Shortfall")
    ax.set_ylabel("Loss fraction")
    ax.set_xlabel("Instrument")
    fig.tight_layout()
    _save_show(fig, output_path, "daily_risk_measures.png")

def plot_monotonic_phases(
    prices: pd.DataFrame,
    window: int = 20,
    output_dir: str | Path = "outputs/plots",
) -> None:
    """Show daily up/down segments separately from rolling trend phases."""
    if window < 2:
        raise ValueError("window must be at least 2.")
    output_path = Path(output_dir)
    fig, axes = plt.subplots(
        len(prices.columns), 1, figsize=(14, 2.8 * len(prices.columns)),
        sharex=True, squeeze=False,
    )
    for ax, ticker in zip(axes[:, 0], prices.columns):
        series = prices[ticker].dropna()
        x = np.arange(len(series), dtype=float)
        y = series.to_numpy(dtype=float)
        points = np.column_stack((x, y))
        segments = np.stack((points[:-1], points[1:]), axis=1)
        segment_colors = np.where(np.diff(y) >= 0, "#18864b", "#c23b3b")
        ax.add_collection(LineCollection(
            segments.tolist(), colors=segment_colors, linewidths=0.8
        ))

        trend = series.rolling(window=min(window, len(series)), min_periods=window).mean()
        slope = trend.diff()
        positive = slope.to_numpy() > 0
        negative = slope.to_numpy() < 0
        ax.fill_between(x, np.nanmin(y), np.nanmax(y), where=positive,
                        color="#18864b", alpha=0.08, step="mid")
        ax.fill_between(x, np.nanmin(y), np.nanmax(y), where=negative,
                        color="#c23b3b", alpha=0.08, step="mid")
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(np.nanmin(y), np.nanmax(y))
        ax.set_ylabel("Adjusted price")
        ax.set_title(f"{ticker}: green/red line = daily up/down; shading = {window}-day trend")
        ticks = np.linspace(0, len(series) - 1, min(6, len(series)), dtype=int)
        date_index = pd.DatetimeIndex(series.index)
        ax.set_xticks(ticks, [date_index[i].strftime("%Y-%m") for i in ticks])
    axes[-1, 0].set_xlabel("Date")
    fig.suptitle("Daily monotonic movements and broader rolling trend phases", y=1.002)
    fig.tight_layout()
    _save_show(fig, output_path, "monotonic_and_trend_phases.png")


def plot_return_distributions(
    log_returns: pd.DataFrame,
    output_dir: str | Path = "outputs/plots",
) -> None:
    """Compare actual daily log-return histograms with fitted normal densities."""
    output_path = Path(output_dir)
    columns = 2
    rows = int(np.ceil(len(log_returns.columns) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 4 * rows), squeeze=False)
    for ax, ticker in zip(axes.flat, log_returns.columns):
        values = log_returns[ticker].dropna().to_numpy(dtype=float)
        mu = float(np.mean(values))
        sigma = float(np.std(values, ddof=1))
        ax.hist(values, bins="fd", density=True, alpha=0.55, color="#4677a8",
                label="Observed log returns")
        x = np.linspace(float(np.min(values)), float(np.max(values)), 400)
        ax.plot(x, (1 / (sigma * np.sqrt(2 * np.pi))) *
                np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                color="#ba3c3c", label="Fitted normal")
        ax.set_title(f"{ticker}: observed vs fitted normal")
        ax.set_xlabel("Daily log return")
        ax.set_ylabel("Density")
        ax.legend()
    for ax in axes.flat[len(log_returns.columns):]:
        ax.set_visible(False)
    fig.tight_layout()
    _save_show(fig, output_path, "return_distributions.png")


def plot_characteristic_functions(
    comparisons: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: str | Path = "outputs/plots",
) -> None:
    """Compare empirical and normal CF components for each instrument."""
    output_path = Path(output_dir)
    fig, axes = plt.subplots(len(comparisons), 2,
                             figsize=(12, max(3, 2.6 * len(comparisons))),
                             squeeze=False)
    for row, (ticker, (frequencies, empirical, fitted)) in enumerate(comparisons.items()):
        axes[row, 0].plot(frequencies, np.real(empirical), label="Empirical")
        axes[row, 0].plot(frequencies, np.real(fitted), "--", label="Normal")
        axes[row, 0].set_title(f"{ticker}: real component")
        axes[row, 1].plot(frequencies, np.imag(empirical), label="Empirical")
        axes[row, 1].plot(frequencies, np.imag(fitted), "--", label="Normal")
        axes[row, 1].set_title(f"{ticker}: imaginary component")
        for ax in axes[row]:
            ax.set_xlabel("Frequency")
            ax.legend()
    fig.tight_layout()
    _save_show(fig, output_path, "characteristic_functions_all_instruments.png")


def plot_math_convergence(
    convergence: pd.DataFrame,
    darboux: pd.DataFrame,
    quadratic_variation: pd.DataFrame,
    output_dir: str | Path = "outputs/plots",
) -> None:
    """Save quadrature, Darboux-sum, and variation-comparison plots."""
    output_path = Path(output_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(convergence.index, convergence["Trapezoid_Error"], "o-",
              label="Trapezoidal")
    ax.loglog(convergence.index, convergence["Simpson_Error"], "s-",
              label="Simpson 1/3")
    ax.set_title("Quadrature error convergence for exp(x) on [0,1]")
    ax.set_xlabel("Number of subintervals")
    ax.set_ylabel("Absolute error")
    ax.legend()
    fig.tight_layout()
    _save_show(fig, output_path, "quadrature_convergence.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(darboux.index, darboux["Lower_Sum"], "o-", label="Lower sum")
    ax.plot(darboux.index, darboux["Upper_Sum"], "s-", label="Upper sum")
    ax.axhline(1 / 3, color="black", linestyle="--", label="Integral = 1/3")
    ax.set_xscale("log")
    ax.set_title("Darboux sums for x^2 on [0,1]")
    ax.set_xlabel("Number of subintervals")
    ax.set_ylabel("Sum")
    ax.legend()
    fig.tight_layout()
    _save_show(fig, output_path, "darboux_sum_convergence.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.loglog(
        quadratic_variation.index,
        quadratic_variation["Smooth_Quadratic_Variation"],
        "o-", label="Smooth path: quadratic variation",
    )
    ax.loglog(
        quadratic_variation.index,
        quadratic_variation["Brownian_Quadratic_Variation"],
        "s-", label="Brownian-inspired: quadratic variation",
    )
    ax.set_title("Illustrative quadratic variation versus partition size")
    ax.set_xlabel("Number of subintervals")
    ax.set_ylabel("Sum of squared increments")
    ax.legend()
    fig.tight_layout()
    _save_show(fig, output_path, "quadratic_variation_examples.png")


def plot_option_sensitivity(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    output_dir: str | Path = "outputs/plots",
) -> None:
    """Plot Black-Scholes call and put values versus spot and volatility."""
    spots = np.linspace(0.7 * spot, 1.3 * spot, 100)
    vol_low = max(0.01, 0.5 * volatility)
    volatilities = np.linspace(vol_low, max(1.5 * volatility, vol_low * 1.01), 100)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for option in ("call", "put"):
        axes[0].plot(
            spots,
            [black_scholes(s, strike, rate, volatility, maturity, option) for s in spots],
            label=option.title(),
        )
        axes[1].plot(
            volatilities,
            [black_scholes(spot, strike, rate, v, maturity, option) for v in volatilities],
            label=option.title(),
        )
    axes[0].set_title("Option value vs. spot")
    axes[0].set_xlabel("Spot")
    axes[1].set_title("Option value vs. volatility")
    axes[1].set_xlabel("Volatility")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    _save_show(fig, Path(output_dir), "black_scholes_sensitivity.png")
