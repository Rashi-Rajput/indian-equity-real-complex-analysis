"""Orchestrate the complete historical-data analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIDENCE_LEVELS, RISK_FREE_RATE, TRADING_DAYS
from .data import (
    calculate_returns,
    clean_prices,
    dataset_summary,
    download_market_data,
    log_return_identity_errors,
)
from .mathematics import run_extended_math_demonstrations, run_math_demonstrations
from .metrics import (
    comparative_summary,
    distribution_summary,
    movement_summary,
    monotonic_runs,
    variation_measures,
)
from .plots import (
    plot_characteristic_functions,
    plot_market_data,
    plot_math_convergence,
    plot_monotonic_phases,
    plot_option_sensitivity,
    plot_return_distributions,
)
from .quantitative_finance import (
    black_scholes,
    black_scholes_greeks,
    carr_madan_call,
    compare_characteristic_functions,
    run_fourier_demonstrations,
)


def main() -> None:
    output_dir = Path("outputs")
    plot_dir = output_dir / "plots"
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    ohlcv, raw_prices = download_market_data()
    print(f"Downloaded price rows: {len(raw_prices)}")
    print(
        f"Raw price date range: {raw_prices.index.min().date()} through "
        f"{raw_prices.index.max().date()}"
    )
    print(
        "Duplicate timestamps removed: "
        f"{raw_prices.attrs.get('duplicate_dates_removed', 'not recorded')}; "
        f"input date index sorted: {raw_prices.attrs.get('dates_were_sorted', 'unknown')}"
    )
    sources = raw_prices.attrs.get("return_price_sources", {})
    if sources:
        print("Return price source by instrument:")
        print(pd.Series(sources, name="Source").to_string())
    print("\nMissing values before complete-case cleaning:")
    print(raw_prices.isna().sum().to_string())

    print("\nOHLCV fields retained (Close and Adjusted Close remain distinct):")
    available_fields = list(dict.fromkeys(ohlcv.columns.get_level_values("Field")))
    print("Available fields:", ", ".join(available_fields))
    ohlcv_quality = pd.DataFrame({
        ticker: {
            field: int(ohlcv[(field, ticker)].isna().sum())
            for field in available_fields
        }
        for ticker in raw_prices.columns
    }).T
    ohlcv_quality.index.name = "Instrument"
    print("Missing OHLCV values by instrument:")
    print(ohlcv_quality.to_string())
    if "Close" in available_fields and "Adj Close" in available_fields:
        def latest_available(values: pd.Series) -> float:
            observed = values.dropna()
            return float(observed.iloc[-1]) if not observed.empty else np.nan

        close_comparison = pd.DataFrame({
            "Latest_Close": ohlcv["Close"].apply(latest_available),
            "Latest_Adj_Close": ohlcv["Adj Close"].apply(latest_available),
        })
        close_comparison["Adj_Close_minus_Close"] = (
            close_comparison["Latest_Adj_Close"] - close_comparison["Latest_Close"]
        )
        print("\nClose versus Adjusted Close (latest available values):")
        print(close_comparison.to_string(float_format=lambda value: f"{value:.6f}"))
    print(
        "Return calculations use Adjusted Close where available; Close is retained "
        "separately because it excludes dividend adjustment."
    )

    prices = clean_prices(raw_prices)
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ArithmeticError("Cleaned prices must have unique, sorted dates.")
    print("\nDataset summary:")
    data_summary = dataset_summary(prices, raw_prices)
    print(data_summary.to_string(float_format=lambda x: f"{x:,.6f}"))
    print(
        f"\nAligned complete-case observations across all five instruments: {len(prices)}. "
        "This common-date sample excludes a date if any instrument has no usable price."
    )

    simple_returns, log_returns = calculate_returns(prices)
    identity_errors = log_return_identity_errors(prices, log_returns)
    print("\nCumulative log-return identity errors:")
    print(identity_errors.to_string())
    if identity_errors.max() >= 1e-10:
        raise ArithmeticError("Cumulative log-return identity check failed.")

    variation = variation_measures(prices)
    comparison = comparative_summary(
        simple_returns, log_returns, CONFIDENCE_LEVELS, TRADING_DAYS, prices=prices
    )
    print(
        "\nComparative financial summary: return and VaR/ES values are daily "
        "decimal fractions; VaR/ES are positive loss amounts unless tail quantiles "
        "are gains. Historical ES averages returns at/below the empirical quantile; "
        "parametric VaR/ES assume normal daily simple returns."
    )
    print(comparison.to_string(float_format=lambda x: f"{x:.6f}"))
    if not comparison[[
        "Total_Variation", "Log_Price_Total_Variation",
        "Normalized_Total_Variation", "Path_Efficiency",
    ]].equals(variation[[
        "Total_Variation", "Log_Price_Variation",
        "Normalized_Variation", "Path_Efficiency",
    ]].rename(columns={
        "Log_Price_Variation": "Log_Price_Total_Variation",
        "Normalized_Variation": "Normalized_Total_Variation",
    })):
        raise ArithmeticError("Consolidated variation measures do not match standalone metrics.")

    print("\nMonotonic runs (first five per instrument):")
    for ticker in prices.columns:
        runs = monotonic_runs(prices[ticker])
        counts = runs["Direction"].value_counts() if not runs.empty else {}
        print(
            f"\n{ticker}: increasing={counts.get('increasing', 0)}, "
            f"decreasing={counts.get('decreasing', 0)}"
        )
        if not runs.empty:
            print(runs.head(5).to_string(index=False))
    movements = movement_summary(prices)
    print("\nConsecutive upward/downward movement totals (adjusted-price units):")
    print(movements.to_string(float_format=lambda value: f"{value:.6f}"))

    print(
        "\nDaily monotonic segments are adjacent observed closes/adjusted prices; "
        "the separate rolling trend shading is descriptive only and does not predict "
        "future direction."
    )

    distributions = distribution_summary(log_returns)
    print("\nObserved log-return distribution statistics (kurtosis is Fisher excess):")
    print(distributions.to_string(float_format=lambda x: f"{x:.6f}"))
    print(
        "Normal VaR/ES rely on symmetric, thin-tailed returns. Skewness, excess "
        "kurtosis, and empirical tail quantiles indicate where that model may be "
        "inadequate; historical estimates also depend on this sample and its regime."
    )

    quadrature = run_math_demonstrations()
    print("\nRiemann quadrature accuracy:")
    print(quadrature.to_string(float_format=lambda x: f"{x:.12g}"))
    convergence, darboux, quadratic_variation, math_validations = (
        run_extended_math_demonstrations()
    )

    # Empirical and Gaussian characteristic functions are calculated separately
    # for all instruments; inversion is validated once against a known normal law.
    characteristic_functions = {
        ticker: compare_characteristic_functions(
            log_returns[ticker].to_numpy(dtype=float)
        )
        for ticker in prices.columns
    }
    characteristic_function_valid = all(
        abs(empirical_cf[np.argmin(np.abs(freqs))] - 1.0) < 1e-12
        and abs(normal_cf[np.argmin(np.abs(freqs))] - 1.0) < 1e-12
        and np.allclose(empirical_cf, np.conjugate(empirical_cf[::-1]), atol=1e-12)
        for freqs, empirical_cf, normal_cf in characteristic_functions.values()
    )
    run_fourier_demonstrations(
        log_returns["^NSEI"].to_numpy(dtype=float),
        comparison=characteristic_functions["^NSEI"],
    )

    # Black-Scholes uses latest index level as a proxy spot, constant historical
    # annualized volatility, a flat rate, and a one-year European maturity.
    spot = float(prices["^NSEI"].iloc[-1])
    volatility = float(log_returns["^NSEI"].std(ddof=1) * np.sqrt(TRADING_DAYS))
    strike = spot
    maturity = 1.0
    call = black_scholes(spot, strike, RISK_FREE_RATE, volatility, maturity, "call")
    put = black_scholes(spot, strike, RISK_FREE_RATE, volatility, maturity, "put")
    parity_residual = call - put - (
        spot - strike * np.exp(-RISK_FREE_RATE * maturity)
    )
    carr_madan = carr_madan_call(
        spot, strike, RISK_FREE_RATE, volatility, maturity
    )
    print("\nBlack-Scholes example (NSEI proxy, ATM, 1 year):")
    print(f"Spot={spot:.6f}, Strike={strike:.6f}, Volatility={volatility:.6f}")
    print(f"Call={call:.6f}, Put={put:.6f}, Put-call parity residual={parity_residual:.3e}")
    print("Call Greeks:", black_scholes_greeks(
        spot, strike, RISK_FREE_RATE, volatility, maturity, "call"
    ))
    print("Put Greeks:", black_scholes_greeks(
        spot, strike, RISK_FREE_RATE, volatility, maturity, "put"
    ))
    print(
        f"Carr-Madan call={carr_madan:.8f}; Black-Scholes call={call:.8f}; "
        f"absolute error={abs(carr_madan-call):.3e}"
    )
    validation_results = {
        "Cumulative_log_return_identity": bool(identity_errors.max() < 1e-10),
        "Characteristic_function_normalization_and_symmetry": bool(
            characteristic_function_valid
        ),
        "Black_Scholes_put_call_parity": bool(abs(parity_residual) < 1e-8 * spot),
        "Carr_Madan_vs_Black_Scholes": bool(abs(carr_madan - call) / spot < 1e-6),
        **math_validations,
    }
    if not all(validation_results.values()):
        failed = [name for name, passed in validation_results.items() if not passed]
        raise ArithmeticError(f"Numerical validation failed: {failed}")

    plot_market_data(
        prices, log_returns, comparison, plot_dir
    )
    plot_monotonic_phases(prices, output_dir=plot_dir)
    plot_return_distributions(log_returns, output_dir=plot_dir)
    plot_characteristic_functions(characteristic_functions, output_dir=plot_dir)
    plot_math_convergence(convergence, darboux, quadratic_variation, output_dir=plot_dir)
    plot_option_sensitivity(
        spot, strike, RISK_FREE_RATE, volatility, maturity, output_dir=plot_dir
    )

    comparison.to_csv(table_dir / "comparative_financial_summary.csv")
    distributions.to_csv(table_dir / "return_distribution_summary.csv")
    movements.to_csv(table_dir / "up_down_movement_summary.csv")
    data_summary.to_csv(table_dir / "dataset_summary.csv")
    ohlcv_quality.to_csv(table_dir / "ohlcv_missing_value_summary.csv")
    quadrature.to_csv(table_dir / "quadrature_accuracy.csv")
    convergence.to_csv(table_dir / "quadrature_convergence.csv")
    darboux.to_csv(table_dir / "darboux_sums.csv")
    quadratic_variation.to_csv(table_dir / "quadratic_variation_examples.csv")
    pd.Series(validation_results, name="Passed").to_csv(
        table_dir / "numerical_validation.csv"
    )

    # Findings use only computed rows; no instrument ordering or conclusions
    # are hardcoded. Negative VaR indicates a gain at the chosen lower quantile.
    print("\nComputed findings:")
    for metric, label in (
        ("Mean_Log_Daily", "Highest mean daily log return"),
        ("Annualized_Volatility", "Highest annualized volatility"),
        ("Historical_VaR_95", "Highest historical 95% VaR"),
        ("Historical_ES_95", "Highest historical 95% ES"),
        ("Mean_Absolute_Log_Return", "Highest mean absolute log return"),
        ("Path_Efficiency", "Highest sampled path efficiency"),
    ):
        values = comparison[metric].dropna()
        if not values.empty:
            numeric_values = values.to_numpy(dtype=float)
            maximum_index = int(np.argmax(numeric_values))
            ticker = str(values.index[maximum_index])
            print(f"{label}: {ticker} ({numeric_values[maximum_index]:.6f})")
    print(
        "Distributional comparison: greatest sample excess kurtosis is "
        f"{distributions['Excess_Kurtosis'].idxmax()} "
        f"({distributions['Excess_Kurtosis'].max():.6f}); "
        "this is a sample statistic, not proof of a particular tail law."
    )
    print("\nNumerical validation results:")
    for name, passed in validation_results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(
        "\nAssumptions and limitations: 252 trading days per year; common-date "
        "complete-case alignment; adjusted prices used for returns when available; "
        "VaR/ES are one-day estimates; the parametric risk model is Gaussian; "
        "Black-Scholes assumes European exercise, constant volatility and a flat "
        "risk-free rate. A continuous cash-flow example is a specified model, not "
        "an interpolation of daily returns. Discrete sampled variation depends on "
        "sampling frequency and is not continuous-path quadratic variation. "
        "Yahoo Finance coverage and corporate-action adjustments may change. "
        f"Plots and CSV tables saved under {output_dir.resolve()}."
    )


if __name__ == "__main__":
    main()
