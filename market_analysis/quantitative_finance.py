"""Characteristic functions, Fourier inversion, and option pricing."""

from __future__ import annotations

from math import exp, isfinite, log, sqrt

import numpy as np
from scipy import integrate, stats


def empirical_cf(samples: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    """Empirical characteristic function E[exp(i*u*X)]."""
    sample_values = np.asarray(samples, dtype=float)
    frequency_values = np.asarray(frequencies, dtype=float)
    if sample_values.size == 0:
        raise ValueError("At least one sample is required.")
    return np.mean(
        np.exp(1j * frequency_values[..., np.newaxis] * sample_values),
        axis=-1,
    )


def normal_cf(frequencies: np.ndarray | float, mu: float, sigma: float) -> np.ndarray:
    """Characteristic function of a normal random variable."""
    if sigma < 0:
        raise ValueError("sigma cannot be negative.")
    u = np.asarray(frequencies, dtype=float)
    return np.exp(1j * u * mu - 0.5 * sigma**2 * u**2)


def normal_fourier_density(x: float, mu: float, sigma: float) -> float:
    """Compute the normal density by Fourier inversion."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    integral = integrate.quad(
        lambda u: float(np.real(np.exp(-1j * u * x) * normal_cf(u, mu, sigma))),
        0,
        np.inf,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=300,
    )[0]
    return float(integral / np.pi)


def gil_pelaez_normal_cdf(x: float, mu: float, sigma: float) -> float:
    """Compute a normal CDF using Gil-Pelaez characteristic-function inversion."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    def integrand(u: float) -> float:
        if u == 0:
            return mu - x
        return float(np.imag(np.exp(-1j * u * x) * normal_cf(u, mu, sigma)) / u)

    integral = integrate.quad(
        integrand, 0, np.inf, epsabs=1e-10, epsrel=1e-10, limit=300
    )[0]
    return float(0.5 - integral / np.pi)


def black_scholes(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    option: str = "call",
) -> float:
    """Black-Scholes European price with a continuously compounded rate."""
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'.")
    if not all(isfinite(value) for value in (spot, strike, rate, volatility, maturity)):
        raise ValueError("Black-Scholes inputs must all be finite.")
    if spot <= 0 or strike <= 0 or volatility <= 0 or maturity <= 0:
        raise ValueError("spot, strike, volatility, and maturity must be positive.")
    d1 = (
        log(spot / strike) + (rate + 0.5 * volatility**2) * maturity
    ) / (volatility * sqrt(maturity))
    d2 = d1 - volatility * sqrt(maturity)
    if option == "call":
        return float(
            spot * stats.norm.cdf(d1)
            - strike * exp(-rate * maturity) * stats.norm.cdf(d2)
        )
    return float(
        strike * exp(-rate * maturity) * stats.norm.cdf(-d2)
        - spot * stats.norm.cdf(-d1)
    )


def black_scholes_greeks(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    option: str = "call",
) -> dict[str, float]:
    """Return standard European Black-Scholes Greeks (theta is per year)."""
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'.")
    if not all(isfinite(value) for value in (spot, strike, rate, volatility, maturity)):
        raise ValueError("Black-Scholes inputs must all be finite.")
    if spot <= 0 or strike <= 0 or volatility <= 0 or maturity <= 0:
        raise ValueError("spot, strike, volatility, and maturity must be positive.")

    d1 = (
        log(spot / strike) + (rate + 0.5 * volatility**2) * maturity
    ) / (volatility * sqrt(maturity))
    d2 = d1 - volatility * sqrt(maturity)
    density = stats.norm.pdf(d1)
    is_call = option == "call"
    delta = stats.norm.cdf(d1) if is_call else stats.norm.cdf(d1) - 1
    gamma = density / (spot * volatility * sqrt(maturity))
    vega = spot * density * sqrt(maturity)
    if is_call:
        theta = (
            -spot * density * volatility / (2 * sqrt(maturity))
            - rate * strike * exp(-rate * maturity) * stats.norm.cdf(d2)
        )
        rho = strike * maturity * exp(-rate * maturity) * stats.norm.cdf(d2)
    else:
        theta = (
            -spot * density * volatility / (2 * sqrt(maturity))
            + rate * strike * exp(-rate * maturity) * stats.norm.cdf(-d2)
        )
        rho = -strike * maturity * exp(-rate * maturity) * stats.norm.cdf(-d2)
    return {
        "Delta": float(delta),
        "Gamma": float(gamma),
        "Theta": float(theta),
        "Vega": float(vega),
        "Rho": float(rho),
    }


def _bs_log_price_cf(
    u: np.ndarray, spot: float, rate: float, volatility: float, maturity: float
) -> np.ndarray:
    """Characteristic function of log(S_T) under risk-neutral Black-Scholes."""
    mean = log(spot) + (rate - 0.5 * volatility**2) * maturity
    return np.exp(1j * u * mean - 0.5 * volatility**2 * maturity * u**2)


def carr_madan_call(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    damping: float = 1.5,
    upper: float = 150.0,
    intervals: int = 30000,
) -> float:
    """Carr-Madan damped call integral, numerically integrated by trapezoids."""
    if not all(np.isfinite(value) for value in (spot, strike, rate, volatility, maturity)):
        raise ValueError("Carr-Madan inputs must all be finite.")
    if spot <= 0 or strike <= 0 or volatility <= 0 or maturity <= 0:
        raise ValueError("spot, strike, volatility, and maturity must be positive.")
    if damping <= 0 or upper <= 0 or intervals < 2:
        raise ValueError("damping and upper must be positive; intervals must be at least 2.")
    log_strike = np.log(strike)
    frequencies = np.linspace(0.0, upper, intervals + 1)
    shifted = frequencies - 1j * (damping + 1)
    phi = _bs_log_price_cf(shifted, spot, rate, volatility, maturity)
    denominator = (
        damping**2 + damping - frequencies**2
        + 1j * (2 * damping + 1) * frequencies
    )
    psi = np.exp(-rate * maturity) * phi / denominator
    integrand = np.exp(-1j * frequencies * log_strike) * psi
    integral = integrate.trapezoid(integrand, frequencies)
    return float(np.exp(-damping * log_strike) * integral.real / np.pi)


def run_fourier_demonstrations(
    samples: np.ndarray,
    comparison: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Print and validate empirical/normal CF comparison and inversion."""
    if samples.size < 2:
        raise ValueError("At least two observations are required for CF analysis.")
    mu = float(np.mean(samples))
    sigma = float(np.std(samples, ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Sample standard deviation must be positive and finite.")

    if comparison is None:
        frequencies = np.linspace(-20, 20, 401)
        empirical = empirical_cf(samples, frequencies)
        gaussian = normal_cf(frequencies, mu, sigma)
    else:
        frequencies, empirical, gaussian = comparison
    selected = np.arange(0, len(frequencies), 100)
    print("\nCharacteristic functions at selected frequencies:")
    for index in selected:
        print(
            f"u={frequencies[index]: .3f}: "
            f"empirical=({empirical[index].real:.6f}, {empirical[index].imag:.6f}), "
            f"normal=({gaussian[index].real:.6f}, {gaussian[index].imag:.6f})"
        )

    points = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 7)
    density_results = np.array([
        [normal_fourier_density(float(x), mu, sigma), stats.norm.pdf(x, mu, sigma)]
        for x in points
    ])
    cdf_results = np.array([
        [gil_pelaez_normal_cdf(float(x), mu, sigma), stats.norm.cdf(x, mu, sigma)]
        for x in points
    ])
    print("\nFourier-inversion checks:")
    for x, density, exact_density, cdf, exact_cdf in zip(
        points, density_results[:, 0], density_results[:, 1],
        cdf_results[:, 0], cdf_results[:, 1],
    ):
        print(
            f"x={x:.8f}: PDF={density:.10f} (exact {exact_density:.10f}), "
            f"CDF={cdf:.10f} (exact {exact_cdf:.10f})"
        )
    if np.max(np.abs(density_results[:, 0] - density_results[:, 1])) >= 1e-7:
        raise ArithmeticError("Fourier density inversion did not match normal density.")
    if np.max(np.abs(cdf_results[:, 0] - cdf_results[:, 1])) >= 1e-7:
        raise ArithmeticError("Gil-Pelaez inversion did not match normal CDF.")
    return frequencies, empirical, gaussian


def compare_characteristic_functions(
    log_returns: np.ndarray,
    frequency_count: int = 401,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frequencies and empirical/normal CFs for a single return series."""
    values = np.asarray(log_returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("At least two finite returns are required.")
    sigma = float(np.std(values, ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Return volatility must be positive and finite.")
    frequencies = np.linspace(-20.0, 20.0, frequency_count)
    return (
        frequencies,
        empirical_cf(values, frequencies),
        normal_cf(frequencies, float(np.mean(values)), sigma),
    )
