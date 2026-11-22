"""Numerical demonstrations for integration and complex analysis topics."""

from __future__ import annotations

from math import factorial

import numpy as np
import pandas as pd
from scipy import integrate, stats


def trapezoidal_rule(function, start: float, end: float, intervals: int) -> float:
    """Composite trapezoidal rule."""
    if intervals < 1:
        raise ValueError("intervals must be positive.")
    x = np.linspace(start, end, intervals + 1)
    y = np.asarray(function(x), dtype=float)
    h = (end - start) / intervals
    return float(h * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1]))


def composite_simpson(function, start: float, end: float, intervals: int = 1000) -> float:
    """Composite Simpson 1/3 rule with an even number of intervals."""
    if intervals < 2:
        raise ValueError("intervals must be at least 2.")
    if intervals % 2:
        intervals += 1
    x = np.linspace(start, end, intervals + 1)
    y = np.asarray(function(x), dtype=float)
    h = (end - start) / intervals
    return float(h / 3 * (
        y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum()
    ))


def cauchy_riemann_error(u, v, x: float, y: float, h: float = 1e-5) -> float:
    """Finite-difference residuals for u_x=v_y and u_y=-v_x."""
    ux = (u(x + h, y) - u(x - h, y)) / (2 * h)
    vy = (v(x, y + h) - v(x, y - h)) / (2 * h)
    uy = (u(x, y + h) - u(x, y - h)) / (2 * h)
    vx = (v(x + h, y) - v(x - h, y)) / (2 * h)
    return float(abs(ux - vy) + abs(uy + vx))


def laplacian(function, x: float, y: float, h: float = 1e-4) -> float:
    """Central finite-difference approximation of the two-dimensional Laplacian."""
    fxx = (function(x + h, y) - 2 * function(x, y) + function(x - h, y)) / h**2
    fyy = (function(x, y + h) - 2 * function(x, y) + function(x, y - h)) / h**2
    return float(fxx + fyy)


def run_math_demonstrations() -> pd.DataFrame:
    """Run and validate the independent numerical-analysis examples."""
    examples = [
        ("x^2", lambda x: x**2, 0.0, 2.0, 8.0 / 3.0),
        ("sin(x)", np.sin, 0.0, np.pi, 2.0),
        ("exp(-x)", lambda x: np.exp(-x), 0.0, 2.0, 1.0 - np.exp(-2.0)),
    ]
    quadrature_rows = []
    for name, function, start, end, exact in examples:
        trapezoid = trapezoidal_rule(function, start, end, 1000)
        simpson = composite_simpson(function, start, end, 1000)
        quadrature_rows.append({
            "Function": name,
            "Analytical": exact,
            "Trapezoidal": trapezoid,
            "Simpson": simpson,
            "Trap_Abs_Error": abs(trapezoid - exact),
            "Simpson_Abs_Error": abs(simpson - exact),
        })
    quadrature = pd.DataFrame(quadrature_rows).set_index("Function")

    normal_mass = integrate.quad(lambda x: stats.norm.pdf(x), -np.inf, np.inf)[0]
    exponential_mass = integrate.quad(lambda x: stats.expon.pdf(x), 0, np.inf)[0]
    pareto_tail = integrate.quad(lambda x: stats.pareto.pdf(x, b=3), 2, np.inf)[0]
    print("\nImproper integral checks:")
    print(f"Normal density mass: {normal_mass:.12f}")
    print(f"Exponential density mass: {exponential_mass:.12f}")
    print(f"Pareto(shape=3) P(X > 2): {pareto_tail:.12f}")
    if abs(normal_mass - 1) > 1e-9 or abs(exponential_mass - 1) > 1e-9:
        raise ArithmeticError("Probability-density normalization check failed.")

    z = np.array([0.3 + 0.4j, -0.5 + 0.8j])
    print("\nComplex-function examples:")
    for name, values in {
        "z^2": z**2,
        "exp(z)": np.exp(z),
        "sin(z)": np.sin(z),
        "conjugate(z)": np.conjugate(z),
    }.items():
        print(f"{name}: {values}")

    u_exp = lambda x, y: np.exp(x) * np.cos(y)
    v_exp = lambda x, y: np.exp(x) * np.sin(y)
    u_z2 = lambda x, y: x**2 - y**2
    v_z2 = lambda x, y: 2 * x * y
    u_conjugate = lambda x, y: x + 0.0 * y
    v_conjugate = lambda x, y: 0.0 * x - y
    cr_errors = {
        "exp(z)": cauchy_riemann_error(u_exp, v_exp, 0.4, -0.3),
        "z^2": cauchy_riemann_error(u_z2, v_z2, 0.4, -0.3),
        "conjugate(z)": cauchy_riemann_error(u_conjugate, v_conjugate, 0.4, -0.3),
    }
    print("\nCauchy-Riemann residuals (analytic functions should approach zero):")
    for name, error in cr_errors.items():
        print(f"{name}: {error:.3e}")
    if cr_errors["exp(z)"] >= 1e-7 or cr_errors["z^2"] >= 1e-7:
        raise ArithmeticError("Cauchy-Riemann check failed for an analytic function.")
    if cr_errors["conjugate(z)"] <= 1:
        raise ArithmeticError("Conjugation should fail the Cauchy-Riemann equations.")

    u_harmonic = lambda x, y: x**2 - y**2
    v_conjugate = lambda x, y: 2 * x * y
    print("\nFinite-difference Laplacians:")
    print(f"u=x^2-y^2: {laplacian(u_harmonic, 0.4, -0.3):.3e}")
    print(f"v=2xy: {laplacian(v_conjugate, 0.4, -0.3):.3e}")
    print(f"w=x^3: {laplacian(lambda x, y: x**3 + 0.0 * y, 0.4, -0.3):.6f}")

    # The geometric series has radius one; the rational expression continues
    # it to the complex plane except at its pole z=1. The exponential series is entire.
    z_inside = 0.5 + 0.2j
    geometric_sum = sum(z_inside**n for n in range(500))
    geometric_formula = 1 / (1 - z_inside)
    continued_value = 1 / (1 - (1.5 + 0.2j))
    exponential_sum = sum((0.7 + 0.4j)**n / factorial(n) for n in range(50))
    print("\nAnalytic continuation:")
    print(f"Geometric series error inside disk: {abs(geometric_sum-geometric_formula):.3e}")
    print(f"Continued geometric value at 1.5+0.2i: {continued_value}")
    print(f"Exponential series error: {abs(exponential_sum-np.exp(0.7+0.4j)):.3e}")

    return quadrature


def run_extended_math_demonstrations(
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, bool]]:
    """Demonstrate convergence, variation, complex analysis, and tests."""
    print("\nQuadrature convergence for exp(x) on [0, 1], exact integral e - 1:")
    exact = float(np.e - 1.0)
    convergence_rows = []
    for intervals in (4, 8, 16, 32, 64):
        trapezoid = trapezoidal_rule(np.exp, 0.0, 1.0, intervals)
        simpson = composite_simpson(np.exp, 0.0, 1.0, intervals)
        convergence_rows.append({
            "Subintervals": intervals,
            "Trapezoid_Error": abs(trapezoid - exact),
            "Simpson_Error": abs(simpson - exact),
        })
    convergence = pd.DataFrame(convergence_rows).set_index("Subintervals")
    print(convergence.to_string(float_format=lambda value: f"{value:.4e}"))

    # Uniform Darboux sums for increasing f(x)=x^2 on [0,1]:
    # the infimum/supremum on each subinterval occur at its left/right endpoint.
    darboux_rows = []
    for intervals in (4, 16, 64, 256):
        partition = np.linspace(0.0, 1.0, intervals + 1)
        lower = float(np.sum(partition[:-1] ** 2) / intervals)
        upper = float(np.sum(partition[1:] ** 2) / intervals)
        darboux_rows.append({
            "Subintervals": intervals,
            "Lower_Sum": lower,
            "Upper_Sum": upper,
            "Gap": upper - lower,
        })
    darboux = pd.DataFrame(darboux_rows).set_index("Subintervals")
    print("\nDarboux sums for f(x)=x^2 on [0,1] (integral=1/3):")
    print(darboux.to_string(float_format=lambda value: f"{value:.8f}"))

    # A modelled continuous cash-flow is not the discrete return series:
    # c(t)=100 exp(-0.02t), discounted continuously at 5%, on [0,5].
    # Its present-value integrand is 100 exp(-0.07t); this is an explicit model.
    cashflow_rate = 0.02
    discount_rate = 0.05
    horizon = 5.0
    pv_integrand = lambda t: 100.0 * np.exp(-(cashflow_rate + discount_rate) * t)
    pv_exact = 100.0 * (
        1.0 - np.exp(-(cashflow_rate + discount_rate) * horizon)
    ) / (cashflow_rate + discount_rate)
    pv_numeric = composite_simpson(pv_integrand, 0.0, horizon, 1000)
    print(
        "\nModelled continuous cash-flow PV on [0,5]: "
        f"numerical={pv_numeric:.10f}, analytical={pv_exact:.10f}, "
        f"absolute error={abs(pv_numeric-pv_exact):.3e}"
    )

    # Total variation and Jordan decomposition are illustrated on f(x)=sin(x)
    # sampled on [0,2pi]. Its exact continuous total variation is 4.
    grid = np.linspace(0.0, 2.0 * np.pi, 4001)
    function_values = np.sin(grid)
    increments = np.diff(function_values)
    cumulative_variation = np.r_[0.0, np.cumsum(np.abs(increments))]
    positive_part = 0.5 * (
        cumulative_variation + function_values - function_values[0]
    )
    negative_part = 0.5 * (
        cumulative_variation - function_values + function_values[0]
    )
    total_variation = float(cumulative_variation[-1])
    jordan_reconstruction_error = float(np.max(
        np.abs((positive_part - negative_part) -
               (function_values - function_values[0]))
    ))
    # Variation additivity for a sampled path at an interior split point.
    split = len(grid) // 3
    variation_whole = float(np.abs(np.diff(function_values)).sum())
    variation_split = float(
        np.abs(np.diff(function_values[:split + 1])).sum()
        + np.abs(np.diff(function_values[split:])).sum()
    )

    # A BV function is Riemann integrable. The bounded Dirichlet function
    # 1_Q is not Riemann integrable: every interval has infimum 0 and supremum 1.
    dirichlet_lower = 0.0
    dirichlet_upper = 1.0

    # Smooth f(t)=t on [0,1] has positive total variation but quadratic
    # variation sum((delta f)^2) -> 0 as the mesh is refined.
    # Brownian-inspired Gaussian increments have expected QV near T, while
    # their path total variation grows with refinement; these are simulations.
    qv_rows = []
    rng = np.random.default_rng(20261118)
    for intervals in (64, 256, 1024, 4096):
        t = np.linspace(0.0, 1.0, intervals + 1)
        smooth = t
        smooth_increments = np.diff(smooth)
        brownian_increments = rng.normal(0.0, np.sqrt(1.0 / intervals), intervals)
        qv_rows.append({
            "Subintervals": intervals,
            "Smooth_Total_Variation": float(np.abs(smooth_increments).sum()),
            "Smooth_Quadratic_Variation": float(np.square(smooth_increments).sum()),
            "Brownian_Total_Variation": float(np.abs(brownian_increments).sum()),
            "Brownian_Quadratic_Variation": float(np.square(brownian_increments).sum()),
        })
    qv = pd.DataFrame(qv_rows).set_index("Subintervals")
    print("\nTotal variation and quadratic variation (Brownian-inspired values are simulated):")
    print(qv.to_string(float_format=lambda value: f"{value:.6f}"))
    print(
        f"Jordan decomposition sampled TV(sin)={total_variation:.8f} "
        f"(continuous exact value 4); reconstruction error="
        f"{jordan_reconstruction_error:.3e}"
    )
    print(f"Variation additivity residual at split: {abs(variation_whole-variation_split):.3e}")
    print(
        "BV/Riemann example: sin(x) has bounded variation and is Riemann integrable; "
        f"Dirichlet function Darboux gap remains {dirichlet_upper-dirichlet_lower:.1f}."
    )

    # Improper integrals: p-test establishes convergence for p>1 and divergence
    # at p=1. The sine integral is conditionally (not absolutely) convergent.
    p2_value = integrate.quad(lambda x: x**-2, 1.0, np.inf)[0]
    p1_partial = integrate.quad(lambda x: x**-1, 1.0, 1000.0)[0]
    sine_integral_finite = integrate.quad(
        lambda x: np.sin(x) / x if x != 0 else 1.0, 0.0, 1.0
    )[0]
    sine_integral_tail = integrate.quad(
        lambda x: 1.0 / x, 1.0, np.inf, weight="sin", wvar=1.0, limlst=200
    )[0]
    absolute_sine_partial = []
    for upper in (100.0, 500.0, 2000.0):
        boundaries = np.arange(1.0, upper, np.pi)
        partition = np.r_[1.0, boundaries, upper]
        absolute_sine_partial.append(sum(
            integrate.quad(
                lambda x: abs(np.sin(x)) / x, left, right, epsabs=1e-8
            )[0]
            for left, right in zip(partition[:-1], partition[1:])
        ))
    absolute_sine_comparison_lower_bounds = [
        2.0 / np.pi * sum(1.0 / k for k in range(2, int(upper / np.pi)))
        for upper in (100.0, 500.0, 2000.0)
    ]
    print("\nImproper-integral convergence tests:")
    print(f"Integral_1^infinity x^-2 dx (p=2, convergent): {p2_value:.10f}")
    print(f"Integral_1^1000 x^-1 dx (p=1, divergent as limit grows): {p1_partial:.10f}")
    print(
        "Integral_0^infinity sin(x)/x dx (conditionally convergent): "
        f"{sine_integral_finite+sine_integral_tail:.10f} (known value pi/2)"
    )
    print(
        "Partial integrals of |sin(x)|/x over [1,R] for R=100,500,2000: "
        f"{absolute_sine_partial}"
    )
    print(
        "Divergence comparison: on [k*pi,(k+1)*pi], the integral is at least "
        "2/((k+1)*pi); summing this harmonic lower bound diverges. "
        f"Partial lower bounds={absolute_sine_comparison_lower_bounds}"
    )

    # Complex differentiability at a point need not imply analyticity nearby:
    # f(z)=|z|^2 is differentiable at z=0 with derivative 0, but not in any
    # neighbourhood. Its real/imaginary parts do not satisfy CR away from 0.
    differentiability_steps = (1e-1, 1e-2, 1e-3, 1e-4)
    point_derivative_quotients = [
        abs((abs(complex(h, 0.0)) ** 2) / complex(h, 0.0))
        for h in differentiability_steps
    ]
    conjugate_cr_residual = cauchy_riemann_error(
        lambda x, y: x + 0.0 * y, lambda x, y: 0.0 * x - y, 0.5, 0.2
    )
    analytic_cr_residual = cauchy_riemann_error(
        lambda x, y: np.exp(x) * np.cos(y),
        lambda x, y: np.exp(x) * np.sin(y),
        0.5,
        0.2,
    )
    print("\nComplex differentiability versus analyticity:")
    print(
        "For f(z)=|z|^2 at z=0, difference quotients along arbitrary h tend to 0; "
        f"tested magnitudes={point_derivative_quotients}."
    )
    point_cr_error = cauchy_riemann_error(
        lambda x, y: x**2 + y**2,
        lambda x, y: 0.0 * x + 0.0 * y,
        0.0,
        0.0,
    )
    nearby_cr_error = cauchy_riemann_error(
        lambda x, y: x**2 + y**2,
        lambda x, y: 0.0 * x + 0.0 * y,
        0.5,
        0.2,
    )
    print(
        "For the geometric series, the original domain is |z|<1 (radius 1); "
        "the continuation 1/(1-z) is defined on C except at the singularity z=1. "
        "Continuation cannot cross or remove that pole."
    )
    print(
        "The function |z|^2 is differentiable at 0 but not analytic in any "
        "neighbourhood of 0; its CR residual is "
        f"{point_cr_error:.3e} at 0 and {nearby_cr_error:.3f} at (0.5,0.2). "
        f"Conjugation away from 0 has residual {conjugate_cr_residual:.3f}."
    )
    print(
        "Analytic exp(z) Cauchy-Riemann residual at (0.5,0.2): "
        f"{analytic_cr_residual:.3e}"
    )
    print(
        "Sufficient condition: if first partial derivatives of u and v are continuous "
        "near a point and satisfy CR there, then u+iv is complex differentiable "
        "there; on an open region this yields analyticity."
    )

    # Cauchy's theorem/formula are numerically demonstrated on the unit circle.
    theta = np.linspace(0.0, 2.0 * np.pi, 20001)
    z_circle = np.exp(1j * theta)
    dz = np.diff(z_circle)
    z_mid = 0.5 * (z_circle[:-1] + z_circle[1:])
    cauchy_integral = np.sum(np.exp(z_mid) * dz)
    cauchy_formula = np.sum((np.exp(z_mid) / z_mid) * dz) / (2j * np.pi)
    print("\nComplex contour-integral checks on the unit circle:")
    print(f"Integral exp(z) dz (should be 0): {cauchy_integral:.3e}")
    print(
        "(1/(2pi i)) integral exp(z)/z dz (should equal exp(0)=1): "
        f"{cauchy_formula:.10f}"
    )

    # For u=Re(z^2)=x^2-y^2, the average on a radius-r circle equals u(0)=0.
    radius = 0.8
    harmonic_on_circle = (radius*np.cos(theta))**2 - (radius*np.sin(theta))**2
    mean_value = float(np.trapezoid(harmonic_on_circle, theta) / (2 * np.pi))
    print(
        f"Harmonic mean-value example: circle average={mean_value:.3e}, "
        "center value=0; maximum/minimum of a nonconstant harmonic function "
        "on a bounded domain occur on its boundary."
    )
    print(
        "Black-Scholes PDE is parabolic, not Laplace's equation. A change to "
        "log-price and time, followed by discounting and scaling, transforms it "
        "to the heat equation; harmonicity is not asserted for option prices."
    )

    validations = {
        "quadrature_errors_decrease": bool(
            np.all(np.diff(convergence["Trapezoid_Error"]) < 0)
            and np.all(np.diff(convergence["Simpson_Error"]) < 0)
        ),
        "darboux_gap_decreases": bool(np.all(np.diff(darboux["Gap"]) < 0)),
        "continuous_cashflow_integral": abs(pv_numeric - pv_exact) < 1e-10,
        "variation_additivity": abs(variation_whole - variation_split) < 1e-12,
        "jordan_reconstruction": jordan_reconstruction_error < 1e-12,
        "improper_p_integral": abs(p2_value - 1.0) < 1e-9,
        "p_test_divergence_example": abs(p1_partial - np.log(1000.0)) < 1e-9,
        "absolute_convergence_comparison": bool(
            np.all(np.diff(absolute_sine_comparison_lower_bounds) > 0)
        ),
        "analytic_cauchy_riemann": analytic_cr_residual < 1e-7,
        "pointwise_cr_not_neighbourhood": (
            point_cr_error < 1e-7 and nearby_cr_error > 0.1
        ),
        "cauchy_theorem": abs(cauchy_integral) < 1e-5,
        "cauchy_formula": abs(cauchy_formula - 1.0) < 1e-5,
        "harmonic_mean_value": abs(mean_value) < 1e-10,
    }
    validations["conditional_sine_integral"] = abs(
        sine_integral_finite + sine_integral_tail - np.pi / 2
    ) < 1e-7
    validations["absolute_sine_integral_growth"] = bool(
        np.all(np.diff(absolute_sine_partial) > 0)
    )
    return convergence, darboux, qv, validations
