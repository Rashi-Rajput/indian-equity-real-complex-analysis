#!/usr/bin/env Rscript
# Historical market-data analysis and mathematical finance demonstrations.
# Dependency: quantmod (install.packages("quantmod") if needed).

options(stringsAsFactors = FALSE, digits = 8)

tickers <- c("RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "^NSEI")
trading_days <- 252
risk_free_rate <- 0.06
confidence_levels <- c(0.95, 0.99)

if (!requireNamespace("quantmod", quietly = TRUE)) {
  stop(
    'Package "quantmod" is not installed for this R installation. Run ',
    'install.packages("quantmod") in the R Console, then restart R.'
  )
}

if (!requireNamespace("zoo", quietly = TRUE)) {
  stop(
    'Package "zoo" is missing. Run install.packages("zoo") in the R Console.'
  )
}

# Data retrieval, cleaning, and summary
end_date <- Sys.Date()
start_date <- end_date - 365.25 * 5
price_series <- list()
missing_counts <- numeric(length(tickers))
names(missing_counts) <- tickers

for (ticker in tickers) {
  downloaded <- tryCatch(
    quantmod::getSymbols(
      ticker,
      src = "yahoo",
      from = start_date,
      to = end_date + 1,
      auto.assign = FALSE,
      warnings = FALSE
    ),
    error = function(e) stop(sprintf("Download failed for %s: %s", ticker, e$message))
  )

  adjusted <- tryCatch(quantmod::Ad(downloaded), error = function(e) NULL)
  close_price <- tryCatch(quantmod::Cl(downloaded), error = function(e) NULL)

  has_adjusted <- !is.null(adjusted) &&
    length(adjusted) > 0 &&
    any(is.finite(as.numeric(adjusted)))

  selected <- if (has_adjusted) {
    adjusted
  } else {
    if (is.null(close_price) || length(close_price) == 0) {
      stop(sprintf("No usable Adjusted Close or Close prices for %s.", ticker))
    }
    warning(sprintf("Adjusted Close unavailable for %s; using Close.", ticker))
    close_price
  }

  values <- as.numeric(selected)
  dates <- as.Date(zoo::index(selected))
  missing_counts[ticker] <- sum(!is.finite(values))
  valid <- is.finite(values) & values > 0 & !is.na(dates)
  series <- values[valid]
  names(series) <- as.character(dates[valid])

  if (length(series) < 3) {
    stop(sprintf(
      "Only %d valid price observations were downloaded for %s. Check the ticker, date range, and Yahoo Finance connection.",
      length(series), ticker
    ))
  }

  price_series[[ticker]] <- series
}

summary_table <- do.call(rbind, lapply(tickers, function(ticker) {
  x <- price_series[[ticker]]
  data.frame(
    Instrument = ticker,
    First_Date = names(x)[1],
    Last_Date = names(x)[length(x)],
    Observations = length(x),
    Missing_Rows_Dropped = unname(missing_counts[ticker]),
    First_Price = unname(x[1]),
    Last_Price = unname(x[length(x)]),
    Mean_Price = mean(x),
    Median_Price = median(x),
    Minimum = min(x),
    Maximum = max(x)
  )
}))
rownames(summary_table) <- summary_table$Instrument
cat("Dataset summary (each instrument uses available observations):\n")
print(summary_table, row.names = FALSE)

# Simple and log returns are aligned to each instrument's actual observed dates.
simple_returns <- list()
log_returns <- list()
identity_errors <- numeric(length(tickers))
names(identity_errors) <- tickers

for (ticker in tickers) {
  p <- price_series[[ticker]]
  simple_returns[[ticker]] <- diff(p) / head(p, -1)
  log_returns[[ticker]] <- diff(log(p))
  names(simple_returns[[ticker]]) <- names(p)[-1]
  names(log_returns[[ticker]]) <- names(p)[-1]
  identity_errors[ticker] <- abs(sum(log_returns[[ticker]]) - log(tail(p, 1) / head(p, 1)))
}

cat("\nCumulative log-return identity errors:\n")
print(identity_errors)
stopifnot(max(identity_errors) < 1e-10)

# Historical and normal VaR/ES are positive loss fractions.
risk_measures <- function(x, confidence) {
  alpha <- 1 - confidence
  cutoff <- as.numeric(stats::quantile(x, probs = alpha, names = FALSE, type = 7))
  tail <- x[x <= cutoff]
  mu <- mean(x)
  sigma <- stats::sd(x)
  z <- stats::qnorm(alpha)
  c(
    Historical_VaR = -cutoff,
    Historical_ES = -mean(tail),
    Normal_VaR = -(mu + sigma * z),
    Normal_ES = -(mu - sigma * stats::dnorm(z) / alpha)
  )
}

comparison_rows <- lapply(tickers, function(ticker) {
  r <- simple_returns[[ticker]]
  lr <- log_returns[[ticker]]
  daily_volatility <- stats::sd(lr)
  row <- c(
    Mean_Simple_Daily = mean(r),
    Mean_Log_Daily = mean(lr),
    Cumulative_Simple = prod(1 + r) - 1,
    Cumulative_Log = sum(lr),
    Daily_Volatility = daily_volatility,
    Annualized_Volatility = daily_volatility * sqrt(trading_days)
  )
  for (confidence in confidence_levels) {
    suffix <- as.integer(confidence * 100)
    row <- c(row, setNames(
      risk_measures(r, confidence),
      paste0(names(risk_measures(r, confidence)), "_", suffix)
    ))
  }
  row
})
comparison <- as.data.frame(do.call(rbind, comparison_rows))
rownames(comparison) <- tickers
cat("\nComparative return and risk summary:\n")
print(comparison)

# Topic 1: contiguous increasing/decreasing runs in sampled daily price data.
monotonic_runs <- function(x) {
  changes <- sign(diff(unname(x)))
  if (!length(changes) || all(changes == 0)) {
    return(data.frame(
      Direction = character(),
      Start = character(),
      End = character(),
      Price_Changes = integer()
    ))
  }

  starts <- c(1, which(changes[-1] != changes[-length(changes)]) + 1)
  ends <- c(starts[-1] - 1, length(changes))

  records <- lapply(seq_along(starts), function(i) {
    direction <- changes[starts[i]]
    if (direction == 0) return(NULL)

    data.frame(
      Direction = if (direction > 0) "increasing" else "decreasing",
      Start = names(x)[starts[i]],
      End = names(x)[ends[i] + 1],
      Price_Changes = ends[i] - starts[i] + 1
    )
  })

  records <- Filter(Negate(is.null), records)
  if (!length(records)) {
    return(data.frame(
      Direction = character(),
      Start = character(),
      End = character(),
      Price_Changes = integer()
    ))
  }
  do.call(rbind, records)
}

monotonic <- lapply(price_series, monotonic_runs)
names(monotonic) <- tickers
cat("\nMonotonic run counts and first observed runs:\n")
for (ticker in tickers) {
  runs <- monotonic[[ticker]]
  if (is.null(runs) || !nrow(runs)) {
    cat(ticker, ": no non-flat runs\n")
    next
  }
  counts <- table(factor(runs$Direction, levels = c("increasing", "decreasing")))
  cat(ticker, ": increasing =", counts[1], "decreasing =", counts[2], "\n")
  print(utils::head(runs, 8), row.names = FALSE)
}

# Topic 2: total variation of the sampled path, normalized variation,
# variation in log prices, and displacement divided by total variation.
variation <- do.call(rbind, lapply(tickers, function(ticker) {
  p <- unname(price_series[[ticker]])
  total <- sum(abs(diff(p)))
  data.frame(
    Instrument = ticker,
    Total_Variation = total,
    Normalized_Variation = total / p[1],
    Log_Price_Variation = sum(abs(diff(log(p)))),
    Path_Efficiency = if (total > 0) abs(tail(p, 1) - p[1]) / total else NA_real_
  )
}))
rownames(variation) <- variation$Instrument
cat("\nSampled bounded-variation measures:\n")
print(variation, row.names = FALSE)

# Topic 3: trapezoidal and composite Simpson 1/3 quadrature.
trapezoid_rule <- function(f, a, b, n = 1000) {
  x <- seq(a, b, length.out = n + 1)
  y <- f(x)
  h <- (b - a) / n
  h * (y[1] / 2 + sum(y[2:n]) + y[n + 1] / 2)
}

simpson_rule <- function(f, a, b, n = 1000) {
  if (n < 2) stop("n must be at least 2")
  if (n %% 2 != 0) n <- n + 1
  x <- seq(a, b, length.out = n + 1)
  y <- f(x)
  h <- (b - a) / n
  even_indexed_interior <- if (n >= 4) seq(3, n - 1, by = 2) else integer(0)
  h / 3 * (y[1] + y[n + 1] +
             4 * sum(y[seq(2, n, by = 2)]) +
             2 * sum(y[even_indexed_interior]))
}

quadrature_functions <- list(
  list(name = "x^2", f = function(x) x^2, a = 0, b = 2, exact = 8 / 3),
  list(name = "sin(x)", f = sin, a = 0, b = pi, exact = 2),
  list(name = "exp(-x)", f = function(x) exp(-x), a = 0, b = 2, exact = 1 - exp(-2))
)
quadrature <- do.call(rbind, lapply(quadrature_functions, function(item) {
  trap <- trapezoid_rule(item$f, item$a, item$b)
  simp <- simpson_rule(item$f, item$a, item$b)
  data.frame(
    Function = item$name, Analytical = item$exact,
    Trapezoidal = trap, Simpson = simp,
    Trap_Abs_Error = abs(trap - item$exact),
    Simpson_Abs_Error = abs(simp - item$exact)
  )
}))
cat("\nRiemann quadrature accuracy:\n")
print(quadrature, row.names = FALSE)

# Topic 4: improper integrals, probability density mass, and tail/risk calculations.
normal_mass <- integrate(stats::dnorm, lower = -Inf, upper = Inf)$value
exponential_mass <- integrate(stats::dexp, lower = 0, upper = Inf)$value
pareto_density <- function(x, shape = 3) ifelse(x >= 1, shape / x^(shape + 1), 0)
pareto_tail <- integrate(pareto_density, lower = 2, upper = Inf)$value
cat("\nImproper-integral checks:\n")
cat("Normal density mass:", normal_mass, "\n")
cat("Exponential density mass:", exponential_mass, "\n")
cat("Pareto(shape=3) P(X > 2):", pareto_tail, "\n")
stopifnot(abs(normal_mass - 1) < 1e-9, abs(exponential_mass - 1) < 1e-9)

# Topics 5-8: complex analysis examples are separate from discrete market data.
z <- c(0.3 + 0.4i, -0.5 + 0.8i)
cat("\nComplex-function examples:\n")
print(list(z_squared = z^2, exponential = exp(z), sine = sin(z), conjugate = Conj(z)))

cauchy_riemann_error <- function(u, v, x, y, h = 1e-5) {
  ux <- (u(x + h, y) - u(x - h, y)) / (2 * h)
  vy <- (v(x, y + h) - v(x, y - h)) / (2 * h)
  uy <- (u(x, y + h) - u(x, y - h)) / (2 * h)
  vx <- (v(x + h, y) - v(x - h, y)) / (2 * h)
  abs(ux - vy) + abs(uy + vx)
}

u_exp <- function(x, y) exp(x) * cos(y)
v_exp <- function(x, y) exp(x) * sin(y)
u_z2 <- function(x, y) x^2 - y^2
v_z2 <- function(x, y) 2 * x * y
u_conjugate <- function(x, y) x
v_conjugate <- function(x, y) -y
cr_errors <- c(
  exp_z = cauchy_riemann_error(u_exp, v_exp, 0.4, -0.3),
  z_squared = cauchy_riemann_error(u_z2, v_z2, 0.4, -0.3),
  conjugate_z = cauchy_riemann_error(u_conjugate, v_conjugate, 0.4, -0.3)
)
cat("Cauchy-Riemann residuals:\n")
print(cr_errors)
stopifnot(cr_errors["exp_z"] < 1e-7, cr_errors["z_squared"] < 1e-7,
          cr_errors["conjugate_z"] > 1)

laplacian <- function(f, x, y, h = 1e-4) {
  (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / h^2 +
    (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / h^2
}
u_harmonic <- function(x, y) x^2 - y^2
v_conjugate <- function(x, y) 2 * x * y
cat("\nFinite-difference Laplacians (zero indicates harmonicity):\n")
print(c(
  `x^2-y^2` = laplacian(u_harmonic, 0.4, -0.3),
  `2xy` = laplacian(v_conjugate, 0.4, -0.3),
  `x^3 (not harmonic)` = laplacian(function(x, y) x^3, 0.4, -0.3)
))

# The geometric power series has radius one. Its rational expression continues
# analytically to the complex plane except at its pole z=1.
z_inside <- 0.5 + 0.2i
geometric_sum <- sum(z_inside^(0:499))
geometric_formula <- 1 / (1 - z_inside)
continued_value <- 1 / (1 - (1.5 + 0.2i))
exp_series <- sum(vapply(
  0:49,
  function(n) (0.7 + 0.4i)^n / factorial(n),
  complex(1)
))
cat("\nAnalytic continuation:\n")
cat("Geometric-series error inside disk:", Mod(geometric_sum - geometric_formula), "\n")
cat("Continued geometric function at 1.5+0.2i:", continued_value, "\n")
cat("Entire exponential-series error:", Mod(exp_series - exp(0.7 + 0.4i)), "\n")

# Characteristic functions: empirical daily NSEI log returns versus fitted normal.
empirical_cf <- function(samples, u) {
  vapply(u, function(freq) mean(exp(1i * freq * samples)), complex(1))
}
normal_cf <- function(u, mu, sigma) exp(1i * u * mu - 0.5 * sigma^2 * u^2)
nsei_returns <- as.numeric(log_returns[["^NSEI"]])
nsei_returns <- nsei_returns[is.finite(nsei_returns)]

if (length(nsei_returns) < 30) {
  stop(
    "Not enough valid ^NSEI returns for characteristic-function analysis. ",
    "Check whether Yahoo Finance returned sufficient NIFTY 50 data."
  )
}

mu <- mean(nsei_returns)
sigma <- stats::sd(nsei_returns)

if (!is.finite(sigma) || sigma <= 0) {
  stop("The ^NSEI return volatility is zero or invalid; Fourier analysis cannot continue.")
}
frequency <- seq(-20, 20, length.out = 401)
empirical <- empirical_cf(nsei_returns, frequency)
gaussian <- normal_cf(frequency, mu, sigma)
cf_comparison <- data.frame(
  Frequency = frequency,
  Empirical_Real = Re(empirical),
  Normal_Real = Re(gaussian),
  Empirical_Imag = Im(empirical),
  Normal_Imag = Im(gaussian)
)
cat("\nCharacteristic-function comparison at selected frequencies:\n")
print(cf_comparison[seq(1, nrow(cf_comparison), by = 100), ], row.names = FALSE)

# Fourier inversion of normal density and Gil-Pelaez inversion of normal CDF.
normal_fourier_density <- function(x, mu, sigma) {
  integrand <- function(u) Re(exp(-1i * u * x) * normal_cf(u, mu, sigma))
  integrate(integrand, lower = 0, upper = Inf, subdivisions = 1000,
            rel.tol = 1e-10)$value / pi
}

gil_pelaez_normal_cdf <- function(x, mu, sigma) {
  integrand <- function(u) {
    out <- numeric(length(u))
    at_zero <- u == 0
    out[at_zero] <- mu - x
    out[!at_zero] <- Im(exp(-1i * u[!at_zero] * x) *
                          normal_cf(u[!at_zero], mu, sigma)) / u[!at_zero]
    out
  }
  0.5 - integrate(integrand, lower = 0, upper = Inf,
                  subdivisions = 1000, rel.tol = 1e-9)$value / pi
}

density_points <- seq(mu - 3 * sigma, mu + 3 * sigma, length.out = 7)
inversion <- do.call(rbind, lapply(density_points, function(x) {
  data.frame(
    x = x,
    Fourier_PDF = normal_fourier_density(x, mu, sigma),
    Exact_PDF = stats::dnorm(x, mu, sigma),
    Gil_Pelaez_CDF = gil_pelaez_normal_cdf(x, mu, sigma),
    Exact_CDF = stats::pnorm(x, mu, sigma)
  )
}))
cat("\nFourier density and Gil-Pelaez CDF inversion validation:\n")
print(inversion, row.names = FALSE)
stopifnot(max(abs(inversion$Fourier_PDF - inversion$Exact_PDF)) < 1e-7,
          max(abs(inversion$Gil_Pelaez_CDF - inversion$Exact_CDF)) < 1e-7)

# Black-Scholes pricing assumes European exercise and constant volatility/rate.
black_scholes <- function(spot, strike, rate, volatility, maturity,
                          option = c("call", "put")) {
  option <- match.arg(option)
  if (any(c(spot, strike, volatility, maturity) <= 0)) {
    stop("spot, strike, volatility, and maturity must be positive")
  }
  d1 <- (log(spot / strike) + (rate + 0.5 * volatility^2) * maturity) /
    (volatility * sqrt(maturity))
  d2 <- d1 - volatility * sqrt(maturity)
  if (option == "call") {
    spot * pnorm(d1) - strike * exp(-rate * maturity) * pnorm(d2)
  } else {
    strike * exp(-rate * maturity) * pnorm(-d2) - spot * pnorm(-d1)
  }
}

black_scholes_greeks <- function(spot, strike, rate, volatility, maturity,
                                 option = c("call", "put")) {
  option <- match.arg(option)
  d1 <- (log(spot / strike) + (rate + 0.5 * volatility^2) * maturity) /
    (volatility * sqrt(maturity))
  d2 <- d1 - volatility * sqrt(maturity)
  delta <- if (option == "call") pnorm(d1) else pnorm(d1) - 1
  gamma <- dnorm(d1) / (spot * volatility * sqrt(maturity))
  vega <- spot * dnorm(d1) * sqrt(maturity)
  if (option == "call") {
    theta <- -spot * dnorm(d1) * volatility / (2 * sqrt(maturity)) -
      rate * strike * exp(-rate * maturity) * pnorm(d2)
    rho <- strike * maturity * exp(-rate * maturity) * pnorm(d2)
  } else {
    theta <- -spot * dnorm(d1) * volatility / (2 * sqrt(maturity)) +
      rate * strike * exp(-rate * maturity) * pnorm(-d2)
    rho <- -strike * maturity * exp(-rate * maturity) * pnorm(-d2)
  }
  c(Delta = delta, Gamma = gamma, Theta = theta, Vega = vega, Rho = rho)
}

bs_log_price_cf <- function(u, spot, rate, volatility, maturity) {
  mean <- log(spot) + (rate - 0.5 * volatility^2) * maturity
  exp(1i * u * mean - 0.5 * volatility^2 * maturity * u^2)
}

# Carr-Madan damped Fourier integral evaluated with the trapezoidal rule.
carr_madan_call <- function(spot, strike, rate, volatility, maturity,
                            damping = 1.5, upper = 150, intervals = 30000) {
  k <- log(strike)
  u <- seq(0, upper, length.out = intervals + 1)
  shifted <- u - 1i * (damping + 1)
  phi <- bs_log_price_cf(shifted, spot, rate, volatility, maturity)
  denominator <- damping^2 + damping - u^2 + 1i * (2 * damping + 1) * u
  psi <- exp(-rate * maturity) * phi / denominator
  integrand <- exp(-1i * u * k) * psi
  exp(-damping * k) / pi * Re(sum(diff(u) * (head(integrand, -1) +
                                                tail(integrand, -1)) / 2))
}

spot <- unname(tail(price_series[["^NSEI"]], 1))
volatility <- unname(comparison["^NSEI", "Annualized_Volatility"])

if (!is.finite(spot) || spot <= 0) {
  stop("Invalid NIFTY 50 spot price. Check the downloaded ^NSEI data.")
}
if (!is.finite(volatility) || volatility <= 0) {
  stop("Invalid annualized volatility. Check the downloaded ^NSEI return data.")
}

strike <- spot
maturity <- 1
call <- black_scholes(spot, strike, risk_free_rate, volatility, maturity, "call")
put <- black_scholes(spot, strike, risk_free_rate, volatility, maturity, "put")
parity_residual <- call - put - (spot - strike * exp(-risk_free_rate * maturity))
cat("\nBlack-Scholes example (NSEI proxy spot, ATM, 1 year):\n")
cat("Spot:", spot, "Strike:", strike, "Volatility:", volatility, "\n")
cat("Call:", call, "Put:", put, "Put-call parity residual:", parity_residual, "\n")
cat("Call Greeks:\n")
print(black_scholes_greeks(spot, strike, risk_free_rate, volatility, maturity, "call"))
cat("Put Greeks:\n")
print(black_scholes_greeks(spot, strike, risk_free_rate, volatility, maturity, "put"))

cm_call <- carr_madan_call(spot, strike, risk_free_rate, volatility, maturity)
cat("Carr-Madan call:", cm_call, "Black-Scholes call:", call,
    "Absolute error:", abs(cm_call - call), "\n")
stopifnot(abs(parity_residual) < 1e-8 * spot,
          abs(cm_call - call) / spot < 1e-6)

# Visualizations
normalized <- lapply(price_series, function(x) x / x[1])
all_dates <- sort(unique(unlist(lapply(normalized, names))))
all_normalized_values <- unlist(normalized, use.names = FALSE)

if (!length(all_dates) || !length(all_normalized_values) ||
    any(!is.finite(all_normalized_values))) {
  stop("The normalized price data are empty or non-finite; plots cannot be created.")
}

plot(
  NA, xlim = range(as.Date(all_dates)), ylim = range(all_normalized_values),
  xlab = "Date", ylab = "Normalized adjusted close",
  main = "Adjusted close normalized to the first observation"
)
colors <- grDevices::rainbow(length(tickers))
for (i in seq_along(tickers)) {
  x <- normalized[[tickers[i]]]
  lines(as.Date(names(x)), unname(x), col = colors[i])
}
legend("topleft", legend = tickers, col = colors, lty = 1, cex = 0.8)

plot(
  NA, xlim = range(as.Date(unlist(lapply(log_returns, names)))),
  ylim = range(unlist(log_returns)), xlab = "Date", ylab = "Daily log return",
  main = "Daily log returns"
)
for (i in seq_along(tickers)) {
  r <- log_returns[[tickers[i]]]
  points(as.Date(names(r)), unname(r), col = colors[i], pch = 16, cex = 0.35)
}
abline(h = 0, col = "gray40")
legend("topright", legend = tickers, col = colors, pch = 16, cex = 0.8)

barplot(
  comparison$Annualized_Volatility,
  names.arg = rownames(comparison), las = 2,
  main = "Annualized historical volatility", ylab = "Annualized volatility"
)

barplot(
  t(as.matrix(comparison[, c(
    "Historical_VaR_95", "Historical_ES_95", "Normal_VaR_95", "Normal_ES_95"
  ), drop = FALSE])),
  beside = TRUE, names.arg = rownames(comparison), las = 2,
  legend.text = TRUE, main = "Daily 95% loss VaR and Expected Shortfall",
  ylab = "Loss fraction"
)

par(mfrow = c(1, 2))
plot(frequency, Re(empirical), type = "l", xlab = "Frequency",
     ylab = "Real component", main = "Characteristic function: real")
lines(frequency, Re(gaussian), lty = 2, col = "red")
legend("topright", legend = c("Empirical", "Normal"), lty = c(1, 2),
       col = c("black", "red"), cex = 0.8)
plot(frequency, Im(empirical), type = "l", xlab = "Frequency",
     ylab = "Imaginary component", main = "Characteristic function: imaginary")
lines(frequency, Im(gaussian), lty = 2, col = "red")
legend("topright", legend = c("Empirical", "Normal"), lty = c(1, 2),
       col = c("black", "red"), cex = 0.8)
par(mfrow = c(1, 1))

# Sensitivity to spot and volatility, holding other model inputs fixed.
spots <- seq(0.7 * spot, 1.3 * spot, length.out = 100)
volatilities <- seq(max(0.05, 0.5 * volatility), 1.5 * volatility, length.out = 100)
plot(spots, vapply(spots, black_scholes, numeric(1), strike = strike,
                   rate = risk_free_rate, volatility = volatility,
                   maturity = maturity, option = "call"),
     type = "l", xlab = "Spot", ylab = "Option value", main = "Value vs. spot")
lines(spots, vapply(spots, black_scholes, numeric(1), strike = strike,
                    rate = risk_free_rate, volatility = volatility,
                    maturity = maturity, option = "put"), col = "red")
legend("topleft", legend = c("Call", "Put"), col = c("black", "red"), lty = 1)

plot(volatilities, vapply(volatilities, black_scholes, numeric(1), spot = spot,
                          strike = strike, rate = risk_free_rate,
                          maturity = maturity, option = "call"),
     type = "l", xlab = "Volatility", ylab = "Option value",
     main = "Value vs. volatility")
lines(volatilities, vapply(volatilities, black_scholes, numeric(1), spot = spot,
                           strike = strike, rate = risk_free_rate,
                           maturity = maturity, option = "put"), col = "red")
legend("topleft", legend = c("Call", "Put"), col = c("black", "red"), lty = 1)
