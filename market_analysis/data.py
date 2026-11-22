"""Market-data download, cleaning, summaries, and return calculations."""

from __future__ import annotations

import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TICKERS


OHLCV_FIELDS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
CACHE_PATH = Path(__file__).resolve().parents[1] / "outputs/cache/yahoo_ohlcv.csv"
USER_CSV_PATH = Path(__file__).resolve().parents[1] / "data/market_ohlcv.csv"


def _offline_fallback(start: pd.Timestamp) -> tuple[pd.DataFrame | None, str | None]:
    """Prefer a supplied real-data CSV, then a recent full OHLCV cache."""
    csv_path = Path(os.environ.get("MARKET_DATA_CSV", USER_CSV_PATH)).expanduser()
    if csv_path.exists():
        try:
            csv_data = _load_long_format_csv(csv_path)
            csv_data = csv_data.loc[csv_data.index >= start]
            if not csv_data.empty:
                return csv_data, f"CSV {csv_path}"
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            warnings.warn(f"Ignoring invalid market-data CSV {csv_path}: {exc}", stacklevel=2)

    cached = _load_cached_data(CACHE_PATH, start)
    if cached is not None:
        return cached, f"recent cache {CACHE_PATH}"
    return None, None


def _normalize_yahoo_frame(
    raw: pd.DataFrame,
    tickers: list[str],
) -> pd.DataFrame:
    """Normalize yfinance's varying field/ticker column layouts."""
    if isinstance(raw.columns, pd.MultiIndex):
        field_level = next(
            (
                level for level in range(raw.columns.nlevels)
                if set(OHLCV_FIELDS).intersection(raw.columns.get_level_values(level))
            ),
            None,
        )
        if field_level is None:
            raise RuntimeError(
                f"Unsupported Yahoo Finance columns: {raw.columns.tolist()[:8]}"
            )
        available_fields = [
            field for field in OHLCV_FIELDS
            if field in raw.columns.get_level_values(field_level)
        ]
        field_frames = {}
        for field in available_fields:
            frame = raw.xs(field, axis=1, level=field_level, drop_level=True)
            if isinstance(frame, pd.Series):
                frame = frame.to_frame(name=tickers[0])
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(-1)
            field_frames[field] = frame.reindex(columns=tickers)
        normalized = pd.concat(field_frames, axis=1)
    else:
        if len(tickers) != 1:
            raise RuntimeError(
                "Yahoo Finance returned single-level columns for multiple tickers."
            )
        available_fields = [field for field in OHLCV_FIELDS if field in raw.columns]
        if not available_fields:
            raise RuntimeError("Yahoo Finance returned no OHLCV columns.")
        normalized = pd.concat(
            {field: raw[field].rename(tickers[0]) for field in available_fields},
            axis=1,
        )
    normalized.columns.names = ["Field", "Ticker"]
    return normalized


def _load_cached_data(cache_path: Path, start: pd.Timestamp) -> pd.DataFrame | None:
    """Load a complete recent-enough OHLCV cache after a network failure."""
    if not cache_path.exists():
        return None
    try:
        cached = pd.read_csv(cache_path, header=[0, 1], index_col=0, parse_dates=True)
        cached.columns.names = ["Field", "Ticker"]
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        warnings.warn(f"Ignoring unreadable market-data cache: {exc}", stacklevel=2)
        return None

    if not isinstance(cached.columns, pd.MultiIndex):
        return None
    tickers_present = set(cached.columns.get_level_values("Ticker"))
    if not set(TICKERS).issubset(tickers_present):
        return None
    cached = cached.loc[~cached.index.duplicated(keep="last")].sort_index()
    cached = cached.loc[cached.index >= start]
    if cached.empty:
        return None

    # Accept at most a week's staleness (weekends and exchange holidays included).
    latest = pd.Timestamp(cached.index.max()).tz_localize(None).normalize()
    today = pd.Timestamp.today().normalize()
    if (today - latest).days > 7:
        return None
    return cached


def _load_long_format_csv(csv_path: Path) -> pd.DataFrame:
    """Load real OHLCV observations from Date,Ticker,field CSV columns."""
    data = pd.read_csv(csv_path)
    normalized_names = {str(column).strip().lower(): column for column in data.columns}
    date_column = normalized_names.get("date")
    ticker_column = normalized_names.get("ticker")
    if date_column is None or ticker_column is None:
        raise ValueError(
            "CSV must be long format with Date and Ticker columns, plus available "
            "Open, High, Low, Close, Adj Close, and Volume columns."
        )

    field_columns = {
        field: normalized_names.get(field.lower())
        for field in OHLCV_FIELDS
        if normalized_names.get(field.lower()) is not None
    }
    if not ({"Adj Close", "Close"} & set(field_columns)):
        raise ValueError("CSV must include at least Adj Close or Close.")

    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data[ticker_column] = data[ticker_column].astype(str).str.strip()
    data = data.loc[
        data[date_column].notna() & data[ticker_column].isin(TICKERS)
    ].copy()
    if data.empty:
        raise ValueError("CSV contains no valid rows for the required instruments.")

    values = {}
    for field, column in field_columns.items():
        data[column] = pd.to_numeric(data[column], errors="coerce")
        values[field] = data.pivot_table(
            index=date_column, columns=ticker_column, values=column, aggfunc="last"
        ).reindex(columns=TICKERS)

    normalized = pd.concat(values, axis=1).sort_index()
    normalized.columns.names = ["Field", "Ticker"]
    missing = [
        ticker for ticker in TICKERS
        if not any(
            (field, ticker) in normalized.columns
            and normalized[(field, ticker)].notna().any()
            for field in ("Adj Close", "Close")
        )
    ]
    if missing:
        raise ValueError(f"CSV has no valid close prices for: {missing}")
    return normalized


def _download_yahoo(yf, tickers: list[str], arguments: dict) -> pd.DataFrame | None:
    """Call yfinance without flooding the console with duplicate logger output."""
    logger = logging.getLogger("yfinance")
    old_level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        result = yf.download(tickers, **arguments)
    finally:
        logger.setLevel(old_level)
    return result


def download_market_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return daily OHLCV data and adjusted-close-preferred return prices.

    The OHLCV frame uses (field, ticker) columns and preserves Close separately
    from Adj Close. The return-price frame prefers Adj Close per instrument,
    falling back to Close only when adjustment data is unavailable.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "Missing yfinance. Install project dependencies with "
            "`python -m pip install yfinance numpy pandas scipy matplotlib`."
        ) from exc

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=5)
    end_exclusive = end + pd.Timedelta(days=1)
    download_args = {
        "start": start.date().isoformat(),
        "end": end_exclusive.date().isoformat(),
        "auto_adjust": False,
        "actions": False,
        "progress": False,
        "group_by": "column",
        "threads": False,
        "timeout": 30,
    }

    raw: pd.DataFrame | None = None
    download_errors: list[str] = []
    # Keep attempts bounded. Persistent SSL resets or HTTP 429 responses are
    # upstream/network failures; repeated calls can worsen rate limiting.
    for attempt in range(2):
        try:
            raw = _download_yahoo(yf, TICKERS, download_args)
            if raw is not None and not raw.empty:
                break
            download_errors.append(f"bulk attempt {attempt + 1}: no data returned")
        except Exception as exc:
            download_errors.append(f"bulk attempt {attempt + 1}: {type(exc).__name__}: {exc}")
        if attempt == 0:
            time.sleep(2)

    retry_missing_tickers = raw is not None and not raw.empty
    if raw is None or raw.empty:
        warnings.warn(
            "Bulk Yahoo Finance download failed; retrying tickers sequentially.",
            stacklevel=2,
        )
        ticker_frames = []
        for ticker in TICKERS:
            ticker_raw = None
            for attempt in range(1):
                try:
                    ticker_raw = _download_yahoo(yf, [ticker], download_args)
                    if ticker_raw is not None and not ticker_raw.empty:
                        break
                    download_errors.append(
                        f"{ticker} attempt {attempt + 1}: no data returned"
                    )
                except Exception as exc:
                    download_errors.append(
                        f"{ticker} attempt {attempt + 1}: {type(exc).__name__}: {exc}"
                    )
            if ticker_raw is not None and not ticker_raw.empty:
                ticker_frames.append(_normalize_yahoo_frame(ticker_raw, [ticker]))

        if ticker_frames:
            raw = pd.concat(ticker_frames, axis=1)
            raw.columns.names = ["Field", "Ticker"]
        else:
            fallback, fallback_source = _offline_fallback(start)
            if fallback is None:
                detail = "; ".join(download_errors[-8:])
                raise RuntimeError(
                    "Yahoo Finance returned no data, and no offline data file or "
                    "recent cache was available. This is consistent with the SSL "
                    "reset/provider throttling in the download logs and cannot be "
                    "repaired by code while the provider connection is blocked. "
                    "Restore access or set "
                    "MARKET_DATA_CSV to a real long-format CSV with columns "
                    "Date,Ticker,Open,High,Low,Close,Adj Close,Volume. "
                    f"Recent download details: {detail}"
                )
            warnings.warn(
                f"Yahoo Finance unavailable; using {fallback_source}.",
                stacklevel=2,
            )
            raw = fallback

    ohlcv = _normalize_yahoo_frame(raw, TICKERS)
    available_price_fields = [
        field for field in ("Adj Close", "Close")
        if field in ohlcv.columns.get_level_values("Field")
    ]
    missing_tickers = [
        ticker for ticker in TICKERS
        if not any(
            (field, ticker) in ohlcv.columns and ohlcv[(field, ticker)].notna().any()
            for field in available_price_fields
        )
    ]
    for ticker in (missing_tickers if retry_missing_tickers else []):
        ticker_raw = None
        for attempt in range(1):
            try:
                ticker_raw = _download_yahoo(yf, [ticker], download_args)
                if ticker_raw is not None and not ticker_raw.empty:
                    break
                download_errors.append(
                    f"{ticker} fallback attempt {attempt + 1}: no data returned"
                )
            except Exception as exc:
                download_errors.append(
                    f"{ticker} fallback attempt {attempt + 1}: "
                    f"{type(exc).__name__}: {exc}"
                )
        if ticker_raw is not None and not ticker_raw.empty:
            fallback_frame = _normalize_yahoo_frame(ticker_raw, [ticker])
            ohlcv = ohlcv.combine_first(fallback_frame)

    # A provider may return a non-empty but incomplete frame. Fill gaps from
    # actual cached/user-supplied observations before declaring an instrument absent.
    missing_tickers = [
        ticker for ticker in TICKERS
        if not any(
            (field, ticker) in ohlcv.columns and ohlcv[(field, ticker)].notna().any()
            for field in ("Adj Close", "Close")
        )
    ]
    if missing_tickers:
        fallback, fallback_source = _offline_fallback(start)
        if fallback is not None:
            ohlcv = ohlcv.combine_first(fallback)
            warnings.warn(
                f"Filling incomplete Yahoo results from {fallback_source}.",
                stacklevel=2,
            )

    remaining_missing = [
        ticker for ticker in TICKERS
        if not any(
            (field, ticker) in ohlcv.columns and ohlcv[(field, ticker)].notna().any()
            for field in ("Adj Close", "Close")
        )
    ]
    if remaining_missing:
        raise RuntimeError(
            "Market data is still unavailable for "
            f"{remaining_missing} after Yahoo retries and offline fallback. "
            "No fabricated or partial substitute will be used."
        )

    ohlcv.columns.names = ["Field", "Ticker"]
    fields = set(ohlcv.columns.get_level_values("Field"))

    def field_frame(field: str) -> pd.DataFrame:
        selected = ohlcv.xs(field, axis=1, level="Field")
        if isinstance(selected, pd.Series):
            selected = selected.to_frame()
        return selected.reindex(columns=TICKERS)

    if "Close" not in fields and "Adj Close" not in fields:
        raise RuntimeError("Yahoo Finance returned neither Close nor Adjusted Close.")
    if "Adj Close" in fields:
        adjusted_prices = field_frame("Adj Close")
        prices = adjusted_prices
        if "Close" in fields:
            # Close is only a per-instrument fallback for missing adjusted values.
            prices = prices.combine_first(field_frame("Close"))
        return_price_sources = {
            ticker: (
                "Close fallback"
                if not adjusted_prices[ticker].notna().any()
                else (
                    "Adj Close with Close fallback"
                    if "Close" in fields and adjusted_prices[ticker].isna().any()
                    else "Adj Close"
                )
            )
            for ticker in TICKERS
        }
    else:
        warnings.warn("Adjusted Close unavailable; using Close.", stacklevel=2)
        prices = field_frame("Close")
        return_price_sources = {ticker: "Close fallback" for ticker in TICKERS}

    duplicate_dates_removed = int(prices.index.duplicated(keep="last").sum())
    dates_were_sorted = bool(prices.index.is_monotonic_increasing)
    prices = prices.reindex(columns=TICKERS)
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    prices.index = pd.to_datetime(prices.index)
    prices = prices.loc[~prices.index.duplicated(keep="last")].sort_index()
    ohlcv.index = pd.to_datetime(ohlcv.index)
    ohlcv = ohlcv.loc[~ohlcv.index.duplicated(keep="last")].sort_index()
    ohlcv = ohlcv.replace([np.inf, -np.inf], np.nan)
    missing_tickers = [ticker for ticker in TICKERS if prices[ticker].notna().sum() == 0]
    if missing_tickers:
        raise RuntimeError(f"No usable prices were returned for: {missing_tickers}")
    prices.attrs["duplicate_dates_removed"] = duplicate_dates_removed
    prices.attrs["dates_were_sorted"] = dates_were_sorted
    prices.attrs["return_price_sources"] = return_price_sources
    ohlcv.attrs["duplicate_dates_removed"] = duplicate_dates_removed
    ohlcv.attrs["dates_were_sorted"] = dates_were_sorted
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ohlcv.to_csv(CACHE_PATH)
    return ohlcv, prices


def download_prices() -> pd.DataFrame:
    """Backward-compatible adjusted-close-preferred price download."""
    return download_market_data()[1]


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Use complete trading dates and reject nonpositive or inadequate data."""
    clean = prices.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if clean.empty:
        raise RuntimeError("No complete observations remain after cleaning.")
    if len(clean) < 3:
        raise RuntimeError("Fewer than three complete price observations remain.")
    if (clean <= 0).any().any():
        raise ValueError("Price data contains non-positive values.")
    return clean.astype(float)


def dataset_summary(
    prices: pd.DataFrame, raw_prices: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Summarize usable data and, if supplied, pre-clean missing row counts."""
    raw = prices if raw_prices is None else raw_prices
    rows = []
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        rows.append({
            "Instrument": ticker,
            "First_Date": series.index.min().date(),
            "Last_Date": series.index.max().date(),
            "Observations": len(series),
            "Available_Observations_Pre_Alignment": int(raw[ticker].notna().sum()),
            "Missing_Rows": int(raw[ticker].isna().sum()),
            "Excluded_By_Common_Date_Alignment": (
                int(raw[ticker].notna().sum()) - len(series)
            ),
            "First_Price": float(series.iloc[0]),
            "Last_Price": float(series.iloc[-1]),
            "Mean_Price": float(series.mean()),
            "Median_Price": float(series.median()),
            "Minimum": float(series.min()),
            "Maximum": float(series.max()),
        })
    return pd.DataFrame(rows).set_index("Instrument")


def calculate_returns(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate aligned simple and log returns from adjusted prices."""
    simple = prices.pct_change(fill_method=None).dropna(how="all")
    log = prices.apply(np.log).diff().dropna(how="all")
    return simple, log


def log_return_identity_errors(
    prices: pd.DataFrame, log_returns: pd.DataFrame
) -> pd.Series:
    """Check sum(log returns) = log(final price / initial price)."""
    errors = {}
    for ticker in prices.columns:
        observed = float(log_returns[ticker].sum())
        endpoint = float(np.log(prices[ticker].iloc[-1] / prices[ticker].iloc[0]))
        errors[ticker] = abs(observed - endpoint)
    return pd.Series(errors, name="Absolute_Error")
