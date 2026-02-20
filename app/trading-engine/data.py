import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import time
import logging
from crypto_agent.core.config import settings

logger = logging.getLogger(__name__)

# ── Exchange Client Setup ─────────────────────────────────────────
_exchanges: dict[str, ccxt.Exchange] = {}

def get_exchange(exchange_id: str = "binance") -> ccxt.Exchange:
    """
    Lazy-initialised exchange client with caching.
    Reuses the same client across calls to avoid re-authentication overhead.
    """
    if exchange_id not in _exchanges:
        exchange_class = getattr(ccxt, exchange_id)
        _exchanges[exchange_id] = exchange_class({
            "apiKey": settings.binance_api_key,
            "secret": settings.binance_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",   # Use perpetual futures for funding rate data
            }
        })
    return _exchanges[exchange_id]


# ── Timeframe Helpers ─────────────────────────────────────────────
TIMEFRAME_MS = {
    "1m":  60_000,
    "5m":  300_000,
    "15m": 900_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
}

def _lookback_to_since(lookback_days: int, timeframe: str) -> int:
    """Convert lookback_days to a UNIX timestamp in milliseconds"""
    since_dt = datetime.utcnow() - timedelta(days=lookback_days)
    return int(since_dt.timestamp() * 1000)


# ── Core OHLCV Fetcher ────────────────────────────────────────────
def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    lookback_days: int = 730,
    exchange_id: str = "binance",
    include_funding_rate: bool = True,
    include_open_interest: bool = True,
) -> pd.DataFrame:
    """
    Fetch full OHLCV history for a symbol, paginating automatically
    to overcome exchange limits (Binance caps at 1500 candles per call).

    Optionally merges in funding rate and open interest data
    so these are available as features in the backtester.

    Args:
        symbol:               e.g. 'BTC/USDT'
        timeframe:            '1m', '5m', '15m', '1h', '4h', '1d'
        lookback_days:        How many days of history to fetch
        exchange_id:          ccxt exchange id, default 'binance'
        include_funding_rate: Merge in 8h funding rate data
        include_open_interest: Merge in open interest snapshots

    Returns:
        pd.DataFrame with columns:
            timestamp, open, high, low, close, volume
            + funding_rate (optional)
            + open_interest (optional)
        Index is a DatetimeIndex in UTC.
    """
    exchange = get_exchange(exchange_id)
    since = _lookback_to_since(lookback_days, timeframe)
    tf_ms = TIMEFRAME_MS.get(timeframe, 3_600_000)
    limit = 1000   # safe page size for most exchanges

    all_candles = []
    current_since = since

    logger.info(f"Fetching {symbol} {timeframe} from {datetime.utcfromtimestamp(since/1000)}")

    while True:
        try:
            candles = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=current_since,
                limit=limit,
            )
        except ccxt.RateLimitExceeded:
            logger.warning("Rate limit hit — sleeping 10s")
            time.sleep(10)
            continue
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching {symbol}: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching {symbol}: {e}")
            raise

        if not candles:
            break

        all_candles.extend(candles)

        # If we got fewer candles than limit, we've reached the present
        if len(candles) < limit:
            break

        # Advance the window
        current_since = candles[-1][0] + tf_ms
        time.sleep(exchange.rateLimit / 1000)  # respect rate limit

    if not all_candles:
        raise ValueError(f"No data returned for {symbol} {timeframe}")

    # Build DataFrame
    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Cast to float to be safe
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")

    # Optionally enrich with funding rate and open interest
    if include_funding_rate:
        try:
            df = _merge_funding_rate(df, symbol, lookback_days, exchange_id)
        except Exception as e:
            logger.warning(f"Could not fetch funding rate for {symbol}: {e}")
            df["funding_rate"] = 0.0

    if include_open_interest:
        try:
            df = _merge_open_interest(df, symbol, lookback_days, exchange_id)
        except Exception as e:
            logger.warning(f"Could not fetch open interest for {symbol}: {e}")
            df["open_interest"] = 0.0

    return df


# ── Funding Rate ──────────────────────────────────────────────────
def _merge_funding_rate(
    df: pd.DataFrame,
    symbol: str,
    lookback_days: int,
    exchange_id: str,
) -> pd.DataFrame:
    """
    Fetch historical funding rates and forward-fill onto the OHLCV index.
    Funding rates are emitted every 8h on Binance perps.
    """
    exchange = get_exchange(exchange_id)
    since = _lookback_to_since(lookback_days, "8h")

    all_rates = []
    current_since = since

    while True:
        try:
            rates = exchange.fetch_funding_rate_history(
                symbol,
                since=current_since,
                limit=1000,
            )
        except Exception:
            break

        if not rates:
            break

        all_rates.extend(rates)

        if len(rates) < 1000:
            break

        current_since = rates[-1]["timestamp"] + 1
        time.sleep(exchange.rateLimit / 1000)

    if not all_rates:
        df["funding_rate"] = 0.0
        return df

    funding_df = pd.DataFrame([
        {
            "timestamp": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
            "funding_rate": float(r["fundingRate"])
        }
        for r in all_rates
    ]).set_index("timestamp").sort_index()

    # Forward fill funding rate onto the OHLCV candle index
    df = df.join(funding_df, how="left")
    df["funding_rate"] = df["funding_rate"].ffill().fillna(0.0)

    return df


# ── Open Interest ─────────────────────────────────────────────────
def _merge_open_interest(
    df: pd.DataFrame,
    symbol: str,
    lookback_days: int,
    exchange_id: str,
) -> pd.DataFrame:
    """
    Fetch open interest history and merge onto OHLCV index.
    Availability depends on exchange — Binance provides this for perps.
    """
    exchange = get_exchange(exchange_id)

    # Binance-specific OI history endpoint
    if exchange_id != "binance":
        df["open_interest"] = 0.0
        return df

    since = _lookback_to_since(lookback_days, "1h")
    all_oi = []
    current_since = since

    while True:
        try:
            # ccxt unified method
            oi_data = exchange.fetch_open_interest_history(
                symbol,
                timeframe="1h",
                since=current_since,
                limit=500,
            )
        except Exception:
            break

        if not oi_data:
            break

        all_oi.extend(oi_data)

        if len(oi_data) < 500:
            break

        current_since = oi_data[-1]["timestamp"] + 3_600_000
        time.sleep(exchange.rateLimit / 1000)

    if not all_oi:
        df["open_interest"] = 0.0
        return df

    oi_df = pd.DataFrame([
        {
            "timestamp": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
            "open_interest": float(r["openInterestAmount"])
        }
        for r in all_oi
    ]).set_index("timestamp").sort_index()

    df = df.join(oi_df, how="left")
    df["open_interest"] = df["open_interest"].ffill().fillna(0.0)

    return df


# ── Multi-Asset Fetcher ───────────────────────────────────────────
def fetch_multiple_assets(
    symbols: list[str],
    timeframe: str = "1h",
    lookback_days: int = 730,
    exchange_id: str = "binance",
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for multiple symbols, returning a dict keyed by symbol.
    Handles failures gracefully — if one asset fails, others still load.
    """
    results = {}
    for symbol in symbols:
        try:
            results[symbol] = fetch_ohlcv(
                symbol, timeframe, lookback_days, exchange_id
            )
            logger.info(f"Successfully loaded {symbol}")
        except Exception as e:
            logger.error(f"Failed to load {symbol}: {e}")
            results[symbol] = pd.DataFrame()  # Empty df signals failure

    loaded = [s for s, df in results.items() if not df.empty]
    failed = [s for s, df in results.items() if df.empty]

    if failed:
        logger.warning(f"Failed to load: {failed}")
    logger.info(f"Loaded {len(loaded)}/{len(symbols)} assets successfully")

    return results


# ── Latest Price ──────────────────────────────────────────────────
def fetch_latest_price(symbol: str, exchange_id: str = "binance") -> float:
    """Fetch the current mid price for a symbol. Used by live signal engine."""
    exchange = get_exchange(exchange_id)
    ticker = exchange.fetch_ticker(symbol)
    return float(ticker["last"])


# ── Data Validation ───────────────────────────────────────────────
def validate_ohlcv(df: pd.DataFrame, symbol: str) -> bool:
    """
    Basic sanity checks on fetched data.
    Logs warnings but does not raise — caller decides how to handle.
    """
    issues = []

    if df.empty:
        issues.append("DataFrame is empty")

    if df.isnull().sum().sum() > len(df) * 0.01:
        issues.append(f"More than 1% null values")

    # Check for suspiciously large price gaps (> 30% move in one candle)
    returns = df["close"].pct_change().abs()
    if (returns > 0.3).any():
        issues.append(f"Found candle(s) with >30% price move — possible bad data")

    # Check for zero volume candles
    zero_vol = (df["volume"] == 0).sum()
    if zero_vol > len(df) * 0.05:
        issues.append(f"{zero_vol} zero-volume candles ({zero_vol/len(df)*100:.1f}%)")

    # Check index is monotonically increasing
    if not df.index.is_monotonic_increasing:
        issues.append("Index is not sorted")

    if issues:
        for issue in issues:
            logger.warning(f"Data quality issue for {symbol}: {issue}")
        return False

    return True