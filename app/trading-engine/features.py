# packages/trading-engine/crypto_agent/engine/features.py
import pandas as pd
import numpy as np
import vectorbt as vbt
from ta import trend, momentum, volatility, volume

FEATURE_REGISTRY = {}

def register_feature(name: str):
    def decorator(fn):
        FEATURE_REGISTRY[name] = fn
        return fn
    return decorator

@register_feature("rsi_14")
def rsi_14(df: pd.DataFrame) -> pd.Series:
    return momentum.RSIIndicator(df["close"], window=14).rsi()

@register_feature("rsi_7")
def rsi_7(df: pd.DataFrame) -> pd.Series:
    return momentum.RSIIndicator(df["close"], window=7).rsi()

@register_feature("macd_signal")
def macd_signal(df: pd.DataFrame) -> pd.Series:
    macd = trend.MACD(df["close"])
    return macd.macd_signal()

@register_feature("macd_diff")
def macd_diff(df: pd.DataFrame) -> pd.Series:
    return trend.MACD(df["close"]).macd_diff()

@register_feature("atr_14")
def atr_14(df: pd.DataFrame) -> pd.Series:
    return volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()

@register_feature("bb_width")
def bb_width(df: pd.DataFrame) -> pd.Series:
    bb = volatility.BollingerBands(df["close"])
    return bb.bollinger_wband()

@register_feature("volume_zscore")
def volume_zscore(df: pd.DataFrame) -> pd.Series:
    vol = df["volume"]
    return (vol - vol.rolling(20).mean()) / vol.rolling(20).std()

@register_feature("obv")
def obv(df: pd.DataFrame) -> pd.Series:
    return volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()

@register_feature("vwap_distance")
def vwap_distance(df: pd.DataFrame) -> pd.Series:
    vwap = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    return (df["close"] - vwap) / vwap

@register_feature("log_return")
def log_return(df: pd.DataFrame) -> pd.Series:
    return np.log(df["close"] / df["close"].shift(1))

@register_feature("rolling_volatility_24")
def rolling_vol(df: pd.DataFrame) -> pd.Series:
    return np.log(df["close"] / df["close"].shift(1)).rolling(24).std()

@register_feature("funding_rate")
def funding_rate(df: pd.DataFrame) -> pd.Series:
    # Assumes funding_rate column injected from exchange data
    return df.get("funding_rate", pd.Series(0, index=df.index))

@register_feature("open_interest_change")
def oi_change(df: pd.DataFrame) -> pd.Series:
    oi = df.get("open_interest", pd.Series(0, index=df.index))
    return oi.pct_change()

def compute_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Compute requested features with zero lookahead bias"""
    result = df.copy()
    for name in feature_names:
        if name not in FEATURE_REGISTRY:
            raise ValueError(f"Unknown feature: {name}. Available: {list(FEATURE_REGISTRY.keys())}")
        result[name] = FEATURE_REGISTRY[name](df)
    return result.dropna()

def get_available_features() -> dict[str, str]:
    """Returns feature catalogue for Claude's context"""
    descriptions = {
        "rsi_14": "Relative Strength Index (14 period) - momentum oscillator 0-100",
        "rsi_7": "RSI short period - more sensitive to recent moves",
        "macd_signal": "MACD signal line - trend following momentum",
        "macd_diff": "MACD histogram - divergence between MACD and signal",
        "atr_14": "Average True Range - volatility measure in price units",
        "bb_width": "Bollinger Band width - measures volatility expansion/contraction",
        "volume_zscore": "Volume Z-score - how unusual current volume is vs 20-period mean",
        "obv": "On Balance Volume - cumulative volume pressure indicator",
        "vwap_distance": "Distance from VWAP as % - mean reversion signal",
        "log_return": "Log return of close price",
        "rolling_volatility_24": "24-period rolling realized volatility",
        "funding_rate": "Perpetual futures funding rate - crowding/sentiment indicator",
        "open_interest_change": "Change in open interest - new money entering market",
    }
    return descriptions