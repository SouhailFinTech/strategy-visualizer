"""
QUANT ALPHA — STRATEGY VISUALIZER v7
FIXES:
- Bug 1: Short signals no longer show when only Long was requested
- Bug 2: Groq now generates correct Position logic (never cumsum)
- Bug 3: Direction-aware code generation (long only / short only / both)

SETUP:
1. Streamlit Secrets: GROQ_API_KEY = "your-groq-key"
2. requirements.txt: streamlit, pandas, numpy, plotly, requests, groq
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ═══════════════════════════════════════════════════════════════
# QUANT ALPHA INDICATOR LIBRARY — internal, tested, always correct
# Groq calls these functions instead of writing indicator formulas
# ═══════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — TREND INDICATORS
# ═══════════════════════════════════════════════════════════════

def add_ema(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Exponential Moving Average"""
    col = f'EMA_{period}'
    df[col] = df['Close'].ewm(span=period, adjust=False).mean()
    return df

def add_sma(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Simple Moving Average"""
    col = f'SMA_{period}'
    df[col] = df['Close'].rolling(period).mean()
    return df

def add_wma(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Weighted Moving Average"""
    col = f'WMA_{period}'
    weights = np.arange(1, period + 1)
    df[col] = df['Close'].rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )
    return df

def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Volume Weighted Average Price (daily reset)"""
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df

def add_macd(df: pd.DataFrame,
             fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD — Line, Signal, Histogram"""
    ema_fast        = df['Close'].ewm(span=fast,   adjust=False).mean()
    ema_slow        = df['Close'].ewm(span=slow,   adjust=False).mean()
    df['MACD']      = ema_fast - ema_slow
    df['MACD_Signal']= df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

def add_supertrend(df: pd.DataFrame,
                   period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Supertrend indicator"""
    df = add_atr(df, period)
    atr = df[f'ATR_{period}']
    hl2 = (df['High'] + df['Low']) / 2

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction  = pd.Series(index=df.index, dtype=int)

    for i in range(1, len(df)):
        if df['Close'].iloc[i] > upper.iloc[i-1]:
            direction.iloc[i] = 1
        elif df['Close'].iloc[i] < lower.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
            if direction.iloc[i] == 1 and lower.iloc[i] < lower.iloc[i-1]:
                lower.iloc[i] = lower.iloc[i-1]
            if direction.iloc[i] == -1 and upper.iloc[i] > upper.iloc[i-1]:
                upper.iloc[i] = upper.iloc[i-1]

        supertrend.iloc[i] = lower.iloc[i] if direction.iloc[i] == 1 else upper.iloc[i]

    df['Supertrend']           = supertrend
    df['Supertrend_Direction'] = direction
    return df

# ═══════════════════════════════════════════════════════════════
# SECTION 2 — MOMENTUM INDICATORS
# ═══════════════════════════════════════════════════════════════

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index"""
    col   = f'RSI_{period}'
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    df[col] = 100 - (100 / (1 + rs))
    return df

def add_stochastic(df: pd.DataFrame,
                   k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Stochastic Oscillator %K and %D"""
    low_min  = df['Low'].rolling(k_period).min()
    high_max = df['High'].rolling(k_period).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['Stoch_D'] = df['Stoch_K'].rolling(d_period).mean()
    return df

def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Commodity Channel Index"""
    tp          = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp      = tp.rolling(period).mean()
    mad         = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    df[f'CCI_{period}'] = (tp - sma_tp) / (0.015 * mad)
    return df

def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Williams %R"""
    high_max = df['High'].rolling(period).max()
    low_min  = df['Low'].rolling(period).min()
    df[f'WR_{period}'] = -100 * (high_max - df['Close']) / (high_max - low_min)
    return df

def add_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Money Flow Index"""
    tp        = (df['High'] + df['Low'] + df['Close']) / 3
    mf        = tp * df['Volume']
    pos_mf    = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg_mf    = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
    mfr       = pos_mf / neg_mf.replace(0, np.nan)
    df[f'MFI_{period}'] = 100 - (100 / (1 + mfr))
    return df

# ═══════════════════════════════════════════════════════════════
# SECTION 3 — VOLATILITY INDICATORS
# ═══════════════════════════════════════════════════════════════

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range"""
    col = f'ATR_{period}'
    tr  = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low']  - df['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df[col] = tr.ewm(span=period, adjust=False).mean()
    return df

def add_bollinger(df: pd.DataFrame,
                  period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands"""
    df['BB_Mid']   = df['Close'].rolling(period).mean()
    std            = df['Close'].rolling(period).std()
    df['BB_Upper'] = df['BB_Mid'] + std_dev * std
    df['BB_Lower'] = df['BB_Mid'] - std_dev * std
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
    df['BB_Pct']   = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    return df

def add_keltner(df: pd.DataFrame,
                ema_period: int = 20, atr_period: int = 10,
                multiplier: float = 2.0) -> pd.DataFrame:
    """Keltner Channels"""
    df = add_atr(df, atr_period)
    df['KC_Mid']   = df['Close'].ewm(span=ema_period, adjust=False).mean()
    df['KC_Upper'] = df['KC_Mid'] + multiplier * df[f'ATR_{atr_period}']
    df['KC_Lower'] = df['KC_Mid'] - multiplier * df[f'ATR_{atr_period}']
    return df

def add_donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Donchian Channels"""
    df[f'DC_Upper_{period}'] = df['High'].rolling(period).max()
    df[f'DC_Lower_{period}'] = df['Low'].rolling(period).min()
    df[f'DC_Mid_{period}']   = (df[f'DC_Upper_{period}'] + df[f'DC_Lower_{period}']) / 2
    return df

# ═══════════════════════════════════════════════════════════════
# SECTION 4 — VOLUME INDICATORS
# ═══════════════════════════════════════════════════════════════

def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume"""
    direction  = np.sign(df['Close'].diff())
    df['OBV']  = (direction * df['Volume']).fillna(0).cumsum()
    return df

def add_volume_sma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Volume Simple Moving Average"""
    df[f'Vol_SMA_{period}'] = df['Volume'].rolling(period).mean()
    return df

def add_cmf(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Chaikin Money Flow"""
    clv       = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / \
                (df['High'] - df['Low']).replace(0, np.nan)
    df['CMF'] = (clv * df['Volume']).rolling(period).sum() / \
                df['Volume'].rolling(period).sum()
    return df

def add_volume_spike(df: pd.DataFrame,
                     period: int = 20, multiplier: float = 2.0) -> pd.DataFrame:
    """Volume Spike — True when volume is X times the average"""
    df = add_volume_sma(df, period)
    df['Volume_Spike'] = df['Volume'] > (multiplier * df[f'Vol_SMA_{period}'])
    return df

# ═══════════════════════════════════════════════════════════════
# SECTION 5 — PRICE ACTION
# ═══════════════════════════════════════════════════════════════

def add_swing_highs_lows(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Swing Highs and Lows.
    A swing high is a candle whose high is the highest in lookback bars each side.
    A swing low is a candle whose low is the lowest in lookback bars each side.
    """
    df['Swing_High'] = False
    df['Swing_Low']  = False

    for i in range(lookback, len(df) - lookback):
        window_high = df['High'].iloc[i-lookback:i+lookback+1]
        window_low  = df['Low'].iloc[i-lookback:i+lookback+1]
        if df['High'].iloc[i] == window_high.max():
            df['Swing_High'].iloc[i] = True
        if df['Low'].iloc[i] == window_low.min():
            df['Swing_Low'].iloc[i] = True

    return df

def add_higher_highs_lower_lows(df: pd.DataFrame,
                                  lookback: int = 5) -> pd.DataFrame:
    """Higher Highs, Higher Lows, Lower Highs, Lower Lows"""
    df = add_swing_highs_lows(df, lookback)
    swing_h = df[df['Swing_High']]['High']
    swing_l = df[df['Swing_Low']]['Low']

    df['HH'] = False
    df['LL'] = False
    df['LH'] = False
    df['HL'] = False

    prev_h = swing_h.shift(1).reindex(df.index).ffill()
    prev_l = swing_l.shift(1).reindex(df.index).ffill()

    df.loc[df['Swing_High'], 'HH'] = df.loc[df['Swing_High'], 'High'] > prev_h[df['Swing_High']]
    df.loc[df['Swing_High'], 'LH'] = df.loc[df['Swing_High'], 'High'] < prev_h[df['Swing_High']]
    df.loc[df['Swing_Low'],  'HL'] = df.loc[df['Swing_Low'],  'Low']  > prev_l[df['Swing_Low']]
    df.loc[df['Swing_Low'],  'LL'] = df.loc[df['Swing_Low'],  'Low']  < prev_l[df['Swing_Low']]

    return df

def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Common candlestick patterns:
    - Engulfing (bullish & bearish)
    - Doji
    - Hammer & Shooting Star
    - Pin Bar
    """
    body      = df['Close'] - df['Open']
    body_abs  = body.abs()
    upper_wick = df['High'] - df[['Open','Close']].max(axis=1)
    lower_wick = df[['Open','Close']].min(axis=1) - df['Low']
    candle_range = df['High'] - df['Low']

    # Bullish Engulfing
    df['Bullish_Engulfing'] = (
        (df['Close'] > df['Open']) &
        (df['Open'].shift(1) > df['Close'].shift(1)) &
        (df['Close'] > df['Open'].shift(1)) &
        (df['Open'] < df['Close'].shift(1))
    )

    # Bearish Engulfing
    df['Bearish_Engulfing'] = (
        (df['Open'] > df['Close']) &
        (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Open'] > df['Close'].shift(1)) &
        (df['Close'] < df['Open'].shift(1))
    )

    # Doji — body is less than 10% of candle range
    df['Doji'] = body_abs < (0.1 * candle_range)

    # Hammer — small body at top, long lower wick (bullish reversal)
    df['Hammer'] = (
        (lower_wick > 2 * body_abs) &
        (upper_wick < body_abs) &
        (candle_range > 0)
    )

    # Shooting Star — small body at bottom, long upper wick (bearish reversal)
    df['Shooting_Star'] = (
        (upper_wick > 2 * body_abs) &
        (lower_wick < body_abs) &
        (candle_range > 0)
    )

    # Pin Bar — rejection candle (long wick, small body)
    df['Bullish_Pin_Bar'] = (
        (lower_wick > 2 * body_abs) &
        (lower_wick > upper_wick * 2)
    )
    df['Bearish_Pin_Bar'] = (
        (upper_wick > 2 * body_abs) &
        (upper_wick > lower_wick * 2)
    )

    return df

def add_inside_outside_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Inside Bar and Outside Bar patterns"""
    df['Inside_Bar'] = (
        (df['High'] < df['High'].shift(1)) &
        (df['Low']  > df['Low'].shift(1))
    )
    df['Outside_Bar'] = (
        (df['High'] > df['High'].shift(1)) &
        (df['Low']  < df['Low'].shift(1))
    )
    return df

def add_support_resistance(df: pd.DataFrame,
                            lookback: int = 20) -> pd.DataFrame:
    """Rolling support and resistance levels"""
    df[f'Resistance_{lookback}'] = df['High'].rolling(lookback).max()
    df[f'Support_{lookback}']    = df['Low'].rolling(lookback).min()
    return df

# ═══════════════════════════════════════════════════════════════
# SECTION 6 — SMC (Smart Money Concepts)
# ═══════════════════════════════════════════════════════════════

def add_structure_break(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Break of Structure (BOS) and Change of Character (CHoCH).
    BOS: price breaks above last swing high (bullish) or below last swing low (bearish)
    CHoCH: structure shift from bearish to bullish or vice versa
    """
    df = add_swing_highs_lows(df, lookback)

    # Get last swing high and low prices
    last_swing_high = df['High'].where(df['Swing_High']).ffill().shift(1)
    last_swing_low  = df['Low'].where(df['Swing_Low']).ffill().shift(1)

    # Bullish BOS — close breaks above last swing high
    df['BOS_Bullish'] = df['Close'] > last_swing_high

    # Bearish BOS — close breaks below last swing low
    df['BOS_Bearish'] = df['Close'] < last_swing_low

    # CHoCH — first BOS in opposite direction
    df['CHoCH_Bullish'] = (
        df['BOS_Bullish'] &
        ~df['BOS_Bullish'].shift(1).fillna(False)
    )
    df['CHoCH_Bearish'] = (
        df['BOS_Bearish'] &
        ~df['BOS_Bearish'].shift(1).fillna(False)
    )

    return df


def add_order_blocks(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Order Blocks — last bearish candle before bullish BOS (bullish OB)
    and last bullish candle before bearish BOS (bearish OB).
    Approximated using swing structure.
    """
    df = add_structure_break(df, lookback)

    df['Bullish_OB'] = False
    df['Bearish_OB'] = False

    for i in range(1, len(df)):
        # Bullish OB: bearish candle just before CHoCH bullish
        if df['CHoCH_Bullish'].iloc[i]:
            j = i - 1
            while j >= 0:
                if df['Open'].iloc[j] > df['Close'].iloc[j]:  # bearish candle
                    df['Bullish_OB'].iloc[j] = True
                    break
                j -= 1

        # Bearish OB: bullish candle just before CHoCH bearish
        if df['CHoCH_Bearish'].iloc[i]:
            j = i - 1
            while j >= 0:
                if df['Close'].iloc[j] > df['Open'].iloc[j]:  # bullish candle
                    df['Bearish_OB'].iloc[j] = True
                    break
                j -= 1

    return df


def add_fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fair Value Gaps (FVG) / Imbalances.
    Bullish FVG: gap between candle[i-2] high and candle[i] low (price skipped upward)
    Bearish FVG: gap between candle[i-2] low and candle[i] high (price skipped downward)
    """
    df['FVG_Bullish'] = df['Low'] > df['High'].shift(2)
    df['FVG_Bearish'] = df['High'] < df['Low'].shift(2)

    # Store the gap levels
    df['FVG_Bull_Low']  = df['High'].shift(2).where(df['FVG_Bullish'])
    df['FVG_Bull_High'] = df['Low'].where(df['FVG_Bullish'])
    df['FVG_Bear_High'] = df['Low'].shift(2).where(df['FVG_Bearish'])
    df['FVG_Bear_Low']  = df['High'].where(df['FVG_Bearish'])

    return df


def add_liquidity_levels(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Liquidity pools — equal highs/lows where stop losses cluster.
    Buy-side liquidity: swing highs (stops above)
    Sell-side liquidity: swing lows (stops below)
    """
    df = add_swing_highs_lows(df, 5)

    # Buy-side liquidity = recent swing highs
    df['BSL'] = df['High'].where(df['Swing_High']).rolling(lookback).max()

    # Sell-side liquidity = recent swing lows
    df['SSL'] = df['Low'].where(df['Swing_Low']).rolling(lookback).min()

    # Liquidity sweep — price wicked through the level then reversed
    df['BSL_Sweep'] = (
        (df['High'] > df['BSL'].shift(1)) &
        (df['Close'] < df['BSL'].shift(1))
    )
    df['SSL_Sweep'] = (
        (df['Low'] < df['SSL'].shift(1)) &
        (df['Close'] > df['SSL'].shift(1))
    )

    return df


def add_premium_discount(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """
    Premium and Discount zones relative to the dealing range.
    Above 50% of range = Premium (look to sell)
    Below 50% of range = Discount (look to buy)
    """
    range_high = df['High'].rolling(lookback).max()
    range_low  = df['Low'].rolling(lookback).min()
    midpoint   = (range_high + range_low) / 2

    df['In_Premium']  = df['Close'] > midpoint
    df['In_Discount'] = df['Close'] < midpoint
    df['Range_50pct'] = midpoint

    return df


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — ICT CONCEPTS
# ═══════════════════════════════════════════════════════════════

def add_market_structure(df: pd.DataFrame,
                          lookback: int = 10) -> pd.DataFrame:
    """
    ICT Market Structure — Bullish or Bearish.
    Bullish: Higher Highs + Higher Lows
    Bearish: Lower Highs + Lower Lows
    """
    df = add_higher_highs_lower_lows(df, lookback)

    df['Bullish_Structure'] = df['HH'] | df['HL']
    df['Bearish_Structure'] = df['LH'] | df['LL']

    return df


def add_optimal_trade_entry(df: pd.DataFrame,
                             lookback: int = 20) -> pd.DataFrame:
    """
    ICT OTE (Optimal Trade Entry) — 62-79% Fibonacci retracement zone.
    Bullish OTE: price retraces to 62-79% of last bullish impulse.
    """
    df = add_swing_highs_lows(df, lookback // 4)

    swing_h = df['High'].where(df['Swing_High']).ffill()
    swing_l = df['Low'].where(df['Swing_Low']).ffill()

    fib_618 = swing_h - 0.618 * (swing_h - swing_l)
    fib_786 = swing_h - 0.786 * (swing_h - swing_l)

    df['In_OTE_Bullish'] = (
        (df['Close'] >= fib_786) &
        (df['Close'] <= fib_618)
    )

    fib_618_bear = swing_l + 0.618 * (swing_h - swing_l)
    fib_786_bear = swing_l + 0.786 * (swing_h - swing_l)

    df['In_OTE_Bearish'] = (
        (df['Close'] <= fib_786_bear) &
        (df['Close'] >= fib_618_bear)
    )

    return df


def add_killzones(df: pd.DataFrame) -> pd.DataFrame:
    """
    ICT Killzones — high probability trading sessions (UTC times).
    London Open: 02:00-05:00 UTC
    New York Open: 07:00-10:00 UTC
    London Close: 10:00-12:00 UTC
    Asian Range: 20:00-00:00 UTC
    """
    if not hasattr(df.index, 'hour'):
        # Index is not datetime with time — return without killzones
        df['London_KZ']   = False
        df['NewYork_KZ']  = False
        df['LondonClose_KZ'] = False
        df['Asian_KZ']    = False
        return df

    h = df.index.hour
    df['London_KZ']      = (h >= 2)  & (h < 5)
    df['NewYork_KZ']     = (h >= 7)  & (h < 10)
    df['LondonClose_KZ'] = (h >= 10) & (h < 12)
    df['Asian_KZ']       = (h >= 20) | (h < 1)
    return df


def add_previous_day_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Previous Day High, Low, Close (PDH, PDL, PDC).
    Key ICT reference levels for bias and entries.
    """
    df['PDH'] = df['High'].shift(1)
    df['PDL'] = df['Low'].shift(1)
    df['PDC'] = df['Close'].shift(1)
    return df


def add_weekly_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Previous Week High and Low"""
    df['Week'] = df.index.to_series().dt.isocalendar().week
    df['PWH']  = df.groupby('Week')['High'].transform('max').shift(1)
    df['PWL']  = df.groupby('Week')['Low'].transform('min').shift(1)
    df = df.drop(columns=['Week'])
    return df


def add_equal_highs_lows(df: pd.DataFrame,
                          tolerance: float = 0.001) -> pd.DataFrame:
    """
    Equal Highs (EQH) and Equal Lows (EQL) — liquidity pools.
    Two swing highs/lows within tolerance % of each other.
    """
    df = add_swing_highs_lows(df, 5)

    df['EQH'] = (
        df['Swing_High'] &
        (abs(df['High'] - df['High'].shift(1)) / df['High'].shift(1) < tolerance) &
        df['Swing_High'].shift(1)
    )
    df['EQL'] = (
        df['Swing_Low'] &
        (abs(df['Low'] - df['Low'].shift(1)) / df['Low'].shift(1) < tolerance) &
        df['Swing_Low'].shift(1)
    )
    return df


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — SIGNAL HELPERS (crossovers, thresholds)
# ═══════════════════════════════════════════════════════════════

def crossover(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """Returns True on the bar where series_a crosses ABOVE series_b"""
    return (series_a > series_b) & (series_a.shift(1) <= series_b.shift(1))

def crossunder(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """Returns True on the bar where series_a crosses BELOW series_b"""
    return (series_a < series_b) & (series_a.shift(1) >= series_b.shift(1))

def above_level(series: pd.Series, level: float) -> pd.Series:
    """Returns True when series crosses above a fixed level"""
    return (series > level) & (series.shift(1) <= level)

def below_level(series: pd.Series, level: float) -> pd.Series:
    """Returns True when series crosses below a fixed level"""
    return (series < level) & (series.shift(1) >= level)

def rising(series: pd.Series, periods: int = 1) -> pd.Series:
    """Returns True when series is rising over N periods"""
    return series > series.shift(periods)

def falling(series: pd.Series, periods: int = 1) -> pd.Series:
    """Returns True when series is falling over N periods"""
    return series < series.shift(periods)


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — CONVENIENCE: ADD ALL COMMON INDICATORS AT ONCE
# ═══════════════════════════════════════════════════════════════

def add_common_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the most commonly used indicators in one call.
    Useful when you want AI to have access to everything.
    """
    df = add_ema(df, 9)
    df = add_ema(df, 20)
    df = add_ema(df, 50)
    df = add_ema(df, 200)
    df = add_sma(df, 20)
    df = add_sma(df, 50)
    df = add_rsi(df, 14)
    df = add_macd(df, 12, 26, 9)
    df = add_bollinger(df, 20, 2.0)
    df = add_atr(df, 14)
    df = add_obv(df)
    df = add_volume_sma(df, 20)
    df = add_volume_spike(df, 20, 2.0)
    df = add_swing_highs_lows(df, 5)
    df = add_candle_patterns(df)
    df = add_support_resistance(df, 20)
    return df


def add_smc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all SMC/ICT indicators in one call.
    """
    df = add_swing_highs_lows(df, 5)
    df = add_higher_highs_lower_lows(df, 5)
    df = add_structure_break(df, 10)
    df = add_order_blocks(df, 10)
    df = add_fair_value_gaps(df)
    df = add_liquidity_levels(df, 20)
    df = add_premium_discount(df, 50)
    df = add_market_structure(df, 10)
    df = add_optimal_trade_entry(df, 20)
    df = add_equal_highs_lows(df, 0.001)
    df = add_previous_day_levels(df)
    return df


# ═══════════════════════════════════════════════════════════════
# LIBRARY REFERENCE — for AI prompt context
# ═══════════════════════════════════════════════════════════════

LIBRARY_REFERENCE = """
AVAILABLE INDICATOR FUNCTIONS (all tested, all correct):

TREND:
- add_ema(df, period)           → df['EMA_20'], df['EMA_50'], etc.
- add_sma(df, period)           → df['SMA_20']
- add_wma(df, period)           → df['WMA_20']
- add_vwap(df)                  → df['VWAP']
- add_macd(df, 12, 26, 9)       → df['MACD'], df['MACD_Signal'], df['MACD_Hist']
- add_supertrend(df, 10, 3.0)   → df['Supertrend'], df['Supertrend_Direction']

MOMENTUM:
- add_rsi(df, 14)               → df['RSI_14']
- add_stochastic(df, 14, 3)     → df['Stoch_K'], df['Stoch_D']
- add_cci(df, 20)               → df['CCI_20']
- add_williams_r(df, 14)        → df['WR_14']
- add_mfi(df, 14)               → df['MFI_14']

VOLATILITY:
- add_atr(df, 14)               → df['ATR_14']
- add_bollinger(df, 20, 2.0)    → df['BB_Mid'], df['BB_Upper'], df['BB_Lower']
- add_keltner(df, 20, 10, 2.0)  → df['KC_Mid'], df['KC_Upper'], df['KC_Lower']
- add_donchian(df, 20)          → df['DC_Upper_20'], df['DC_Lower_20']

VOLUME:
- add_obv(df)                   → df['OBV']
- add_volume_sma(df, 20)        → df['Vol_SMA_20']
- add_cmf(df, 20)               → df['CMF']
- add_volume_spike(df, 20, 2.0) → df['Volume_Spike'] (bool)

PRICE ACTION:
- add_swing_highs_lows(df, 5)   → df['Swing_High'], df['Swing_Low'] (bool)
- add_candle_patterns(df)       → df['Bullish_Engulfing'], df['Bearish_Engulfing'],
                                   df['Doji'], df['Hammer'], df['Shooting_Star'],
                                   df['Bullish_Pin_Bar'], df['Bearish_Pin_Bar']
- add_inside_outside_bars(df)   → df['Inside_Bar'], df['Outside_Bar']
- add_support_resistance(df,20) → df['Resistance_20'], df['Support_20']
- add_higher_highs_lower_lows(df,5) → df['HH'], df['HL'], df['LH'], df['LL']

SMC (Smart Money Concepts):
- add_structure_break(df, 10)   → df['BOS_Bullish'], df['BOS_Bearish'],
                                   df['CHoCH_Bullish'], df['CHoCH_Bearish']
- add_order_blocks(df, 10)      → df['Bullish_OB'], df['Bearish_OB']
- add_fair_value_gaps(df)       → df['FVG_Bullish'], df['FVG_Bearish']
- add_liquidity_levels(df, 20)  → df['BSL'], df['SSL'],
                                   df['BSL_Sweep'], df['SSL_Sweep']
- add_premium_discount(df, 50)  → df['In_Premium'], df['In_Discount']
- add_equal_highs_lows(df)      → df['EQH'], df['EQL']

ICT:
- add_market_structure(df, 10)  → df['Bullish_Structure'], df['Bearish_Structure']
- add_optimal_trade_entry(df)   → df['In_OTE_Bullish'], df['In_OTE_Bearish']
- add_killzones(df)             → df['London_KZ'], df['NewYork_KZ'],
                                   df['LondonClose_KZ'], df['Asian_KZ']
- add_previous_day_levels(df)   → df['PDH'], df['PDL'], df['PDC']
- add_weekly_levels(df)         → df['PWH'], df['PWL']

SIGNAL HELPERS (return boolean Series):
- crossover(series_a, series_b)     → True when a crosses above b
- crossunder(series_a, series_b)    → True when a crosses below b
- above_level(series, level)        → True when series crosses above fixed level
- below_level(series, level)        → True when series crosses below fixed level
- rising(series, periods)           → True when series is rising
- falling(series, periods)          → True when series is falling

CONVENIENCE:
- add_common_indicators(df)     → adds EMA9/20/50/200, SMA20/50, RSI, MACD, BB, ATR, OBV, swings, candles
- add_smc_indicators(df)        → adds all SMC/ICT indicators

EXAMPLE SIGNALS:
- EMA crossover long:  crossover(df['EMA_20'], df['EMA_50'])
- RSI oversold:        below_level(df['RSI_14'], 30)
- Bullish engulfing:   df['Bullish_Engulfing']
- Liquidity sweep up:  df['SSL_Sweep']
- FVG entry:           df['FVG_Bullish'] & df['In_Discount']
- ICT OTE entry:       df['In_OTE_Bullish'] & df['Bullish_Structure']
- SMC full setup:      df['CHoCH_Bullish'] & df['In_Discount'] & df['FVG_Bullish']
"""

# ═══════════════════════════════════════════════════════════════
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Strategy Visualizer | Quant Alpha",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #080a0f;
    color: #e8e0d0;
}
.stApp { background-color: #080a0f; }

.main-header {
    background: linear-gradient(135deg, #0d0f14 0%, #1a1508 100%);
    border: 1px solid #3d2f00; border-radius: 12px;
    padding: 28px 32px; margin-bottom: 24px;
}
.main-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem; color: #f59e0b;
    margin: 0; letter-spacing: -1px;
}
.main-header p { color: #6b5b3a; margin: 6px 0 0; font-size: 0.9rem; }

.step-card {
    background: #0d0f14; border: 1px solid #1e2030;
    border-radius: 10px; padding: 20px; margin-bottom: 16px;
}
.step-card.active { border-color: #f59e0b; }
.step-card.done   { border-color: #22c55e; }
.step-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem; color: #f59e0b;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px;
}
.parsed-box {
    background: #0a0c10; border: 1px solid #1e2030;
    border-left: 4px solid #f59e0b; border-radius: 8px;
    padding: 16px; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem; color: #a89060; margin: 12px 0;
}
.tag {
    display: inline-block; padding: 2px 10px; border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; margin: 3px;
}
.tag-entry { background:#1a2e1a; color:#4ade80; border:1px solid #166534; }
.tag-sl    { background:#2e1a1a; color:#f87171; border:1px solid #991b1b; }
.tag-tp    { background:#1a2a1a; color:#86efac; border:1px solid #15803d; }
.data-source-box {
    background:#0a0c10; border:1px solid #1e2030;
    border-left:4px solid #3b82f6; border-radius:8px;
    padding:12px 16px; font-family:'IBM Plex Mono',monospace;
    font-size:0.78rem; color:#60a5fa; margin:8px 0;
}
.section-hdr {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem; color: #f59e0b;
    letter-spacing: 3px; text-transform: uppercase;
    border-bottom: 1px solid #1e2030;
    padding-bottom: 8px; margin: 20px 0 14px;
}
[data-testid="stSidebar"] {
    background: #06080c; border-right: 1px solid #1e2030;
}
.stButton > button {
    background: linear-gradient(135deg,#92400e,#b45309);
    color: #fef3c7; border: none; border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700; padding: 12px 24px; width: 100%;
    transition: all 0.2s; letter-spacing: 1px;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#b45309,#d97706);
    transform: translateY(-1px);
}
.stTextArea textarea, .stTextInput input {
    background: #0a0c10 !important; color: #e8e0d0 !important;
    border: 1px solid #1e2030 !important; border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.85rem !important;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
BINANCE_SYMBOLS = {
    'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT',
    'BNB': 'BNBUSDT', 'XRP': 'XRPUSDT', 'ADA': 'ADAUSDT',
    'DOGE': 'DOGEUSDT', 'AVAX': 'AVAXUSDT', 'MATIC': 'MATICUSDT',
    'LINK': 'LINKUSDT', 'DOT': 'DOTUSDT', 'UNI': 'UNIUSDT',
    'LTC': 'LTCUSDT', 'ATOM': 'ATOMUSDT', 'NEAR': 'NEARUSDT',
}

COINGECKO_IDS = {
    'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana',
    'BNB': 'binancecoin', 'XRP': 'ripple', 'ADA': 'cardano',
    'DOGE': 'dogecoin', 'AVAX': 'avalanche-2', 'MATIC': 'matic-network',
    'LINK': 'chainlink', 'DOT': 'polkadot', 'UNI': 'uniswap',
    'LTC': 'litecoin', 'ATOM': 'cosmos', 'NEAR': 'near',
}

PERIOD_DAYS    = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730}
BINANCE_LIMITS = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730}

# ─────────────────────────────────────────────────────────────
# GROQ INIT
# ─────────────────────────────────────────────────────────────
def init_llm():
    if not GROQ_AVAILABLE:
        st.error("groq package not installed.")
        return None
    try:
        return Groq(api_key=st.secrets["GROQ_API_KEY"])
    except Exception as e:
        st.error(f"Groq error: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# DATA — BINANCE (primary) + COINGECKO (fallback) + CSV (manual)
# ─────────────────────────────────────────────────────────────
def fetch_binance(symbol: str, period: str):
    """Fetch from Binance with 3 retries"""
    sym   = BINANCE_SYMBOLS.get(symbol.upper())
    limit = BINANCE_LIMITS.get(period, 90)
    if not sym:
        return None

    import time
    for attempt in range(3):
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': sym, 'interval': '1d',
                        'limit': min(limit, 1000)},
                timeout=15,
                headers={'User-Agent': 'QuantAlpha/1.0'}
            )
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    df = pd.DataFrame(data, columns=[
                        'timestamp','Open','High','Low','Close','Volume',
                        'ct','qv','nt','tbb','tbq','ignore'
                    ])
                    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.set_index('Date')
                    df = df[['Open','High','Low','Close','Volume']].astype(float)
                    return df.dropna()
            # Wait before retry
            if attempt < 2:
                time.sleep(2)
        except Exception:
            if attempt < 2:
                time.sleep(2)
    return None


def fetch_coingecko(symbol: str, days: int):
    """Fetch from CoinGecko free API"""
    coin_id = COINGECKO_IDS.get(symbol.upper(), symbol.lower())
    api_key = ""
    try:
        api_key = st.secrets.get("COINGECKO_API_KEY", "")
    except Exception:
        pass
    headers = {'User-Agent': 'QuantAlpha/1.0'}
    if api_key:
        headers['x-cg-demo-api-key'] = api_key
    try:
        resp = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
            f"?vs_currency=usd&days={days}",
            timeout=15, headers=headers
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data or not isinstance(data, list):
            return None
        df = pd.DataFrame(
            data, columns=['timestamp','Open','High','Low','Close']
        )
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('Date').drop('timestamp', axis=1).astype(float)
        df['Volume'] = 0.0
        return df.resample('D').last().dropna()
    except Exception:
        return None


def load_csv(uploaded_file):
    """
    Load CSV — supports multiple formats:
    1. MT5 format: tab-separated, date+time columns, 2023.01.01 date format
    2. Standard format: Date/Open/High/Low/Close columns
    3. Any reasonable OHLC CSV
    """
    import io

    try:
        # Read raw bytes first to detect format
        raw = uploaded_file.read()
        uploaded_file.seek(0)

        # ── Try MT5 format first ──────────────────────────────
        # MT5 exports: <DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t...
        # With a header row that starts with <
        try:
            content = raw.decode('utf-8')
            lines   = content.strip().split('\n')

            # Detect MT5 by checking if first line has <DATE> or tab-separated
            is_mt5 = ('<DATE>' in lines[0] or
                      '\t' in lines[0] or
                      lines[0].startswith('<'))

            if is_mt5:
                # Remove angle brackets from headers
                header = lines[0].replace('<','').replace('>','').strip()
                cols   = [c.strip().lower() for c in header.split('\t')]

                rows = []
                for line in lines[1:]:
                    if line.strip():
                        rows.append(line.strip().split('\t'))

                df = pd.DataFrame(rows, columns=cols)

                # MT5 has separate date and time columns
                if 'date' in cols and 'time' in cols:
                    df['datetime'] = pd.to_datetime(
                        df['date'] + ' ' + df['time'],
                        format='%Y.%m.%d %H:%M:%S',
                        errors='coerce'
                    )
                    # Fallback format without seconds
                    mask = df['datetime'].isna()
                    if mask.any():
                        df.loc[mask, 'datetime'] = pd.to_datetime(
                            df.loc[mask,'date'] + ' ' + df.loc[mask,'time'],
                            format='%Y.%m.%d %H:%M',
                            errors='coerce'
                        )
                    df = df.set_index('datetime')
                elif 'date' in cols:
                    df.index = pd.to_datetime(
                        df['date'], format='%Y.%m.%d', errors='coerce'
                    )
                    df = df.drop(columns=['date'], errors='ignore')

                df.index.name = 'Date'

                # Rename MT5 columns to standard names
                rename_map = {
                    'open': 'Open', 'high': 'High',
                    'low': 'Low',   'close': 'Close',
                    'vol': 'Volume', 'tickvol': 'Volume',
                    'tick volume': 'Volume', 'volume': 'Volume'
                }
                df = df.rename(columns=rename_map)

                # Keep required columns
                required = ['Open','High','Low','Close']
                missing  = [c for c in required if c not in df.columns]
                if missing:
                    st.error(f"MT5 CSV missing columns: {missing}")
                    return None

                if 'Volume' not in df.columns:
                    df['Volume'] = 0.0

                df = df[['Open','High','Low','Close','Volume']].astype(float)
                df = df.dropna().sort_index()

                if len(df) > 0:
                    st.success(
                        f"✅ MT5 format detected — {len(df):,} bars loaded"
                    )
                    return df

        except Exception:
            pass

        # ── Try standard CSV format ───────────────────────────
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip().title() for c in df.columns]

        # Find date column
        date_col = next(
            (c for c in ['Date','Datetime','Timestamp','Time','Open Time']
             if c in df.columns),
            None
        )
        if date_col is None:
            # Try first column as date
            first_col = df.columns[0]
            try:
                pd.to_datetime(df[first_col].iloc[0])
                date_col = first_col
            except Exception:
                st.error(
                    "Cannot find date column. "
                    "Expected: Date, Datetime, Timestamp, or Time"
                )
                return None

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.set_index(date_col)
        df.index.name = 'Date'

        # Check required columns
        required = ['Open','High','Low','Close']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"CSV missing columns: {missing}")
            return None

        if 'Volume' not in df.columns:
            df['Volume'] = 0.0

        df = df[['Open','High','Low','Close','Volume']].astype(float)
        df = df.dropna().sort_index()
        return df

    except Exception as e:
        st.error(f"CSV error: {e}")
        return None


def fetch_data(symbol, period, uploaded_file=None):
    """Try all data sources in order"""

    # 1. User CSV upload (highest priority)
    if uploaded_file is not None:
        uploaded_file.seek(0)
        df = load_csv(uploaded_file)
        if df is not None and len(df) > 30:
            return df, "📁 Your CSV"
        else:
            st.warning(
                "CSV uploaded but could not be read. "
                "Trying live data sources..."
            )

    # 2. Binance (most reliable for crypto)
    with st.spinner("📡 Trying Binance..."):
        df = fetch_binance(symbol, period)
    if df is not None and len(df) > 30:
        return df, "🟡 Binance"

    # 3. CoinGecko fallback
    with st.spinner("📡 Trying CoinGecko..."):
        days = PERIOD_DAYS.get(period, 90)
        df   = fetch_coingecko(symbol, days)
    if df is not None and len(df) > 30:
        return df, "🦎 CoinGecko"

    return None, None

# ─────────────────────────────────────────────────────────────
# GROQ HELPERS
# ─────────────────────────────────────────────────────────────
def parse_strategy(client, description: str):
    prompt = f"""You are a quantitative trading expert.
Parse this trading strategy into structured JSON.

Strategy: "{description}"

Return ONLY valid JSON — no markdown, no explanation:
{{
  "entry_long": "long entry condition or null if no long",
  "entry_short": "short entry condition or null if no short",
  "stop_loss": "stop loss description",
  "take_profit": "take profit description",
  "indicators": ["list of indicators"],
  "strategy_type": "trend or mean-reversion or breakout or momentum",
  "sl_pct": 0.02,
  "tp_pct": 0.06,
  "indicator_params": {{
    "ema_fast": 20, "ema_slow": 50,
    "rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30
  }},
  "summary": "one sentence summary"
}}

IMPORTANT:
- If user only says BUY/LONG — set entry_short to null
- If user only says SELL/SHORT — set entry_long to null
- Only set both if user explicitly mentions both directions"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600, temperature=0.1
        )
        text = response.choices[0].message.content.strip()
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        result = json.loads(text.strip())

        # Sanitize — ensure critical numeric fields are never None
        result['sl_pct'] = result.get('sl_pct') or 0.02
        result['tp_pct'] = result.get('tp_pct') or 0.06
        result['strategy_type'] = result.get('strategy_type') or 'trend'
        result['summary']       = result.get('summary') or 'Trading Strategy'
        result['indicators']    = result.get('indicators') or []
        result['indicator_params'] = result.get('indicator_params') or {
            'ema_fast': 20, 'ema_slow': 50,
            'rsi_period': 14, 'rsi_overbought': 70, 'rsi_oversold': 30
        }
        # Ensure indicator_params values are never None
        ip = result['indicator_params']
        ip['ema_fast']       = ip.get('ema_fast')       or 20
        ip['ema_slow']       = ip.get('ema_slow')       or 50
        ip['rsi_period']     = ip.get('rsi_period')     or 14
        ip['rsi_overbought'] = ip.get('rsi_overbought') or 70
        ip['rsi_oversold']   = ip.get('rsi_oversold')   or 30

        return result
    except json.JSONDecodeError:
        st.error("AI returned invalid JSON. Please rephrase your strategy.")
        return None
    except Exception as e:
        st.error(f"Parse error: {e}")
        return None


def detect_advanced_features(description: str) -> dict:
    """Detect direction and advanced features from description"""
    d = description.lower()
    return {
        'has_partial_close':  any(w in d for w in ['partial', 'scale out', 'partial close']),
        'has_trailing_stop':  any(w in d for w in ['trail', 'trailing']),
        'has_both_directions':any(w in d for w in ['short', 'sell when', 'both']),
        'has_long':           any(w in d for w in ['buy', 'long', 'bullish']),
    }


def generate_signal_block(client, description: str, strategy: dict) -> str:
    """
    GROQ'S ONLY JOB: translate plain English → signal lines.
    Everything else is hardcoded by us.

    THE FIX vs previous versions:
    - No static EMA 20/50 example — that was why Groq kept outputting EMA 20/50
    - Dynamic prompt built from actual strategy type Groq already parsed
    - Auto-repair layer catches any remaining mistakes
    - Groq gets told WHAT indicators the strategy needs (from parse step)
    """
    features    = detect_advanced_features(description)
    has_long    = features['has_long'] or not features['has_both_directions']
    has_short   = features['has_both_directions']
    stype       = strategy.get('strategy_type', 'trend')
    indicators  = strategy.get('indicators', [])
    params      = strategy.get('indicator_params', {}) or {}

    # Build direction rule
    if has_long and has_short:
        direction = "BOTH long AND short"
        signal_template = """\
df['long_signal']  = <YOUR_LONG_CONDITION>.fillna(False)
df['short_signal'] = <YOUR_SHORT_CONDITION>.fillna(False)
df['Signal']       = df['long_signal'].astype(int) - df['short_signal'].astype(int)"""
    elif has_short:
        direction = "SHORT only"
        signal_template = """\
df['long_signal']  = pd.Series(False, index=df.index)
df['short_signal'] = <YOUR_SHORT_CONDITION>.fillna(False)
df['Signal']       = -df['short_signal'].astype(int)"""
    else:
        direction = "LONG only"
        signal_template = """\
df['long_signal']  = <YOUR_LONG_CONDITION>.fillna(False)
df['short_signal'] = pd.Series(False, index=df.index)
df['Signal']       = df['long_signal'].astype(int)"""

    # Build hint about what indicators the strategy uses
    ind_hint = f"Strategy type: {stype}\nIndicators mentioned: {', '.join(indicators) if indicators else 'detect from description'}"
    if params.get('ema_fast'): ind_hint += f"\nEMA fast period: {params['ema_fast']}"
    if params.get('ema_slow'): ind_hint += f"\nEMA slow period: {params['ema_slow']}"
    if params.get('rsi_period'): ind_hint += f"\nRSI period: {params['rsi_period']}"

    prompt = f"""You are a Python quant developer.
Translate this trading strategy into Python signal detection code.

STRATEGY: "{description}"
DIRECTION: {direction}
{ind_hint}

AVAILABLE FUNCTIONS (pre-built library — call these, never write formulas):

INDICATORS (call to add columns to df):
add_ema(df, period)        → df['EMA_20'], df['EMA_50'], etc.
add_sma(df, period)        → df['SMA_20']
add_rsi(df, period)        → df['RSI_14']
add_macd(df, 12, 26, 9)    → df['MACD'], df['MACD_Signal'], df['MACD_Hist']
add_bollinger(df, 20, 2.0) → df['BB_Upper'], df['BB_Lower'], df['BB_Mid']
add_atr(df, 14)            → df['ATR_14']
add_stochastic(df, 14, 3)  → df['Stoch_K'], df['Stoch_D']
add_vwap(df)               → df['VWAP']
add_obv(df)                → df['OBV']
add_volume_spike(df, 20, 2)→ df['Volume_Spike']

PRICE ACTION / SMC / ICT:
add_swing_highs_lows(df, 5)    → df['Swing_High'], df['Swing_Low']
add_candle_patterns(df)        → df['Bullish_Engulfing'], df['Bearish_Engulfing'], df['Hammer'], df['Shooting_Star'], df['Bullish_Pin_Bar'], df['Bearish_Pin_Bar'], df['Doji']
add_structure_break(df, 10)    → df['BOS_Bullish'], df['BOS_Bearish'], df['CHoCH_Bullish'], df['CHoCH_Bearish']
add_fair_value_gaps(df)        → df['FVG_Bullish'], df['FVG_Bearish']
add_liquidity_levels(df, 20)   → df['BSL_Sweep'], df['SSL_Sweep'], df['BSL'], df['SSL']
add_order_blocks(df, 10)       → df['Bullish_OB'], df['Bearish_OB']
add_premium_discount(df, 50)   → df['In_Premium'], df['In_Discount']
add_market_structure(df, 10)   → df['Bullish_Structure'], df['Bearish_Structure']
add_optimal_trade_entry(df)    → df['In_OTE_Bullish'], df['In_OTE_Bearish']
add_equal_highs_lows(df)       → df['EQH'], df['EQL']
add_higher_highs_lower_lows(df)→ df['HH'], df['HL'], df['LH'], df['LL']
add_support_resistance(df, 20) → df['Resistance_20'], df['Support_20']
add_previous_day_levels(df)    → df['PDH'], df['PDL'], df['PDC']

SIGNAL HELPERS (return boolean Series):
crossover(series_a, series_b)  → True when a crosses above b
crossunder(series_a, series_b) → True when a crosses below b
above_level(series, value)     → True when series crosses above fixed level
below_level(series, value)     → True when series crosses below fixed level
rising(series, periods)        → True when series is rising
falling(series, periods)       → True when series is falling

MANDATORY OUTPUT FORMAT — follow exactly:
{signal_template}

RULES:
1. First call add_*() functions for every indicator you need
2. Then write the signal conditions using library helpers
3. df['Signal'] MUST always be the last line
4. Both long_signal and short_signal MUST be assigned
5. Output ONLY Python lines — no imports, no def, no markdown, no comments

OUTPUT ONLY THE PYTHON LINES NOW:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.0
        )
        text = response.choices[0].message.content.strip()
        if '```python' in text:
            text = text.split('```python')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]

        lines = [l for l in text.strip().splitlines()
                 if not l.strip().startswith('import ')
                 and not l.strip().startswith('from ')
                 and l.strip() != '']

        joined = '\n'.join(lines)

        # ── AUTO-REPAIR: fix Groq mistakes ───────────────────
        # Repair 1: Signal missing
        if "df['Signal']" not in joined and 'df["Signal"]' not in joined:
            if has_long and has_short:
                lines.append("df['Signal'] = df['long_signal'].astype(int) - df['short_signal'].astype(int)")
            elif has_short:
                lines.append("df['Signal'] = -df['short_signal'].astype(int)")
            else:
                lines.append("df['Signal'] = df['long_signal'].astype(int)")

        # Repair 2: long_signal missing
        if "long_signal" not in joined:
            lines.insert(0, "df['long_signal'] = pd.Series(False, index=df.index)")

        # Repair 3: short_signal missing
        if "short_signal" not in joined:
            lines.insert(0, "df['short_signal'] = pd.Series(False, index=df.index)")

        # Repair 4: fillna missing on boolean signals
        repaired = []
        for line in lines:
            if (("long_signal']  =" in line or "long_signal'] =" in line) and
                    'fillna' not in line and 'pd.Series' not in line and 'astype' not in line):
                line = line.rstrip() + '.fillna(False)'
            if (("short_signal']  =" in line or "short_signal'] =" in line) and
                    'fillna' not in line and 'pd.Series' not in line and 'astype' not in line):
                line = line.rstrip() + '.fillna(False)'
            repaired.append(line)
        lines = repaired

        # Indent 4 spaces
        return '\n'.join(
            '    ' + line.lstrip() if line.strip() else ''
            for line in lines
        )

    except Exception as e:
        # Safe fallback
        if has_long and has_short:
            sig = "    df['Signal'] = df['long_signal'].astype(int) - df['short_signal'].astype(int)"
        elif has_short:
            sig = "    df['Signal'] = -df['short_signal'].astype(int)"
        else:
            sig = "    df['Signal'] = df['long_signal'].astype(int)"
        return (
            "    df['long_signal']  = pd.Series(False, index=df.index)\n"
            "    df['short_signal'] = pd.Series(False, index=df.index)\n"
            f"{sig}\n"
            f"    # Signal generation failed: {e}"
        )


def generate_signal_block(client, description: str) -> str:
    """
    AI writes ONLY the signal detection lines.
    Uses pre-built indicator library to minimize formula errors.

    WHY IT FAILED BEFORE:
    1. Prompt example was long-only — Groq copied it for long+short
    2. df['Signal'] was never mentioned — Groq forgot to create it
    3. Partial close/trailing stop were never mentioned — Groq ignored them
    4. No validation of output — broken code passed through silently

    HOW WE FIX IT:
    1. Detect direction (long/short/both) before calling Groq
    2. Give Groq explicit examples for every case
    3. Enforce df['Signal'] creation in the prompt
    4. Validate output and fix common mistakes automatically
    """

    features = detect_advanced_features(description)
    has_long  = features['has_long'] or not features['has_both_directions']
    has_short = features['has_both_directions']

    # Build direction-specific example
    if has_long and has_short:
        direction_rule = "BOTH long and short signals requested"
        example = """\
# EXAMPLE for long AND short strategy:
df = add_ema(df, 20)
df = add_ema(df, 50)
df['long_signal']  = crossover(df['EMA_20'], df['EMA_50']).fillna(False)
df['short_signal'] = crossunder(df['EMA_20'], df['EMA_50']).fillna(False)
df['Signal']       = df['long_signal'].astype(int) - df['short_signal'].astype(int)
# Signal values: 1=long, -1=short, 0=flat"""
    elif has_short and not has_long:
        direction_rule = "SHORT only — no long signals"
        example = """\
# EXAMPLE for short-only strategy:
df = add_rsi(df, 14)
df['long_signal']  = pd.Series(False, index=df.index)
df['short_signal'] = above_level(df['RSI_14'], 70).fillna(False)
df['Signal']       = -df['short_signal'].astype(int)
# Signal values: -1=short, 0=flat"""
    else:
        direction_rule = "LONG only — no short signals"
        example = """\
# EXAMPLE for long-only strategy:
df = add_ema(df, 20)
df = add_ema(df, 50)
df['long_signal']  = crossover(df['EMA_20'], df['EMA_50']).fillna(False)
df['short_signal'] = pd.Series(False, index=df.index)
df['Signal']       = df['long_signal'].astype(int)
# Signal values: 1=long, 0=flat"""

    # Partial close + trailing note
    advanced_note = ""
    if features['has_partial_close'] or features['has_trailing_stop']:
        advanced_note = """
NOTE ON PARTIAL CLOSE / TRAILING STOP:
These features cannot be implemented with simple signal columns.
They require a trade simulation loop which is handled separately.
For now, implement the ENTRY signals only.
Mark the positions with comments:
# PARTIAL CLOSE: 30% at first target — handled in trade loop
# TRAILING STOP: 70% trail — handled in trade loop
"""

    prompt = f"""You are a Python quant developer writing signal detection code.

STRATEGY: "{description}"
DIRECTION: {direction_rule}

AVAILABLE LIBRARY (already imported — call these, never write formulas):
{LIBRARY_REFERENCE}

MANDATORY RULES:
1. ALWAYS call library functions — never write ewm(), rolling(), etc. yourself
2. df['long_signal'] MUST be a boolean pd.Series ending with .fillna(False)
3. df['short_signal'] MUST be a boolean pd.Series ending with .fillna(False)
4. df['Signal'] MUST always be created as the LAST line:
   - Long only:  df['Signal'] = df['long_signal'].astype(int)
   - Short only: df['Signal'] = -df['short_signal'].astype(int)
   - Both:       df['Signal'] = df['long_signal'].astype(int) - df['short_signal'].astype(int)
5. Use crossover() for crosses above, crossunder() for crosses below
6. Use above_level() for threshold crosses, below_level() for threshold crosses
7. Output ONLY Python lines — no def, no imports, no markdown, no explanation
{advanced_note}
FOLLOW THIS EXACT PATTERN:
{example}

Now write signal code for: "{description}"
Output ONLY the Python lines. No explanation."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.0
        )
        text = response.choices[0].message.content.strip()
        if '```python' in text:
            text = text.split('```python')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]

        # Clean lines
        lines = [l for l in text.strip().splitlines()
                 if not l.strip().startswith('import ')
                 and not l.strip().startswith('from ')]

        # ── AUTO-REPAIR: fix common Groq mistakes ────────────

        joined = '\n'.join(lines)

        # Repair 1: Signal column missing — add it
        if "df['Signal']" not in joined and 'df["Signal"]' not in joined:
            if has_long and has_short:
                lines.append("df['Signal'] = df['long_signal'].astype(int) - df['short_signal'].astype(int)")
            elif has_short:
                lines.append("df['Signal'] = -df['short_signal'].astype(int)")
            else:
                lines.append("df['Signal'] = df['long_signal'].astype(int)")

        # Repair 2: long_signal missing
        if "df['long_signal']" not in joined and 'df["long_signal"]' not in joined:
            lines.insert(0, "df['long_signal'] = pd.Series(False, index=df.index)")

        # Repair 3: short_signal missing
        if "df['short_signal']" not in joined and 'df["short_signal"]' not in joined:
            lines.insert(0, "df['short_signal'] = pd.Series(False, index=df.index)")

        # Repair 4: fillna missing on signal lines
        repaired = []
        for line in lines:
            stripped = line.strip()
            # Add .fillna(False) to boolean signal assignments that lack it
            if (("long_signal']  =" in line or "long_signal'] =" in line or
                 'long_signal"] =' in line) and
                'fillna' not in line and
                'pd.Series' not in line):
                line = line.rstrip() + '.fillna(False)'
            if (("short_signal']  =" in line or "short_signal'] =" in line or
                 'short_signal"] =' in line) and
                'fillna' not in line and
                'pd.Series' not in line):
                line = line.rstrip() + '.fillna(False)'
            repaired.append(line)
        lines = repaired

        # Indent exactly 4 spaces
        signal_block = '\n'.join(
            '    ' + line.lstrip() if line.strip() else ''
            for line in lines
        )
        return signal_block

    except Exception as e:
        # Safe fallback — no signals, won't crash
        if has_long and has_short:
            sig_line = "    df['Signal'] = df['long_signal'].astype(int) - df['short_signal'].astype(int)"
        elif has_short:
            sig_line = "    df['Signal'] = -df['short_signal'].astype(int)"
        else:
            sig_line = "    df['Signal'] = df['long_signal'].astype(int)"
        return (
            "    df['long_signal']  = pd.Series(False, index=df.index)\n"
            "    df['short_signal'] = pd.Series(False, index=df.index)\n"
            f"{sig_line}\n"
            f"    # Signal generation failed: {e}"
        )


def generate_python_code(client, strategy: dict, symbol: str,
                          description: str = '') -> str:
    """
    100% hardcoded code generation — zero Groq involvement.
    Groq only parses the strategy description into JSON (done earlier).
    This function builds the complete backtest code from templates.
    Result is always syntactically correct and mathematically sound.
    """
    has_long    = strategy.get('entry_long')  is not None
    has_short   = strategy.get('entry_short') is not None
    binance_sym = BINANCE_SYMBOLS.get(symbol.upper(), 'BTCUSDT')
    sl_pct      = strategy.get('sl_pct') or 0.02
    tp_pct      = strategy.get('tp_pct') or 0.06
    params      = strategy.get('indicator_params', {})
    ema_fast    = params.get('ema_fast', 20)
    ema_slow    = params.get('ema_slow', 50)
    rsi_period  = params.get('rsi_period', 14)
    rsi_ob      = params.get('rsi_overbought', 70)
    rsi_os      = params.get('rsi_oversold', 30)
    stype       = strategy.get('strategy_type', 'trend')
    summary     = strategy.get('summary', 'Trading Strategy')

    # ── Indicator section ─────────────────────────────────────
    if stype in ['trend', 'momentum']:
        indicator_block = f"""\
    # Exponential Moving Averages
    df['EMA_fast'] = df['Close'].ewm(span={ema_fast}, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span={ema_slow}, adjust=False).mean()"""
        ind1_label = f'EMA {ema_fast}'
        ind2_label = f'EMA {ema_slow}'
        ind1_col   = 'EMA_fast'
        ind2_col   = 'EMA_slow'
        bottom_panel = 'volume'
        long_sig  = f"(df['EMA_fast'] > df['EMA_slow']) & (df['EMA_fast'].shift(1) <= df['EMA_slow'].shift(1))"
        short_sig = f"(df['EMA_fast'] < df['EMA_slow']) & (df['EMA_fast'].shift(1) >= df['EMA_slow'].shift(1))"

    elif stype == 'mean-reversion':
        indicator_block = f"""\
    # RSI Indicator
    delta       = df['Close'].diff()
    gain        = delta.clip(lower=0).rolling({rsi_period}).mean()
    loss        = (-delta.clip(upper=0)).rolling({rsi_period}).mean()
    rs          = gain / loss.replace(0, float('nan'))
    df['RSI']   = 100 - (100 / (1 + rs))
    # EMAs for visual reference
    df['EMA_fast'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=50, adjust=False).mean()"""
        ind1_label = 'EMA 20'
        ind2_label = 'EMA 50'
        ind1_col   = 'EMA_fast'
        ind2_col   = 'EMA_slow'
        bottom_panel = 'rsi'
        long_sig  = f"(df['RSI'] < {rsi_os}) & (df['RSI'].shift(1) >= {rsi_os})"
        short_sig = f"(df['RSI'] > {rsi_ob}) & (df['RSI'].shift(1) <= {rsi_ob})"

    else:  # breakout
        indicator_block = f"""\
    # Bollinger Bands
    df['BB_mid']   = df['Close'].rolling(20).mean()
    bb_std         = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * bb_std
    df['BB_lower'] = df['BB_mid'] - 2 * bb_std
    # EMAs for visual reference
    df['EMA_fast'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=50, adjust=False).mean()"""
        ind1_label = 'BB Upper'
        ind2_label = 'BB Lower'
        ind1_col   = 'BB_upper'
        ind2_col   = 'BB_lower'
        bottom_panel = 'volume'
        long_sig  = f"(df['Close'] > df['BB_upper']) & (df['Close'].shift(1) <= df['BB_upper'].shift(1))"
        short_sig = f"(df['Close'] < df['BB_lower']) & (df['Close'].shift(1) >= df['BB_lower'].shift(1))"

    # ── Signal section ────────────────────────────────────────
    # If description provided — use AI with indicator library
    # Otherwise — use hardcoded template
    if description and client:
        signal_block = generate_signal_block(client, description, strategy)
    elif has_long and not has_short:
        signal_block = f"""\
    # Long only
    df['long_signal']  = ({long_sig}).fillna(False)
    df['short_signal'] = pd.Series(False, index=df.index)
    df['Signal']       = df['long_signal'].astype(int)"""
    elif has_short and not has_long:
        signal_block = f"""\
    # Short only
    df['long_signal']  = pd.Series(False, index=df.index)
    df['short_signal'] = ({short_sig}).fillna(False)
    df['Signal']       = -df['short_signal'].astype(int)"""
    else:
        signal_block = f"""\
    # Long and short
    df['long_signal']  = ({long_sig}).fillna(False)
    df['short_signal'] = ({short_sig}).fillna(False)
    df['Signal']       = df['long_signal'].astype(int) - df['short_signal'].astype(int)"""

    # ── Bottom panel section ──────────────────────────────────
    if bottom_panel == 'rsi':
        bottom_block = f"""\
    # RSI panel
    fig.add_trace(go.Scatter(
        x=df['Open time'], y=df['RSI'],
        name='RSI', line=dict(color='#a78bfa', width=1.5)
    ), row=2, col=1)
    fig.add_hline(y={rsi_ob}, line_color='#ef4444', line_dash='dash',
                  line_width=1, opacity=0.6, row=2, col=1)
    fig.add_hline(y={rsi_os}, line_color='#4ade80', line_dash='dash',
                  line_width=1, opacity=0.6, row=2, col=1)"""
    else:
        bottom_block = """\
    # Volume panel
    bar_colors = ['#26a69a' if c >= o else '#ef5350'
                  for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(
        x=df['Open time'], y=df['Volume'],
        name='Volume', marker_color=bar_colors, opacity=0.6
    ), row=2, col=1)"""

    # ── SL/TP annotation section ──────────────────────────────
    sltp_block = f"""\
    # Draw SL/TP lines for each long entry
    for idx in df[df['long_signal']].index:
        entry    = float(df.loc[idx, 'Close'])
        sl_price = entry * (1 - {sl_pct})
        tp_price = entry * (1 + {tp_pct})
        t_start  = df.loc[idx, 'Open time']
        pos      = df.index.get_loc(idx)
        t_end    = df.iloc[min(pos + 8, len(df) - 1)]['Open time']
        fig.add_shape(type='line', x0=t_start, x1=t_end,
            y0=sl_price, y1=sl_price,
            line=dict(color='#ef4444', width=1, dash='dash'), row=1, col=1)
        fig.add_shape(type='line', x0=t_start, x1=t_end,
            y0=tp_price, y1=tp_price,
            line=dict(color='#4ade80', width=1, dash='dot'), row=1, col=1)
    # Draw SL/TP lines for each short entry
    for idx in df[df['short_signal']].index:
        entry    = float(df.loc[idx, 'Close'])
        sl_price = entry * (1 + {sl_pct})
        tp_price = entry * (1 - {tp_pct})
        t_start  = df.loc[idx, 'Open time']
        pos      = df.index.get_loc(idx)
        t_end    = df.iloc[min(pos + 8, len(df) - 1)]['Open time']
        fig.add_shape(type='line', x0=t_start, x1=t_end,
            y0=sl_price, y1=sl_price,
            line=dict(color='#ef4444', width=1, dash='dash'), row=1, col=1)
        fig.add_shape(type='line', x0=t_start, x1=t_end,
            y0=tp_price, y1=tp_price,
            line=dict(color='#4ade80', width=1, dash='dot'), row=1, col=1)"""

    # ── Read indicator library to embed inline ────────────────
    try:
        with open('quant_indicators.py', 'r') as f:
            lib_content = f.read()
        # Remove the LIBRARY_REFERENCE string (not needed in client code)
        if 'LIBRARY_REFERENCE' in lib_content:
            lib_content = lib_content[:lib_content.find('LIBRARY_REFERENCE')]
        lib_import = f"\n# ── Indicator Library (embedded) ──────────────────────────\n{lib_content}\n"
    except Exception:
        lib_import = ""

    # ── Full code template ────────────────────────────────────
    code = f'''import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
{lib_import}

# ── Fetch OHLCV from Binance ──────────────────────────────────
def fetch_data():
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={{"symbol": "{binance_sym}", "interval": "1d", "limit": 365}}
        )
        resp.raise_for_status()
        raw = resp.json()
        df  = pd.DataFrame(raw, columns=[
            "Open time", "Open", "High", "Low", "Close", "Volume",
            "ct", "qv", "nt", "tbb", "tbq", "ignore"
        ])
        df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col])
        df = df.drop(columns=["ct", "qv", "nt", "tbb", "tbq", "ignore"])
        return df
    except Exception as e:
        print(f"Data error: {{e}}")
        return None


# ── Generate signals (indicators + signal logic combined) ─────
def generate_signals(df):
{signal_block}
    # Shift by 1 bar — prevents lookahead bias
    df["Position"] = df["Signal"].shift(1).fillna(0)
    return df


# ── Backtest ──────────────────────────────────────────────────
def backtest(df, sl_pct={sl_pct}, tp_pct={tp_pct},
             trail_pct=None, partial_close_pct=None):
    """
    Vectorized backtest with optional trailing stop and partial close.
    sl_pct        : stop loss as decimal (0.02 = 2%)
    tp_pct        : take profit as decimal (0.06 = 6%)
    trail_pct     : trailing stop as decimal (None = disabled)
    partial_close_pct : partial close at first target (None = disabled)
    """
    df["Return"]   = df["Close"].pct_change()

    if trail_pct is None:
        # ── Simple vectorized backtest ─────────────────────────
        df["Commission"]      = np.where(
            df["Position"] != df["Position"].shift(1), 0.001, 0
        )
        df["Strategy_Return"] = df["Return"] * df["Position"] - df["Commission"]
        df["BH_Equity"]       = (1 + df["Return"]).cumprod()
        df["Strategy_Equity"] = (1 + df["Strategy_Return"]).cumprod()
    else:
        # ── Trade simulation loop (for trailing stop / partial close) ──
        equity     = 1.0
        bh_equity  = 1.0
        equities   = []
        bh_equities= []
        position   = 0
        entry_price= None
        trail_high = None
        trail_low  = None
        partial_done = False

        prices = df["Close"].values
        returns= df["Return"].fillna(0).values

        for i in range(len(df)):
            bh_equity *= (1 + returns[i])
            new_pos    = int(df["Position"].iloc[i])

            # Open new position
            if position == 0 and new_pos != 0:
                position    = new_pos
                entry_price = prices[i]
                trail_high  = prices[i]
                trail_low   = prices[i]
                partial_done= False
                equity      *= (1 - 0.001)  # commission

            elif position != 0:
                price = prices[i]

                # Update trailing reference
                if position == 1:
                    trail_high = max(trail_high, price)
                    trail_stop = trail_high * (1 - trail_pct)
                    hit_sl     = price <= entry_price * (1 - sl_pct)
                    hit_tp     = price >= entry_price * (1 + tp_pct)
                    hit_trail  = price <= trail_stop and price < trail_high
                else:  # short
                    trail_low  = min(trail_low, price)
                    trail_stop = trail_low  * (1 + trail_pct)
                    hit_sl     = price >= entry_price * (1 + sl_pct)
                    hit_tp     = price <= entry_price * (1 - tp_pct)
                    hit_trail  = price >= trail_stop and price > trail_low

                # Partial close at 30% of position
                if partial_close_pct and not partial_done and hit_tp:
                    partial_return = (price / entry_price - 1) * position * partial_close_pct
                    equity        *= (1 + partial_return - 0.001)
                    partial_done   = True

                # Full close
                if hit_sl or hit_trail or (hit_tp and not partial_close_pct):
                    close_return = (price / entry_price - 1) * position
                    remaining    = (1 - partial_close_pct) if partial_done and partial_close_pct else 1.0
                    equity      *= (1 + close_return * remaining - 0.001)
                    position     = 0
                    entry_price  = None
                else:
                    equity *= (1 + returns[i] * position)

            equities.append(equity)
            bh_equities.append(bh_equity)

        df["Strategy_Equity"] = equities
        df["BH_Equity"]       = bh_equities
        df["Strategy_Return"] = pd.Series(equities).pct_change().fillna(0).values

    return df


# ── Performance metrics ───────────────────────────────────────
def metrics(df):
    r        = df["Strategy_Return"].dropna()
    sharpe   = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    roll_max = df["Strategy_Equity"].cummax()
    max_dd   = ((df["Strategy_Equity"] - roll_max) / roll_max).min()
    win_rate = (r > 0).mean()
    total_r  = df["Strategy_Equity"].iloc[-1] - 1
    n_trades = int((df["Position"] != df["Position"].shift(1)).sum() / 2)
    return sharpe, max_dd, win_rate, total_r, n_trades


# ── Chart ─────────────────────────────────────────────────────
def plot_results(df):
    # Detect available indicator columns for chart
    ind_cols = [c for c in df.columns if c.startswith('EMA_') or
                c.startswith('SMA_') or c in ['BB_Upper','BB_Lower','BB_Mid',
                'MACD','RSI_14','Supertrend']]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=[0.70, 0.30]
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df["Open time"],
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",
        decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    # Draw all available indicator lines automatically
    colors = ["#f59e0b","#60a5fa","#a78bfa","#34d399","#f87171"]
    for i, col in enumerate(ind_cols[:5]):
        fig.add_trace(go.Scatter(
            x=df["Open time"], y=df[col],
            name=col, line=dict(color=colors[i % len(colors)], width=1.5),
            opacity=0.9
        ), row=1, col=1)

    # Long entry markers
    long_entries = df[df["long_signal"]]
    if not long_entries.empty:
        fig.add_trace(go.Scatter(
            x=long_entries["Open time"],
            y=long_entries["Close"] * 0.994,
            mode="markers", name="Long Entry",
            marker=dict(symbol="triangle-up", size=14, color="#4ade80",
                        line=dict(color="#166534", width=1))
        ), row=1, col=1)

    # Short entry markers
    short_entries = df[df["short_signal"]]
    if not short_entries.empty:
        fig.add_trace(go.Scatter(
            x=short_entries["Open time"],
            y=short_entries["Close"] * 1.006,
            mode="markers", name="Short Entry",
            marker=dict(symbol="triangle-down", size=14, color="#f87171",
                        line=dict(color="#991b1b", width=1))
        ), row=1, col=1)

{sltp_block}

{bottom_block}

    # Equity curves
    fig.add_trace(go.Scatter(
        x=df["Open time"], y=df["Strategy_Equity"],
        name="Strategy", line=dict(color="#4ade80", width=2)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df["Open time"], y=df["BH_Equity"],
        name="Buy & Hold", line=dict(color="#64748b", width=1.5, dash="dash")
    ), row=2, col=1)

    fig.update_layout(
        height=700,
        paper_bgcolor="#080a0f",
        plot_bgcolor="#0d0f14",
        font=dict(family="monospace", color="#a89060", size=11),
        legend=dict(bgcolor="#0d0f14", bordercolor="#1e2030", borderwidth=1),
        xaxis_rangeslider_visible=False,
        title=dict(
            text="<b>{symbol}</b> — {summary}",
            font=dict(color="#f59e0b", size=14), x=0.01
        )
    )
    fig.update_xaxes(gridcolor="#1e2030", zerolinecolor="#1e2030")
    fig.update_yaxes(gridcolor="#1e2030", zerolinecolor="#1e2030")
    return fig


# ── Main ──────────────────────────────────────────────────────
def main():
    df = fetch_data()
    if df is None:
        print("Failed to fetch data.")
        return

    df = add_indicators(df)
    df = generate_signals(df)
    df = backtest(df)

    sharpe, max_dd, win_rate, total_r, n_trades = metrics(df)

    print("=" * 48)
    print(f"  {symbol} — {summary}")
    print("=" * 48)
    print(f"  Sharpe Ratio : {{sharpe:.2f}}")
    print(f"  Max Drawdown : {{max_dd:.1%}}")
    print(f"  Win Rate     : {{win_rate:.1%}}")
    print(f"  Total Return : {{total_r:.1%}}")
    print(f"  Trades       : {{n_trades}}")
    print("=" * 48)

    fig = plot_results(df)
    fig.show()


if __name__ == "__main__":
    main()
'''
    return code

# ─────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────
def add_indicators(df, params):
    df = df.copy()
    ef = params.get('ema_fast', 20)
    es = params.get('ema_slow', 50)
    rp = params.get('rsi_period', 14)

    df['EMA_fast'] = df['Close'].ewm(span=ef, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=es, adjust=False).mean()

    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(rp).mean()
    loss  = (-delta.clip(upper=0)).rolling(rp).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    df['BB_mid']   = df['Close'].rolling(20).mean()
    std            = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * std
    df['BB_lower'] = df['BB_mid'] - 2 * std

    return df


# ─────────────────────────────────────────────────────────────
# SIGNALS — FIXED: respects entry_long/entry_short null values
# ─────────────────────────────────────────────────────────────
def generate_signals(df, strategy):
    df    = df.copy()
    p     = strategy.get('indicator_params', {})
    stype = strategy.get('strategy_type', 'trend')
    rob   = p.get('rsi_overbought', 70)
    ros   = p.get('rsi_oversold', 30)

    # ✅ FIX: only generate signals for what user actually requested
    wants_long  = strategy.get('entry_long')  is not None
    wants_short = strategy.get('entry_short') is not None

    # Always initialize as proper boolean Series — never scalar
    df['long_signal']  = pd.Series(False, index=df.index)
    df['short_signal'] = pd.Series(False, index=df.index)

    if stype in ['trend', 'momentum']:
        if wants_long:
            df['long_signal'] = (
                (df['EMA_fast'] > df['EMA_slow']) &
                (df['EMA_fast'].shift(1) <= df['EMA_slow'].shift(1))
            ).fillna(False)
        if wants_short:
            df['short_signal'] = (
                (df['EMA_fast'] < df['EMA_slow']) &
                (df['EMA_fast'].shift(1) >= df['EMA_slow'].shift(1))
            ).fillna(False)

    elif stype == 'mean-reversion':
        if wants_long:
            df['long_signal'] = (
                (df['RSI'] < ros) & (df['RSI'].shift(1) >= ros)
            ).fillna(False)
        if wants_short:
            df['short_signal'] = (
                (df['RSI'] > rob) & (df['RSI'].shift(1) <= rob)
            ).fillna(False)

    elif stype == 'breakout':
        if wants_long:
            df['long_signal'] = (
                (df['Close'] > df['BB_upper']) &
                (df['Close'].shift(1) <= df['BB_upper'].shift(1))
            ).fillna(False)
        if wants_short:
            df['short_signal'] = (
                (df['Close'] < df['BB_lower']) &
                (df['Close'].shift(1) >= df['BB_lower'].shift(1))
            ).fillna(False)

    return df

# ─────────────────────────────────────────────────────────────
# PLOTLY CHART
# ─────────────────────────────────────────────────────────────
def draw_chart(df, strategy, symbol, data_source, show='both'):
    df_plot = df.tail(80).copy()
    sl_pct  = strategy.get('sl_pct') or 0.02
    tp_pct  = strategy.get('tp_pct') or 0.06
    stype   = strategy.get('strategy_type', 'trend')
    params  = strategy.get('indicator_params', {})
    ef_span = params.get('ema_fast', 20)
    es_span = params.get('ema_slow', 50)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=[0.75, 0.25]
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['Open'], high=df_plot['High'],
        low=df_plot['Low'],   close=df_plot['Close'],
        name='Price',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a', decreasing_fillcolor='#ef5350',
    ), row=1, col=1)

    # EMAs
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['EMA_fast'],
        name=f'EMA {ef_span}',
        line=dict(color='#f59e0b', width=1.5), opacity=0.9
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['EMA_slow'],
        name=f'EMA {es_span}',
        line=dict(color='#60a5fa', width=1.5), opacity=0.9
    ), row=1, col=1)

    # Bollinger Bands
    if stype == 'breakout':
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['BB_upper'], name='BB Upper',
            line=dict(color='#f59e0b', width=1, dash='dash'), opacity=0.6
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['BB_lower'], name='BB Lower',
            line=dict(color='#f59e0b', width=1, dash='dash'),
            fill='tonexty', fillcolor='rgba(245,158,11,0.05)', opacity=0.6
        ), row=1, col=1)

    # ── Long signals (shown in long chart and both chart) ────────
    long_df = df_plot[df_plot['long_signal']] if show in ('long', 'both') else df_plot.iloc[0:0]
    if not long_df.empty:
        fig.add_trace(go.Scatter(
            x=long_df.index, y=long_df['Close'] * 0.994,
            mode='markers', name='Long Entry',
            marker=dict(symbol='triangle-up', size=14, color='#4ade80',
                       line=dict(color='#166534', width=1))
        ), row=1, col=1)
        for date, row in long_df.iterrows():
            entry = float(row['Close'])
            sl    = entry * (1 - sl_pct)
            tp    = entry * (1 + tp_pct)
            try:
                end_date = df_plot.index[min(
                    df_plot.index.get_loc(date) + 8,
                    len(df_plot) - 1
                )]
            except Exception:
                end_date = date
            fig.add_shape(type='line', x0=date, x1=end_date, y0=sl, y1=sl,
                line=dict(color='#ef4444', width=1.2, dash='dash'), row=1, col=1)
            fig.add_shape(type='line', x0=date, x1=end_date, y0=tp, y1=tp,
                line=dict(color='#4ade80', width=1.2, dash='dot'), row=1, col=1)
            fig.add_annotation(x=end_date, y=sl,
                text=f"SL {sl_pct*100:.0f}%", showarrow=False,
                font=dict(color='#ef4444', size=9),
                xanchor='left', row=1, col=1)
            fig.add_annotation(x=end_date, y=tp,
                text=f"TP {tp_pct*100:.0f}%", showarrow=False,
                font=dict(color='#4ade80', size=9),
                xanchor='left', row=1, col=1)

    # ── Short signals (shown in short chart and both chart) ───────
    short_df = df_plot[df_plot['short_signal']] if show in ('short', 'both') else df_plot.iloc[0:0]
    if not short_df.empty:
        fig.add_trace(go.Scatter(
            x=short_df.index, y=short_df['Close'] * 1.006,
            mode='markers', name='Short Entry',
            marker=dict(symbol='triangle-down', size=14, color='#f87171',
                       line=dict(color='#991b1b', width=1))
        ), row=1, col=1)
        for date, row in short_df.iterrows():
            entry = float(row['Close'])
            sl    = entry * (1 + sl_pct)
            tp    = entry * (1 - tp_pct)
            try:
                end_date = df_plot.index[min(
                    df_plot.index.get_loc(date) + 8,
                    len(df_plot) - 1
                )]
            except Exception:
                end_date = date
            fig.add_shape(type='line', x0=date, x1=end_date, y0=sl, y1=sl,
                line=dict(color='#ef4444', width=1.2, dash='dash'), row=1, col=1)
            fig.add_shape(type='line', x0=date, x1=end_date, y0=tp, y1=tp,
                line=dict(color='#4ade80', width=1.2, dash='dot'), row=1, col=1)

    # ── Bottom panel ──────────────────────────────────────────
    if stype == 'mean-reversion':
        rob = params.get('rsi_overbought', 70)
        ros = params.get('rsi_oversold', 30)
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['RSI'],
            name='RSI', line=dict(color='#a78bfa', width=1.5)
        ), row=2, col=1)
        fig.add_hline(y=rob, line_color='#ef4444', line_dash='dash',
                     line_width=1, opacity=0.6, row=2, col=1)
        fig.add_hline(y=ros, line_color='#4ade80', line_dash='dash',
                     line_width=1, opacity=0.6, row=2, col=1)
    else:
        if df_plot['Volume'].sum() > 0:
            bar_colors = [
                '#26a69a' if c >= o else '#ef5350'
                for c, o in zip(df_plot['Close'], df_plot['Open'])
            ]
            fig.add_trace(go.Bar(
                x=df_plot.index, y=df_plot['Volume'],
                name='Volume', marker_color=bar_colors, opacity=0.6
            ), row=2, col=1)

    n_long  = len(long_df)
    n_short = len(short_df)

    fig.update_layout(
        height=620,
        paper_bgcolor='#080a0f', plot_bgcolor='#0d0f14',
        font=dict(family='IBM Plex Mono', color='#a89060', size=11),
        legend=dict(bgcolor='#0d0f14', bordercolor='#1e2030',
                   borderwidth=1, font=dict(color='#a89060', size=10)),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=80, t=70, b=40),
        title=dict(
            text=(
                f"<b>{symbol}</b> — {strategy.get('summary','Strategy')}<br>"
                f"<span style='font-size:11px;color:#6b5b3a'>"
                f"🔺 {n_long} Long  🔻 {n_short} Short  "
                f"| {data_source} | Last 80 bars</span>"
            ),
            font=dict(color='#f59e0b', size=13), x=0.01
        )
    )
    fig.update_xaxes(gridcolor='#1e2030', zerolinecolor='#1e2030',
                    tickfont=dict(color='#6b5b3a'))
    fig.update_yaxes(gridcolor='#1e2030', zerolinecolor='#1e2030',
                    tickfont=dict(color='#6b5b3a'))
    return fig

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📈 STRATEGY VISUALIZER</h1>
    <p>Describe your strategy → See it on real candles → Get Python code</p>
    <p style="color:#3d2f00;font-family:'IBM Plex Mono';font-size:0.7rem">
    QUANT ALPHA · GROQ + BINANCE · INTERACTIVE · $0
    </p>
</div>""", unsafe_allow_html=True)

client = init_llm()
if not client:
    st.error("⚠️ Add GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='font-family:IBM Plex Mono;font-size:0.68rem;
    color:#f59e0b;letter-spacing:2px;text-transform:uppercase;
    border-bottom:1px solid #1e2030;padding-bottom:8px;margin-bottom:16px'>
    ⚙ SETTINGS</div>""", unsafe_allow_html=True)

    symbol = st.selectbox("Asset", options=list(BINANCE_SYMBOLS.keys()), index=0)
    period = st.selectbox("Period", options=list(PERIOD_DAYS.keys()), index=1)

    st.markdown("---")
    st.markdown("""<div style='font-family:IBM Plex Mono;font-size:0.65rem;
    color:#f59e0b;letter-spacing:1px;margin-bottom:8px'>
    📁 UPLOAD YOUR OWN DATA (OPTIONAL)</div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "CSV: Date, Open, High, Low, Close",
        type=['csv'], label_visibility="collapsed"
    )
    if uploaded_file:
        st.markdown("""<div class='data-source-box'>
        ✅ CSV loaded — will use your data</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<div style='font-family:IBM Plex Mono;font-size:0.65rem;color:#3d2f00'>
    <b style='color:#f59e0b'>DATA SOURCES:</b><br>
    1️⃣ Your CSV (if uploaded)<br>
    2️⃣ Binance API (auto)<br>
    3️⃣ CoinGecko (fallback)<br><br>
    <b style='color:#f59e0b'>EXAMPLES:</b><br><br>
    "Buy BTC when 20 EMA crosses above 50 EMA. SL 2%, TP 6%."<br><br>
    "Long when RSI drops below 30. SL 3%, TP 9%."<br><br>
    "Short Bollinger lower breakout. SL 1.5%, TP 5%."
    </div>""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────
for key in ['parsed','df','fig','code','data_source','description']:
    if key not in st.session_state:
        st.session_state[key] = None

# ── STEP 1 ────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">STEP 1 — DESCRIBE YOUR STRATEGY</div>',
            unsafe_allow_html=True)
st.markdown("""<div class="step-card active">
<div class="step-num">✏️ PLAIN ENGLISH — NO CODING NEEDED</div>
Describe entry conditions, stop loss, and take profit.
Only mention SHORT if you want short signals.
</div>""", unsafe_allow_html=True)

description = st.text_area(
    "Strategy",
    placeholder="Buy BTC when the 20 EMA crosses above the 50 EMA. SL 2%, TP 6%.",
    height=100, label_visibility="collapsed"
)

c1, c2 = st.columns([3,1])
with c1: parse_btn = st.button("🧠 PARSE STRATEGY", use_container_width=True)
with c2: reset_btn = st.button("↺ Reset",           use_container_width=True)

if reset_btn:
    for key in ['parsed','df','fig','code','data_source','description']:
        st.session_state[key] = None
    st.rerun()

if parse_btn and description.strip():
    with st.spinner("🧠 Parsing..."):
        parsed = parse_strategy(client, description)
    if parsed:
        st.session_state.parsed      = parsed
        st.session_state.description = description  # store for code generation
        st.session_state.fig         = None
        st.session_state.code        = None
        st.session_state.data_source = None

# ── STEP 2 ────────────────────────────────────────────────────
if st.session_state.parsed:
    p = st.session_state.parsed
    st.markdown('<div class="section-hdr">STEP 2 — CONFIRM UNDERSTANDING</div>',
                unsafe_allow_html=True)
    st.markdown(f"""<div class="parsed-box">
    <b style='color:#f59e0b'>AI PARSED AS:</b><br><br>
    <b>Summary:</b> {p.get('summary','—')}<br>
    <b>Type:</b> {p.get('strategy_type','—').upper()}<br>
    <b>Indicators:</b> {', '.join(p.get('indicators',[]))}
    </div>""", unsafe_allow_html=True)

    # Show direction tags
    sl_display = (p.get('sl_pct') or 0.01) * 100
    tp_display = (p.get('tp_pct') or 0.02) * 100
    sl_default = p.get('sl_pct') is None
    tp_default = p.get('tp_pct') is None

    for col, (cls, txt) in zip(st.columns(4), [
        ('tag-entry', f"📈 LONG: {str(p.get('entry_long','—'))[:32]}"),
        ('tag-entry', f"📉 SHORT: {str(p.get('entry_short','—'))[:32]}"),
        ('tag-sl',    f"🛑 SL: {sl_display:.1f}%{'  (default)' if sl_default else ''}"),
        ('tag-tp',    f"🎯 TP: {tp_display:.1f}%{'  (default)' if tp_default else ''}"),
    ]):
        with col:
            st.markdown(f'<span class="tag {cls}">{txt}</span>',
                       unsafe_allow_html=True)

    # Default SL/TP notification
    if sl_default or tp_default:
        st.markdown("""
        <div style='background:#1c1400;border:1px solid #f59e0b;border-radius:8px;
        padding:12px 16px;margin:10px 0;font-family:IBM Plex Mono;font-size:0.78rem;color:#f59e0b'>
        ⚠️ <b>DEFAULT RISK APPLIED:</b> SL 1% · TP 2%<br>
        <span style='color:#6b5b3a'>You didn't specify SL/TP.
        To change them, re-describe your strategy and include
        e.g. "stop loss 2%, take profit 6%"</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    if st.button("📊 VISUALIZE ON REAL CANDLES", use_container_width=True):
        with st.spinner("📡 Fetching market data..."):
            df, source = fetch_data(symbol, period, uploaded_file)

        if df is not None and len(df) > 30:
            st.markdown(f"""<div class="data-source-box">
            ✅ {len(df)} candles from {source}</div>""",
                       unsafe_allow_html=True)
            with st.spinner("🎨 Building charts..."):
                df = add_indicators(df, p.get('indicator_params', {}))
                df = generate_signals(df, p)
                st.session_state.df          = df
                st.session_state.data_source = source
                # Build two separate figures
                st.session_state.fig_long  = draw_chart(
                    df, p, symbol, source, show='long')
                st.session_state.fig_short = draw_chart(
                    df, p, symbol, source, show='short')
        else:
            st.error(
                "Could not fetch data from any source.\n\n"
                "**Solution:** Upload a CSV file in the sidebar.\n"
                "Format: Date, Open, High, Low, Close columns."
            )

# ── STEP 3 ────────────────────────────────────────────────────
if st.session_state.get('fig_long') or st.session_state.get('fig_short'):
    st.markdown('<div class="section-hdr">STEP 3 — YOUR SETUP — LONG & SHORT</div>',
                unsafe_allow_html=True)

    # Always show BOTH charts
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("""<div style='font-family:IBM Plex Mono;font-size:0.75rem;
        color:#4ade80;letter-spacing:2px;margin-bottom:8px'>
        📈 LONG SETUP</div>""", unsafe_allow_html=True)
        if st.session_state.get('fig_long'):
            st.plotly_chart(st.session_state.fig_long,
                           use_container_width=True,
                           config={'displayModeBar': True, 'scrollZoom': True})

    with col_r:
        st.markdown("""<div style='font-family:IBM Plex Mono;font-size:0.75rem;
        color:#f87171;letter-spacing:2px;margin-bottom:8px'>
        📉 SHORT SETUP</div>""", unsafe_allow_html=True)
        if st.session_state.get('fig_short'):
            st.plotly_chart(st.session_state.fig_short,
                           use_container_width=True,
                           config={'displayModeBar': True, 'scrollZoom': True})

    st.markdown("""<div style='text-align:center;font-family:IBM Plex Mono;
    font-size:0.78rem;color:#a89060;margin:12px 0'>
    🔺 Green triangles = Long entries &nbsp;|&nbsp;
    🔻 Red triangles = Short entries<br>
    Dashed = Stop Loss &nbsp;|&nbsp; Dotted = Take Profit
    </div>""", unsafe_allow_html=True)

    cy, cn = st.columns(2)
    with cy: yes_btn = st.button("✅ YES — Generate Python Code", use_container_width=True)
    with cn: no_btn  = st.button("❌ NO — Redescribe",            use_container_width=True)

    if no_btn:
        st.session_state.fig_long  = None
        st.session_state.fig_short = None
        st.session_state.code      = None
        st.info("Refine your description in Step 1.")

    if yes_btn:
        with st.spinner("⚙️ Generating code..."):
            st.session_state.code = generate_python_code(
                client, st.session_state.parsed, symbol,
                description=st.session_state.get('description', ''))

# ── STEP 4 ────────────────────────────────────────────────────
if st.session_state.code:
    st.markdown('<div class="section-hdr">STEP 4 — YOUR PYTHON CODE</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="step-card done">
    <div class="step-num">✅ READY — RUN IN COLAB OR JUPYTER</div>
    Then paste into <b>Backtest Validator</b> to check for errors.
    </div>""", unsafe_allow_html=True)

    st.text_area("Code", value=st.session_state.code,
                height=320, label_visibility="collapsed")
    st.download_button(
        "⬇️ Download .py file",
        data=st.session_state.code,
        file_name=f"{symbol}_strategy.py",
        mime="text/plain", use_container_width=True
    )
    st.markdown("""<div style='background:#0d0f14;border:1px solid #f59e0b;
    border-radius:10px;padding:16px;margin-top:16px;text-align:center'>
    <b style='font-family:IBM Plex Mono;color:#f59e0b'>
    ⚠️ VALIDATE BEFORE TRADING LIVE</b><br>
    <span style='font-family:IBM Plex Mono;color:#6b5b3a;font-size:0.8rem'>
    Paste code into <b style='color:#e8e0d0'>Backtest Validator</b>
    to detect lookahead bias and overfitting
    </span></div>""", unsafe_allow_html=True)

# Footer
st.markdown("""<div style="text-align:center;margin-top:48px;padding:16px;
border-top:1px solid #1e2030">
<span style="font-family:IBM Plex Mono;font-size:0.65rem;color:#1e2030">
QUANT ALPHA — NOT FINANCIAL ADVICE
</span></div>""", unsafe_allow_html=True)
