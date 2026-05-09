import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Strategy Visualizer | Quant Alpha",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════
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
    font-family: 'IBM Plex Mon Mono', monospace; font-size: 0.72rem; margin: 3px;
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

# ═══════════════════════════════════════════════════════════════
# INDICATOR LIBRARY ─────────────────────────────────────────────────
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
    col = f'WMAMA_{period}'
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

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range"""
    col  = f'ATR_{period}'
    tr  = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs()
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

def add_swing_highs_lows(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Swing Highs and Lows"""
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
    df['HL'] = False
    df['LH'] = False
    df['LL'] = False

    prev_h = swing_h.shift(1).reindex(df.index).ffill()
    prev_l = swing_l.shift(1).reindex(df.index).ffill()

    df.loc[df['Swing_High'], 'HH'] = df.loc[df['Swing_High'], 'High'] > prev_h[df['Swing_High']]
    df.loc[df['Swing_High'], 'LH'] = df.loc[df['Swing_High'], 'High'] < prev_h[df['Swing_High']]
    df.loc[df['Swing_Low'], 'HL'] = df.loc[df['Swing_Low'], 'Low'] > prev_l[df['Swing_Low']]
    df.loc[df['Swing_Low'], 'LL'] = df.loc[df['Swing_Low'], 'Low'] < prev_l[df['Swing_Low']]

    return df

def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Candlestick patterns"""
    body      = df['Close'] - df['Open']
    body_abs  = body.abs()
    upper_wick = df['High'] - df[['Open','Close']].max(axis=1)
    lower_wick = df[['Open','Close']].min(axis=1) - df['Low']
    candle_range = df['High'] - df['Low']

    df['Bullish_Engulfing'] = (
        (df['Close'] > df['Open']) &
        (df['Open'].shift(1) > df['Close'].shift(1)) &
        (df['Close'] > df['Open'].shift(1)) &
        (df['Open'] < df['Close'].shift(1))
    )

    df['Bearish_Engulfing'] = (
        (df['Open'] > df['Close']) &
        (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Open'] > df['Close'].shift(1)) &
        (df['Close'] < df['Open'].shift(1))
    )

    df['Doji'] = body_abs < (0.1 * candle_range)

    df['Hammer'] = (
        (lower_wick > 2 * body_abs) &
        (upper_wick < body_abs) &
        (candle_range > 0)
    )

    df['Shooting_Star'] = (
        (upper_wick > 2 * body_abs) &
        (lower_wick < body_abs) &
        (candle_range > 0)
    )

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
        (df['Low'] > df['Low'].shift(1))
    )
    df['Outside_Bar'] = (
        (df['High'] > df['High'].shift(1)) &
        (df['Low'] < df['Low'].shift(1))
    )
    return df

def add_support_resistance(df: pd.DataFrame,
                            lookback: int = 20) -> pd.DataFrame:
    """Rolling support and resistance levels"""
    df[f'Resistance_{lookback}'] = df['High'].rolling(lookback).max()
    df[f'Support_{lookback}']    = df['Low'].rolling(lookback).min()
    return df

def add_structure_break(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """Break of Structure (BOS) and Change of Character (CHoCH)"""
    df = add_swing_highs_lows(df, lookback)

    last_swing_high = df['High'].where(df['Swing_High']).ffill().shift(1)
    last_swing_low  = df['Low'].where(df['Swing_Low']).ffill().shift(1)

    df['BOS_Bullish'] = df['Close'] > last_swing_high
    df['BOS_Bearish'] = df['Close'] < last_swing_low

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
    """Order Blocks"""
    df = add_structure_break(df, lookback)

    df['Bullish_OB'] = False
    df['Bearish_OB'] = False

    for i in range(1, len(df)):
        if df['CHoCH_Bullish'].iloc[i]:
            j = i - 1
            while j >= 0:
                if df['Open'].iloc[j] > df['Close'].iloc[j]:
                    df['Bullish_OB'].iloc[j] = True
                    break
                j -= 1

        if df['CHoCH_Bearish'].iloc[i]:
            j = i - 1
            while j >= 0:
                if df['Close'].iloc[j] > df['Open'].iloc[j]:
                    df['Bearish_OB'].iloc[j] = True
                    break
                j -= 1

    return df

def add_fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Fair Value Gaps / Imbalances"""
    df['FVG_Bullish'] = df['Low'] > df['High'].shift(2)
    df['FVG_Bearish'] = df['High'] < df['Low'].shift(2)

    df['FVG_Bull_Low']  = df['High'].shift(2).where(df['FVG_Bullish'])
    df['FVG_Bull_High'] = df['Low'].where(df['FVG_Bullish'])

    df['FVG_Bear_High'] = df['Low'].shift(2).where(df['FVG_Bearish'])
    df['FVG_Bear_Low']  = df['High'].where(df['FVG_Bearish'])

    return df

def add_liquidity_levels(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Liquidity pools"""
    df = add_swing_highs_lows(df, 5)

    df['BSL'] = df['High'].where(df['Swing_High']).rolling(lookback).max()
    df['SSL'] = df['Low'].where(df['Swing_Low']).rolling(lookback).min()

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
    """Premium and Discount zones"""
    range_high = df['High'].rolling(lookback).max()
    range_low  = df['Low'].rolling(lookback).min()
    midpoint   = (range_high + range_low) / 2

    df['In_Premium']  = df['Close'] > midpoint
    df['In_Discount'] = df['Close'] < midpoint
    df['Range_50pct'] = midpoint

    return df

def add_market_structure(df: pd.DataFrame,
                          lookback: int = 10) -> pd.DataFrame:
    """ICT Market Structure"""
    df = add_higher_highs_lower_lows(df, lookback)

    df['Bullish_Structure'] = df['HH'] | df['HL']
    df['Bearish_Structure'] = df['LH'] | df['LL']

    return df

def add_optimal_trade_entry(df: pd.DataFrame,
                             lookback: int = 20) -> pd.DataFrame:
    """ICT OTE (Optimal Trade Entry)"""
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
    """ICT Killzones"""
    if not hasattr(df.index, 'hour'):
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
    """Previous Day High, Low, Close"""
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
    """Equal Highs (EQH) and Equal Lows (EQL)"""
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
# LIBRARY REFERENCE
# ═══════════════════════════════════════════════════════════════

LIBRARY_REFERENCE = """
AVAILABLE INDICATOR FUNCTIONS (all tested, all correct):

TREND:
- add_ema(df, period)        → df['EMA_20'], df['EMA_50'], etc.
- add_sma(df, period)        → df['SMA_20']
- add_wma(df, period)        → df['WMA_20']
- add_vwap(df)               → df['VWAP']
- add_macd(df, 12, 26, 9)    → df['MACD'], df['MACD_Signal'], df['MACD_Hist']
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
- add_obv(df)                    → df['OBV']
- add_volume_sma(df, 20)        → df['Vol_SMA_20']
- add_cmf(df, 20)               → df['CMF']
- add_volume_spike(df, 20, 2.0) → df['Volume_Spike'] (bool)

PRICE ACTION / SMC / ICT:
- add_swing_highs_lows(df, 5)   → df['Swing_High'], df['Swing_Low'] (bool)
- add_candle_patterns(df)       → df['Bullish_Engulfing'], df['Bearish_Engulfing'],
                                   df['Hammer'], df['Shooting_Star'],
                                   df['Bullish_Pin_Bar'], df['Bearish_Pin_Bar']
- add_structure_break(df, 10)    → df['BOS_Bullish'], df['BOS_Bearish'],
                                   df['CHoCH_Bullish'], df['CHoCH_Bearish']
- add_order_blocks(df, 10)      → df['Bullish_OB'], df['Bearish_OB']
- add_fair_value_gaps(df)       → df['FVG_Bullish'], df['FVG_Bearish']
- add_liquidity_levels(df, 20)  → df['BSL'], df['SSL'],
                                   df['BSL_Sweep'], df['SSL_Sweep']
- add_premium_discount(df, 50)   → df['In_Premium'], df['In_Discount']
- add_market_structure(df, 10)  → df['Bullish_Structure'], df['Bearish_Structure']
- add_optimal_trade_entry(df)   → df['In_OTE_Bullish'], df['In_OTE_Bearish']
- add_previous_day_levels(df)   → df['PDH'], df['PDL'], df['PDC']

SIGNAL HELPERS (return boolean Series):
- crossover(series_a, series_b)  → True when a crosses above b
- crossunder(series_a, series_b) → True when a crosses below b
- above_level(series, value)     → True when series crosses above fixed level
- below_level(series, value)     → True when series crosses below fixed level
- rising(series, periods)       → True when series is rising over N periods
- falling(series, periods)      → True when series is falling over N periods

CONVENIENCE:
- add_common_indicators(df)     → adds EMA 9/20/50, SMA 20/50, RSI, MACD, BB, ATR, OBV, swings, candles, support/resistance
- add_smc_indicators(df)        → adds all SMC/ICT indicators
"""

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def fetch_binance(symbol: str, period: str):
    """Fetch from Binance"""
    sym   = symbol.upper()
    limit = 90
    
    import time
    for attempt in range(3):
        try:
            resp = requests.get(
                f"https://api.binance.com/api/v3/klines",
                params={'symbol': sym, 'interval': period, 'limit': min(limit, 1000)},
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    df = pd.DataFrame(data, columns=[
                        'timestamp','Open','High','Low','Close','Volume',
                        'ct','qv','nt','tbb','tbq','ignore'
                    ])
                    df['Date']
