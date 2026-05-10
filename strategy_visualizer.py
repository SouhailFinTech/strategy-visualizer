import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    border: 1px solid #3d2f00;
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 24px;
}
.main-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    color: #f59e0b;
    margin: 0;
    letter-spacing: -1px;
}
.main-header p { color: #6b5b3a; margin: 6px 0 0; font-size: 0.9rem; }
.step-card {
    background: #0d0f14;
    border: 1px solid #1e2030;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
}
.step-card.active { border-color: #f59e0b; }
.step-card.done { border-color: #22c55e; }
.step-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #f59e0b;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.parsed-box {
    background: #0a0c10;
    border: 1px solid #1e2030;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #a89060;
    margin: 12px 0;
}
.tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    margin: 3px;
}
.tag-entry { background: #1a2e1a; color: #4ade80; border: 1px solid #166534; }
.tag-sl { background: #2e1a1a; color: #f87171; border: 1px solid #991b1b; }
.tag-tp { background: #1a2a1a; color: #86efac; border: 1px solid #15803d; }
.data-source-box {
    background: #0a0c10;
    border: 1px solid #1e2030;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #60a5fa;
    margin: 8px 0;
}
.section-hdr {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #f59e0b;
    letter-spacing: 3px;
    text-transform: uppercase;
    border-bottom: 1px solid #1e2030;
    padding-bottom: 8px;
    margin: 20px 0 14px;
}
[data-testid="stSidebar"] {
    background: #06080c;
    border-right: 1px solid #1e2030;
}
.stButton > button {
    background: linear-gradient(135deg, #92400e, #b45309);
    color: #fef3c7;
    border: none;
    border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    padding: 12px 24px;
    width: 100%;
    transition: all 0.2s;
    letter-spacing: 1px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #b45309, #d97706);
    transform: translateY(-1px);
}
.stTextArea textarea, .stTextInput input {
    background: #0a0c10 !important;
    color: #e8e0d0 !important;
    border: 1px solid #1e2030 !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
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

PERIOD_DAYS = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730}
BINANCE_LIMITS = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730}

# ═══════════════════════════════════════════════════════════════
# GROQ INIT
# ═══════════════════════════════════════════════════════════════
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

def init_llm():
    if not GROQ_AVAILABLE:
        st.error("groq package not installed.")
        return None
    try:
        return Groq(api_key=st.secrets["GROQ_API_KEY"])
    except Exception as e:
        st.error(f"Groq error: {e}")
        return None

# ═══════════════════════════════════════════════════════════════
# INDICATOR LIBRARY
# ═══════════════════════════════════════════════════════════════

def add_ema(df: pd.DataFrame, period: int) -> pd.DataFrame:
    col = f'EMA_{period}'
    df[col] = df['Close'].ewm(span=period, adjust=False).mean()
    return df

def add_sma(df: pd.DataFrame, period: int) -> pd.DataFrame:
    col = f'SMA_{period}'
    df[col] = df['Close'].rolling(period).mean()
    return df

def add_wma(df: pd.DataFrame, period: int) -> pd.DataFrame:
    col = f'WMA_{period}'
    weights = np.arange(1, period + 1)
    df[col] = df['Close'].rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )
    return df

def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df

def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

def add_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    df = add_atr(df, period)
    atr = df[f'ATR_{period}']
    hl2 = (df['High'] + df['Low']) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
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
    df['Supertrend'] = supertrend
    df['Supertrend_Direction'] = direction
    return df

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    col = f'RSI_{period}'
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    df[col] = 100 - (100 / (1 + rs))
    return df

def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    low_min = df['Low'].rolling(k_period).min()
    high_max = df['High'].rolling(k_period).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['Stoch_D'] = df['Stoch_K'].rolling(d_period).mean()
    return df

def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    df[f'CCI_{period}'] = (tp - sma_tp) / (0.015 * mad)
    return df

def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high_max = df['High'].rolling(period).max()
    low_min = df['Low'].rolling(period).min()
    df[f'WR_{period}'] = -100 * (high_max - df['Close']) / (high_max - low_min)
    return df

def add_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
    mfr = pos_mf / neg_mf.replace(0, np.nan)
    df[f'MFI_{period}'] = 100 - (100 / (1 + mfr))
    return df

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    col = f'ATR_{period}'
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df[col] = tr.ewm(span=period, adjust=False).mean()
    return df

def add_bollinger(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    df['BB_Mid'] = df['Close'].rolling(period).mean()
    std = df['Close'].rolling(period).std()
    df['BB_Upper'] = df['BB_Mid'] + std_dev * std
    df['BB_Lower'] = df['BB_Mid'] - std_dev * std
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
    df['BB_Pct'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    return df

def add_keltner(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0) -> pd.DataFrame:
    df = add_atr(df, atr_period)
    df['KC_Mid'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    df['KC_Upper'] = df['KC_Mid'] + multiplier * df[f'ATR_{atr_period}']
    df['KC_Lower'] = df['KC_Mid'] - multiplier * df[f'ATR_{atr_period}']
    return df

def add_donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df[f'DC_Upper_{period}'] = df['High'].rolling(period).max()
    df[f'DC_Lower_{period}'] = df['Low'].rolling(period).min()
    df[f'DC_Mid_{period}'] = (df[f'DC_Upper_{period}'] + df[f'DC_Lower_{period}']) / 2
    return df

def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    direction = np.sign(df['Close'].diff())
    df['OBV'] = (direction * df['Volume']).fillna(0).cumsum()
    return df

def add_volume_sma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df[f'Vol_SMA_{period}'] = df['Volume'].rolling(period).mean()
    return df

def add_cmf(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / \
          (df['High'] - df['Low']).replace(0, np.nan)
    df['CMF'] = (clv * df['Volume']).rolling(period).sum() / \
                df['Volume'].rolling(period).sum()
    return df

def add_volume_spike(df: pd.DataFrame, period: int = 20, multiplier: float = 2.0) -> pd.DataFrame:
    df = add_volume_sma(df, period)
    df['Volume_Spike'] = df['Volume'] > (multiplier * df[f'Vol_SMA_{period}'])
    return df

def add_swing_highs_lows(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    df['Swing_High'] = False
    df['Swing_Low'] = False
    for i in range(lookback, len(df) - lookback):
        window_high = df['High'].iloc[i-lookback:i+lookback+1]
        window_low = df['Low'].iloc[i-lookback:i+lookback+1]
        if df['High'].iloc[i] == window_high.max():
            df['Swing_High'].iloc[i] = True
        if df['Low'].iloc[i] == window_low.min():
            df['Swing_Low'].iloc[i] = True
    return df

def add_higher_highs_lower_lows(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
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
    df.loc[df['Swing_Low'], 'HL'] = df.loc[df['Swing_Low'], 'Low'] > prev_l[df['Swing_Low']]
    df.loc[df['Swing_Low'], 'LL'] = df.loc[df['Swing_Low'], 'Low'] < prev_l[df['Swing_Low']]
    return df

def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    body = df['Close'] - df['Open']
    body_abs = body.abs()
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
        (df['Close'] < df['Open']) &
        (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Open'] > df['Close'].shift(1)) &
        (df['Close'] < df['Open'].shift(1))
    )
    df['Doji'] = body_abs < (0.1 * candle_range)
    df['Hammer'] = (lower_wick > 2 * body_abs) & (upper_wick < body_abs) & (candle_range > 0)
    df['Shooting_Star'] = (upper_wick > 2 * body_abs) & (lower_wick < body_abs) & (candle_range > 0)
    df['Bullish_Pin_Bar'] = (lower_wick > 2 * body_abs) & (lower_wick > upper_wick * 2)
    df['Bearish_Pin_Bar'] = (upper_wick > 2 * body_abs) & (upper_wick > lower_wick * 2)
    return df

def add_inside_outside_bars(df: pd.DataFrame) -> pd.DataFrame:
    df['Inside_Bar'] = (df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))
    df['Outside_Bar'] = (df['High'] > df['High'].shift(1)) & (df['Low'] < df['Low'].shift(1))
    return df

def add_support_resistance(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    df[f'Resistance_{lookback}'] = df['High'].rolling(lookback).max()
    df[f'Support_{lookback}'] = df['Low'].rolling(lookback).min()
    return df

def add_structure_break(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    df = add_swing_highs_lows(df, lookback)
    last_swing_high = df['High'].where(df['Swing_High']).ffill().shift(1)
    last_swing_low = df['Low'].where(df['Swing_Low']).ffill().shift(1)
    df['BOS_Bullish'] = df['Close'] > last_swing_high
    df['BOS_Bearish'] = df['Close'] < last_swing_low
    df['CHoCH_Bullish'] = df['BOS_Bullish'] & ~df['BOS_Bullish'].shift(1).fillna(False)
    df['CHoCH_Bearish'] = df['BOS_Bearish'] & ~df['BOS_Bearish'].shift(1).fillna(False)
    return df

def add_order_blocks(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
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
    df['FVG_Bullish'] = df['Low'] > df['High'].shift(2)
    df['FVG_Bearish'] = df['High'] < df['Low'].shift(2)
    df['FVG_Bull_Low'] = df['High'].shift(2).where(df['FVG_Bullish'])
    df['FVG_Bull_High'] = df['Low'].where(df['FVG_Bullish'])
    df['FVG_Bear_High'] = df['Low'].shift(2).where(df['FVG_Bearish'])
    df['FVG_Bear_Low'] = df['High'].where(df['FVG_Bearish'])
    return df

def add_liquidity_levels(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    df = add_swing_highs_lows(df, 5)
    df['BSL'] = df['High'].where(df['Swing_High']).rolling(lookback).max()
    df['SSL'] = df['Low'].where(df['Swing_Low']).rolling(lookback).min()
    df['BSL_Sweep'] = (df['High'] > df['BSL'].shift(1)) & (df['Close'] < df['BSL'].shift(1))
    df['SSL_Sweep'] = (df['Low'] < df['SSL'].shift(1)) & (df['Close'] > df['SSL'].shift(1))
    return df

def add_premium_discount(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    range_high = df['High'].rolling(lookback).max()
    range_low = df['Low'].rolling(lookback).min()
    midpoint = (range_high + range_low) / 2
    df['In_Premium'] = df['Close'] > midpoint
    df['In_Discount'] = df['Close'] < midpoint
    df['Range_50pct'] = midpoint
    return df

def add_market_structure(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    df = add_higher_highs_lower_lows(df, lookback)
    df['Bullish_Structure'] = df['HH'] | df['HL']
    df['Bearish_Structure'] = df['LH'] | df['LL']
    return df

def add_optimal_trade_entry(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    df = add_swing_highs_lows(df, lookback // 4)
    swing_h = df['High'].where(df['Swing_High']).ffill()
    swing_l = df['Low'].where(df['Swing_Low']).ffill()
    fib_618 = swing_h - 0.618 * (swing_h - swing_l)
    fib_786 = swing_h - 0.786 * (swing_h - swing_l)
    df['In_OTE_Bullish'] = (df['Close'] >= fib_786) & (df['Close'] <= fib_618)
    fib_618_bear = swing_l + 0.618 * (swing_h - swing_l)
    fib_786_bear = swing_l + 0.786 * (swing_h - swing_l)
    df['In_OTE_Bearish'] = (df['Close'] <= fib_786_bear) & (df['Close'] >= fib_618_bear)
    return df

def add_killzones(df: pd.DataFrame) -> pd.DataFrame:
    if not hasattr(df.index, 'hour'):
        df['London_KZ'] = False
        df['NewYork_KZ'] = False
        df['LondonClose_KZ'] = False
        df['Asian_KZ'] = False
        return df
    h = df.index.hour
    df['London_KZ'] = (h >= 2) & (h < 5)
    df['NewYork_KZ'] = (h >= 7) & (h < 10)
    df['LondonClose_KZ'] = (h >= 10) & (h < 12)
    df['Asian_KZ'] = (h >= 20) | (h < 1)
    return df

def add_previous_day_levels(df: pd.DataFrame) -> pd.DataFrame:
    df['PDH'] = df['High'].shift(1)
    df['PDL'] = df['Low'].shift(1)
    df['PDC'] = df['Close'].shift(1)
    return df

def add_weekly_levels(df: pd.DataFrame) -> pd.DataFrame:
    df['Week'] = df.index.to_series().dt.isocalendar().week
    df['PWH'] = df.groupby('Week')['High'].transform('max').shift(1)
    df['PWL'] = df.groupby('Week')['Low'].transform('min').shift(1)
    df = df.drop(columns=['Week'])
    return df

def add_equal_highs_lows(df: pd.DataFrame, tolerance: float = 0.001) -> pd.DataFrame:
    df = add_swing_highs_lows(df, 5)
    df['EQH'] = df['Swing_High'] & (abs(df['High'] - df['High'].shift(1)) / df['High'].shift(1) < tolerance) & df['Swing_High'].shift(1)
    df['EQL'] = df['Swing_Low'] & (abs(df['Low'] - df['Low'].shift(1)) / df['Low'].shift(1) < tolerance) & df['Swing_Low'].shift(1)
    return df

def crossover(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    return (series_a > series_b) & (series_a.shift(1) <= series_b.shift(1))

def crossunder(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    return (series_a < series_b) & (series_a.shift(1) >= series_b.shift(1))

def above_level(series: pd.Series, level: float) -> pd.Series:
    return (series > level) & (series.shift(1) <= level)

def below_level(series: pd.Series, level: float) -> pd.Series:
    return (series < level) & (series.shift(1) >= level)

def rising(series: pd.Series, periods: int = 1) -> pd.Series:
    return series > series.shift(periods)

def falling(series: pd.Series, periods: int = 1) -> pd.Series:
    return series < series.shift(periods)

def add_common_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════
def fetch_binance(symbol: str, period: str):
    sym = BINANCE_SYMBOLS.get(symbol.upper())
    limit = BINANCE_LIMITS.get(period, 90)
    if not sym:
        return None
    import time
    for attempt in range(3):
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': sym, 'interval': '1d', 'limit': min(limit, 1000)},
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
            if attempt < 2:
                time.sleep(2)
        except Exception:
            if attempt < 2:
                time.sleep(2)
    return None

def fetch_coingecko(symbol: str, days: int):
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
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}",
            timeout=15, headers=headers
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data or not isinstance(data, list):
            return None
        df = pd.DataFrame(data, columns=['timestamp','Open','High','Low','Close'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('Date').drop('timestamp', axis=1).astype(float)
        df['Volume'] = 0.0
        return df.resample('D').last().dropna()
    except Exception:
        return None

def clean_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate column names by appending suffix"""
    cols = df.columns
    if cols.duplicated().any():
        cols = pd.io.parsers.ParserBase({'names': cols})._maybe_dedup_names(cols)
        df.columns = cols
    return df

def clean_duplicate_index(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate index values by keeping first occurrence"""
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep='first')]
    return df

def load_csv(uploaded_file):
    """Load CSV with MT5 and standard format support"""
    import io
    try:
        raw = uploaded_file.read()
        uploaded_file.seek(0)
        
        try:
            content = raw.decode('utf-8')
            lines = content.strip().split('\n')
            is_mt5 = ('<DATE>' in lines[0] or '\t' in lines[0] or lines[0].startswith('<'))
            
            if is_mt5:
                header = lines[0].replace('<','').replace('>','').strip()
                cols = [c.strip().lower() for c in header.split('\t')]
                rows = []
                for line in lines[1:]:
                    if line.strip():
                        rows.append(line.strip().split('\t'))
                df = pd.DataFrame(rows, columns=cols)
                
                if 'date' in cols and 'time' in cols:
                    df['datetime'] = pd.to_datetime(
                        df['date'] + ' ' + df['time'],
                        format='%Y.%m.%d %H:%M:%S', errors='coerce'
                    )
                    mask = df['datetime'].isna()
                    if mask.any():
                        df.loc[mask, 'datetime'] = pd.to_datetime(
                            df.loc[mask,'date'] + ' ' + df.loc[mask,'time'],
                            format='%Y.%m.%d %H:%M', errors='coerce'
                        )
                    df = df.set_index('datetime')
                elif 'date' in cols:
                    df.index = pd.to_datetime(df['date'], format='%Y.%m.%d', errors='coerce')
                    df = df.drop(columns=['date'], errors='ignore')
                
                df.index.name = 'Date'
                rename_map = {
                    'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                    'vol': 'Volume', 'tickvol': 'Volume', 'tick volume': 'Volume', 'volume': 'Volume'
                }
                df = df.rename(columns=rename_map)
                required = ['Open','High','Low','Close']
                if any(c not in df.columns for c in required):
                    st.error(f"MT5 CSV missing columns: {[c for c in required if c not in df.columns]}")
                    return None
                if 'Volume' not in df.columns:
                    df['Volume'] = 0.0
                df = df[['Open','High','Low','Close','Volume']].astype(float)
                df = df.dropna().sort_index()
                
                df = clean_duplicate_columns(df)
                df = clean_duplicate_index(df)
                
                if len(df) > 0:
                    st.success(f"✅ MT5 format detected — {len(df):,} bars loaded")
                    return df
        except Exception:
            pass
        
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip().title() for c in df.columns]
        
        date_col = next(
            (c for c in ['Date','Datetime','Timestamp','Time','Open Time'] if c in df.columns),
            None
        )
        if date_col is None:
            first_col = df.columns[0]
            try:
                pd.to_datetime(df[first_col].iloc[0])
                date_col = first_col
            except Exception:
                st.error("Cannot find date column. Expected: Date, Datetime, Timestamp, or Time")
                return None
        
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.set_index(date_col)
        df.index.name = 'Date'
        
        required = ['Open','High','Low','Close']
        if any(c not in df.columns for c in required):
            st.error(f"CSV missing columns: {[c for c in required if c not in df.columns]}")
            return None
        if 'Volume' not in df.columns:
            df['Volume'] = 0.0
        
        df = df[['Open','High','Low','Close','Volume']].astype(float)
        df = df.dropna().sort_index()
        
        df = clean_duplicate_columns(df)
        df = clean_duplicate_index(df)
        
        return df
    except Exception as e:
        st.error(f"CSV error: {e}")
        return None

def fetch_data(symbol, period, uploaded_file=None):
    if uploaded_file is not None:
        uploaded_file.seek(0)
        df = load_csv(uploaded_file)
        if df is not None and len(df) > 30:
            return df, "📁 Your CSV"
        st.warning("CSV uploaded but could not be read. Trying live data sources...")
    
    with st.spinner("📡 Trying Binance..."):
        df = fetch_binance(symbol, period)
    if df is not None and len(df) > 30:
        return df, "🟡 Binance"
    
    with st.spinner("📡 Trying CoinGecko..."):
        days = PERIOD_DAYS.get(period, 90)
        df = fetch_coingecko(symbol, days)
    if df is not None and len(df) > 30:
        return df, "🦎 CoinGecko"
    
    return None, None

# ═══════════════════════════════════════════════════════════════
# GROQ HELPERS
# ═══════════════════════════════════════════════════════════════
def parse_strategy(client, description: str):
    prompt = """You are a quantitative trading expert.
Parse this trading strategy into structured JSON.
Strategy: \"""" + description + "\""""
Return ONLY valid JSON:
{
  "entry_long": "long entry condition or null if no long",
  "entry_short": "short entry condition or null if no short",
  "stop_loss": "stop loss description",
  "take_profit": "take profit description",
  "indicators": ["list of indicators"],
  "strategy_type": "trend or mean-reversion or breakout or momentum",
  "sl_pct": 0.02,
  "tp_pct": 0.06,
  "indicator_params": {"ema_fast": 20, "ema_slow": 50, "rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30},
  "summary": "one sentence summary"
}
IMPORTANT: If user only says BUY/LONG — set entry_short to null. If user only says SELL/SHORT — set entry_long to null."""

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
        result['sl_pct'] = result.get('sl_pct') or 0.02
        result['tp_pct'] = result.get('tp_pct') or 0.06
        result['strategy_type'] = result.get('strategy_type') or 'trend'
        result['summary'] = result.get('summary') or 'Trading Strategy'
        result['indicators'] = result.get('indicators') or []
        result['indicator_params'] = result.get('indicator_params') or {
            'ema_fast': 20, 'ema_slow': 50, 'rsi_period': 14, 'rsi_overbought': 70, 'rsi_oversold': 30
        }
        ip = result['indicator_params']
        ip['ema_fast'] = ip.get('ema_fast') or 20
        ip['ema_slow'] = ip.get('ema_slow') or 50
        ip['rsi_period'] = ip.get('rsi_period') or 14
        ip['rsi_overbought'] = ip.get('rsi_overbought') or 70
        ip['rsi_oversold'] = ip.get('rsi_oversold') or 30
        return result
    except Exception as e:
        st.error(f"Parse error: {e}")
        return None

def detect_advanced_features(description: str) -> dict:
    d = description.lower()
    return {
        'has_partial_close': any(w in d for w in ['partial', 'scale out', 'partial close']),
        'has_trailing_stop': any(w in d for w in ['trail', 'trailing']),
        'has_both_directions': any(w in d for w in ['short', 'sell when', 'both']),
        'has_long': any(w in d for w in ['buy', 'long', 'bullish']),
    }

def generate_signal_block(client, description: str, strategy: dict) -> str:
    features = detect_advanced_features(description)
    has_long = features['has_long'] or not features['has_both_directions']
    has_short = features['has_both_directions']
    stype = strategy.get('strategy_type', 'trend')
    indicators = strategy.get('indicators', [])
    params = strategy.get('indicator_params', {}) or {}

    if has_long and has_short:
        direction = "BOTH long AND short"
        signal_template = """df['long_signal'] = <YOUR_LONG_CONDITION>.fillna(False)
df['short_signal'] = <YOUR_SHORT_CONDITION>.fillna(False)
df['Signal'] = df['long_signal'].astype(int) - df['short_signal'].astype(int)"""
    elif has_short:
        direction = "SHORT only"
        signal_template = """df['long_signal'] = pd.Series(False, index=df.index)
df['short_signal'] = <YOUR_SHORT_CONDITION>.fillna(False)
df['Signal'] = -df['short_signal'].astype(int)"""
    else:
        direction = "LONG only"
        signal_template = """df['long_signal'] = <YOUR_LONG_CONDITION>.fillna(False)
df['short_signal'] = pd.Series(False, index=df.index)
df['Signal'] = df['long_signal'].astype(int)"""

    ind_hint = "Strategy type: " + stype + "\nIndicators: " + ', '.join(indicators) if indicators else 'detect from description'
    if params.get('ema_fast'):
        ind_hint += "\nEMA fast: " + str(params['ema_fast'])
    if params.get('ema_slow'):
        ind_hint += "\nEMA slow: " + str(params['ema_slow'])
    if params.get('rsi_period'):
        ind_hint += "\nRSI: " + str(params['rsi_period'])

    prompt = """You are a Python quant developer. Translate this trading strategy into Python signal detection code.
STRATEGY: \"""" + description + "\""""
DIRECTION: """ + direction + """
""" + ind_hint + """

AVAILABLE FUNCTIONS:
INDICATORS: add_ema, add_sma, add_rsi, add_macd, add_bollinger, add_atr, add_stochastic, add_vwap, add_obv, add_volume_spike, add_swing_highs_lows, add_candle_patterns, add_structure_break, add_fair_value_gaps, add_liquidity_levels, add_order_blocks, add_premium_discount, add_market_structure, add_optimal_trade_entry, add_equal_highs_lows, add_higher_highs_lower_lows, add_support_resistance, add_previous_day_levels, add_supertrend, add_cci, add_williams_r, add_mfi, add_donchian, add_keltner, add_inside_outside_bars, add_common_indicators, add_smc_indicators
SIGNAL HELPERS: crossover, crossunder, above_level, below_level, rising, falling

OUTPUT FORMAT:
""" + signal_template + """
RULES: First call add_*() for indicators, then write signal conditions. Both long_signal and short_signal MUST be assigned. df['Signal'] MUST be last line. Output ONLY Python lines — no imports, no def, no markdown.
OUTPUT ONLY THE PYTHON LINES NOW:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800, temperature=0.0
        )
        text = response.choices[0].message.content.strip()
        if '```python' in text:
            text = text.split('```python')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        lines = [l for l in text.strip().splitlines() if not l.strip().startswith(('import ', 'from ')) and l.strip()]
        joined = '\n'.join(lines)

        if "df['Signal']" not in joined and 'df["Signal"]' not in joined:
            if has_long and has_short:
                lines.append("df['Signal'] = df['long_signal'].astype(int) - df['short_signal'].astype(int)")
            elif has_short:
                lines.append("df['Signal'] = -df['short_signal'].astype(int)")
            else:
                lines.append("df['Signal'] = df['long_signal'].astype(int)")
        if "long_signal" not in joined:
            lines.insert(0, "df['long_signal'] = pd.Series(False, index=df.index)")
        if "short_signal" not in joined:
            lines.insert(0, "df['short_signal'] = pd.Series(False, index=df.index)")
        
        repaired = []
        for line in lines:
            line_str = str(line)
            if (("long_signal'] =" in line_str or "long_signal'] =" in line_str) and 'fillna' not in line_str and 'pd.Series' not in line_str and 'astype' not in line_str:
                line_str = line_str.rstrip() + '.fillna(False)'
            if (("short_signal'] =" in line_str or "short_signal'] =" in line_str) and 'fillna' not in line_str and 'pd.Series' not in line_str and 'astype' not in line_str:
                line_str = line_str.rstrip() + '.fillna(False)'
            repaired.append(line_str)
        lines = repaired

        return '\n'.join('    ' + line.lstrip() if line.strip() else '' for line in lines)
    except Exception as e:
        if has_long and has_short:
            sig = "    df['Signal'] = df['long_signal'].astype(int) - df['short_signal'].astype(int)"
        elif has_short:
            sig = "    df['Signal'] = -df['short_signal'].astype(int)"
        else:
            sig = "    df['Signal'] = df['long_signal'].astype(int)"
        return "    df['long_signal'] = pd.Series(False, index=df.index)\n    df['short_signal'] = pd.Series(False, index=df.index)\n" + sig + "\n    # Signal generation failed: " + str(e)

def generate_python_code(client, strategy: dict, symbol: str, description: str = '') -> str:
    binance_sym = BINANCE_SYMBOLS.get(symbol.upper(), 'BTCUSDT')
    sl_pct = strategy.get('sl_pct') or 0.01
    tp_pct = strategy.get('tp_pct') or 0.02
    summary = strategy.get('summary', 'Trading Strategy')
    features = detect_advanced_features(description)
    has_trailing = features['has_trailing_stop']
    has_partial = features['has_partial_close']

    signal_block = generate_signal_block(client, description, strategy)

    try:
        lib_lines = []
        in_lib = False
        with open(__file__, 'r') as f:
            for line in f:
                if '# ═══' in line and 'QUANT ALPHA INDICATOR LIBRARY' in line:
                    in_lib = True
                if in_lib:
                    lib_lines.append(line)
                if in_lib and 'LIBRARY_REFERENCE' in line and '"""' in line and len(lib_lines) > 5:
                    break
        lib_code = ''.join(lib_lines[:400])
    except Exception:
        lib_code = "# indicator library not found"

    trail_arg = "trail_pct=0.03" if has_trailing else "trail_pct=None"
    partial_arg = "partial_close_pct=0.3" if has_partial else "partial_close_pct=None"

    # FIXED: Use regular string concatenation instead of triple-quoted f-string
    code_part1 = """import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Indicator Library
"""
    
    code_part2 = lib_code
    
    code_part3 = """def fetch_data():
    try:
        resp = requests.get("https://api.binance.com/api/v3/klines", params={"symbol": """"
    
    code_part4 = binance_sym
    
    code_part5 = """", "interval": "1d", "limit": 365}})
        resp.raise_for_status()
        raw = resp.json()
        df = pd.DataFrame(raw, columns=[
            "Open time","Open","High","Low","Close","Volume",
            "ct","qv","nt","tbb","tbq","ignore"
        ])
        df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
        for col in ["Open","High","Low","Close","Volume"]:
            df[col] = pd.to_numeric(df[col])
        return df.drop(columns=["ct","qv","nt","tbb","tbq","ignore"])
    except Exception as e:
        print(f"Data error: {e}")
        return None

def generate_signals(df):
"""
    
    code_part6 = signal_block
    
    code_part7 = """    df["Position"] = df["Signal"].shift(1).fillna(0)
    return df

def backtest(df, sl_pct="""
    
    code_part8 = str(sl_pct)
    
    code_part9 = """, tp_pct="""
    
    code_part10 = str(tp_pct)
    
    code_part11 = """, trail_pct=None, partial_close_pct=None):
    df["Return"] = df["Close"].pct_change()
    if trail_pct is None:
        df["Commission"] = np.where(df["Position"] != df["Position"].shift(1), 0.001, 0)
        df["Strategy_Return"] = df["Return"] * df["Position"] - df["Commission"]
        df["BH_Equity"] = (1 + df["Return"]).cumprod()
        df["Strategy_Equity"] = (1 + df["Strategy_Return"]).cumprod()
    else:
        equity, bh_eq = 1.0, 1.0
        equities, bh_equities = [], []
        position, entry_price = 0, None
        trail_high, trail_low = None, None
        partial_done = False
        prices = df["Close"].values
        returns = df["Return"].fillna(0).values
        for i in range(len(df)):
            bh_eq *= (1 + returns[i])
            new_pos = int(df["Position"].iloc[i])
            if position == 0 and new_pos != 0:
                position, entry_price = new_pos, prices[i]
                trail_high = trail_low = prices[i]
                partial_done = False
                equity *= (1 - 0.001)
            elif position != 0:
                price = prices[i]
                if position == 1:
                    trail_high = max(trail_high, price)
                    trail_stop = trail_high * (1 - trail_pct)
                    hit_sl = price <= entry_price * (1 - sl_pct)
                    hit_tp = price >= entry_price * (1 + tp_pct)
                    hit_trail = price <= trail_stop
                else:
                    trail_low = min(trail_low, price)
                    trail_stop = trail_low * (1 + trail_pct)
                    hit_sl = price >= entry_price * (1 + sl_pct)
                    hit_tp = price <= entry_price * (1 - tp_pct)
                    hit_trail = price >= trail_stop
                if partial_close_pct and not partial_done and hit_tp:
                    equity *= (1 + (price/entry_price-1)*position*partial_close_pct - 0.001)
                    partial_done = True
                if hit_sl or hit_trail or (hit_tp and not partial_close_pct):
                    remaining = (1-partial_close_pct) if partial_done and partial_close_pct else 1.0
                    equity *= (1 + (price/entry_price-1)*position*remaining - 0.001)
                    position, entry_price = 0, None
                else:
                    equity *= (1 + returns[i] * position)
            equities.append(equity)
            bh_equities.append(bh_eq)
        df["Strategy_Equity"] = equities
        df["BH_Equity"] = bh_equities
        df["Strategy_Return"] = pd.Series(equities).pct_change().fillna(0).values
    return df

def metrics(df):
    r = df["Strategy_Return"].dropna()
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    roll_max = df["Strategy_Equity"].cummax()
    max_dd = ((df["Strategy_Equity"] - roll_max) / roll_max).min()
    win_rate = (r > 0).mean()
    total_r = df["Strategy_Equity"].iloc[-1] - 1
    n_trades = int((df["Position"] != df["Position"].shift(1)).sum() / 2)
    return sharpe, max_dd, win_rate, total_r, n_trades

def plot_results(df):
    ind_cols = [c for c in df.columns if any(c.startswith(p) for p in ["EMA_","SMA_","BB_","RSI_","MACD","Stoch","ATR_","WMA_","VWAP"])]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.70, 0.30])
    fig.add_trace(go.Candlestick(x=df["Open time"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350", increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350"), row=1, col=1)
    colors = ["#f59e0b","#60a5fa","#a78bfa","#34d399","#f87171"]
    for i, col in enumerate(ind_cols[:5]):
        if col in df.columns and not df[col].isna().all():
            if col == 'BB_Upper' and 'BB_Lower' in df.columns:
                fig.add_trace(go.Scatter(x=df["Open time"], y=df["BB_Upper"], name="BB Upper", line=dict(color="#f59e0b", width=1, dash="dash"), opacity=0.6), row=1, col=1)
                fig.add_trace(go.Scatter(x=df["Open time"], y=df["BB_Lower"], name="BB Lower", line=dict(color="#f59e0b", width=1, dash="dash"), fill="tonexty", fillcolor="rgba(245,158,11,0.05)", opacity=0.6), row=1, col=1)
            elif col not in ['BB_Lower','BB_Mid']:
                fig.add_trace(go.Scatter(x=df["Open time"], y=df[col], name=col, line=dict(color=colors[i % len(colors)], width=1.5), opacity=0.9), row=1, col=1)
    rsi_cols = [c for c in df.columns if c.startswith('RSI_')]
    has_rsi = len(rsi_cols) > 0
    long_df = df[df['long_signal']] if 'long_signal' in df.columns else df.iloc[0:0]
    if not long_df.empty:
        fig.add_trace(go.Scatter(x=long_df.index, y=long_df['Close']*0.994, mode='markers', name='Long Entry', marker=dict(symbol='triangle-up', size=14, color='#4ade80', line=dict(color='#166534', width=1))), row=1, col=1)
        for date, row in long_df.iterrows():
            entry, sl, tp = float(row['Close']), entry * (1 - sl_pct), entry * (1 + tp_pct)
            try:
                end_date = df.index[min(df.index.get_loc(date)+8, len(df)-1)]
            except: end_date = date
            fig.add_shape(type='line', x0=date, x1=end_date, y0=sl, y1=sl, line=dict(color='#ef4444', width=1.2, dash='dash'), row=1, col=1)
            fig.add_shape(type='line', x0=date, x1=end_date, y0=tp, y1=tp, line=dict(color='#4ade80', width=1.2, dash='dot'), row=1, col=1)
    short_df = df[df['short_signal']] if 'short_signal' in df.columns else df.iloc[0:0]
    if not short_df.empty:
        fig.add_trace(go.Scatter(x=short_df.index, y=short_df['Close']*1.006, mode='markers', name='Short Entry', marker=dict(symbol='triangle-down', size=14, color='#f87171', line=dict(color='#991b1b', width=1))), row=1, col=1)
        for date, row in short_df.iterrows():
            entry = float(row['Close'])
            sl = entry * (1 + sl_pct)
            tp = entry * (1 - tp_pct)
            try:
                end_date = df.index[min(df.index.get_loc(date)+8, len(df)-1)]
            except: end_date = date
            fig.add_shape(type='line', x0=date, x1=end_date, y0=sl, y1=sl, line=dict(color='#ef4444', width=1.2, dash='dash'), row=1, col=1)
            fig.add_shape(type='line', x0=date, x1=end_date, y0=tp, y1=tp, line=dict(color='#4ade80', width=1.2, dash='dot'), row=1, col=1)
    
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        bar_colors_list = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['Close'], df['Open'])]
        unique_colors = []
        seen_indices = set()
        for i, color in enumerate(bar_colors_list):
            if i not in seen_indices:
                unique_colors.append(color)
                seen_indices.add(i)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=unique_colors, opacity=0.6), row=2, col=1)
    elif has_rsi:
        rsi_col = rsi_cols[0]
        fig.add_trace(go.Scatter(x=df.index, y=df[rsi_col], name=rsi_col, line=dict(color='#a78bfa', width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_color='#ef4444', line_dash='dash', line_width=1, opacity=0.6, row=2, col=1)
        fig.add_hline(y=30, line_color='#4ade80', line_dash='dash', line_width=1, opacity=0.6, row=2, col=1)

    fig.add_trace(go.Scatter(x=df["Open time"], y=df["Strategy_Equity"], name="Strategy", line=dict(color="#4ade80", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Open time"], y=df["BH_Equity"], name="Buy & Hold", line=dict(color="#64748b", width=1.5, dash="dash")), row=2, col=1)
    fig.update_layout(height=700, paper_bgcolor="#080a0f", plot_bgcolor="#0d0f14", xaxis_rangeslider_visible=False, font=dict(color="#a89060"), title=dict(text="<b>" + symbol + "</b> — " + summary + "<br><span style='font-size:11px;color:#6b5b3a'>" + str(n_long) + " Long  " + str(n_short) + " Short  | " + "Data Source" + " | Last 80 bars</span>", font=dict(color="#f59e0b", size=13), x=0.01))
    fig.update_xaxes(gridcolor="#1e2030", zerolinecolor="#1e2030", tickfont=dict(color="#6b5b3a"))
    fig.update_yaxes(gridcolor="#1e2030", zerolinecolor="#1e2030", tickfont=dict(color="#6b5b3a"))
    return fig

def main():
    df = fetch_data()
    if df is None:
        print("Failed to fetch data.")
        return
    df = generate_signals(df)
    df = backtest(df, trail_pct=None, partial_close_pct=None)
    sharpe, max_dd, win_rate, total_r, n_trades = metrics(df)
    print("=" * 50)
    print("  " + summary)
    print("=" * 50)
    print("  Sharpe Ratio : {:.2f}".format(sharpe))
    print("  Max Drawdown : {:.1%}".format(max_dd))
    print("  Win Rate     : {:.1%}".format(win_rate))
    print("  Total Return : {:.1%}".format(total_r))
    print("  Trades       : " + str(n_trades))
    print("=" * 50)
    fig = plot_results(df)
    fig.show()

if __name__ == "__main__":
    main()
'''
    
    code = code_part1 + code_part2 + code_part3 + code_part4 + code_part5 + code_part6 + code_part7 + code_part8 + code_part9 + code_part10 + code_part11
    return code

# ═══════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════
def add_indicators(df, params):
    df = df.copy()
    ef = params.get('ema_fast', 20)
    es = params.get('ema_slow', 50)
    rp = params.get('rsi_period', 14)
    df['EMA_fast'] = df['Close'].ewm(span=ef, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=es, adjust=False).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(rp).mean()
    loss = (-delta.clip(upper=0)).rolling(rp).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['BB_mid'] = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * std
    df['BB_lower'] = df['BB_mid'] - 2 * std
    return df

def generate_signals(df, strategy, client=None, description=''):
    df = df.copy()
    df['long_signal'] = pd.Series(False, index=df.index)
    df['short_signal'] = pd.Series(False, index=df.index)
    df['Signal'] = pd.Series(0, index=df.index)
    
    if not client or not description:
        p = strategy.get('indicator_params', {}) or {}
        stype = strategy.get('strategy_type', 'trend')
        wants_long = strategy.get('entry_long') is not None
        wants_short = strategy.get('entry_short') is not None
        if 'EMA_fast' in df.columns and 'EMA_slow' in df.columns:
            if wants_long:
                df['long_signal'] = ((df['EMA_fast'] > df['EMA_slow']) & (df['EMA_fast'].shift(1) <= df['EMA_slow'].shift(1))).fillna(False)
            if wants_short:
                df['short_signal'] = ((df['EMA_fast'] < df['EMA_slow']) & (df['EMA_fast'].shift(1) >= df['EMA_slow'].shift(1))).fillna(False)
        df['Signal'] = df['long_signal'].astype(int) - df['short_signal'].astype(int)
        return df
    
    signal_block = generate_signal_block(client, description, strategy)
    exec_globals = {
        'df': df, 'pd': pd, 'np': np,
        'add_ema': add_ema, 'add_sma': add_sma, 'add_rsi': add_rsi, 'add_macd': add_macd,
        'add_bollinger': add_bollinger, 'add_atr': add_atr, 'add_stochastic': add_stochastic,
        'add_vwap': add_vwap, 'add_obv': add_obv, 'add_volume_spike': add_volume_spike,
        'add_volume_sma': add_volume_sma, 'add_wma': add_wma, 'add_swing_highs_lows': add_swing_highs_lows,
        'add_candle_patterns': add_candle_patterns, 'add_structure_break': add_structure_break,
        'add_fair_value_gaps': add_fair_value_gaps, 'add_liquidity_levels': add_liquidity_levels,
        'add_order_blocks': add_order_blocks, 'add_premium_discount': add_premium_discount,
        'add_market_structure': add_market_structure, 'add_optimal_trade_entry': add_optimal_trade_entry,
        'add_equal_highs_lows': add_equal_highs_lows, 'add_higher_highs_lower_lows': add_higher_highs_lower_lows,
        'add_support_resistance': add_support_resistance, 'add_previous_day_levels': add_previous_day_levels,
        'add_supertrend': add_supertrend, 'add_cci': add_cci, 'add_williams_r': add_williams_r,
        'add_mfi': add_mfi, 'add_donchian': add_donchian, 'add_keltner': add_keltner,
        'add_inside_outside_bars': add_inside_outside_bars, 'add_common_indicators': add_common_indicators,
        'add_smc_indicators': add_smc_indicators, 'crossover': crossover, 'crossunder': crossunder,
        'above_level': above_level, 'below_level': below_level, 'rising': rising, 'falling': falling,
    }
    clean_block = '\n'.join(line[4:] if line.startswith('    ') else line for line in signal_block.splitlines())
    try:
        exec(clean_block, exec_globals)
        df = exec_globals['df']
        if 'long_signal' in df.columns:
            df['long_signal'] = df['long_signal'].fillna(False).astype(bool)
        if 'short_signal' in df.columns:
            df['short_signal'] = df['short_signal'].fillna(False).astype(bool)
        if 'Signal' not in df.columns:
            df['Signal'] = df['long_signal'].astype(int) - df['short_signal'].astype(int)
    except Exception as e:
        st.warning(f"Signal execution error: {e}. Using empty signals.")
    return df

# ═══════════════════════════════════════════════════════════════
# PLOTLY CHART - FIXED VERSION
# ═══════════════════════════════════════════════════════════════
def draw_chart(df, strategy, symbol, data_source, show='both'):
    df_plot = df.tail(80).copy()
    
    df_plot = clean_duplicate_columns(df_plot)
    df_plot = clean_duplicate_index(df_plot)
    
    sl_pct = strategy.get('sl_pct') or 0.02
    tp_pct = strategy.get('tp_pct') or 0.06
    stype = strategy.get('strategy_type', 'trend')
    params = strategy.get('indicator_params', {})
    ef_span = params.get('ema_fast', 20)
    es_span = params.get('ema_slow', 50)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])

    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot['Open'], high=df_plot['High'],
        low=df_plot['Low'], close=df_plot['Close'], name='Price',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a', decreasing_fillcolor='#ef5350',
    ), row=1, col=1)

    ind_prefixes = ['EMA_','SMA_','BB_','RSI_','MACD','Stoch','ATR_','WMA_','VWAP','KC_','DC_','Supertrend']
    ind_cols = [c for c in df_plot.columns if any(c.startswith(p) for p in ind_prefixes) and c not in ['BB_Pct','BB_Width']]
    colors_ind = ['#f59e0b','#60a5fa','#a78bfa','#34d399','#f472b6','#fb923c']
    for i, col in enumerate(ind_cols[:6]):
        if col in df_plot.columns and not df_plot[col].isna().all():
            if col == 'BB_Upper' and 'BB_Lower' in df_plot.columns:
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_Upper'], name='BB Upper', line=dict(color='#f59e0b', width=1, dash='dash'), opacity=0.6), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_Lower'], name='BB Lower', line=dict(color='#f59e0b', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(245,158,11,0.05)', opacity=0.6), row=1, col=1)
            elif col not in ['BB_Lower','BB_Mid']:
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[col], name=col, line=dict(color=colors_ind[i % len(colors_ind)], width=1.5), opacity=0.9), row=1, col=1)

    rsi_cols = [c for c in df_plot.columns if c.startswith('RSI_')]
    has_rsi = len(rsi_cols) > 0

    long_df = df_plot[df_plot['long_signal']] if show in ('long', 'both') else df_plot.iloc[0:0]
    if not long_df.empty:
        fig.add_trace(go.Scatter(x=long_df.index, y=long_df['Close']*0.994, mode='markers', name='Long Entry', marker=dict(symbol='triangle-up', size=14, color='#4ade80', line=dict(color='#166534', width=1))), row=1, col=1)
        for date, row in long_df.iterrows():
            entry = float(row['Close'])
            sl = entry * (1 - sl_pct)
            tp = entry * (1 + tp_pct)
            try:
                end_date = df_plot.index[min(df_plot.index.get_loc(date)+8, len(df_plot)-1)]
            except: end_date = date
            fig.add_shape(type='line', x0=date, x1=end_date, y0=sl, y1=sl, line=dict(color='#ef4444', width=1.2, dash='dash'), row=1, col=1)
            fig.add_shape(type='line', x0=date, x1=end_date, y0=tp, y1=tp, line=dict(color='#4ade80', width=1.2, dash='dot'), row=1, col=1)

    short_df = df_plot[df_plot['short_signal']] if show in ('short', 'both') else df_plot.iloc[0:0]
    if not short_df.empty:
        fig.add_trace(go.Scatter(x=short_df.index, y=short_df['Close']*1.006, mode='markers', name='Short Entry', marker=dict(symbol='triangle-down', size=14, color='#f87171', line=dict(color='#991b1b', width=1))), row=1, col=1)
        for date, row in short_df.iterrows():
            entry = float(row['Close'])
            sl = entry * (1 + sl_pct)
            tp = entry * (1 - tp_pct)
            try:
                end_date = df_plot.index[min(df_plot.index.get_loc(date)+8, len(df_plot)-1)]
            except: end_date = date
            fig.add_shape(type='line', x0=date, x1=end_date, y0=sl, y1=sl, line=dict(color='#ef4444', width=1.2, dash='dash'), row=1, col=1)
            fig.add_shape(type='line', x0=date, x1=end_date, y0=tp, y1=tp, line=dict(color='#4ade80', width=1.2, dash='dot'), row=1, col=1)

    if 'Volume' in df_plot.columns and df_plot['Volume'].sum() > 0:
        bar_colors_list = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df_plot['Close'], df_plot['Open'])]
        unique_colors = []
        seen = set()
        for i, color in enumerate(bar_colors_list):
            if i not in seen:
                unique_colors.append(color)
                seen.add(i)
        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], name='Volume', marker_color=unique_colors, opacity=0.6), row=2, col=1)
    elif has_rsi:
        rsi_col = rsi_cols[0]
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[rsi_col], name=rsi_col, line=dict(color='#a78bfa', width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_color='#ef4444', line_dash='dash', line_width=1, opacity=0.6, row=2, col=1)
        fig.add_hline(y=30, line_color='#4ade80', line_dash='dash', line_width=1, opacity=0.6, row=2, col=1)

    n_long = len(long_df)
    n_short = len(short_df)

    fig.update_layout(
        height=620, paper_bgcolor='#080a0f', plot_bgcolor='#0d0f14',
        font=dict(family='IBM Plex Mono', color='#a89060', size=11),
        legend=dict(bgcolor='#0d0f14', bordercolor='#1e2030', borderwidth=1, font=dict(color='#a89060', size=10)),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=80, t=70, b=40),
        title=dict(text="<b>" + symbol + "</b> — " + strategy.get('summary','Strategy') + "<br><span style='font-size:11px;color:#6b5b3a'>" + str(n_long) + " Long  " + str(n_short) + " Short  | " + data_source + " | Last 80 bars</span>", font=dict(color='#f59e0b', size=13), x=0.01)
    )
    fig.update_xaxes(gridcolor='#1e2030', zerolinecolor='#1e2030', tickfont=dict(color='#6b5b3a'))
    fig.update_yaxes(gridcolor='#1e2030', zerolinecolor='#1e2030', tickfont=dict(color='#6b5b3a'))
    return fig

# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>📈 STRATEGY VISUALIZER</h1>
    <p>Describe your strategy → See it on real candles → Get Python code</p>
    <p style="color:#3d2f00;font-family:'IBM Plex Mono';font-size:0.7rem">QUANT ALPHA · GROQ + BINANCE · INTERACTIVE · $0</p>
</div>""", unsafe_allow_html=True)

client = init_llm()
if not client:
    st.error("⚠️ Add GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

with st.sidebar:
    st.markdown("""<div style='font-family:IBM Plex Mono;font-size:0.68rem;color:#f59e0b;letter-spacing:2px;text-transform:uppercase;border-bottom:1px solid #1e2030;padding-bottom:8px;margin-bottom:16px'>&TINGS</div>""", unsafe_allow_html=True)
    symbol = st.selectbox("Asset", options=list(BINANCE_SYMBOLS.keys()), index=0)
    period = st.selectbox("Period", options=list(PERIOD_DAYS.keys()), index=1)
    st.markdown("---")
    st.markdown("""<div style='font-family:IBM Plex Mono;font-size:0.65rem;color:#f59e0b;letter-spacing:1px;margin-bottom:8px'>📁 UPLOAD YOUR OWN DATA (OPTIONAL)</div>""", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("CSV: Date, Open, High, Low, Close", type=['csv'], label_visibility="collapsed")
    if uploaded_file:
        st.markdown("""<div class='data-source-box'>✅ CSV loaded — will use your data</div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""<div style='font-family:IBM Plex Mono;font-size:0.65rem;color:#3d2f00'><b style='color:#f59e0b'>DATA SOURCES:</b><br>1️⃣ Your CSV (if uploaded)<br>2️⃣ Binance API (auto)<br>3️⃣ CoinGecko (fallback)<br><br><b style='color:#f59e0b'>EXAMPLES:</b><br><br>"Buy BTC when 20 EMA crosses above 50 EMA. SL 2%, TP 6%."<br><br>"Long when RSI drops below 30. SL 3%, TP 9%."<br><br>"Short Bollinger lower breakout. SL 1.5%, TP 5%."</div>""", unsafe_allow_html=True)

for key in ['parsed','df','fig_long','fig_short','code','data_source','description']:
    if key not in st.session_state:
        st.session_state[key] = None

st.markdown('<div class="section-hdr">STEP 1 — DESCRIBE YOUR STRATEGY</div>', unsafe_allow_html=True)
st.markdown("""<div class="step-card active"><div class="step-num">✏️ PLAIN ENGLISH — NO CODING NEEDED</div>Describe entry conditions, stop loss, and take profit. Only mention SHORT if you want short signals.</div>""", unsafe_allow_html=True)

description = st.text_area("Strategy", placeholder="Buy BTC when the 20 EMA crosses above the 50 EMA. SL 2%, TP 6%.", height=100, label_visibility="collapsed")

c1, c2 = st.columns([3,1])
with c1: parse_btn = st.button("🧠 PARSE STRATEGY", use_container_width=True)
with c2: reset_btn = st.button("Reset", use_container_width=True)

if reset_btn:
    for key in ['parsed','df','fig_long','fig_short','code','data_source','description']:
        st.session_state[key] = None
    st.rerun()

if parse_btn and description.strip():
    with st.spinner("🧠 Parsing..."):
        parsed = parse_strategy(client, description)
    if parsed:
        st.session_state.parsed = parsed
        st.session_state.description = description
        st.session_state.fig = None
        st.session_state.code = None
        st.session_state.data_source = None

if st.session_state.parsed:
    p = st.session_state.parsed
    st.markdown('<div class="section-hdr">STEP 2 — CONFIRM UNDERSTANDING</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="parsed-box"><b style='color:#f59e0b'>AI PARSED AS:</b><br><br><b>Summary:</b> {p.get('summary','—')}<br><b>Type:</b> {p.get('strategy_type','—').upper()}<br><b>Indicators:</b> {', '.join(p.get('indicators',[]))}</div>""", unsafe_allow_html=True)

    sl_display = (p.get('sl_pct') or 0.01) * 100
    tp_display = (p.get('tp_pct') or 0.02) * 100
    for col, (cls, txt) in zip(st.columns(4), [
        ('tag-entry', f"📈 LONG: {str(p.get('entry_long','—'))[:32]}"),
        ('tag-entry', f"📉 SHORT: {str(p.get('entry_short','—'))[:32]}"),
        ('tag-sl', f"🛑 SL: {sl_display:.1f}%" + ("  (default)" if p.get('sl_pct') is None else "")),
        ('tag-tp', f"🎯 TP: {tp_display:.1f}%" + ("  (default)" if p.get('tp_pct') is None else "")),
    ]):
        with col:
            st.markdown(f'<span class="tag {cls}">{txt}</span>', unsafe_allow_html=True)

    if p.get('sl_pct') is None or p.get('tp_pct') is None:
        st.markdown("""<div style='background:#1c1400;border:1px solid #f59e0b;border-radius:8px;padding:12px 16px;margin:10px 0;font-family:IBM Plex Mono;font-size:0.78rem;color:#f59e0b'>⚠️ <b>DEFAULT RISK APPLIED:</b> SL 1% · TP 2%<br><span style='color:#6b5b3a'>You didn't specify SL/TP. To change them, re-describe your strategy and include e.g. "stop loss 2%, take profit 6%"</span></div>""", unsafe_allow_html=True)

    st.markdown("")
    if st.button("📊 VISUALIZE ON REAL CANDLES", use_container_width=True):
        with st.spinner("📡 Fetching market data..."):
            df, source = fetch_data(symbol, period, uploaded_file)
        if df is not None and len(df) > 30:
            st.markdown(f"""<div class="data-source-box">✅ {len(df)} candles from {source}</div>""", unsafe_allow_html=True)
            with st.spinner("🎨 Building charts..."):
                df = add_indicators(df, p.get('indicator_params', {}))
                df = generate_signals(df, p, client=client, description=st.session_state.get('description',''))
                st.session_state.df = df
                st.session_state.data_source = source
                st.session_state.fig_long = draw_chart(df, p, symbol, source, show='long')
                st.session_state.fig_short = draw_chart(df, p, symbol, source, show='short')
        else:
            st.error("Could not fetch data from any source. **Solution**: Upload a CSV file in the sidebar. Format: Date, Open, High, Low, Close columns.")

if st.session_state.get('fig_long') or st.session_state.get('fig_short'):
    st.markdown('<div class="section-hdr">STEP 3 — YOUR SETUP — LONG & SHORT</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""<div style='font-family:IBM Plex Mono;font-size:0.75rem;color:#4ade80;letter-spacing:2px;margin-bottom:8px'>📈 LONG SETUP</div>""", unsafe_allow_html=True)
        if st.session_state.get('fig_long'):
            st.plotly_chart(st.session_state.fig_long, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})
    with col_r:
        st.markdown("""<div style='font-family:IBM Plex Mono;font-size:0.75rem;color:#f87171;letter-spacing:2px;margin-bottom:8px'>📉 SHORT SETUP</div>""", unsafe_allow_html=True)
        if st.session_state.get('fig_short'):
            n_short = int(st.session_state.df['short_signal'].sum()) if st.session_state.get('df') is not None else 0
            if n_short > 0:
                st.plotly_chart(st.session_state.fig_short, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})
            else:
                st.markdown("""<div style='background:#0d0f14;border:1px solid #1e2030;border-radius:10px;padding:40px;text-align:center;font-family:IBM Plex Mono;color:#334155;font-size:0.8rem'>📉 No short signals detected<br><br><span style='color:#1e2030'>You didn't ask for short entries.<br>Re-describe with "short when..." to add them.</span></div>""", unsafe_allow_html=True)

    st.markdown("""<div style='text-align:center;font-family:IBM Plex Mono;font-size:0.78rem;color:#a89060;margin:12px 0'>🔺 Green triangles = Long entries &nbsp;|&nbsp; 🔻 Red triangles = Short entries<br>Dashed = Stop Loss &nbsp;|&nbsp; Dotted = Take Profit</div>""", unsafe_allow_html=True)

    cy, cn = st.columns(2)
    with cy: yes_btn = st.button("✅ YES — Generate Python Code", use_container_width=True)
    with cn: no_btn = st.button("❌ NO — Redescribe", use_container_width=True)

    if no_btn:
        st.session_state.fig_long = None
        st.session_state.fig_short = None
        st.session_state.code = None
        st.info("Refine your description in Step 1.")
    if yes_btn:
        with st.spinner("& Generating code..."):
            st.session_state.code = generate_python_code(client, st.session_state.parsed, symbol, description=st.session_state.get('description', ''))

if st.session_state.code:
    st.markdown('<div class="section-hdr">STEP 4 — YOUR PYTHON CODE</div>', unsafe_allow_html=True)
    st.markdown("""<div class="step-card done"><div class="step-num">✅ READY — RUN IN COLAB OR JUPYTER</div>Then paste into <b>Backtest Validator</b> to check for errors.</div>""", unsafe_allow_html=True)
    st.text_area("Code", value=st.session_state.code, height=320, label_visibility="collapsed")
    st.download_button("⬇️ Download .py file", data=st.session_state.code, file_name=f"{symbol}_strategy.py", mime="text/plain", use_container_width=True)
    st.markdown("""<div style='background:#0d0f14;border:1px solid #f59e0b;border-radius:10px;padding:16px;margin-top:16px;text-align:center'><b style='font-family:IBM Plex Mono;color:#f59e0b'>⚠️ VALIDATE BEFORE TRADING LIVE</b><br><span style='font-family:IBM Plex Mono;color:#6b5b3a;font-size:0.8rem'>Paste code into <b style='color:#e8e0d0'>Backtest Validator</b> to detect lookahead bias and overfitting</span></div>""", unsafe_allow_html=True)

st.markdown("""<div style="text-align:center;margin-top:48px;padding:16px;border-top:1px solid #1e2030"><span style="font-family:IBM Plex Mono;font-size:0.65rem;color:#1e2030">QUANT ALPHA — NOT FINANCIAL ADVICE</span></div>""", unsafe_allow_html=True)
