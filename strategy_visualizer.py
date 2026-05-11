"""
QUANT ALPHA — STRATEGY VISUALIZER v8
FIXES:
- Only shows indicators explicitly requested by user
- Clean code generation without broken library extraction
- Smart indicator filtering based on parsed strategy
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
# DATA FETCHING
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
    """Load CSV file"""
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip().title() for c in df.columns]

        date_col = next(
            (c for c in ['Date','Datetime','Timestamp','Time','Open Time']
             if c in df.columns),
            None
        )
        if date_col is None:
            first_col = df.columns[0]
            try:
                pd.to_datetime(df[first_col].iloc[0])
                date_col = first_col
            except Exception:
                st.error("Cannot find date column.")
                return None

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.set_index(date_col)
        df.index.name = 'Date'

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
    if uploaded_file is not None:
        uploaded_file.seek(0)
        df = load_csv(uploaded_file)
        if df is not None and len(df) > 30:
            return df, "📁 Your CSV"

    with st.spinner("📡 Trying Binance..."):
        df = fetch_binance(symbol, period)
    if df is not None and len(df) > 30:
        return df, "🟡 Binance"

    with st.spinner("📡 Trying CoinGecko..."):
        days = PERIOD_DAYS.get(period, 90)
        df   = fetch_coingecko(symbol, days)
    if df is not None and len(df) > 30:
        return df, "🦎 CoinGecko"

    return None, None

# ─────────────────────────────────────────────────────────────
# INDICATOR LIBRARY
# ─────────────────────────────────────────────────────────────
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

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index"""
    col   = f'RSI_{period}'
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    df[col] = 100 - (100 / (1 + rs))
    return df

def add_macd(df: pd.DataFrame,
             fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD"""
    ema_fast        = df['Close'].ewm(span=fast,   adjust=False).mean()
    ema_slow        = df['Close'].ewm(span=slow,   adjust=False).mean()
    df['MACD']      = ema_fast - ema_slow
    df['MACD_Signal']= df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

def add_bollinger(df: pd.DataFrame,
                  period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands"""
    df['BB_Mid']   = df['Close'].rolling(period).mean()
    std            = df['Close'].rolling(period).std()
    df['BB_Upper'] = df['BB_Mid'] + std_dev * std
    df['BB_Lower'] = df['BB_Mid'] - std_dev * std
    return df

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

def add_stochastic(df: pd.DataFrame,
                   k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Stochastic Oscillator"""
    low_min  = df['Low'].rolling(k_period).min()
    high_max = df['High'].rolling(k_period).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['Stoch_D'] = df['Stoch_K'].rolling(d_period).mean()
    return df

def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Volume Weighted Average Price"""
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df

def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume"""
    direction  = np.sign(df['Close'].diff())
    df['OBV']  = (direction * df['Volume']).fillna(0).cumsum()
    return df

def add_wma(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Weighted Moving Average"""
    col = f'WMA_{period}'
    weights = np.arange(1, period + 1)
    df[col] = df['Close'].rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )
    return df

def crossover(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """Returns True when series_a crosses ABOVE series_b"""
    return (series_a > series_b) & (series_a.shift(1) <= series_b.shift(1))

def crossunder(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """Returns True when series_a crosses BELOW series_b"""
    return (series_a < series_b) & (series_a.shift(1) >= series_b.shift(1))

# ─────────────────────────────────────────────────────────────
# SMART INDICATOR ADDITION - ONLY ADDS WHAT'S REQUESTED
# ─────────────────────────────────────────────────────────────
def add_indicators(df, params, indicators_list=None):
    """
    Only add indicators that are explicitly requested.
    """
    df = df.copy()
    indicators_list = indicators_list or []
    params = params or {}
    
    # Convert to lowercase for matching
    indicators_lower = [ind.lower() for ind in indicators_list]
    
    # Check what indicators are needed
    needs_ema = any('ema' in ind for ind in indicators_lower) or len(indicators_list) == 0
    needs_sma = any('sma' in ind for ind in indicators_lower)
    needs_rsi = any('rsi' in ind for ind in indicators_lower)
    needs_bb = any('bollinger' in ind or 'bb' in ind for ind in indicators_lower)
    needs_macd = any('macd' in ind for ind in indicators_lower)
    needs_atr = any('atr' in ind for ind in indicators_lower)
    needs_stoch = any('stoch' in ind for ind in indicators_lower)
    
    # Add only requested indicators
    if needs_ema:
        ef = params.get('ema_fast', 20)
        es = params.get('ema_slow', 50)
        df = add_ema(df, ef)
        df = add_ema(df, es)
    
    if needs_sma:
        period = params.get('sma_period', 20)
        df = add_sma(df, period)
    
    if needs_rsi:
        rp = params.get('rsi_period', 14)
        df = add_rsi(df, rp)
    
    if needs_bb:
        period = params.get('bb_period', 20)
        std_dev = params.get('bb_std', 2.0)
        df = add_bollinger(df, period, std_dev)
    
    if needs_macd:
        df = add_macd(df)
    
    if needs_atr:
        period = params.get('atr_period', 14)
        df = add_atr(df, period)
    
    if needs_stoch:
        df = add_stochastic(df)
    
    return df

# ─────────────────────────────────────────────────────────────
# GROQ PARSING
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
  "indicators": ["list of indicators mentioned"],
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

        # Sanitize
        result['sl_pct'] = result.get('sl_pct') or 0.02
        result['tp_pct'] = result.get('tp_pct') or 0.06
        result['strategy_type'] = result.get('strategy_type') or 'trend'
        result['summary']       = result.get('summary') or 'Trading Strategy'
        result['indicators']    = result.get('indicators') or []
        result['indicator_params'] = result.get('indicator_params') or {
            'ema_fast': 20, 'ema_slow': 50,
            'rsi_period': 14, 'rsi_overbought': 70, 'rsi_oversold': 30
        }
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
    """Generate signal code from Groq"""
    features    = detect_advanced_features(description)
    has_long    = features['has_long'] or not features['has_both_directions']
    has_short   = features['has_both_directions']
    stype       = strategy.get('strategy_type', 'trend')
    indicators  = strategy.get('indicators', [])
    params      = strategy.get('indicator_params', {}) or {}

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

    ind_hint = f"Strategy type: {stype}\nIndicators mentioned: {', '.join(indicators) if indicators else 'detect from description'}"
    if params.get('ema_fast'): ind_hint += f"\nEMA fast period: {params['ema_fast']}"
    if params.get('ema_slow'): ind_hint += f"\nEMA slow period: {params['ema_slow']}"
    if params.get('rsi_period'): ind_hint += f"\nRSI period: {params['rsi_period']}"

    prompt = f"""You are a Python quant developer.
Translate this trading strategy into Python signal detection code.

STRATEGY: "{description}"
DIRECTION: {direction}
{ind_hint}

AVAILABLE FUNCTIONS:
add_ema(df, period)        → df['EMA_20'], df['EMA_50']
add_sma(df, period)        → df['SMA_20']
add_rsi(df, period)        → df['RSI_14']
add_macd(df, 12, 26, 9)    → df['MACD'], df['MACD_Signal']
add_bollinger(df, 20, 2.0) → df['BB_Upper'], df['BB_Lower']
add_atr(df, 14)            → df['ATR_14']
add_stochastic(df, 14, 3)  → df['Stoch_K'], df['Stoch_D']

SIGNAL HELPERS:
crossover(series_a, series_b)  → True when a crosses above b
crossunder(series_a, series_b) → True when a crosses below b

MANDATORY OUTPUT FORMAT:
{signal_template}

RULES:
1. First call add_*() functions for every indicator you need
2. Then write the signal conditions
3. df['Signal'] MUST always be the last line
4. Both long_signal and short_signal MUST be assigned
5. Output ONLY Python lines — no imports, no def, no markdown

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

        # Auto-repair
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
            if (("long_signal']  =" in line or "long_signal'] =" in line) and
                    'fillna' not in line and 'pd.Series' not in line and 'astype' not in line):
                line = line.rstrip() + '.fillna(False)'
            if (("short_signal']  =" in line or "short_signal'] =" in line) and
                    'fillna' not in line and 'pd.Series' not in line and 'astype' not in line):
                line = line.rstrip() + '.fillna(False)'
            repaired.append(line)
        lines = repaired

        return '\n'.join(
            '    ' + line.lstrip() if line.strip() else ''
            for line in lines
        )

    except Exception as e:
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
        )


def generate_python_code(client, strategy: dict, symbol: str,
                          description: str = '') -> str:
    """
    Generates complete backtest code with CLEAN library.
    """
    binance_sym = BINANCE_SYMBOLS.get(symbol.upper(), 'BTCUSDT')
    sl_pct      = strategy.get('sl_pct') or 0.01
    tp_pct      = strategy.get('tp_pct') or 0.02
    summary     = strategy.get('summary', 'Trading Strategy')
    features    = detect_advanced_features(description)
    has_trailing = features['has_trailing_stop']
    has_partial  = features['has_partial_close']

    signal_block = generate_signal_block(client, description, strategy)

    # CLEAN STATIC LIBRARY - No file reading
    lib_code = '''import pandas as pd
import numpy as np

def add_ema(df, period):
    df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    return df

def add_sma(df, period):
    df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
    return df

def add_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    return df

def add_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

def add_bollinger(df, period=20, std_dev=2.0):
    df['BB_Mid'] = df['Close'].rolling(period).mean()
    std = df['Close'].rolling(period).std()
    df['BB_Upper'] = df['BB_Mid'] + std_dev * std
    df['BB_Lower'] = df['BB_Mid'] - std_dev * std
    return df

def add_atr(df, period=14):
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df[f'ATR_{period}'] = tr.ewm(span=period, adjust=False).mean()
    return df

def add_stochastic(df, k_period=14, d_period=3):
    low_min = df['Low'].rolling(k_period).min()
    high_max = df['High'].rolling(k_period).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['Stoch_D'] = df['Stoch_K'].rolling(d_period).mean()
    return df

def crossover(series_a, series_b):
    return (series_a > series_b) & (series_a.shift(1) <= series_b.shift(1))

def crossunder(series_a, series_b):
    return (series_a < series_b) & (series_a.shift(1) >= series_b.shift(1))
'''

    trail_arg   = "trail_pct=0.03"  if has_trailing else "trail_pct=None"
    partial_arg = "partial_close_pct=0.3" if has_partial else "partial_close_pct=None"

    code = f'''import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Indicator Library ─────────────────────────────────────────
{lib_code}

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
            "Open time","Open","High","Low","Close","Volume",
            "ct","qv","nt","tbb","tbq","ignore"
        ])
        df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
        for col in ["Open","High","Low","Close","Volume"]:
            df[col] = pd.to_numeric(df[col])
        return df.drop(columns=["ct","qv","nt","tbb","tbq","ignore"])
    except Exception as e:
        print(f"Data error: {{e}}")
        return None


# ── Generate signals (AI-written, library-backed) ─────────────
def generate_signals(df):
{signal_block}
    # Shift by 1 bar — prevents lookahead bias
    df["Position"] = df["Signal"].shift(1).fillna(0)
    return df


# ── Backtest ──────────────────────────────────────────────────
def backtest(df, sl_pct={sl_pct}, tp_pct={tp_pct},
             trail_pct=None, partial_close_pct=None):
    df["Return"] = df["Close"].pct_change()
    if trail_pct is None:
        df["Commission"]      = np.where(
            df["Position"] != df["Position"].shift(1), 0.001, 0)
        df["Strategy_Return"] = df["Return"] * df["Position"] - df["Commission"]
        df["BH_Equity"]       = (1 + df["Return"]).cumprod()
        df["Strategy_Equity"] = (1 + df["Strategy_Return"]).cumprod()
    else:
        equity, bh_eq = 1.0, 1.0
        equities, bh_equities = [], []
        position, entry_price = 0, None
        trail_high, trail_low = None, None
        partial_done = False
        prices  = df["Close"].values
        returns = df["Return"].fillna(0).values
        for i in range(len(df)):
            bh_eq  *= (1 + returns[i])
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
                    hit_sl     = price <= entry_price * (1 - sl_pct)
                    hit_tp     = price >= entry_price * (1 + tp_pct)
                    hit_trail  = price <= trail_stop
                else:
                    trail_low  = min(trail_low, price)
                    trail_stop = trail_low * (1 + trail_pct)
                    hit_sl     = price >= entry_price * (1 + sl_pct)
                    hit_tp     = price <= entry_price * (1 - tp_pct)
                    hit_trail  = price >= trail_stop
                if partial_close_pct and not partial_done and hit_tp:
                    equity      *= (1 + (price/entry_price-1)*position*partial_close_pct - 0.001)
                    partial_done = True
                if hit_sl or hit_trail or (hit_tp and not partial_close_pct):
                    remaining = (1-partial_close_pct) if partial_done and partial_close_pct else 1.0
                    equity   *= (1 + (price/entry_price-1)*position*remaining - 0.001)
                    position, entry_price = 0, None
                else:
                    equity *= (1 + returns[i] * position)
            equities.append(equity)
            bh_equities.append(bh_eq)
        df["Strategy_Equity"] = equities
        df["BH_Equity"]       = bh_equities
        df["Strategy_Return"] = pd.Series(equities).pct_change().fillna(0).values
    return df


# ── Metrics ───────────────────────────────────────────────────
def metrics(df):
    r        = df["Strategy_Return"].dropna()
    sharpe   = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    roll_max = df["Strategy_Equity"].cummax()
    max_dd   = ((df["Strategy_Equity"] - roll_max) / roll_max).min()
    win_rate = (r > 0).mean()
    total_r  = df["Strategy_Equity"].iloc[-1] - 1
    n_trades = int((df["Position"] != df["Position"].shift(1)).sum() / 2)
    return sharpe, max_dd, win_rate, total_r, n_trades


# ── Chart ────────────────────────────────────────────────────
def plot_results(df):
    ind_cols = [c for c in df.columns if any(
        c.startswith(p) for p in
        ["EMA_","SMA_","BB_","RSI_","MACD","Stoch","ATR_","WMA_","VWAP"]
    )]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=[0.70, 0.30])
    fig.add_trace(go.Candlestick(
        x=df["Open time"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
    ), row=1, col=1)
    colors = ["#f59e0b","#60a5fa","#a78bfa","#34d399","#f87171"]
    for i, col in enumerate(ind_cols[:5]):
        fig.add_trace(go.Scatter(x=df["Open time"], y=df[col],
            name=col, line=dict(color=colors[i%len(colors)], width=1.5),
            opacity=0.9), row=1, col=1)
    long_e = df[df["long_signal"]]
    if not long_e.empty:
        fig.add_trace(go.Scatter(x=long_e["Open time"],
            y=long_e["Close"]*0.994, mode="markers", name="Long Entry",
            marker=dict(symbol="triangle-up", size=14, color="#4ade80")), row=1, col=1)
    short_e = df[df["short_signal"]]
    if not short_e.empty:
        fig.add_trace(go.Scatter(x=short_e["Open time"],
            y=short_e["Close"]*1.006, mode="markers", name="Short Entry",
            marker=dict(symbol="triangle-down", size=14, color="#f87171")), row=1, col=1)
    for idx in df[df["long_signal"]].index:
        e = float(df.loc[idx,"Close"])
        t0 = df.loc[idx,"Open time"]
        t1 = df.iloc[min(df.index.get_loc(idx)+8, len(df)-1)]["Open time"]
        fig.add_shape(type="line",x0=t0,x1=t1,y0=e*(1-{sl_pct}),y1=e*(1-{sl_pct}),
            line=dict(color="#ef4444",width=1,dash="dash"),row=1,col=1)
        fig.add_shape(type="line",x0=t0,x1=t1,y0=e*(1+{tp_pct}),y1=e*(1+{tp_pct}),
            line=dict(color="#4ade80",width=1,dash="dot"),row=1,col=1)
    for idx in df[df["short_signal"]].index:
        e = float(df.loc[idx,"Close"])
        t0 = df.loc[idx,"Open time"]
        t1 = df.iloc[min(df.index.get_loc(idx)+8, len(df)-1)]["Open time"]
        fig.add_shape(type="line",x0=t0,x1=t1,y0=e*(1+{sl_pct}),y1=e*(1+{sl_pct}),
            line=dict(color="#ef4444",width=1,dash="dash"),row=1,col=1)
        fig.add_shape(type="line",x0=t0,x1=t1,y0=e*(1-{tp_pct}),y1=e*(1-{tp_pct}),
            line=dict(color="#4ade80",width=1,dash="dot"),row=1,col=1)
    bar_colors = ["#26a69a" if c>=o else "#ef5350"
                  for c,o in zip(df["Close"],df["Open"])]
    fig.add_trace(go.Bar(x=df["Open time"],y=df["Volume"],
        name="Volume",marker_color=bar_colors,opacity=0.6),row=2,col=1)
    fig.add_trace(go.Scatter(x=df["Open time"],y=df["Strategy_Equity"],
        name="Strategy",line=dict(color="#4ade80",width=2)),row=2,col=1)
    fig.add_trace(go.Scatter(x=df["Open time"],y=df["BH_Equity"],
        name="Buy & Hold",line=dict(color="#64748b",width=1.5,dash="dash")),row=2,col=1)
    fig.update_layout(height=700,paper_bgcolor="#080a0f",plot_bgcolor="#0d0f14",
        xaxis_rangeslider_visible=False,
        font=dict(color="#a89060"),
        title=dict(text=f"<b>{symbol}</b> — {summary}",
                   font=dict(color="#f59e0b",size=14),x=0.01))
    fig.update_xaxes(gridcolor="#1e2030")
    fig.update_yaxes(gridcolor="#1e2030")
    return fig


# ── Main ──────────────────────────────────────────────────────
def main():
    df = fetch_data()
    if df is None:
        print("Failed to fetch data.")
        return
    df = generate_signals(df)
    df = backtest(df, {trail_arg}, {partial_arg})
    sharpe, max_dd, win_rate, total_r, n_trades = metrics(df)
    print("=" * 50)
    print(f"  {summary}")
    print("=" * 50)
    print(f"  Sharpe Ratio : {{sharpe:.2f}}")
    print(f"  Max Drawdown : {{max_dd:.1%}}")
    print(f"  Win Rate     : {{win_rate:.1%}}")
    print(f"  Total Return : {{total_r:.1%}}")
    print(f"  Trades       : {{n_trades}}")
    print("=" * 50)
    fig = plot_results(df)
    fig.show()

if __name__ == "__main__":
    main()
'''
    return code

# ─────────────────────────────────────────────────────────────
# APP SIGNAL RUNNER
# ─────────────────────────────────────────────────────────────
def generate_signals_app(df, strategy, client=None, description=''):
    """
    Run Groq signal block on the dataframe for the app chart.
    """
    df = df.copy()

    df['long_signal']  = pd.Series(False, index=df.index)
    df['short_signal'] = pd.Series(False, index=df.index)
    df['Signal']       = pd.Series(0, index=df.index)

    if not client or not description:
        p     = strategy.get('indicator_params', {}) or {}
        stype = strategy.get('strategy_type', 'trend')
        wants_long  = strategy.get('entry_long')  is not None
        wants_short = strategy.get('entry_short') is not None

        if 'EMA_20' in df.columns and 'EMA_50' in df.columns:
            if wants_long:
                df['long_signal'] = (
                    (df['EMA_20'] > df['EMA_50']) &
                    (df['EMA_20'].shift(1) <= df['EMA_50'].shift(1))
                ).fillna(False)
            if wants_short:
                df['short_signal'] = (
                    (df['EMA_20'] < df['EMA_50']) &
                    (df['EMA_20'].shift(1) >= df['EMA_50'].shift(1))
                ).fillna(False)
        df['Signal'] = df['long_signal'].astype(int) - df['short_signal'].astype(int)
        return df

    signal_block = generate_signal_block(client, description, strategy)

    try:
        exec_globals = {
            'df': df, 'pd': pd, 'np': np,
            'add_ema': add_ema, 'add_sma': add_sma,
            'add_rsi': add_rsi, 'add_macd': add_macd,
            'add_bollinger': add_bollinger, 'add_atr': add_atr,
            'add_stochastic': add_stochastic, 'add_vwap': add_vwap,
            'add_obv': add_obv, 'add_wma': add_wma,
            'crossover': crossover, 'crossunder': crossunder,
        }
        clean_block = '\n'.join(
            line[4:] if line.startswith('    ') else line
            for line in signal_block.splitlines()
        )
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

# ─────────────────────────────────────────────────────────────
# PLOTLY CHART - ONLY SHOWS REQUESTED INDICATORS
# ─────────────────────────────────────────────────────────────
def draw_chart(df, strategy, symbol, data_source, show='both'):
    df_plot = df.tail(80).copy()
    sl_pct  = strategy.get('sl_pct') or 0.02
    tp_pct  = strategy.get('tp_pct') or 0.06
    params  = strategy.get('indicator_params', {})
    indicators_requested = strategy.get('indicators', [])
    
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=[0.75, 0.25]
    )

    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['Open'], high=df_plot['High'],
        low=df_plot['Low'],   close=df_plot['Close'],
        name='Price',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a', decreasing_fillcolor='#ef5350',
    ), row=1, col=1)

    # ── ONLY SHOW REQUESTED INDICATORS ──
    ind_prefixes = ['EMA_','SMA_','BB_','RSI_','MACD','Stoch','ATR_','WMA_','VWAP']
    ind_cols = [c for c in df_plot.columns
                if any(c.startswith(p) for p in ind_prefixes)
                and c not in ['BB_Pct','BB_Width']]
    
    # Filter to only requested indicators
    if indicators_requested:
        requested_lower = [ind.lower() for ind in indicators_requested]
        filtered_cols = []
        for col in ind_cols:
            col_lower = col.lower()
            # Match indicator names
            if any(ind in col_lower for ind in requested_lower):
                filtered_cols.append(col)
            # Also match by prefix (e.g., "ema" matches "EMA_20")
            elif any(col_lower.startswith(prefix.lower()) for prefix in ['ema','sma','bb','rsi','macd','stoch','atr']):
                for req in requested_lower:
                    if req in col_lower or col_lower.startswith(req):
                        filtered_cols.append(col)
                        break
        ind_cols = filtered_cols if filtered_cols else ind_cols
    
    colors_ind = ['#f59e0b','#60a5fa','#a78bfa','#34d399','#f472b6','#fb923c']
    for i, col in enumerate(ind_cols[:6]):
        if col in df_plot.columns and not df_plot[col].isna().all():
            if col == 'BB_Upper' and 'BB_Lower' in df_plot.columns:
                fig.add_trace(go.Scatter(
                    x=df_plot.index, y=df_plot['BB_Upper'], name='BB Upper',
                    line=dict(color='#f59e0b', width=1, dash='dash'), opacity=0.6
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df_plot.index, y=df_plot['BB_Lower'], name='BB Lower',
                    line=dict(color='#f59e0b', width=1, dash='dash'),
                    fill='tonexty', fillcolor='rgba(245,158,11,0.05)', opacity=0.6
                ), row=1, col=1)
            elif col not in ['BB_Lower','BB_Mid']:
                fig.add_trace(go.Scatter(
                    x=df_plot.index, y=df_plot[col], name=col,
                    line=dict(color=colors_ind[i % len(colors_ind)], width=1.5),
                    opacity=0.9
                ), row=1, col=1)

    rsi_cols = [c for c in df_plot.columns if c.startswith('RSI_')]
    has_rsi  = len(rsi_cols) > 0

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

    if rsi_cols:
        rsi_col = rsi_cols[0]
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot[rsi_col],
            name=rsi_col, line=dict(color='#a78bfa', width=1.5)
        ), row=2, col=1)
        fig.add_hline(y=70, line_color='#ef4444', line_dash='dash',
                      line_width=1, opacity=0.6, row=2, col=1)
        fig.add_hline(y=30, line_color='#4ade80', line_dash='dash',
                      line_width=1, opacity=0.6, row=2, col=1)
    else:
        bar_colors = ['#26a69a' if c >= o else '#ef5350'
                      for c, o in zip(df_plot['Close'], df_plot['Open'])]
        fig.add_trace(go.Bar(
            x=df_plot.index, y=df_plot['Volume'],
            name='Volume', marker_color=bar_colors, opacity=0.6
        ), row=2, col=1)

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
for key in ['parsed','df','fig_long','fig_short','code','data_source','description']:
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
    for key in ['parsed','df','fig_long','fig_short','code','data_source','description']:
        st.session_state[key] = None
    st.rerun()

if parse_btn and description.strip():
    with st.spinner("🧠 Parsing..."):
        parsed = parse_strategy(client, description)
    if parsed:
        st.session_state.parsed      = parsed
        st.session_state.description = description
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
                # FIXED: Pass indicators list to only add what's requested
                df = add_indicators(
                    df, 
                    p.get('indicator_params', {}),
                    indicators_list=p.get('indicators', [])
                )
                df = generate_signals_app(df, p,
                    client=client,
                    description=st.session_state.get('description',''))
                st.session_state.df          = df
                st.session_state.data_source = source
                st.session_state.fig_long  = draw_chart(
                    df, p, symbol, source, show='long')
                st.session_state.fig_short = draw_chart(
                    df, p, symbol, source, show='short')
        else:
            st.error(
                "Could not fetch data from any source.\n\n"
                "**Solution:** Upload a CSV file in the sidebar."
            )

# ── STEP 3 ────────────────────────────────────────────────────
if st.session_state.get('fig_long') or st.session_state.get('fig_short'):
    st.markdown('<div class="section-hdr">STEP 3 — YOUR SETUP — LONG & SHORT</div>',
                unsafe_allow_html=True)

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
            n_short = int(st.session_state.df['short_signal'].sum()) if st.session_state.get('df') is not None else 0
            if n_short > 0:
                st.plotly_chart(st.session_state.fig_short,
                               use_container_width=True,
                               config={'displayModeBar': True, 'scrollZoom': True})
            else:
                st.markdown("""<div style='background:#0d0f14;border:1px solid #1e2030;
                border-radius:10px;padding:40px;text-align:center;
                font-family:IBM Plex Mono;color:#334155;font-size:0.8rem'>
                📉 No short signals detected<br><br>
                <span style='color:#1e2030'>You didn't ask for short entries.<br>
                Re-describe with "short when..." to add them.</span>
                </div>""", unsafe_allow_html=True)

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
    Backtest thoroughly before live trading
    </span></div>""", unsafe_allow_html=True)

st.markdown("""<div style="text-align:center;margin-top:48px;padding:16px;
border-top:1px solid #1e2030">
<span style="font-family:IBM Plex Mono;font-size:0.65rem;color:#1e2030">
QUANT ALPHA — NOT FINANCIAL ADVICE
</span></div>""", unsafe_allow_html=True)
