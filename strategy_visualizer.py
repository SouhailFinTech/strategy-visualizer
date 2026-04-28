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
        return json.loads(text.strip())
    except json.JSONDecodeError:
        st.error("AI returned invalid JSON. Please rephrase your strategy.")
        return None
    except Exception as e:
        st.error(f"Parse error: {e}")
        return None


def generate_signal_block(client, description: str) -> str:
    """
    AI writes ONLY the signal lines.
    Everything else (backtest, metrics, chart) is hardcoded by us.
    AI is given the full indicator library so it calls functions
    instead of writing formulas — minimizes errors dramatically.
    """

    # Import the library reference
    try:
        from quant_indicators import LIBRARY_REFERENCE
    except ImportError:
        LIBRARY_REFERENCE = "Use standard pandas/numpy indicator calculations."

    prompt = f"""You are a Python quant developer.
Your ONLY job: write the signal code for this strategy.

STRATEGY: "{description}"

You have access to a pre-built indicator library already imported as:
from quant_indicators import *

{LIBRARY_REFERENCE}

RULES — follow every rule exactly:
1. Call library functions to calculate indicators — do NOT write indicator formulas yourself
2. Example: df = add_ema(df, 20) then use df['EMA_20'] — never write ewm() yourself
3. df['long_signal'] must be a boolean pd.Series
4. df['short_signal'] must be a boolean pd.Series
5. If only long requested: df['short_signal'] = pd.Series(False, index=df.index)
6. If only short requested: df['long_signal'] = pd.Series(False, index=df.index)
7. Use crossover() / crossunder() helpers for crossovers — never write (a > b) & (a.shift(1) <= b.shift(1)) yourself
8. End both signal lines with .fillna(False)
9. Return ONLY Python lines — no def, no imports, no explanation, no markdown

EXAMPLE OUTPUT for "buy when EMA 20 crosses EMA 50":
df = add_ema(df, 20)
df = add_ema(df, 50)
df['long_signal']  = crossover(df['EMA_20'], df['EMA_50']).fillna(False)
df['short_signal'] = pd.Series(False, index=df.index)

Now write signal code for: "{description}"
Output ONLY the Python lines."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.0
        )
        text = response.choices[0].message.content.strip()
        if '```python' in text:
            text = text.split('```python')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]

        # Strip all existing indentation then add exactly 4 spaces
        lines = text.strip().splitlines()
        # Remove import lines — library already imported in generated code
        lines = [l for l in lines
                 if not l.strip().startswith('import ')
                 and not l.strip().startswith('from ')]
        signal_block = '\n'.join(
            '    ' + line.lstrip() if line.strip() else ''
            for line in lines
        )
        return signal_block

    except Exception as e:
        # Safe fallback — no signals, won't crash
        return (
            "    df['long_signal']  = pd.Series(False, index=df.index)\n"
            "    df['short_signal'] = pd.Series(False, index=df.index)\n"
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
    sl_pct      = strategy.get('sl_pct', 0.02)
    tp_pct      = strategy.get('tp_pct', 0.06)
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
    if has_long and not has_short:
        signal_block = f"""\
    # Long only — entry when crossover detected
    df['long_signal']  = {long_sig}
    df['short_signal'] = pd.Series(False, index=df.index)
    df['Signal']       = df['long_signal'].astype(int)"""
    elif has_short and not has_long:
        signal_block = f"""\
    # Short only — entry when crossunder detected
    df['long_signal']  = pd.Series(False, index=df.index)
    df['short_signal'] = {short_sig}
    df['Signal']       = -df['short_signal'].astype(int)"""
    else:
        signal_block = f"""\
    # Long and short signals
    df['long_signal']  = {long_sig}
    df['short_signal'] = {short_sig}
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

    # ── Full code template ────────────────────────────────────
    code = f'''import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


# ── Add indicators ────────────────────────────────────────────
def add_indicators(df):
{indicator_block}
    return df


# ── Generate signals ──────────────────────────────────────────
def generate_signals(df):
{signal_block}
    # Shift by 1 bar — prevents lookahead bias
    df["Position"] = df["Signal"].shift(1).fillna(0)
    return df


# ── Backtest ──────────────────────────────────────────────────
def backtest(df):
    # Daily close-to-close returns
    df["Return"] = df["Close"].pct_change()

    # Commission 0.1% only when position changes (entry/exit)
    df["Commission"] = np.where(
        df["Position"] != df["Position"].shift(1), 0.001, 0
    )

    # Strategy daily return = market return × position − commission
    df["Strategy_Return"] = df["Return"] * df["Position"] - df["Commission"]

    # Cumulative equity curves starting at 1.0
    df["BH_Equity"]       = (1 + df["Return"]).cumprod()
    df["Strategy_Equity"] = (1 + df["Strategy_Return"]).cumprod()

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

    # Indicator lines
    fig.add_trace(go.Scatter(
        x=df["Open time"], y=df["{ind1_col}"],
        name="{ind1_label}",
        line=dict(color="#f59e0b", width=1.5), opacity=0.9
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["Open time"], y=df["{ind2_col}"],
        name="{ind2_label}",
        line=dict(color="#60a5fa", width=1.5), opacity=0.9
    ), row=1, col=1)

    # Long entry markers
    long_entries = df[df["long_signal"]]
    if not long_entries.empty:
        fig.add_trace(go.Scatter(
            x=long_entries["Open time"],
            y=long_entries["Close"] * 0.994,
            mode="markers", name="Long Entry",
            marker=dict(symbol="triangle-up", size=14,
                        color="#4ade80",
                        line=dict(color="#166534", width=1))
        ), row=1, col=1)

    # Short entry markers
    short_entries = df[df["short_signal"]]
    if not short_entries.empty:
        fig.add_trace(go.Scatter(
            x=short_entries["Open time"],
            y=short_entries["Close"] * 1.006,
            mode="markers", name="Short Entry",
            marker=dict(symbol="triangle-down", size=14,
                        color="#f87171",
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
def draw_chart(df, strategy, symbol, data_source):
    df_plot = df.tail(80).copy()
    sl_pct  = strategy.get('sl_pct', 0.02)
    tp_pct  = strategy.get('tp_pct', 0.06)
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

    # ── Long signals ──────────────────────────────────────────
    long_df = df_plot[df_plot['long_signal']]
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

    # ── Short signals ─────────────────────────────────────────
    short_df = df_plot[df_plot['short_signal']]
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
for key in ['parsed','df','fig','code','data_source']:
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
    for key in ['parsed','df','fig','code','data_source']:
        st.session_state[key] = None
    st.rerun()

if parse_btn and description.strip():
    with st.spinner("🧠 Parsing..."):
        parsed = parse_strategy(client, description)
    if parsed:
        st.session_state.parsed      = parsed
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

    for col, (cls, txt) in zip(st.columns(4), [
        ('tag-entry', f"📈 LONG: {str(p.get('entry_long','None'))[:32]}"),
        ('tag-entry', f"📉 SHORT: {str(p.get('entry_short','None'))[:32]}"),
        ('tag-sl',    f"🛑 SL: {p.get('sl_pct',0.02)*100:.1f}%"),
        ('tag-tp',    f"🎯 TP: {p.get('tp_pct',0.06)*100:.1f}%"),
    ]):
        with col:
            st.markdown(f'<span class="tag {cls}">{txt}</span>',
                       unsafe_allow_html=True)

    st.markdown("")
    if st.button("📊 VISUALIZE ON REAL CANDLES", use_container_width=True):
        with st.spinner("📡 Fetching market data..."):
            df, source = fetch_data(symbol, period, uploaded_file)

        if df is not None and len(df) > 30:
            st.markdown(f"""<div class="data-source-box">
            ✅ {len(df)} candles from {source}</div>""",
                       unsafe_allow_html=True)
            with st.spinner("🎨 Building chart..."):
                df = add_indicators(df, p.get('indicator_params',{}))
                df = generate_signals(df, p)
                st.session_state.df          = df
                st.session_state.data_source = source
                st.session_state.fig         = draw_chart(df, p, symbol, source)
        else:
            st.error(
                "Could not fetch data from any source.\n\n"
                "**Solution:** Upload a CSV file in the sidebar.\n"
                "Format: Date, Open, High, Low, Close columns.\n"
                "Download from Binance, TradingView, or any exchange."
            )

# ── STEP 3 ────────────────────────────────────────────────────
if st.session_state.fig:
    st.markdown('<div class="section-hdr">STEP 3 — IS THIS YOUR SETUP?</div>',
                unsafe_allow_html=True)
    st.plotly_chart(
        st.session_state.fig, use_container_width=True,
        config={
            'displayModeBar': True, 'scrollZoom': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'{symbol}_strategy', 'scale': 2
            }
        }
    )
    st.markdown("""<div style='text-align:center;font-family:IBM Plex Mono;
    font-size:0.82rem;color:#a89060;margin:12px 0'>
    🔺 Green = Long entries &nbsp;|&nbsp; 🔻 Red = Short entries<br>
    Dashed = SL &nbsp;|&nbsp; Dotted = TP &nbsp;|&nbsp;
    🖱️ Scroll to zoom · Drag to pan
    </div>""", unsafe_allow_html=True)

    cy, cn = st.columns(2)
    with cy: yes_btn = st.button("✅ YES — Generate Python Code", use_container_width=True)
    with cn: no_btn  = st.button("❌ NO — Redescribe",            use_container_width=True)

    if no_btn:
        st.session_state.fig  = None
        st.session_state.code = None
        st.info("Refine your description in Step 1.")

    if yes_btn:
        with st.spinner("⚙️ Generating code..."):
            st.session_state.code = generate_python_code(
                client, st.session_state.parsed, symbol)

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
