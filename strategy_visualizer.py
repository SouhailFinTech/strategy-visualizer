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
@st.cache_data(ttl=300)
def fetch_binance(symbol: str, period: str):
    sym   = BINANCE_SYMBOLS.get(symbol.upper())
    limit = BINANCE_LIMITS.get(period, 90)
    if not sym:
        return None
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={'symbol': sym, 'interval': '1d', 'limit': min(limit, 1000)},
            timeout=15, headers={'User-Agent': 'QuantAlpha/1.0'}
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data:
            return None
        df = pd.DataFrame(data, columns=[
            'timestamp','Open','High','Low','Close','Volume',
            'ct','qv','nt','tbb','tbq','ignore'
        ])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('Date')
        df = df[['Open','High','Low','Close','Volume']].astype(float)
        return df.dropna()
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_coingecko(symbol: str, days: int):
    coin_id = COINGECKO_IDS.get(symbol.upper(), symbol.lower())
    api_key = st.secrets.get("COINGECKO_API_KEY", "")
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
        df = pd.DataFrame(data, columns=['timestamp','Open','High','Low','Close'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('Date').drop('timestamp', axis=1).astype(float)
        df['Volume'] = 0.0
        return df.resample('D').last().dropna()
    except Exception:
        return None


def load_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip().title() for c in df.columns]
        date_col = next(
            (c for c in ['Date','Timestamp','Time','Datetime'] if c in df.columns),
            None
        )
        if not date_col:
            st.error("CSV needs a Date or Timestamp column")
            return None
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df.index.name = 'Date'
        for col in ['Open','High','Low','Close']:
            if col not in df.columns:
                st.error(f"CSV missing column: {col}")
                return None
        if 'Volume' not in df.columns:
            df['Volume'] = 0.0
        return df[['Open','High','Low','Close','Volume']].astype(float).dropna().sort_index()
    except Exception as e:
        st.error(f"CSV error: {e}")
        return None


def fetch_data(symbol, period, uploaded_file=None):
    if uploaded_file is not None:
        df = load_csv(uploaded_file)
        if df is not None and len(df) > 30:
            return df, "📁 Your CSV"

    df = fetch_binance(symbol, period)
    if df is not None and len(df) > 30:
        return df, "🟡 Binance"

    days = PERIOD_DAYS.get(period, 90)
    df = fetch_coingecko(symbol, days)
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


def generate_python_code(client, strategy: dict, symbol: str) -> str:
    has_long  = strategy.get('entry_long')  is not None
    has_short = strategy.get('entry_short') is not None

    if has_long and not has_short:
        direction = "LONG ONLY. Position = 1 when in trade, 0 when flat. Never go short."
    elif has_short and not has_long:
        direction = "SHORT ONLY. Position = -1 when in trade, 0 when flat. Never go long."
    else:
        direction = "LONG AND SHORT. Position = 1 for long, -1 for short, 0 for flat."

    binance_sym = BINANCE_SYMBOLS.get(symbol.upper(), 'BTCUSDT')

    prompt = f"""Generate a complete runnable Python backtesting script.

Strategy: {json.dumps(strategy, indent=2)}
Asset: {symbol}
Binance symbol: {binance_sym}
Direction: {direction}

Fetch data from Binance:
import requests
resp = requests.get('https://api.binance.com/api/v3/klines',
    params={{'symbol': '{binance_sym}', 'interval': '1d', 'limit': 365}})

STRICT RULES — every rule is mandatory:
1. Direction is {direction} — do not add any signals not requested
2. Signal column = 1 when entry condition met, 0 otherwise
3. Position column = Signal.shift(1) directly — NEVER use cumsum()
4. Position values must only be: 0, 1, or -1
5. Entry price = next bar Open (use shift to avoid lookahead)
6. Commission = 0.1% applied only when position changes
7. Calculate and print: Sharpe ratio, max drawdown, win rate, total return
8. Plot equity curve with plotly dark theme
9. Add clear comments explaining each section
10. Handle API errors gracefully with try/except

Return ONLY Python code. No explanation. No markdown."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2500, temperature=0.1
        )
        text = response.choices[0].message.content.strip()
        if '```python' in text:
            text = text.split('```python')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        return text.strip()
    except Exception as e:
        return f"# Error generating code: {e}"

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

    df['long_signal']  = False
    df['short_signal'] = False

    if stype in ['trend', 'momentum']:
        if wants_long:
            df['long_signal'] = (
                (df['EMA_fast'] > df['EMA_slow']) &
                (df['EMA_fast'].shift(1) <= df['EMA_slow'].shift(1))
            )
        if wants_short:
            df['short_signal'] = (
                (df['EMA_fast'] < df['EMA_slow']) &
                (df['EMA_fast'].shift(1) >= df['EMA_slow'].shift(1))
            )

    elif stype == 'mean-reversion':
        if wants_long:
            df['long_signal'] = (
                (df['RSI'] < ros) & (df['RSI'].shift(1) >= ros)
            )
        if wants_short:
            df['short_signal'] = (
                (df['RSI'] > rob) & (df['RSI'].shift(1) <= rob)
            )

    elif stype == 'breakout':
        if wants_long:
            df['long_signal'] = (
                (df['Close'] > df['BB_upper']) &
                (df['Close'].shift(1) <= df['BB_upper'].shift(1))
            )
        if wants_short:
            df['short_signal'] = (
                (df['Close'] < df['BB_lower']) &
                (df['Close'].shift(1) >= df['BB_lower'].shift(1))
            )

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
