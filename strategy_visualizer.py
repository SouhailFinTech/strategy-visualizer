"""
╔══════════════════════════════════════════════════════════════╗
║         QUANT ALPHA — STRATEGY VISUALIZER                    ║
║   Describe your strategy → See it on real candles → Get code ║
║                                                              ║
║   Stack: Streamlit + Gemini Flash (free) + mplfinance        ║
║   Deploy: streamlit.io (free)                                ║
║                                                              ║
║   SETUP:                                                     ║
║   1. Create .streamlit/secrets.toml                          ║
║      GEMINI_API_KEY = "your-key-here"                        ║
║   2. pip install -r requirements.txt                         ║
║   3. streamlit run strategy_visualizer.py                    ║
╚══════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import re
import io
from datetime import datetime, timedelta

# ── Gemini API ────────────────────────────────────────────────
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

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
# CSS — dark terminal with amber accent
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
    transition: border-color 0.2s;
}
.step-card.active { border-color: #f59e0b; }
.step-card.done   { border-color: #22c55e; }

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
.tag-entry  { background:#1a2e1a; color:#4ade80; border:1px solid #166534; }
.tag-sl     { background:#2e1a1a; color:#f87171; border:1px solid #991b1b; }
.tag-tp     { background:#1a2a1a; color:#86efac; border:1px solid #15803d; }
.tag-filter { background:#1a1a2e; color:#93c5fd; border:1px solid #1d4ed8; }

.code-block {
    background: #050608;
    border: 1px solid #1e2030;
    border-radius: 8px;
    padding: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #a8b4c0;
    overflow-x: auto;
    white-space: pre;
    line-height: 1.6;
}

.verdict-yes {
    background: linear-gradient(135deg,#052e16,#14532d);
    border: 1px solid #22c55e;
    border-radius: 10px; padding: 20px; text-align: center;
}
.verdict-no {
    background: linear-gradient(135deg,#1c0606,#3a0d0d);
    border: 1px solid #ef4444;
    border-radius: 10px; padding: 20px; text-align: center;
}

[data-testid="stSidebar"] {
    background: #06080c;
    border-right: 1px solid #1e2030;
}

.stButton > button {
    background: linear-gradient(135deg,#92400e,#b45309);
    color: #fef3c7; border: none; border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700; padding: 12px 24px;
    width: 100%; transition: all 0.2s;
    letter-spacing: 1px;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#b45309,#d97706);
    transform: translateY(-1px);
}

.stTextArea textarea, .stTextInput input, .stSelectbox select {
    background: #0a0c10 !important;
    color: #e8e0d0 !important;
    border: 1px solid #1e2030 !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}

.section-hdr {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem; color: #f59e0b;
    letter-spacing: 3px; text-transform: uppercase;
    border-bottom: 1px solid #1e2030;
    padding-bottom: 8px; margin: 20px 0 14px;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# GEMINI SETUP
# ─────────────────────────────────────────────────────────────
def init_gemini():
    if not GEMINI_AVAILABLE:
        return None
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        st.error(f"Gemini init error: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# GEMINI HELPERS
# ─────────────────────────────────────────────────────────────
def parse_strategy(model, description: str) -> dict:
    """Parse plain English strategy into structured JSON"""

    prompt = f"""You are a quantitative trading expert.
Parse this trading strategy description into structured JSON.

Strategy: "{description}"

Return ONLY valid JSON with this exact structure:
{{
  "entry_long": "condition for long entry (or null)",
  "entry_short": "condition for short entry (or null)",
  "stop_loss": "stop loss description (e.g. '2% below entry')",
  "take_profit": "take profit description (e.g. '6% above entry')",
  "indicators": ["list", "of", "indicators", "used"],
  "timeframe": "suggested timeframe (1h/4h/1d)",
  "asset_type": "crypto/forex/stocks",
  "strategy_type": "trend/mean-reversion/breakout/momentum",
  "sl_pct": 0.02,
  "tp_pct": 0.06,
  "indicator_params": {{
    "ema_fast": 20,
    "ema_slow": 50,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30
  }},
  "summary": "one sentence summary of the strategy"
}}

Rules:
- sl_pct and tp_pct must be decimal numbers (0.02 = 2%)
- If not specified, use reasonable defaults
- indicator_params only include what's relevant
- Return ONLY the JSON object, no explanation, no markdown"""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Clean markdown if present
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        return json.loads(text.strip())
    except Exception as e:
        st.error(f"Parse error: {e}")
        return None


def generate_python_code(model, strategy: dict, symbol: str) -> str:
    """Generate complete Python backtest code from parsed strategy"""

    prompt = f"""You are an expert Python quant developer.
Generate a complete, runnable Python backtesting script for this strategy:

{json.dumps(strategy, indent=2)}

Asset: {symbol}

Requirements:
- Use yfinance for data (free)
- Use pandas and numpy
- Implement the exact strategy described
- Always use .shift(1) on signals to prevent lookahead bias
- Include realistic transaction costs (0.1% per trade)
- Calculate: Sharpe ratio, max drawdown, total return, win rate
- Plot equity curve using matplotlib
- Clean, commented code
- Entry uses next bar Open after signal (realistic)

Return ONLY the complete Python code, no explanation."""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if '```python' in text:
            text = text.split('```python')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        return text.strip()
    except Exception as e:
        return f"# Error generating code: {e}"


# ─────────────────────────────────────────────────────────────
# CHART GENERATOR
# ─────────────────────────────────────────────────────────────
def fetch_data(symbol: str, period: str = '3mo',
               interval: str = '1d') -> pd.DataFrame:
    """Fetch OHLCV data from yfinance"""
    try:
        df = yf.download(symbol, period=period,
                        interval=interval, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open','High','Low','Close','Volume']].copy()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None


def add_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Add indicators based on parsed strategy params"""
    df = df.copy()

    ema_fast = params.get('ema_fast', 20)
    ema_slow = params.get('ema_slow', 50)
    rsi_period = params.get('rsi_period', 14)

    df['EMA_fast'] = df['Close'].ewm(span=ema_fast).mean()
    df['EMA_slow'] = df['Close'].ewm(span=ema_slow).mean()

    # RSI
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(rsi_period).mean()
    loss  = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_mid']   = df['Close'].rolling(20).mean()
    df['BB_upper'] = df['BB_mid'] + 2 * df['Close'].rolling(20).std()
    df['BB_lower'] = df['BB_mid'] - 2 * df['Close'].rolling(20).std()

    return df


def generate_signals(df: pd.DataFrame,
                     strategy: dict) -> pd.DataFrame:
    """Generate entry signals based on strategy type"""
    df = df.copy()
    params = strategy.get('indicator_params', {})
    stype  = strategy.get('strategy_type', 'trend')

    rsi_ob = params.get('rsi_overbought', 70)
    rsi_os = params.get('rsi_oversold', 30)

    df['long_signal']  = False
    df['short_signal'] = False

    if stype in ['trend', 'momentum']:
        # EMA crossover
        df['long_signal']  = (
            (df['EMA_fast'] > df['EMA_slow']) &
            (df['EMA_fast'].shift(1) <= df['EMA_slow'].shift(1))
        )
        df['short_signal'] = (
            (df['EMA_fast'] < df['EMA_slow']) &
            (df['EMA_fast'].shift(1) >= df['EMA_slow'].shift(1))
        )
    elif stype == 'mean-reversion':
        # RSI oversold/overbought
        df['long_signal']  = (
            (df['RSI'] < rsi_os) &
            (df['RSI'].shift(1) >= rsi_os)
        )
        df['short_signal'] = (
            (df['RSI'] > rsi_ob) &
            (df['RSI'].shift(1) <= rsi_ob)
        )
    elif stype == 'breakout':
        # Bollinger breakout
        df['long_signal']  = (
            (df['Close'] > df['BB_upper']) &
            (df['Close'].shift(1) <= df['BB_upper'].shift(1))
        )
        df['short_signal'] = (
            (df['Close'] < df['BB_lower']) &
            (df['Close'].shift(1) >= df['BB_lower'].shift(1))
        )

    return df


def draw_chart(df: pd.DataFrame, strategy: dict,
               symbol: str) -> plt.Figure:
    """Draw professional candlestick chart with signals"""

    # Use last 60 candles for clarity
    df_plot = df.tail(60).copy().reset_index()

    sl_pct = strategy.get('sl_pct', 0.02)
    tp_pct = strategy.get('tp_pct', 0.06)
    stype  = strategy.get('strategy_type', 'trend')

    fig = plt.figure(figsize=(14, 10), facecolor='#080a0f')

    # Layout: price chart (top) + indicator (bottom)
    ax1 = fig.add_subplot(3, 1, (1, 2))
    ax2 = fig.add_subplot(3, 1, 3, sharex=ax1)

    for ax in [ax1, ax2]:
        ax.set_facecolor('#0d0f14')
        ax.tick_params(colors='#6b5b3a', labelsize=8)
        ax.spines['bottom'].set_color('#1e2030')
        ax.spines['top'].set_color('#1e2030')
        ax.spines['left'].set_color('#1e2030')
        ax.spines['right'].set_color('#1e2030')
        ax.yaxis.label.set_color('#6b5b3a')
        ax.grid(color='#1e2030', linewidth=0.5, alpha=0.5)

    x = np.arange(len(df_plot))

    # ── Candlesticks ──────────────────────────────────────────
    for i, row in df_plot.iterrows():
        o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
        color = '#26a69a' if c >= o else '#ef5350'
        # Wick
        ax1.plot([i, i], [l, h], color=color, linewidth=0.8, alpha=0.8)
        # Body
        body_h = abs(c - o)
        body_y = min(o, c)
        rect = plt.Rectangle((i - 0.3, body_y), 0.6,
                              max(body_h, (h-l)*0.01),
                              color=color, alpha=0.9)
        ax1.add_patch(rect)

    # ── EMAs ──────────────────────────────────────────────────
    ax1.plot(x, df_plot['EMA_fast'].values,
             color='#f59e0b', linewidth=1.2,
             label=f"EMA {strategy.get('indicator_params',{}).get('ema_fast',20)}",
             alpha=0.9)
    ax1.plot(x, df_plot['EMA_slow'].values,
             color='#60a5fa', linewidth=1.2,
             label=f"EMA {strategy.get('indicator_params',{}).get('ema_slow',50)}",
             alpha=0.9)

    # Bollinger Bands (for breakout strategies)
    if stype == 'breakout':
        ax1.fill_between(x,
            df_plot['BB_upper'].values,
            df_plot['BB_lower'].values,
            alpha=0.08, color='#f59e0b')
        ax1.plot(x, df_plot['BB_upper'].values,
                 color='#f59e0b', linewidth=0.8, alpha=0.5, linestyle='--')
        ax1.plot(x, df_plot['BB_lower'].values,
                 color='#f59e0b', linewidth=0.8, alpha=0.5, linestyle='--')

    # ── Entry signals + SL/TP ─────────────────────────────────
    long_indices  = df_plot[df_plot['long_signal']].index.tolist()
    short_indices = df_plot[df_plot['short_signal']].index.tolist()

    for idx in long_indices:
        row = df_plot.loc[idx]
        entry = row['Close']
        sl    = entry * (1 - sl_pct)
        tp    = entry * (1 + tp_pct)

        # Entry arrow
        ax1.annotate('', xy=(idx, entry * 0.997),
                     xytext=(idx, entry * (1 - sl_pct*0.6)),
                     arrowprops=dict(arrowstyle='->', color='#4ade80',
                                   lw=2.5))
        # SL line
        ax1.hlines(sl, idx, min(idx+8, len(df_plot)-1),
                   colors='#ef4444', linewidth=1,
                   linestyles='dashed', alpha=0.7)
        # TP line
        ax1.hlines(tp, idx, min(idx+8, len(df_plot)-1),
                   colors='#4ade80', linewidth=1,
                   linestyles='dotted', alpha=0.7)
        # Labels
        ax1.text(idx+0.3, sl * 0.9985,
                 f'SL {sl_pct*100:.0f}%',
                 color='#ef4444', fontsize=6.5,
                 fontfamily='monospace')
        ax1.text(idx+0.3, tp * 1.001,
                 f'TP {tp_pct*100:.0f}%',
                 color='#4ade80', fontsize=6.5,
                 fontfamily='monospace')

    for idx in short_indices:
        row = df_plot.loc[idx]
        entry = row['Close']
        sl    = entry * (1 + sl_pct)
        tp    = entry * (1 - tp_pct)

        # Entry arrow (pointing down)
        ax1.annotate('', xy=(idx, entry * 1.003),
                     xytext=(idx, entry * (1 + sl_pct*0.6)),
                     arrowprops=dict(arrowstyle='->', color='#f87171',
                                   lw=2.5))
        # SL line
        ax1.hlines(sl, idx, min(idx+8, len(df_plot)-1),
                   colors='#ef4444', linewidth=1,
                   linestyles='dashed', alpha=0.7)
        # TP line
        ax1.hlines(tp, idx, min(idx+8, len(df_plot)-1),
                   colors='#4ade80', linewidth=1,
                   linestyles='dotted', alpha=0.7)

    # ── Bottom indicator ──────────────────────────────────────
    if stype == 'mean-reversion':
        ax2.plot(x, df_plot['RSI'].values,
                 color='#a78bfa', linewidth=1.2)
        rsi_ob = strategy.get('indicator_params',{}).get('rsi_overbought',70)
        rsi_os = strategy.get('indicator_params',{}).get('rsi_oversold',30)
        ax2.axhline(rsi_ob, color='#ef4444', linewidth=0.8,
                    linestyle='--', alpha=0.6)
        ax2.axhline(rsi_os, color='#4ade80', linewidth=0.8,
                    linestyle='--', alpha=0.6)
        ax2.fill_between(x, df_plot['RSI'].values, rsi_os,
                         where=(df_plot['RSI'].values < rsi_os),
                         alpha=0.2, color='#4ade80')
        ax2.fill_between(x, df_plot['RSI'].values, rsi_ob,
                         where=(df_plot['RSI'].values > rsi_ob),
                         alpha=0.2, color='#ef4444')
        ax2.set_ylabel('RSI', color='#6b5b3a', fontsize=8)
        ax2.set_ylim(0, 100)
    else:
        # Volume bars
        vol_colors = ['#26a69a' if
                      df_plot.loc[i,'Close'] >= df_plot.loc[i,'Open']
                      else '#ef5350'
                      for i in df_plot.index]
        ax2.bar(x, df_plot['Volume'].values,
                color=vol_colors, alpha=0.6)
        ax2.set_ylabel('Volume', color='#6b5b3a', fontsize=8)

    # ── Labels ────────────────────────────────────────────────
    n_ticks = min(8, len(df_plot))
    tick_idx = np.linspace(0, len(df_plot)-1, n_ticks, dtype=int)
    tick_labels = []
    for i in tick_idx:
        try:
            d = df_plot.loc[i, 'Date'] if 'Date' in df_plot.columns \
                else df_plot.index[i]
            tick_labels.append(pd.Timestamp(d).strftime('%b %d'))
        except Exception:
            tick_labels.append(str(i))

    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels(tick_labels, rotation=30,
                        ha='right', fontsize=7)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#f59e0b', label='EMA Fast'),
        mpatches.Patch(color='#60a5fa', label='EMA Slow'),
        mpatches.Patch(color='#4ade80', label='Long Entry ↑'),
        mpatches.Patch(color='#f87171', label='Short Entry ↓'),
        mpatches.Patch(color='#ef4444', label='Stop Loss ---'),
        mpatches.Patch(color='#4ade80', label='Take Profit ···'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left',
               facecolor='#0d0f14', edgecolor='#1e2030',
               labelcolor='#a89060', fontsize=7.5)

    summary = strategy.get('summary', 'Trading Strategy')
    n_long  = len(long_indices)
    n_short = len(short_indices)
    fig.suptitle(
        f'{symbol} — {summary}\n'
        f'{n_long} Long signals  |  {n_short} Short signals  |  '
        f'Last 60 bars',
        color='#f59e0b', fontsize=10,
        fontfamily='monospace', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>📈 STRATEGY VISUALIZER</h1>
    <p>Describe your trading strategy in plain English →
    See it on real candles → Get the Python backtest code</p>
    <p style="color:#3d2f00;font-family:'IBM Plex Mono';font-size:0.7rem">
        QUANT ALPHA — NO CODING REQUIRED
    </p>
</div>
""", unsafe_allow_html=True)

# Init Gemini
model = init_gemini()
if not model:
    st.error("⚠️ Gemini API not configured. "
             "Add GEMINI_API_KEY to .streamlit/secrets.toml")
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:IBM Plex Mono;font-size:0.68rem;
    color:#f59e0b;letter-spacing:2px;text-transform:uppercase;
    border-bottom:1px solid #1e2030;padding-bottom:8px;margin-bottom:16px'>
    ⚙ SETTINGS
    </div>""", unsafe_allow_html=True)

    symbol = st.text_input(
        "Asset Symbol",
        value="BTC-USD",
        help="Examples: BTC-USD, ETH-USD, AAPL, EURUSD=X"
    )

    period = st.selectbox(
        "Data Period",
        options=['1mo','3mo','6mo','1y','2y'],
        index=1
    )

    interval = st.selectbox(
        "Timeframe",
        options=['1d','1h','4h','1wk'],
        index=0
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-family:IBM Plex Mono;font-size:0.68rem;
    color:#3d2f00'>
    <b style='color:#f59e0b'>HOW IT WORKS:</b><br><br>
    1️⃣ Describe your strategy<br>
    2️⃣ AI parses your rules<br>
    3️⃣ See signals on real candles<br>
    4️⃣ Confirm the setup looks right<br>
    5️⃣ Get complete Python code<br>
    6️⃣ Validate with Backtest Validator<br>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:IBM Plex Mono;font-size:0.65rem;color:#3d2f00'>
    <b style='color:#f59e0b'>EXAMPLES:</b><br><br>
    "Buy when RSI drops below 30 on BTC, sell when RSI hits 70.
    Stop loss 3% below entry."<br><br>
    "Long when 20 EMA crosses above 50 EMA. TP 8%, SL 2%."<br><br>
    "Short when price breaks below Bollinger lower band.
    SL 1.5%, TP 4.5%."
    </div>""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────
for key in ['parsed','df','chart_done','code']:
    if key not in st.session_state:
        st.session_state[key] = None

# ─────────────────────────────────────────────────────────────
# STEP 1 — DESCRIBE STRATEGY
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">STEP 1 — DESCRIBE YOUR STRATEGY</div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="step-card active">
<div class="step-num">✏️ PLAIN ENGLISH — NO CODING NEEDED</div>
Describe your entry conditions, stop loss, and take profit.
Be as specific or as vague as you want.
</div>""", unsafe_allow_html=True)

description = st.text_area(
    "Strategy Description",
    placeholder=(
        "Example: Buy Bitcoin when the 20-period EMA crosses above "
        "the 50-period EMA on the daily chart. Set stop loss 2% below "
        "entry and take profit at 6%. Exit when EMA crosses back down."
    ),
    height=120,
    label_visibility="collapsed"
)

col1, col2 = st.columns([3,1])
with col1:
    parse_btn = st.button("🧠 PARSE STRATEGY", use_container_width=True)
with col2:
    reset_btn = st.button("↺ Reset", use_container_width=True)

if reset_btn:
    for key in ['parsed','df','chart_done','code']:
        st.session_state[key] = None
    st.rerun()

if parse_btn and description.strip():
    with st.spinner("🧠 AI is reading your strategy..."):
        parsed = parse_strategy(model, description)

    if parsed:
        st.session_state.parsed = parsed
        st.session_state.chart_done = None
        st.session_state.code = None

# ─────────────────────────────────────────────────────────────
# STEP 2 — SHOW PARSED STRATEGY
# ─────────────────────────────────────────────────────────────
if st.session_state.parsed:
    p = st.session_state.parsed

    st.markdown('<div class="section-hdr">STEP 2 — CONFIRM UNDERSTANDING</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="parsed-box">
    <b style='color:#f59e0b'>📋 AI PARSED YOUR STRATEGY AS:</b><br><br>
    <b>Summary:</b> {p.get('summary','—')}<br>
    <b>Type:</b> {p.get('strategy_type','—').upper()}<br>
    <b>Asset:</b> {p.get('asset_type','—')} | <b>Timeframe:</b> {p.get('timeframe','—')}<br><br>
    <b>Indicators:</b> {', '.join(p.get('indicators', []))}<br>
    </div>""", unsafe_allow_html=True)

    # Tags
    col_tags = st.columns(4)
    with col_tags[0]:
        if p.get('entry_long'):
            st.markdown(f'<span class="tag tag-entry">📈 LONG: {p["entry_long"][:40]}</span>',
                       unsafe_allow_html=True)
    with col_tags[1]:
        if p.get('entry_short'):
            st.markdown(f'<span class="tag tag-entry">📉 SHORT: {p["entry_short"][:40]}</span>',
                       unsafe_allow_html=True)
    with col_tags[2]:
        st.markdown(f'<span class="tag tag-sl">🛑 SL: {p.get("sl_pct",0.02)*100:.1f}%</span>',
                   unsafe_allow_html=True)
    with col_tags[3]:
        st.markdown(f'<span class="tag tag-tp">🎯 TP: {p.get("tp_pct",0.06)*100:.1f}%</span>',
                   unsafe_allow_html=True)

    st.markdown("")

    # Visualize button
    viz_btn = st.button(
        "📊 VISUALIZE ON REAL CANDLES",
        use_container_width=True
    )

    if viz_btn:
        with st.spinner(f"📡 Fetching {symbol} data..."):
            df = fetch_data(symbol, period, interval)

        if df is not None and len(df) > 60:
            with st.spinner("🎨 Drawing chart..."):
                df = add_indicators(df, p.get('indicator_params', {}))
                df = generate_signals(df, p)
                st.session_state.df = df

                fig = draw_chart(df, p, symbol)

                # Save chart to buffer
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=130,
                           facecolor='#080a0f', bbox_inches='tight')
                buf.seek(0)
                st.session_state.chart_done = buf
                plt.close(fig)
        else:
            st.error(f"Could not fetch data for {symbol}. "
                    f"Try BTC-USD, ETH-USD, or AAPL.")

# ─────────────────────────────────────────────────────────────
# STEP 3 — SHOW CHART + CONFIRM
# ─────────────────────────────────────────────────────────────
if st.session_state.chart_done:
    st.markdown('<div class="section-hdr">STEP 3 — IS THIS YOUR SETUP?</div>',
                unsafe_allow_html=True)

    st.image(st.session_state.chart_done, use_column_width=True)

    # Download chart
    st.download_button(
        "⬇️ Download Chart",
        data=st.session_state.chart_done,
        file_name=f"{symbol}_strategy_chart.png",
        mime="image/png"
    )

    st.markdown("")
    st.markdown("""
    <div style='text-align:center;font-family:IBM Plex Mono;
    font-size:0.9rem;color:#a89060;margin:16px 0'>
    🔍 Green arrows = Long entries &nbsp;|&nbsp;
    Red arrows = Short entries<br>
    Dashed lines = Stop Loss &nbsp;|&nbsp;
    Dotted lines = Take Profit
    </div>""", unsafe_allow_html=True)

    col_yes, col_no = st.columns(2)
    with col_yes:
        confirm_btn = st.button(
            "✅ YES — Generate Python Code",
            use_container_width=True
        )
    with col_no:
        deny_btn = st.button(
            "❌ NO — Redescribe Strategy",
            use_container_width=True
        )

    if deny_btn:
        st.session_state.chart_done = None
        st.session_state.code = None
        st.info("👆 Go back to Step 1 and refine your description.")

    if confirm_btn:
        with st.spinner("⚙️ Generating Python backtest code..."):
            code = generate_python_code(
                model,
                st.session_state.parsed,
                symbol
            )
            st.session_state.code = code

# ─────────────────────────────────────────────────────────────
# STEP 4 — GENERATED CODE
# ─────────────────────────────────────────────────────────────
if st.session_state.code:
    st.markdown('<div class="section-hdr">STEP 4 — YOUR PYTHON BACKTEST CODE</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="step-card done">
    <div class="step-num">✅ CODE GENERATED — READY TO USE</div>
    Copy this code, run it in Jupyter or Google Colab,
    then paste it into the <b>Backtest Validator</b> to check for errors.
    </div>""", unsafe_allow_html=True)

    st.markdown(
        f'<div class="code-block">{st.session_state.code}</div>',
        unsafe_allow_html=True
    )

    col_dl, col_copy = st.columns(2)
    with col_dl:
        st.download_button(
            "⬇️ Download .py file",
            data=st.session_state.code,
            file_name=f"{symbol}_strategy.py",
            mime="text/plain",
            use_container_width=True
        )
    with col_copy:
        st.text_area(
            "Copy from here",
            value=st.session_state.code,
            height=200,
            label_visibility="visible"
        )

    # CTA to validator
    st.markdown("""
    <div style='background:#0d0f14;border:1px solid #f59e0b;
    border-radius:10px;padding:20px;margin-top:16px;text-align:center'>
        <div style='font-family:IBM Plex Mono;color:#f59e0b;
        font-size:0.9rem;font-weight:700;margin-bottom:8px'>
        ⚠️ NEXT STEP — VALIDATE YOUR BACKTEST
        </div>
        <div style='font-family:IBM Plex Mono;color:#6b5b3a;font-size:0.8rem'>
        Paste this code into the <b style='color:#e8e0d0'>
        Backtest Validator</b> to check for lookahead bias,
        overfitting, and unrealistic assumptions before trading it live.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:48px;padding:16px;
border-top:1px solid #1e2030">
    <span style="font-family:IBM Plex Mono;font-size:0.65rem;color:#1e2030">
    QUANT ALPHA STRATEGY VISUALIZER — NOT FINANCIAL ADVICE
    </span>
</div>""", unsafe_allow_html=True)
