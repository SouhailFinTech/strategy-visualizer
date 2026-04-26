"""
QUANT ALPHA — STRATEGY VISUALIZER v4
- Replaced matplotlib with Plotly (pre-installed on Streamlit Cloud)
- Replaced yfinance with CoinGecko API (only needs requests)
- Works on Python 3.14+
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    border: 1px solid #3d2f00;
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 24px;
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
    letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 8px;
}

.parsed-box {
    background: #0a0c10; border: 1px solid #1e2030;
    border-left: 4px solid #f59e0b; border-radius: 8px;
    padding: 16px; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem; color: #a89060; margin: 12px 0;
}

.tag { display:inline-block; padding:2px 10px; border-radius:4px;
       font-family:'IBM Plex Mono',monospace; font-size:0.72rem; margin:3px; }
.tag-entry { background:#1a2e1a; color:#4ade80; border:1px solid #166534; }
.tag-sl    { background:#2e1a1a; color:#f87171; border:1px solid #991b1b; }
.tag-tp    { background:#1a2a1a; color:#86efac; border:1px solid #15803d; }

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
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
COINGECKO_IDS = {
    'BTC': 'bitcoin',       'ETH': 'ethereum',
    'SOL': 'solana',        'BNB': 'binancecoin',
    'XRP': 'ripple',        'ADA': 'cardano',
    'DOGE': 'dogecoin',     'AVAX': 'avalanche-2',
    'MATIC': 'matic-network','LINK': 'chainlink',
    'DOT': 'polkadot',      'UNI': 'uniswap',
    'LTC': 'litecoin',      'ATOM': 'cosmos',
    'NEAR': 'near',
}
PERIOD_DAYS = {
    '1mo': 30, '3mo': 90,
    '6mo': 180, '1y': 365, '2y': 730
}

# ─────────────────────────────────────────────────────────────
# GEMINI
# ─────────────────────────────────────────────────────────────
def init_gemini():
    if not GEMINI_AVAILABLE:
        return None
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash-8b')
    except Exception as e:
        st.error(f"Gemini error: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# DATA — COINGECKO (only uses requests)
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_crypto_data(symbol: str, days: int = 90):
    coin_id = COINGECKO_IDS.get(symbol.upper(), symbol.lower())
    url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}"
           f"/ohlc?vs_currency=usd&days={days}")
    try:
        resp = requests.get(
            url, timeout=15,
            headers={'User-Agent': 'QuantAlpha/1.0'}
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
        df = df.set_index('Date').drop('timestamp', axis=1)
        df = df.astype(float)
        df['Volume'] = 0.0
        df = df.resample('D').last().dropna()
        return df
    except Exception as e:
        st.error(f"Data error: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# GEMINI HELPERS
# ─────────────────────────────────────────────────────────────
def parse_strategy(model, description: str):
    prompt = f"""Parse this trading strategy into JSON.
Strategy: "{description}"

Return ONLY valid JSON — no markdown, no explanation:
{{
  "entry_long": "long entry condition or null",
  "entry_short": "short entry condition or null",
  "stop_loss": "stop loss description",
  "take_profit": "take profit description",
  "indicators": ["indicator1"],
  "strategy_type": "trend or mean-reversion or breakout or momentum",
  "sl_pct": 0.02,
  "tp_pct": 0.06,
  "indicator_params": {{
    "ema_fast": 20, "ema_slow": 50,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30
  }},
  "summary": "one sentence summary"
}}"""
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        return json.loads(text.strip())
    except Exception as e:
        st.error(f"Parse error: {e}")
        return None


def generate_python_code(model, strategy: dict, symbol: str) -> str:
    prompt = f"""Generate complete Python backtesting code.
Strategy: {json.dumps(strategy, indent=2)}
Asset: {symbol}

Use CoinGecko API for data:
https://api.coingecko.com/api/v3/coins/COIN_ID/ohlc?vs_currency=usd&days=365

Requirements:
- pandas and numpy only (no yfinance, no matplotlib)
- .shift(1) on all signals — no lookahead bias
- 0.1% commission per trade
- Entry on next bar Open
- Calculate: Sharpe, max drawdown, win rate, total return
- Use plotly for charts
- Clean commented professional code

Return ONLY Python code, no explanation."""
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
# INDICATORS
# ─────────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    ef = params.get('ema_fast', 20)
    es = params.get('ema_slow', 50)
    rp = params.get('rsi_period', 14)

    df['EMA_fast'] = df['Close'].ewm(span=ef).mean()
    df['EMA_slow'] = df['Close'].ewm(span=es).mean()

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


def generate_signals(df: pd.DataFrame, strategy: dict) -> pd.DataFrame:
    df    = df.copy()
    p     = strategy.get('indicator_params', {})
    stype = strategy.get('strategy_type', 'trend')
    rob   = p.get('rsi_overbought', 70)
    ros   = p.get('rsi_oversold', 30)

    df['long_signal']  = False
    df['short_signal'] = False

    if stype in ['trend', 'momentum']:
        df['long_signal'] = (
            (df['EMA_fast'] > df['EMA_slow']) &
            (df['EMA_fast'].shift(1) <= df['EMA_slow'].shift(1))
        )
        df['short_signal'] = (
            (df['EMA_fast'] < df['EMA_slow']) &
            (df['EMA_fast'].shift(1) >= df['EMA_slow'].shift(1))
        )
    elif stype == 'mean-reversion':
        df['long_signal'] = (
            (df['RSI'] < ros) & (df['RSI'].shift(1) >= ros)
        )
        df['short_signal'] = (
            (df['RSI'] > rob) & (df['RSI'].shift(1) <= rob)
        )
    elif stype == 'breakout':
        df['long_signal'] = (
            (df['Close'] > df['BB_upper']) &
            (df['Close'].shift(1) <= df['BB_upper'].shift(1))
        )
        df['short_signal'] = (
            (df['Close'] < df['BB_lower']) &
            (df['Close'].shift(1) >= df['BB_lower'].shift(1))
        )
    return df

# ─────────────────────────────────────────────────────────────
# PLOTLY CHART
# ─────────────────────────────────────────────────────────────
def draw_chart(df: pd.DataFrame,
               strategy: dict, symbol: str):
    df_plot = df.tail(80).copy()
    sl_pct  = strategy.get('sl_pct', 0.02)
    tp_pct  = strategy.get('tp_pct', 0.06)
    stype   = strategy.get('strategy_type', 'trend')
    params  = strategy.get('indicator_params', {})

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.75, 0.25],
        subplot_titles=[
            f"{symbol} — {strategy.get('summary','Strategy')}",
            'RSI' if stype == 'mean-reversion' else 'Volume'
        ]
    )

    # ── Candlesticks ──────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['Open'],
        high=df_plot['High'],
        low=df_plot['Low'],
        close=df_plot['Close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350',
    ), row=1, col=1)

    # ── EMAs ──────────────────────────────────────────────────
    ef_span = params.get('ema_fast', 20)
    es_span = params.get('ema_slow', 50)

    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['EMA_fast'],
        name=f'EMA {ef_span}',
        line=dict(color='#f59e0b', width=1.5),
        opacity=0.9
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['EMA_slow'],
        name=f'EMA {es_span}',
        line=dict(color='#60a5fa', width=1.5),
        opacity=0.9
    ), row=1, col=1)

    # ── Bollinger Bands ───────────────────────────────────────
    if stype == 'breakout':
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['BB_upper'],
            name='BB Upper',
            line=dict(color='#f59e0b', width=1, dash='dash'),
            opacity=0.5
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['BB_lower'],
            name='BB Lower',
            line=dict(color='#f59e0b', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(245,158,11,0.05)',
            opacity=0.5
        ), row=1, col=1)

    # ── Long signals ──────────────────────────────────────────
    long_df = df_plot[df_plot['long_signal']]
    if not long_df.empty:
        fig.add_trace(go.Scatter(
            x=long_df.index,
            y=long_df['Close'] * 0.995,
            mode='markers',
            name='Long Entry',
            marker=dict(
                symbol='triangle-up',
                size=14,
                color='#4ade80',
                line=dict(color='#166534', width=1)
            )
        ), row=1, col=1)

        # SL / TP lines for each long signal
        for date, row in long_df.iterrows():
            entry = row['Close']
            sl    = entry * (1 - sl_pct)
            tp    = entry * (1 + tp_pct)

            # Find end date (8 bars forward)
            idx_pos = df_plot.index.get_loc(date)
            end_idx = min(idx_pos + 8, len(df_plot) - 1)
            end_date = df_plot.index[end_idx]

            fig.add_shape(
                type='line',
                x0=date, x1=end_date, y0=sl, y1=sl,
                line=dict(color='#ef4444', width=1, dash='dash'),
                row=1, col=1
            )
            fig.add_shape(
                type='line',
                x0=date, x1=end_date, y0=tp, y1=tp,
                line=dict(color='#4ade80', width=1, dash='dot'),
                row=1, col=1
            )
            fig.add_annotation(
                x=end_date, y=sl,
                text=f"SL {sl_pct*100:.0f}%",
                showarrow=False,
                font=dict(color='#ef4444', size=9),
                xanchor='left', row=1, col=1
            )
            fig.add_annotation(
                x=end_date, y=tp,
                text=f"TP {tp_pct*100:.0f}%",
                showarrow=False,
                font=dict(color='#4ade80', size=9),
                xanchor='left', row=1, col=1
            )

    # ── Short signals ─────────────────────────────────────────
    short_df = df_plot[df_plot['short_signal']]
    if not short_df.empty:
        fig.add_trace(go.Scatter(
            x=short_df.index,
            y=short_df['Close'] * 1.005,
            mode='markers',
            name='Short Entry',
            marker=dict(
                symbol='triangle-down',
                size=14,
                color='#f87171',
                line=dict(color='#991b1b', width=1)
            )
        ), row=1, col=1)

    # ── Bottom panel ──────────────────────────────────────────
    if stype == 'mean-reversion':
        rob = params.get('rsi_overbought', 70)
        ros = params.get('rsi_oversold', 30)

        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['RSI'],
            name='RSI',
            line=dict(color='#a78bfa', width=1.5)
        ), row=2, col=1)

        fig.add_hline(y=rob, line_color='#ef4444',
                      line_dash='dash', line_width=1,
                      opacity=0.6, row=2, col=1)
        fig.add_hline(y=ros, line_color='#4ade80',
                      line_dash='dash', line_width=1,
                      opacity=0.6, row=2, col=1)
    else:
        colors = [
            '#26a69a' if c >= o else '#ef5350'
            for c, o in zip(df_plot['Close'], df_plot['Open'])
        ]
        fig.add_trace(go.Bar(
            x=df_plot.index,
            y=df_plot['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.6
        ), row=2, col=1)

    # ── Layout ────────────────────────────────────────────────
    fig.update_layout(
        height=600,
        paper_bgcolor='#080a0f',
        plot_bgcolor='#0d0f14',
        font=dict(family='IBM Plex Mono', color='#a89060', size=11),
        legend=dict(
            bgcolor='#0d0f14',
            bordercolor='#1e2030',
            borderwidth=1,
            font=dict(color='#a89060', size=10)
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=60, b=40),
        title=dict(
            text=(f"<b>{symbol}</b> — "
                  f"{strategy.get('summary','Strategy')}<br>"
                  f"<span style='font-size:11px;color:#6b5b3a'>"
                  f"🟢 {len(long_df)} Long  "
                  f"🔴 {len(short_df)} Short  "
                  f"| Last 80 bars</span>"),
            font=dict(color='#f59e0b', size=13),
            x=0.01
        )
    )

    fig.update_xaxes(
        gridcolor='#1e2030',
        zerolinecolor='#1e2030',
        tickfont=dict(color='#6b5b3a')
    )
    fig.update_yaxes(
        gridcolor='#1e2030',
        zerolinecolor='#1e2030',
        tickfont=dict(color='#6b5b3a')
    )

    return fig

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📈 STRATEGY VISUALIZER</h1>
    <p>Describe your strategy → See it on real candles → Get Python code</p>
    <p style="color:#3d2f00;font-family:'IBM Plex Mono';font-size:0.7rem">
    QUANT ALPHA · GEMINI + COINGECKO · INTERACTIVE CHARTS · $0
    </p>
</div>""", unsafe_allow_html=True)

model = init_gemini()
if not model:
    st.error("⚠️ Add GEMINI_API_KEY to Streamlit Secrets.")
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='font-family:IBM Plex Mono;font-size:0.68rem;
    color:#f59e0b;letter-spacing:2px;text-transform:uppercase;
    border-bottom:1px solid #1e2030;padding-bottom:8px;margin-bottom:16px'>
    ⚙ SETTINGS</div>""", unsafe_allow_html=True)

    symbol = st.selectbox(
        "Asset", options=list(COINGECKO_IDS.keys()), index=0
    )
    period = st.selectbox(
        "Period", options=list(PERIOD_DAYS.keys()), index=1
    )

    st.markdown("---")
    st.markdown("""<div style='font-family:IBM Plex Mono;
    font-size:0.65rem;color:#3d2f00'>
    <b style='color:#f59e0b'>EXAMPLES:</b><br><br>
    "Buy when 20 EMA crosses above
    50 EMA. SL 2%, TP 6%."<br><br>
    "Long when RSI drops below 30.
    SL 3%, TP 9%."<br><br>
    "Short Bollinger lower break.
    SL 1.5%, TP 5%."
    </div>""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────
for key in ['parsed', 'df', 'fig', 'code']:
    if key not in st.session_state:
        st.session_state[key] = None

# ── STEP 1 ────────────────────────────────────────────────────
st.markdown(
    '<div class="section-hdr">STEP 1 — DESCRIBE YOUR STRATEGY</div>',
    unsafe_allow_html=True
)
st.markdown("""<div class="step-card active">
<div class="step-num">✏️ PLAIN ENGLISH — NO CODING NEEDED</div>
Describe entry, stop loss, and take profit in plain language.
</div>""", unsafe_allow_html=True)

description = st.text_area(
    "Strategy",
    placeholder=(
        "Buy when 20 EMA crosses above 50 EMA. "
        "Stop loss 2%, take profit 6%."
    ),
    height=100,
    label_visibility="collapsed"
)

c1, c2 = st.columns([3, 1])
with c1:
    parse_btn = st.button("🧠 PARSE STRATEGY", use_container_width=True)
with c2:
    reset_btn = st.button("↺ Reset", use_container_width=True)

if reset_btn:
    for key in ['parsed', 'df', 'fig', 'code']:
        st.session_state[key] = None
    st.rerun()

if parse_btn and description.strip():
    with st.spinner("Parsing your strategy..."):
        parsed = parse_strategy(model, description)
    if parsed:
        st.session_state.parsed = parsed
        st.session_state.fig    = None
        st.session_state.code   = None

# ── STEP 2 ────────────────────────────────────────────────────
if st.session_state.parsed:
    p = st.session_state.parsed

    st.markdown(
        '<div class="section-hdr">STEP 2 — CONFIRM UNDERSTANDING</div>',
        unsafe_allow_html=True
    )
    st.markdown(f"""<div class="parsed-box">
    <b style='color:#f59e0b'>AI PARSED AS:</b><br><br>
    <b>Summary:</b> {p.get('summary', '—')}<br>
    <b>Type:</b> {p.get('strategy_type', '—').upper()}<br>
    <b>Indicators:</b> {', '.join(p.get('indicators', []))}
    </div>""", unsafe_allow_html=True)

    for col, (cls, txt) in zip(st.columns(4), [
        ('tag-entry', f"📈 LONG: {str(p.get('entry_long','—'))[:30]}"),
        ('tag-entry', f"📉 SHORT: {str(p.get('entry_short','—'))[:30]}"),
        ('tag-sl',    f"🛑 SL: {p.get('sl_pct', 0.02)*100:.1f}%"),
        ('tag-tp',    f"🎯 TP: {p.get('tp_pct', 0.06)*100:.1f}%"),
    ]):
        with col:
            st.markdown(
                f'<span class="tag {cls}">{txt}</span>',
                unsafe_allow_html=True
            )

    st.markdown("")
    if st.button("📊 VISUALIZE ON REAL CANDLES",
                 use_container_width=True):
        days = PERIOD_DAYS.get(period, 90)
        with st.spinner(f"Fetching {symbol} from CoinGecko..."):
            df = fetch_crypto_data(symbol, days)

        if df is not None and len(df) > 60:
            with st.spinner("Building interactive chart..."):
                df = add_indicators(df, p.get('indicator_params', {}))
                df = generate_signals(df, p)
                st.session_state.df  = df
                st.session_state.fig = draw_chart(df, p, symbol)
        else:
            st.error(f"No data for {symbol}. Try BTC or ETH.")

# ── STEP 3 ────────────────────────────────────────────────────
if st.session_state.fig:
    st.markdown(
        '<div class="section-hdr">STEP 3 — IS THIS YOUR SETUP?</div>',
        unsafe_allow_html=True
    )

    st.plotly_chart(
        st.session_state.fig,
        use_container_width=True,
        config={'displayModeBar': True,
                'scrollZoom': True}
    )

    st.markdown("""<div style='text-align:center;
    font-family:IBM Plex Mono;font-size:0.82rem;
    color:#a89060;margin:12px 0'>
    🔺 Green triangles = Long entries &nbsp;|&nbsp;
    🔻 Red triangles = Short entries<br>
    Dashed = Stop Loss &nbsp;|&nbsp; Dotted = Take Profit
    &nbsp;|&nbsp; 🖱️ Scroll to zoom · Drag to pan
    </div>""", unsafe_allow_html=True)

    cy, cn = st.columns(2)
    with cy:
        yes_btn = st.button(
            "✅ YES — Generate Python Code",
            use_container_width=True
        )
    with cn:
        no_btn = st.button(
            "❌ NO — Redescribe",
            use_container_width=True
        )

    if no_btn:
        st.session_state.fig  = None
        st.session_state.code = None
        st.info("Go back to Step 1 and refine your description.")

    if yes_btn:
        with st.spinner("Generating Python backtest code..."):
            st.session_state.code = generate_python_code(
                model, st.session_state.parsed, symbol
            )

# ── STEP 4 ────────────────────────────────────────────────────
if st.session_state.code:
    st.markdown(
        '<div class="section-hdr">STEP 4 — YOUR PYTHON CODE</div>',
        unsafe_allow_html=True
    )
    st.markdown("""<div class="step-card done">
    <div class="step-num">✅ READY — RUN IN COLAB OR JUPYTER</div>
    Then validate it in the <b>Backtest Validator</b>
    to detect lookahead bias and overfitting.
    </div>""", unsafe_allow_html=True)

    st.text_area(
        "Code",
        value=st.session_state.code,
        height=300,
        label_visibility="collapsed"
    )

    st.download_button(
        "⬇️ Download .py file",
        data=st.session_state.code,
        file_name=f"{symbol}_strategy.py",
        mime="text/plain",
        use_container_width=True
    )

    st.markdown("""<div style='background:#0d0f14;
    border:1px solid #f59e0b;border-radius:10px;
    padding:16px;margin-top:16px;text-align:center'>
    <b style='font-family:IBM Plex Mono;color:#f59e0b'>
    ⚠️ VALIDATE BEFORE TRADING
    </b><br>
    <span style='font-family:IBM Plex Mono;
    color:#6b5b3a;font-size:0.8rem'>
    Paste code into <b style='color:#e8e0d0'>
    Backtest Validator</b> to check for errors
    </span></div>""", unsafe_allow_html=True)

# Footer
st.markdown("""<div style="text-align:center;margin-top:48px;
padding:16px;border-top:1px solid #1e2030">
<span style="font-family:IBM Plex Mono;
font-size:0.65rem;color:#1e2030">
QUANT ALPHA — NOT FINANCIAL ADVICE
</span></div>""", unsafe_allow_html=True)
