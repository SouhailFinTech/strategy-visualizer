"""
QUANT ALPHA — STRATEGY VISUALIZER v3
Fixed: replaced yfinance with CoinGecko API (no install needed)
Uses only: streamlit, pandas, numpy, matplotlib, requests, google-generativeai
All available on Streamlit Cloud without issues.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
import json
import io
from datetime import datetime

# ── Gemini ────────────────────────────────────────────────────
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
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px;
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
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.85rem !important;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SUPPORTED ASSETS
# ─────────────────────────────────────────────────────────────
COINGECKO_IDS = {
    'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana',
    'BNB': 'binancecoin', 'XRP': 'ripple', 'ADA': 'cardano',
    'DOGE': 'dogecoin', 'AVAX': 'avalanche-2', 'MATIC': 'matic-network',
    'LINK': 'chainlink', 'DOT': 'polkadot', 'UNI': 'uniswap',
    'LTC': 'litecoin', 'ATOM': 'cosmos', 'NEAR': 'near',
}
PERIOD_DAYS = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730}

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
        st.error(f"Gemini error: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# DATA — COINGECKO FREE API (only needs requests, no yfinance)
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_crypto_data(symbol: str, days: int = 90) -> pd.DataFrame:
    coin_id = COINGECKO_IDS.get(symbol.upper(), symbol.lower())
    url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}"
           f"/ohlc?vs_currency=usd&days={days}")
    try:
        resp = requests.get(url, timeout=15,
                           headers={'User-Agent': 'QuantAlpha/1.0'})
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data or not isinstance(data, list):
            return None
        df = pd.DataFrame(data,
                         columns=['timestamp','Open','High','Low','Close'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('Date').drop('timestamp', axis=1)
        df = df.astype(float)
        df['Volume'] = 0
        df = df.resample('D').last().dropna()
        return df
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# GEMINI HELPERS
# ─────────────────────────────────────────────────────────────
def parse_strategy(model, description: str) -> dict:
    prompt = f"""Parse this trading strategy into JSON.
Strategy: "{description}"
Return ONLY valid JSON:
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
    "rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30
  }},
  "summary": "one sentence summary"
}}
Return ONLY the JSON object. No markdown. No explanation."""
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
Use CoinGecko API for data (NOT yfinance):
https://api.coingecko.com/api/v3/coins/COIN_ID/ohlc?vs_currency=usd&days=365
Requirements:
- pandas and numpy only
- .shift(1) on all signals (no lookahead)
- 0.1% commission per trade
- Entry on next bar Open
- Calculate Sharpe, max drawdown, win rate, total return
- matplotlib dark theme chart
Return ONLY the Python code."""
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if '```python' in text:
            text = text.split('```python')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        return text.strip()
    except Exception as e:
        return f"# Error: {e}"

# ─────────────────────────────────────────────────────────────
# INDICATORS + SIGNALS
# ─────────────────────────────────────────────────────────────
def add_indicators(df, params):
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


def generate_signals(df, strategy):
    df = df.copy()
    p     = strategy.get('indicator_params', {})
    stype = strategy.get('strategy_type', 'trend')
    rob   = p.get('rsi_overbought', 70)
    ros   = p.get('rsi_oversold', 30)
    df['long_signal']  = False
    df['short_signal'] = False
    if stype in ['trend', 'momentum']:
        df['long_signal']  = ((df['EMA_fast'] > df['EMA_slow']) &
                              (df['EMA_fast'].shift(1) <= df['EMA_slow'].shift(1)))
        df['short_signal'] = ((df['EMA_fast'] < df['EMA_slow']) &
                              (df['EMA_fast'].shift(1) >= df['EMA_slow'].shift(1)))
    elif stype == 'mean-reversion':
        df['long_signal']  = ((df['RSI'] < ros) & (df['RSI'].shift(1) >= ros))
        df['short_signal'] = ((df['RSI'] > rob) & (df['RSI'].shift(1) <= rob))
    elif stype == 'breakout':
        df['long_signal']  = ((df['Close'] > df['BB_upper']) &
                              (df['Close'].shift(1) <= df['BB_upper'].shift(1)))
        df['short_signal'] = ((df['Close'] < df['BB_lower']) &
                              (df['Close'].shift(1) >= df['BB_lower'].shift(1)))
    return df

# ─────────────────────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────────────────────
def draw_chart(df, strategy, symbol):
    df_plot = df.tail(60).copy().reset_index()
    sl      = strategy.get('sl_pct', 0.02)
    tp      = strategy.get('tp_pct', 0.06)
    stype   = strategy.get('strategy_type', 'trend')
    fig     = plt.figure(figsize=(14, 10), facecolor='#080a0f')
    ax1     = fig.add_subplot(3, 1, (1, 2))
    ax2     = fig.add_subplot(3, 1, 3, sharex=ax1)

    for ax in [ax1, ax2]:
        ax.set_facecolor('#0d0f14')
        ax.tick_params(colors='#6b5b3a', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#1e2030')
        ax.grid(color='#1e2030', linewidth=0.5, alpha=0.5)

    x = np.arange(len(df_plot))

    for i, row in df_plot.iterrows():
        o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
        col = '#26a69a' if c >= o else '#ef5350'
        ax1.plot([i,i], [l,h], color=col, linewidth=0.8, alpha=0.8)
        ax1.add_patch(plt.Rectangle((i-0.3, min(o,c)), 0.6,
                      max(abs(c-o),(h-l)*0.01), color=col, alpha=0.9))

    ax1.plot(x, df_plot['EMA_fast'].values, color='#f59e0b',
             linewidth=1.2, label=f"EMA {strategy.get('indicator_params',{}).get('ema_fast',20)}")
    ax1.plot(x, df_plot['EMA_slow'].values, color='#60a5fa',
             linewidth=1.2, label=f"EMA {strategy.get('indicator_params',{}).get('ema_slow',50)}")

    for idx in df_plot[df_plot['long_signal']].index:
        row = df_plot.loc[idx]
        e   = row['Close']
        ax1.annotate('', xy=(idx, e*0.997),
                     xytext=(idx, e*(1-sl*0.6)),
                     arrowprops=dict(arrowstyle='->', color='#4ade80', lw=2.5))
        ax1.hlines(e*(1-sl), idx, min(idx+8,len(df_plot)-1),
                   colors='#ef4444', linewidth=1, linestyles='dashed', alpha=0.7)
        ax1.hlines(e*(1+tp), idx, min(idx+8,len(df_plot)-1),
                   colors='#4ade80', linewidth=1, linestyles='dotted', alpha=0.7)
        ax1.text(idx+0.3, e*(1-sl)*0.9985,
                 f'SL {sl*100:.0f}%', color='#ef4444',
                 fontsize=6.5, fontfamily='monospace')
        ax1.text(idx+0.3, e*(1+tp)*1.001,
                 f'TP {tp*100:.0f}%', color='#4ade80',
                 fontsize=6.5, fontfamily='monospace')

    for idx in df_plot[df_plot['short_signal']].index:
        row = df_plot.loc[idx]
        e   = row['Close']
        ax1.annotate('', xy=(idx, e*1.003),
                     xytext=(idx, e*(1+sl*0.6)),
                     arrowprops=dict(arrowstyle='->', color='#f87171', lw=2.5))
        ax1.hlines(e*(1+sl), idx, min(idx+8,len(df_plot)-1),
                   colors='#ef4444', linewidth=1, linestyles='dashed', alpha=0.7)
        ax1.hlines(e*(1-tp), idx, min(idx+8,len(df_plot)-1),
                   colors='#4ade80', linewidth=1, linestyles='dotted', alpha=0.7)

    if stype == 'mean-reversion':
        ax2.plot(x, df_plot['RSI'].values, color='#a78bfa', linewidth=1.2)
        rob = strategy.get('indicator_params',{}).get('rsi_overbought',70)
        ros = strategy.get('indicator_params',{}).get('rsi_oversold',30)
        ax2.axhline(rob, color='#ef4444', linewidth=0.8, linestyle='--', alpha=0.6)
        ax2.axhline(ros, color='#4ade80', linewidth=0.8, linestyle='--', alpha=0.6)
        ax2.set_ylabel('RSI', color='#6b5b3a', fontsize=8)
        ax2.set_ylim(0, 100)
    else:
        ax2.bar(x, df_plot['Volume'].values, color='#26a69a', alpha=0.5)
        ax2.set_ylabel('Volume', color='#6b5b3a', fontsize=8)

    n_ticks  = min(8, len(df_plot))
    tick_idx = np.linspace(0, len(df_plot)-1, n_ticks, dtype=int)
    tick_lbl = []
    for i in tick_idx:
        try:
            d = df_plot.loc[i,'Date'] if 'Date' in df_plot.columns \
                else df_plot.index[i]
            tick_lbl.append(pd.Timestamp(d).strftime('%b %d'))
        except Exception:
            tick_lbl.append(str(i))
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels(tick_lbl, rotation=30, ha='right', fontsize=7)

    ax1.legend(handles=[
        mpatches.Patch(color='#f59e0b', label='EMA Fast'),
        mpatches.Patch(color='#60a5fa', label='EMA Slow'),
        mpatches.Patch(color='#4ade80', label='Long ↑'),
        mpatches.Patch(color='#f87171', label='Short ↓'),
        mpatches.Patch(color='#ef4444', label='SL ---'),
        mpatches.Patch(color='#4ade80', label='TP ···'),
    ], loc='upper left', facecolor='#0d0f14',
       edgecolor='#1e2030', labelcolor='#a89060', fontsize=7.5)

    n_long  = int(df_plot['long_signal'].sum())
    n_short = int(df_plot['short_signal'].sum())
    fig.suptitle(
        f"{symbol} — {strategy.get('summary','Strategy')}\n"
        f"{n_long} Long  |  {n_short} Short  |  Last 60 bars",
        color='#f59e0b', fontsize=10, fontfamily='monospace', y=0.98)
    plt.tight_layout(rect=[0,0,1,0.96])
    return fig

# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📈 STRATEGY VISUALIZER</h1>
    <p>Describe your strategy → See it on real candles → Get Python code</p>
    <p style="color:#3d2f00;font-family:'IBM Plex Mono';font-size:0.7rem">
    QUANT ALPHA · GEMINI + COINGECKO · $0 COST
    </p>
</div>""", unsafe_allow_html=True)

model = init_gemini()
if not model:
    st.error("⚠️ Add GEMINI_API_KEY to Streamlit Secrets.")
    st.stop()

with st.sidebar:
    st.markdown("""<div style='font-family:IBM Plex Mono;font-size:0.68rem;
    color:#f59e0b;letter-spacing:2px;text-transform:uppercase;
    border-bottom:1px solid #1e2030;padding-bottom:8px;margin-bottom:16px'>
    ⚙ SETTINGS</div>""", unsafe_allow_html=True)

    symbol = st.selectbox("Asset", options=list(COINGECKO_IDS.keys()), index=0)
    period = st.selectbox("Period", options=list(PERIOD_DAYS.keys()), index=1)

    st.markdown("---")
    st.markdown("""<div style='font-family:IBM Plex Mono;font-size:0.65rem;color:#3d2f00'>
    <b style='color:#f59e0b'>EXAMPLES:</b><br><br>
    "Buy when 20 EMA crosses above 50 EMA. SL 2%, TP 6%."<br><br>
    "Long when RSI drops below 30. SL 3%, TP 9%."<br><br>
    "Short on Bollinger lower band break. SL 1.5%, TP 5%."
    </div>""", unsafe_allow_html=True)

for key in ['parsed','df','chart_done','code']:
    if key not in st.session_state:
        st.session_state[key] = None

# STEP 1
st.markdown('<div class="section-hdr">STEP 1 — DESCRIBE YOUR STRATEGY</div>',
            unsafe_allow_html=True)
st.markdown("""<div class="step-card active">
<div class="step-num">✏️ PLAIN ENGLISH — NO CODING NEEDED</div>
Describe entry conditions, stop loss, and take profit.
</div>""", unsafe_allow_html=True)

description = st.text_area("Strategy",
    placeholder="Buy when 20 EMA crosses above 50 EMA. SL 2%, TP 6%.",
    height=100, label_visibility="collapsed")

c1, c2 = st.columns([3,1])
with c1: parse_btn = st.button("🧠 PARSE STRATEGY", use_container_width=True)
with c2: reset_btn = st.button("↺ Reset", use_container_width=True)

if reset_btn:
    for key in ['parsed','df','chart_done','code']:
        st.session_state[key] = None
    st.rerun()

if parse_btn and description.strip():
    with st.spinner("Parsing..."):
        parsed = parse_strategy(model, description)
    if parsed:
        st.session_state.parsed = parsed
        st.session_state.chart_done = None
        st.session_state.code = None

# STEP 2
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
        ('tag-entry', f"📈 LONG: {str(p.get('entry_long','—'))[:30]}"),
        ('tag-entry', f"📉 SHORT: {str(p.get('entry_short','—'))[:30]}"),
        ('tag-sl',    f"🛑 SL: {p.get('sl_pct',0.02)*100:.1f}%"),
        ('tag-tp',    f"🎯 TP: {p.get('tp_pct',0.06)*100:.1f}%"),
    ]):
        with col:
            st.markdown(f'<span class="tag {cls}">{txt}</span>',
                       unsafe_allow_html=True)

    st.markdown("")
    if st.button("📊 VISUALIZE ON REAL CANDLES", use_container_width=True):
        days = PERIOD_DAYS.get(period, 90)
        with st.spinner(f"Fetching {symbol} from CoinGecko..."):
            df = fetch_crypto_data(symbol, days)
        if df is not None and len(df) > 60:
            with st.spinner("Drawing chart..."):
                df = add_indicators(df, p.get('indicator_params',{}))
                df = generate_signals(df, p)
                st.session_state.df = df
                fig = draw_chart(df, p, symbol)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=130,
                           facecolor='#080a0f', bbox_inches='tight')
                buf.seek(0)
                st.session_state.chart_done = buf
                plt.close(fig)
        else:
            st.error(f"No data for {symbol}. Try BTC or ETH.")

# STEP 3
if st.session_state.chart_done:
    st.markdown('<div class="section-hdr">STEP 3 — IS THIS YOUR SETUP?</div>',
                unsafe_allow_html=True)
    st.image(st.session_state.chart_done, use_column_width=True)
    st.download_button("⬇️ Download Chart",
                       data=st.session_state.chart_done,
                       file_name=f"{symbol}_strategy.png",
                       mime="image/png")
    st.markdown("""<div style='text-align:center;font-family:IBM Plex Mono;
    font-size:0.85rem;color:#a89060;margin:16px 0'>
    🟢 Green = Long  |  🔴 Red = Short  |
    Dashed = SL  |  Dotted = TP
    </div>""", unsafe_allow_html=True)

    cy, cn = st.columns(2)
    with cy: yes_btn = st.button("✅ YES — Generate Code", use_container_width=True)
    with cn: no_btn  = st.button("❌ NO — Redescribe",     use_container_width=True)

    if no_btn:
        st.session_state.chart_done = None
        st.session_state.code = None
        st.info("Refine your description in Step 1.")
    if yes_btn:
        with st.spinner("Generating Python code..."):
            st.session_state.code = generate_python_code(
                model, st.session_state.parsed, symbol)

# STEP 4
if st.session_state.code:
    st.markdown('<div class="section-hdr">STEP 4 — YOUR PYTHON CODE</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="step-card done">
    <div class="step-num">✅ READY — RUN IN COLAB OR JUPYTER</div>
    Then validate in the <b>Backtest Validator</b>.
    </div>""", unsafe_allow_html=True)
    st.text_area("Code", value=st.session_state.code,
                height=300, label_visibility="collapsed")
    st.download_button("⬇️ Download .py",
                       data=st.session_state.code,
                       file_name=f"{symbol}_strategy.py",
                       mime="text/plain",
                       use_container_width=True)
    st.markdown("""<div style='background:#0d0f14;border:1px solid #f59e0b;
    border-radius:10px;padding:16px;margin-top:16px;text-align:center'>
    <b style='font-family:IBM Plex Mono;color:#f59e0b'>
    ⚠️ VALIDATE THIS BACKTEST BEFORE TRADING
    </b><br>
    <span style='font-family:IBM Plex Mono;color:#6b5b3a;font-size:0.8rem'>
    Paste code into Backtest Validator → detect lookahead + overfitting
    </span></div>""", unsafe_allow_html=True)

st.markdown("""<div style="text-align:center;margin-top:48px;
padding:16px;border-top:1px solid #1e2030">
<span style="font-family:IBM Plex Mono;font-size:0.65rem;color:#1e2030">
QUANT ALPHA — NOT FINANCIAL ADVICE
</span></div>""", unsafe_allow_html=True)
