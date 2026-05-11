"""
QUANT ALPHA — STRATEGY VISUALIZER v9
FIXES:
- Only shows/generates indicators explicitly requested by user
- Clean static library in exported code (no __file__ reading bugs)
- Full MT5 CSV support (<DATE>, <TIME>, YYYY.MM.DD format)
- Smart indicator filtering in charts & backtests
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
html, body, [class*="css"] { font-family: 'Syne', sans-serif; background-color: #080a0f; color: #e8e0d0; }
.stApp { background-color: #080a0f; }
.main-header { background: linear-gradient(135deg, #0d0f14 0%, #1a1508 100%); border: 1px solid #3d2f00; border-radius: 12px; padding: 28px 32px; margin-bottom: 24px; }
.main-header h1 { font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; color: #f59e0b; margin: 0; letter-spacing: -1px; }
.main-header p { color: #6b5b3a; margin: 6px 0 0; font-size: 0.9rem; }
.step-card { background: #0d0f14; border: 1px solid #1e2030; border-radius: 10px; padding: 20px; margin-bottom: 16px; }
.step-card.active { border-color: #f59e0b; }
.step-card.done   { border-color: #22c55e; }
.step-num { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #f59e0b; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px; }
.parsed-box { background: #0a0c10; border: 1px solid #1e2030; border-left: 4px solid #f59e0b; border-radius: 8px; padding: 16px; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: #a89060; margin: 12px 0; }
.tag { display: inline-block; padding: 2px 10px; border-radius: 4px; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; margin: 3px; }
.tag-entry { background:#1a2e1a; color:#4ade80; border:1px solid #166534; }
.tag-sl    { background:#2e1a1a; color:#f87171; border:1px solid #991b1b; }
.tag-tp    { background:#1a2a1a; color:#86efac; border:1px solid #15803d; }
.data-source-box { background:#0a0c10; border:1px solid #1e2030; border-left:4px solid #3b82f6; border-radius:8px; padding:12px 16px; font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#60a5fa; margin:8px 0; }
.section-hdr { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: #f59e0b; letter-spacing: 3px; text-transform: uppercase; border-bottom: 1px solid #1e2030; padding-bottom: 8px; margin: 20px 0 14px; }
[data-testid="stSidebar"] { background: #06080c; border-right: 1px solid #1e2030; }
.stButton > button { background: linear-gradient(135deg,#92400e,#b45309); color: #fef3c7; border: none; border-radius: 8px; font-family: 'IBM Plex Mono', monospace; font-weight: 700; padding: 12px 24px; width: 100%; transition: all 0.2s; letter-spacing: 1px; }
.stButton > button:hover { background: linear-gradient(135deg,#b45309,#d97706); transform: translateY(-1px); }
.stTextArea textarea, .stTextInput input { background: #0a0c10 !important; color: #e8e0d0 !important; border: 1px solid #1e2030 !important; border-radius: 8px !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.85rem !important; }
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
# DATA FETCHING (MT5 FIXED)
# ─────────────────────────────────────────────────────────────
def fetch_binance(symbol: str, period: str):
    sym = BINANCE_SYMBOLS.get(symbol.upper())
    limit = BINANCE_LIMITS.get(period, 90)
    if not sym: return None
    import time
    for attempt in range(3):
        try:
            resp = requests.get("https://api.binance.com/api/v3/klines",
                params={'symbol': sym, 'interval': '1d', 'limit': min(limit, 1000)},
                timeout=15, headers={'User-Agent': 'QuantAlpha/1.0'})
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    df = pd.DataFrame(data, columns=['timestamp','Open','High','Low','Close','Volume','ct','qv','nt','tbb','tbq','ignore'])
                    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.set_index('Date')[['Open','High','Low','Close','Volume']].astype(float).dropna()
                    return df
            if attempt < 2: time.sleep(2)
        except: 
            if attempt < 2: time.sleep(2)
    return None

def fetch_coingecko(symbol: str, days: int):
    coin_id = COINGECKO_IDS.get(symbol.upper(), symbol.lower())
    headers = {'User-Agent': 'QuantAlpha/1.0'}
    try:
        resp = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}",
            timeout=15, headers=headers)
        if resp.status_code != 200: return None
        data = resp.json()
        if not data or not isinstance(data, list): return None
        df = pd.DataFrame(data, columns=['timestamp','Open','High','Low','Close'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('Date').drop('timestamp', axis=1).astype(float)
        df['Volume'] = 0.0
        return df.resample('D').last().dropna()
    except: return None

def load_csv(uploaded_file):
    """Supports MT5 (<DATE>,<TIME>), standard CSV, and common OHLC formats"""
    try:
        uploaded_file.seek(0)
        raw = uploaded_file.read()
        content = raw.decode('utf-8', errors='ignore')
        lines = content.strip().split('\n')
        first_line = lines[0].strip()
        
        # ─ MT5 DETECTION ──────────────────────────────────────
        is_mt5 = '<DATE>' in first_line.upper() or '<TIME>' in first_line.upper() or first_line.startswith('<')
        
        if is_mt5:
            st.info("📊 MT5 format detected")
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=None, engine='python', skipinitialspace=True)
            df.columns = [c.strip().strip('<>').upper() for c in df.columns]
            
            col_map = {'DATE':'Date','TIME':'Time','OPEN':'Open','HIGH':'High','LOW':'Low','CLOSE':'Close','VOL':'Volume','TICKVOL':'Volume','VOLUME':'Volume'}
            df = df.rename(columns=col_map)
            
            required = ['Date','Open','High','Low','Close']
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"MT5 CSV missing: {missing} | Found: {list(df.columns)}")
                return None
                
            if 'Date' in df.columns and 'Time' in df.columns:
                try:
                    df['Date'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), 
                                                format='%Y.%m.%d %H:%M:%S', errors='coerce')
                    if df['Date'].isna().any():
                        df['Date'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), 
                                                    format='%Y.%m.%d %H:%M', errors='coerce')
                except:
                    df['Date'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
                df = df.drop(columns=['Time'])
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                
            df = df.set_index('Date')
            df.index.name = 'Date'
            if 'Volume' not in df.columns: df['Volume'] = 0.0
            df = df[['Open','High','Low','Close','Volume']].apply(pd.to_numeric, errors='coerce').dropna().sort_index()
            if len(df) > 0: st.success(f"✅ MT5 loaded: {len(df):,} bars")
            return df

        # ── STANDARD CSV ──────────────────────────────────────
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip().title() for c in df.columns]
        
        date_col = next((c for c in ['Date','Datetime','Timestamp','Time','Open Time'] if c in df.columns), None)
        if not date_col:
            date_col = df.columns[0]
            try: pd.to_datetime(df[date_col].iloc[0])
            except: 
                st.error(f"Cannot find date column. Available: {list(df.columns)}")
                return None
                
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.set_index(date_col)
        df.index.name = 'Date'
        
        required = ['Open','High','Low','Close']
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"CSV missing: {missing} | Available: {list(df.columns)}")
            return None
            
        if 'Volume' not in df.columns: df['Volume'] = 0.0
        df = df[['Open','High','Low','Close','Volume']].astype(float).dropna().sort_index()
        return df
    except Exception as e:
        st.error(f"CSV error: {e}")
        return None

def fetch_data(symbol, period, uploaded_file=None):
    if uploaded_file is not None:
        uploaded_file.seek(0)
        df = load_csv(uploaded_file)
        if df is not None and len(df) > 30: return df, "📁 Your CSV"
    with st.spinner("📡 Trying Binance..."):
        df = fetch_binance(symbol, period)
    if df is not None and len(df) > 30: return df, " Binance"
    with st.spinner("📡 Trying CoinGecko..."):
        df = fetch_coingecko(symbol, PERIOD_DAYS.get(period, 90))
    if df is not None and len(df) > 30: return df, " CoinGecko"
    return None, None

# ─────────────────────────────────────────────────────────────
# INDICATOR LIBRARY (ESSENTIAL ONLY)
# ─────────────────────────────────────────────────────────────
def add_ema(df, period): df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean(); return df
def add_sma(df, period): df[f'SMA_{period}'] = df['Close'].rolling(period).mean(); return df
def add_rsi(df, period=14):
    delta = df['Close'].diff(); gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean(); rs = gain / loss.replace(0, np.nan)
    df[f'RSI_{period}'] = 100 - (100 / (1 + rs)); return df
def add_macd(df, fast=12, slow=26, signal=9):
    ef = df['Close'].ewm(span=fast, adjust=False).mean(); es = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ef - es; df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']; return df
def add_bollinger(df, period=20, std_dev=2.0):
    df['BB_Mid'] = df['Close'].rolling(period).mean(); std = df['Close'].rolling(period).std()
    df['BB_Upper'] = df['BB_Mid'] + std_dev * std; df['BB_Lower'] = df['BB_Mid'] - std_dev * std; return df
def add_atr(df, period=14):
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    df[f'ATR_{period}'] = tr.ewm(span=period, adjust=False).mean(); return df
def add_stochastic(df, k=14, d=3):
    low, high = df['Low'].rolling(k).min(), df['High'].rolling(k).max()
    df['Stoch_K'] = 100*(df['Close']-low)/(high-low); df['Stoch_D'] = df['Stoch_K'].rolling(d).mean(); return df

def crossover(a, b): return (a > b) & (a.shift(1) <= b.shift(1))
def crossunder(a, b): return (a < b) & (a.shift(1) >= b.shift(1))

# ────────────────────────────────────────────────────────────
# SMART INDICATOR LOADER (ONLY ADDS REQUESTED)
# ─────────────────────────────────────────────────────────────
def add_indicators(df, params, indicators_list=None):
    df = df.copy()
    indicators_list = indicators_list or []
    params = params or {}
    ind_lower = [i.lower() for i in indicators_list]
    
    if any('ema' in i for i in ind_lower) or not indicators_list:
        df = add_ema(df, params.get('ema_fast', 20))
        df = add_ema(df, params.get('ema_slow', 50))
    if any('sma' in i for i in ind_lower): df = add_sma(df, params.get('sma_period', 20))
    if any('rsi' in i for i in ind_lower): df = add_rsi(df, params.get('rsi_period', 14))
    if any('bb' in i or 'bollinger' in i for i in ind_lower): df = add_bollinger(df, params.get('bb_period', 20), params.get('bb_std', 2.0))
    if any('macd' in i for i in ind_lower): df = add_macd(df)
    if any('atr' in i for i in ind_lower): df = add_atr(df, params.get('atr_period', 14))
    if any('stoch' in i for i in ind_lower): df = add_stochastic(df)
    return df

# ─────────────────────────────────────────────────────────────
# GROQ PARSING & SIGNAL GEN
# ─────────────────────────────────────────────────────────────
def parse_strategy(client, description):
    prompt = f"""Parse this trading strategy into JSON. Return ONLY valid JSON.
Strategy: "{description}"
Format:
{{
  "entry_long": "condition or null",
  "entry_short": "condition or null",
  "indicators": ["list mentioned"],
  "strategy_type": "trend/mean-reversion/breakout",
  "sl_pct": 0.02, "tp_pct": 0.06,
  "indicator_params": {{"ema_fast":20,"ema_slow":50,"rsi_period":14}},
  "summary": "one sentence"
}}
Rules: null if not requested. Only list indicators actually mentioned."""
    try:
        resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role":"user","content":prompt}], max_tokens=500, temperature=0.1)
        text = resp.choices[0].message.content.strip()
        if '```json' in text: text = text.split('```json')[1].split('```')[0]
        elif '```' in text: text = text.split('```')[1].split('```')[0]
        res = json.loads(text)
        res['sl_pct'] = res.get('sl_pct') or 0.02; res['tp_pct'] = res.get('tp_pct') or 0.06
        res['indicators'] = res.get('indicators') or []; res['indicator_params'] = res.get('indicator_params') or {}
        return res
    except Exception as e:
        st.error(f"Parse error: {e}"); return None

def detect_features(desc):
    d = desc.lower()
    return {'partial': any(w in d for w in ['partial','scale']), 'trail': any(w in d for w in ['trail']),
            'both': any(w in d for w in ['short','sell when','both']), 'long': any(w in d for w in ['buy','long','bullish'])}

def gen_signal_block(client, desc, strat):
    f = detect_features(desc)
    dir_rule = "BOTH long AND short" if f['long'] and f['both'] else ("SHORT only" if f['both'] else "LONG only")
    tmpl = """df['long_signal']  = <LONG_COND>.fillna(False)
df['short_signal'] = <SHORT_COND>.fillna(False)
df['Signal']       = df['long_signal'].astype(int) - df['short_signal'].astype(int)"""
    if not f['long']: tmpl = """df['long_signal']  = pd.Series(False, index=df.index)
df['short_signal'] = <SHORT_COND>.fillna(False)
df['Signal']       = -df['short_signal'].astype(int)"""
    elif not f['both']: tmpl = """df['long_signal']  = <LONG_COND>.fillna(False)
df['short_signal'] = pd.Series(False, index=df.index)
df['Signal']       = df['long_signal'].astype(int)"""
    
    prompt = f"""Translate to Python signal code.
Strategy: "{desc}"
Direction: {dir_rule}
Indicators: {', '.join(strat.get('indicators',[]))}
Params: {strat.get('indicator_params',{})}

Available: add_ema, add_sma, add_rsi, add_macd, add_bollinger, add_atr, add_stochastic, crossover, crossunder
Format:
{tmpl}
Rules: Call add_* first. Assign both signals. df['Signal'] last. NO markdown. ONLY Python lines."""
    try:
        resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role":"user","content":prompt}], max_tokens=600, temperature=0.0)
        text = resp.choices[0].message.content.strip()
        if '```' in text: text = text.split('```')[-2] if text.count('```')>=2 else text.split('```')[1]
        lines = [l for l in text.splitlines() if l.strip() and not l.strip().startswith(('import','from'))]
        # Auto-repair
        if "df['Signal']" not in '\n'.join(lines):
            lines.append("df['Signal'] = df['long_signal'].astype(int) - df['short_signal'].astype(int)")
        if "long_signal" not in '\n'.join(lines): lines.insert(0, "df['long_signal'] = pd.Series(False, index=df.index)")
        if "short_signal" not in '\n'.join(lines): lines.insert(0, "df['short_signal'] = pd.Series(False, index=df.index)")
        return '\n'.join('    '+l.lstrip() for l in lines if l.strip())
    except Exception as e:
        return f"    df['long_signal'] = pd.Series(False, index=df.index)\n    df['short_signal'] = pd.Series(False, index=df.index)\n    df['Signal'] = 0\n    # Error: {e}"

# ─────────────────────────────────────────────────────────────
# CODE GENERATION (CLEAN STATIC LIBRARY)
# ─────────────────────────────────────────────────────────────
def generate_python_code(client, strat, symbol, desc=''):
    sym = BINANCE_SYMBOLS.get(symbol.upper(), 'BTCUSDT')
    sl, tp = strat.get('sl_pct',0.02), strat.get('tp_pct',0.06)
    sig = gen_signal_block(client, desc, strat)
    f = detect_features(desc)
    
    lib = '''import pandas as pd, numpy as np
def add_ema(df, p): df[f'EMA_{p}']=df['Close'].ewm(span=p,adjust=False).mean(); return df
def add_sma(df, p): df[f'SMA_{p}']=df['Close'].rolling(p).mean(); return df
def add_rsi(df, p=14): d=df['Close'].diff(); g=d.clip(lower=0).rolling(p).mean(); l=(-d.clip(upper=0)).rolling(p).mean(); df[f'RSI_{p}']=100-(100/(1+g/l.replace(0,np.nan))); return df
def add_macd(df, f=12, s=26, sg=9): ef=df['Close'].ewm(f,adjust=False).mean(); es=df['Close'].ewm(s,adjust=False).mean(); df['MACD']=ef-es; df['MACD_Signal']=df['MACD'].ewm(sg,adjust=False).mean(); df['MACD_Hist']=df['MACD']-df['MACD_Signal']; return df
def add_bollinger(df, p=20, sd=2.0): m=df['Close'].rolling(p).mean(); s=df['Close'].rolling(p).std(); df['BB_Mid']=m; df['BB_Upper']=m+sd*s; df['BB_Lower']=m-sd*s; return df
def add_atr(df, p=14): tr=pd.concat([df['High']-df['Low'],(df['High']-df['Close'].shift(1)).abs(),(df['Low']-df['Close'].shift(1)).abs()],axis=1).max(1); df[f'ATR_{p}']=tr.ewm(p,adjust=False).mean(); return df
def add_stochastic(df, k=14, d=3): lo,hi=df['Low'].rolling(k).min(),df['High'].rolling(k).max(); df['Stoch_K']=100*(df['Close']-lo)/(hi-lo); df['Stoch_D']=df['Stoch_K'].rolling(d).mean(); return df
def crossover(a,b): return (a>b)&(a.shift(1)<=b.shift(1))
def crossunder(a,b): return (a<b)&(a.shift(1)>=b.shift(1))'''

    code = f'''import requests, pandas as pd, numpy as np, plotly.graph_objects as go
from plotly.subplots import make_subplots

{lib}

def fetch():
    r = requests.get("https://api.binance.com/api/v3/klines", params={{"symbol":"{sym}","interval":"1d","limit":365}}).json()
    df = pd.DataFrame(r, columns=["t","O","H","L","C","V","x","y","z","a","b","c"])
    df["Date"] = pd.to_datetime(df["t"], unit="ms")
    return df.set_index("Date")[["O","H","L","C","V"]].astype(float).rename(columns={"O":"Open","H":"High","L":"Low","C":"Close","V":"Volume"})

def signals(df):
{sig}
    df["Position"] = df["Signal"].shift(1).fillna(0)
    return df

def backtest(df, sl={sl}, tp={tp}, trail=None, partial=None):
    df["Ret"] = df["Close"].pct_change()
    eq, bh = 1.0, 1.0; pos, entry = 0, None; th, tl = None, None; done = False
    res, bhr = [], []
    for i in range(len(df)):
        bh *= (1 + df["Ret"].iloc[i])
        np_ = int(df["Position"].iloc[i])
        if pos == 0 and np_ != 0:
            pos, entry = np_, df["Close"].iloc[i]; th=tl=entry; done=False; eq*=0.999
        elif pos != 0:
            p = df["Close"].iloc[i]
            if pos==1: th=max(th,p); sl_h=th*(1-trail) if trail else 0; hit= p<=entry*(1-sl) or p>=entry*(1+tp) or (trail and p<=sl_h)
            else: tl=min(tl,p); sl_l=tl*(1+trail) if trail else 0; hit= p>=entry*(1+sl) or p<=entry*(1-tp) or (trail and p>=sl_l)
            if partial and not done and (pos==1 and p>=entry*(1+tp) or pos==-1 and p<=entry*(1-tp)):
                eq *= (1 + (p/entry-1)*pos*partial - 0.001); done=True
            if hit or (not partial and (pos==1 and p>=entry*(1+tp) or pos==-1 and p<=entry*(1-tp))):
                rem = (1-partial) if done and partial else 1.0
                eq *= (1 + (p/entry-1)*pos*rem - 0.001); pos=entry=0
            else: eq *= (1 + df["Ret"].iloc[i]*pos)
        res.append(eq); bhr.append(bh)
    df["Eq"]=res; df["BH"]=bhr; df["StrRet"]=pd.Series(res).pct_change().fillna(0).values
    return df

def metrics(df):
    r=df["StrRet"].dropna(); sh=r.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0
    mx=((df["Eq"]-df["Eq"].cummax())/df["Eq"].cummax()).min(); wr=(r>0).mean()
    tr=df["Eq"].iloc[-1]-1; nt=int((df["Position"]!=df["Position"].shift(1)).sum()/2)
    return sh,mx,wr,tr,nt

def plot(df, sym="{symbol}", summary="{strat.get('summary','Strategy')}"):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.05,row_heights=[0.7,0.3])
    fig.add_trace(go.Candlestick(x=df.index,o=df["Open"],h=df["High"],l=df["Low"],c=df["Close"],name="Price",
        increasing_line_color="#26a69a",decreasing_line_color="#ef5350"),row=1,col=1)
    for c in ["EMA_20","EMA_50","RSI_14","BB_Upper","BB_Lower","MACD","ATR_14","Stoch_K"]:
        if c in df.columns and not df[c].isna().all(): fig.add_trace(go.Scatter(x=df.index,y=df[c],name=c),row=1,col=1)
    le=df[df["long_signal"]]; se=df[df["short_signal"]]
    if not le.empty: fig.add_trace(go.Scatter(x=le.index,y=le["Close"]*0.995,mode="markers",name="Long",marker=dict(symbol="triangle-up",size=12,color="#4ade80")),row=1,col=1)
    if not se.empty: fig.add_trace(go.Scatter(x=se.index,y=se["Close"]*1.005,mode="markers",name="Short",marker=dict(symbol="triangle-down",size=12,color="#f87171")),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["Eq"],name="Strategy",line=dict(color="#4ade80")),row=2,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["BH"],name="B&H",line=dict(color="#64748b",dash="dash")),row=2,col=1)
    fig.update_layout(height=650,paper_bgcolor="#080a0f",plot_bgcolor="#0d0f14",font=dict(color="#a89060"),
        title=f"<b>{sym}</b> — {{summary}}", xaxis_rangeslider_visible=False)
    return fig

if __name__=="__main__":
    df=fetch(); df=signals(df); df=backtest(df)
    sh,mx,wr,tr,nt=metrics(df)
    print(f"Sharpe:{{sh:.2f}} DD:{{mx:.1%}} WR:{{wr:.1%}} Ret:{{tr:.1%}} Trades:{{nt}}")
    plot(df).show()'''
    return code

# ─────────────────────────────────────────────────────────────
# APP SIGNAL RUNNER & CHART
# ─────────────────────────────────────────────────────────────
def run_signals_app(df, strat, client=None, desc=''):
    df = df.copy()
    df['long_signal'] = pd.Series(False, index=df.index)
    df['short_signal'] = pd.Series(False, index=df.index)
    df['Signal'] = pd.Series(0, index=df.index)
    
    if not client or not desc:
        p = strat.get('indicator_params', {})
        if 'EMA_20' in df.columns and 'EMA_50' in df.columns:
            df['long_signal'] = crossover(df['EMA_20'], df['EMA_50']).fillna(False)
        df['Signal'] = df['long_signal'].astype(int)
        return df
        
    sig = gen_signal_block(client, desc, strat)
    try:
        g = {'df':df,'pd':pd,'np':np,'add_ema':add_ema,'add_sma':add_sma,'add_rsi':add_rsi,
             'add_macd':add_macd,'add_bollinger':add_bollinger,'add_atr':add_atr,'add_stochastic':add_stochastic,
             'crossover':crossover,'crossunder':crossunder}
        exec('\n'.join(l[4:] if l.startswith('    ') else l for l in sig.splitlines()), g)
        df = g['df']
        df['long_signal'] = df.get('long_signal', pd.Series(False, index=df.index)).fillna(False).astype(bool)
        df['short_signal'] = df.get('short_signal', pd.Series(False, index=df.index)).fillna(False).astype(bool)
        df['Signal'] = df['long_signal'].astype(int) - df['short_signal'].astype(int)
    except Exception as e:
        st.warning(f"Signal error: {e}")
    return df

def draw_chart(df, strat, symbol, source, show='both'):
    dp = df.tail(80).copy()
    sl, tp = strat.get('sl_pct',0.02), strat.get('tp_pct',0.06)
    req = strat.get('indicators', [])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(x=dp.index,o=dp['Open'],h=dp['High'],l=dp['Low'],c=dp['Close'],name="Price",
        increasing_line_color="#26a69a",decreasing_line_color="#ef5350"),row=1,col=1)
    
    # Filter indicators
    ind_cols = [c for c in dp.columns if any(c.startswith(p) for p in ['EMA_','SMA_','BB_','RSI_','MACD','Stoch','ATR_'])]
    if req:
        rl = [r.lower() for r in req]
        ind_cols = [c for c in ind_cols if any(r in c.lower() for r in rl)]
    colors = ["#f59e0b","#60a5fa","#a78bfa","#34d399","#f472b6"]
    for i,c in enumerate(ind_cols[:5]):
        if c in dp.columns and not dp[c].isna().all():
            fig.add_trace(go.Scatter(x=dp.index,y=dp[c],name=c,line=dict(color=colors[i%5],width=1.5)),row=1,col=1)
            
    long_df = dp[dp['long_signal']] if show in ('long','both') else dp.iloc[:0]
    short_df = dp[dp['short_signal']] if show in ('short','both') else dp.iloc[:0]
    if not long_df.empty:
        fig.add_trace(go.Scatter(x=long_df.index,y=long_df['Close']*0.994,mode="markers",name="Long",marker=dict(symbol="triangle-up",size=12,color="#4ade80")),row=1,col=1)
    if not short_df.empty:
        fig.add_trace(go.Scatter(x=short_df.index,y=short_df['Close']*1.006,mode="markers",name="Short",marker=dict(symbol="triangle-down",size=12,color="#f87171")),row=1,col=1)
        
    fig.add_trace(go.Bar(x=dp.index,y=dp['Volume'],name="Volume",marker_color=["#26a69a" if c>=o else "#ef5350" for c,o in zip(dp['Close'],dp['Open'])],opacity=0.6),row=2,col=1)
    
    fig.update_layout(height=620,paper_bgcolor="#080a0f",plot_bgcolor="#0d0f14",font=dict(family="IBM Plex Mono",color="#a89060"),
        title=f"<b>{symbol}</b> — {strat.get('summary','Strategy')} | 🔺{len(long_df)} 🔻{len(short_df)} | {source}",
        xaxis_rangeslider_visible=False, margin=dict(l=50,r=80,t=60,b=40))
    return fig

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-header"><h1>📈 STRATEGY VISUALIZER</h1><p>Describe → Visualize → Export Python</p><p style="color:#3d2f00;font-size:0.7rem">QUANT ALPHA · GROQ + BINANCE · INTERACTIVE</p></div>', unsafe_allow_html=True)
client = init_llm()
if not client: st.error("⚠️ Set GROQ_API_KEY in Streamlit Secrets"); st.stop()

with st.sidebar:
    symbol = st.selectbox("Asset", list(BINANCE_SYMBOLS.keys()), index=0)
    period = st.selectbox("Period", list(PERIOD_DAYS.keys()), index=1)
    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV (MT5/Standard)", type=['csv'])
    if uploaded: st.markdown('<div class="data-source-box">✅ CSV Ready</div>', unsafe_allow_html=True)

for k in ['parsed','df','fig_l','fig_s','code','source','desc']:
    if k not in st.session_state: st.session_state[k] = None

st.markdown('<div class="section-hdr">STEP 1 — DESCRIBE</div>', unsafe_allow_html=True)
desc = st.text_area("Strategy", placeholder="Buy BTC when 20 EMA crosses above 50 EMA. SL 2%, TP 6%.", height=90)
c1,c2 = st.columns([3,1])
with c1: parse = st.button(" PARSE", use_container_width=True)
with c2: reset = st.button("↺ Reset", use_container_width=True)
if reset:
    for k in ['parsed','df','fig_l','fig_s','code','source','desc']: st.session_state[k] = None
    st.rerun()
if parse and desc.strip():
    with st.spinner("🧠 Parsing..."): st.session_state.parsed = parse_strategy(client, desc)
    if st.session_state.parsed:
        st.session_state.desc = desc; st.session_state.fig_l = st.session_state.fig_s = st.session_state.code = None

if st.session_state.parsed:
    p = st.session_state.parsed
    st.markdown('<div class="section-hdr">STEP 2 — CONFIRM</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="parsed-box"><b>AI PARSED:</b><br>
    Summary: {p.get('summary')}<br>Type: {p.get('strategy_type','trend').upper()}<br>
    Indicators: {', '.join(p.get('indicators',[]))}<br>
    SL: {(p.get('sl_pct') or 0.01)*100:.1f}% | TP: {(p.get('tp_pct') or 0.02)*100:.1f}%</div>""", unsafe_allow_html=True)
    if st.button("📊 VISUALIZE", use_container_width=True):
        with st.spinner("📡 Fetching..."): df, src = fetch_data(symbol, period, uploaded)
        if df is not None and len(df)>30:
            with st.spinner(" Charting..."):
                df = add_indicators(df, p.get('indicator_params',{}), p.get('indicators',[]))
                df = run_signals_app(df, p, client, st.session_state.desc)
                st.session_state.df, st.session_state.source = df, src
                st.session_state.fig_l = draw_chart(df, p, symbol, src, 'long')
                st.session_state.fig_s = draw_chart(df, p, symbol, src, 'short')
        else: st.error("Data fetch failed. Try CSV upload.")

if st.session_state.fig_l or st.session_state.fig_s:
    st.markdown('<div class="section-hdr">STEP 3 — CHARTS</div>', unsafe_allow_html=True)
    l,r = st.columns(2)
    with l: st.markdown("📈 LONG"); st.plotly_chart(st.session_state.fig_l, use_container_width=True)
    with r: 
        st.markdown("📉 SHORT")
        n_s = int(st.session_state.df['short_signal'].sum()) if st.session_state.df is not None else 0
        if n_s > 0: st.plotly_chart(st.session_state.fig_s, use_container_width=True)
        else: st.info("No short signals (add 'short when...' to enable)")
        
    y,n = st.columns(2)
    with y: 
        if st.button("✅ GENERATE CODE", use_container_width=True):
            with st.spinner("⚙️ Building..."): st.session_state.code = generate_python_code(client, st.session_state.parsed, symbol, st.session_state.desc)
    with n: 
        if st.button("❌ REDESCRIBE", use_container_width=True): st.session_state.fig_l = st.session_state.fig_s = st.session_state.code = None

if st.session_state.code:
    st.markdown('<div class="section-hdr">STEP 4 — EXPORT</div>', unsafe_allow_html=True)
    st.text_area("Python Code", value=st.session_state.code, height=350)
    st.download_button("️ Download .py", data=st.session_state.code, file_name=f"{symbol}_strategy.py", mime="text/plain", use_container_width=True)

st.markdown('<div style="text-align:center;margin-top:40px;color:#1e2030;font-size:0.7rem">QUANT ALPHA — NOT FINANCIAL ADVICE</div>', unsafe_allow_html=True)
