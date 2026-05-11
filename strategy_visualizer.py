import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Indicator Library ─────────────────────────────────────────
                if '# ═══' in line and 'QUANT ALPHA INDICATOR LIBRARY' in line:
                    in_lib = True
                if in_lib:
                    lib_lines.append(line)
                if in_lib and 'LIBRARY_REFERENCE' in line and '"""' in line and len(lib_lines) > 5:
                    break
        lib_code = ''.join(lib_lines[:400])  # safety limit
    except Exception:
        lib_code = "# indicator library not found"

    # ── Trailing / partial close config ───────────────────────
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


# ── Chart ─────────────────────────────────────────────────────
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
# APP SIGNAL RUNNER
# Executes the same Groq-generated signal block that goes into
# the downloaded code — chart shows exactly what client gets
# ─────────────────────────────────────────────────────────────
def generate_signals(df, strategy, client=None, description=''):
    """
    Run Groq signal block on the dataframe for the app chart.
    Same code as the downloaded backtest — what you see = what you get.
    """
    df = df.copy()

    # Initialize safe defaults
    df['long_signal']  = pd.Series(False, index=df.index)
    df['short_signal'] = pd.Series(False, index=df.index)
    df['Signal']       = pd.Series(0, index=df.index)

    if not client or not description:
        # Fallback: basic signal from parsed strategy
        p     = strategy.get('indicator_params', {}) or {}
        stype = strategy.get('strategy_type', 'trend')
        wants_long  = strategy.get('entry_long')  is not None
        wants_short = strategy.get('entry_short') is not None

        if 'EMA_fast' in df.columns and 'EMA_slow' in df.columns:
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
        df['Signal'] = df['long_signal'].astype(int) - df['short_signal'].astype(int)
        return df

    # Get Groq signal block
    signal_block = generate_signal_block(client, description, strategy)

    # Execute signal block in dataframe context
    # Library functions are in global scope — available here
    try:
        exec_globals = {
            'df': df, 'pd': pd, 'np': np,
            # inject all library functions
            'add_ema': add_ema, 'add_sma': add_sma,
            'add_rsi': add_rsi, 'add_macd': add_macd,
            'add_bollinger': add_bollinger, 'add_atr': add_atr,
            'add_stochastic': add_stochastic, 'add_vwap': add_vwap,
            'add_obv': add_obv, 'add_volume_spike': add_volume_spike,
            'add_volume_sma': add_volume_sma, 'add_wma': add_wma,
            'add_swing_highs_lows': add_swing_highs_lows,
            'add_candle_patterns': add_candle_patterns,
            'add_structure_break': add_structure_break,
            'add_fair_value_gaps': add_fair_value_gaps,
            'add_liquidity_levels': add_liquidity_levels,
            'add_order_blocks': add_order_blocks,
            'add_premium_discount': add_premium_discount,
            'add_market_structure': add_market_structure,
            'add_optimal_trade_entry': add_optimal_trade_entry,
            'add_equal_highs_lows': add_equal_highs_lows,
            'add_higher_highs_lower_lows': add_higher_highs_lower_lows,
            'add_support_resistance': add_support_resistance,
            'add_previous_day_levels': add_previous_day_levels,
            'add_supertrend': add_supertrend, 'add_cci': add_cci,
            'add_williams_r': add_williams_r, 'add_mfi': add_mfi,
            'add_donchian': add_donchian, 'add_keltner': add_keltner,
            'add_inside_outside_bars': add_inside_outside_bars,
            'add_common_indicators': add_common_indicators,
            'add_smc_indicators': add_smc_indicators,
            'crossover': crossover, 'crossunder': crossunder,
            'above_level': above_level, 'below_level': below_level,
            'rising': rising, 'falling': falling,
        }
        # Remove 4-space indent from signal block for exec
        clean_block = '\n'.join(
            line[4:] if line.startswith('    ') else line
            for line in signal_block.splitlines()
        )
        exec(clean_block, exec_globals)
        df = exec_globals['df']

        # Ensure proper types
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

    # ── Auto-detect indicator columns — draw whatever Groq calculated ──
    ind_prefixes = ['EMA_','SMA_','BB_','RSI_','MACD','Stoch','ATR_','WMA_','VWAP',
                    'KC_','DC_','Supertrend']
    ind_cols = [c for c in df_plot.columns
                if any(c.startswith(p) for p in ind_prefixes)
                and c not in ['BB_Pct','BB_Width']]
    colors_ind = ['#f59e0b','#60a5fa','#a78bfa','#34d399','#f472b6','#fb923c']
    for i, col in enumerate(ind_cols[:6]):
        if col in df_plot.columns and not df_plot[col].isna().all():
            # BB bands get fill
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

    # ── RSI on bottom panel if present ───────────────────────────
    rsi_cols = [c for c in df_plot.columns if c.startswith('RSI_')]
    has_rsi  = len(rsi_cols) > 0

    # ── Long signals (shown in long chart and both chart) ────────
    long_df = df_plot[df_plot['long_signal']] if show in ('long', 'both') else df_plot.iloc[0:0]
    if not long_df.empty:
        fig.add_trace(go.Scatter(
            x=long_df.index, y=long_df['Close'] * 0.994,
            mode='markers', name='Long Entry',
            marker=dict(symbol='triangle-up', size=14, color='#4ade80',
                       line=dict(color='#166534', width=1))


# ── Fetch OHLCV from Binance ──────────────────────────────────
def fetch_data():
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1d", "limit": 365}
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
        print(f"Data error: {e}")
        return None


# ── Generate signals (AI-written, library-backed) ─────────────
def generate_signals(df):
    df = add_ema(df, 20)
    df = add_ema(df, 50)
    df = add_rsi(df, 14)
    long_condition = below_level(df['RSI_14'], 30)
    short_condition = above_level(df['RSI_14'], 70)
    df['long_signal']  = long_condition.fillna(False)
    df['short_signal'] = short_condition.fillna(False)
    df['Signal']       = df['long_signal'].astype(int) - df['short_signal'].astype(int)
    # Shift by 1 bar — prevents lookahead bias
    df["Position"] = df["Signal"].shift(1).fillna(0)
    return df


# ── Backtest ──────────────────────────────────────────────────
def backtest(df, sl_pct=0.03, tp_pct=0.09,
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


# ── Chart ─────────────────────────────────────────────────────
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
        fig.add_shape(type="line",x0=t0,x1=t1,y0=e*(1-0.03),y1=e*(1-0.03),
            line=dict(color="#ef4444",width=1,dash="dash"),row=1,col=1)
        fig.add_shape(type="line",x0=t0,x1=t1,y0=e*(1+0.09),y1=e*(1+0.09),
            line=dict(color="#4ade80",width=1,dash="dot"),row=1,col=1)
    for idx in df[df["short_signal"]].index:
        e = float(df.loc[idx,"Close"])
        t0 = df.loc[idx,"Open time"]
        t1 = df.iloc[min(df.index.get_loc(idx)+8, len(df)-1)]["Open time"]
        fig.add_shape(type="line",x0=t0,x1=t1,y0=e*(1+0.03),y1=e*(1+0.03),
            line=dict(color="#ef4444",width=1,dash="dash"),row=1,col=1)
        fig.add_shape(type="line",x0=t0,x1=t1,y0=e*(1-0.09),y1=e*(1-0.09),
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
        title=dict(text=f"<b>BTC</b> — Mean-reversion strategy based on RSI oversold and overbought conditions",
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
    df = backtest(df, trail_pct=None, partial_close_pct=None)
    sharpe, max_dd, win_rate, total_r, n_trades = metrics(df)
    print("=" * 50)
    print(f"  Mean-reversion strategy based on RSI oversold and overbought conditions")
    print("=" * 50)
    print(f"  Sharpe Ratio : {sharpe:.2f}")
    print(f"  Max Drawdown : {max_dd:.1%}")
    print(f"  Win Rate     : {win_rate:.1%}")
    print(f"  Total Return : {total_r:.1%}")
    print(f"  Trades       : {n_trades}")
    print("=" * 50)
    fig = plot_results(df)
    fig.show()

if __name__ == "__main__":
    main()
