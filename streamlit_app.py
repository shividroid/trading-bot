import math, requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(page_title="ETH Trading Bot", page_icon="📈", layout="wide")

ATR_PERIOD=21; ST_FACTOR=0.75

TIMEFRAMES = {
    "1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m",
    "45m":"45m","1h":"1h","2h":"2h","4h":"4h","6h":"6h","1D":"1d","1W":"1w"
}
# Binance interval codes
TF_MAP = {
    "1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m",
    "45m":"45m","1h":"1h","2h":"2h","4h":"4h","6h":"6h","1D":"1d","1W":"1w"
}

# Delta India resolution codes
DELTA_RES = {
    "1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m",
    "1h":"1h","2h":"2h","4h":"4h","6h":"6h","1D":"1d","1W":"1w"
}

@st.cache_data(ttl=120)
def fetch_candles(interval="1h", limit=550):
    """
    Fetch ETHUSD candles from Delta India (exact same data as TradingView chart).
    Falls back to Binance if Delta fails.
    """
    import time

    # ── Delta India (primary — exact match with TradingView) ──────────────
    # 45m not available on Delta either — resample from 15m
    if interval == "45m":
        df = _fetch_delta("15m", limit * 3)
        if df is not None:
            df = df.resample("45min").agg({
                "open":"first","high":"max","low":"min","close":"last","volume":"sum"
            }).dropna().iloc[-limit:]
            return df
    else:
        res = DELTA_RES.get(interval, "1h")
        df  = _fetch_delta(res, limit)
        if df is not None:
            return df

    # ── Binance fallback (if Delta unreachable) ───────────────────────────
    st.warning("⚠️ Delta India API unavailable — using Binance data (slight price difference)")
    binance_res = TF_MAP.get(interval, "1h")
    if interval == "45m":
        for url in ["https://api.binance.us/api/v3/klines","https://api.binance.com/api/v3/klines"]:
            try:
                r=requests.get(url,params={"symbol":"ETHUSDT","interval":"15m","limit":limit*3},timeout=15)
                data=r.json()
                if isinstance(data,list) and len(data)>50:
                    df=_parse_binance(data)
                    return df.resample("45min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna().iloc[-limit:]
            except: pass
    else:
        for url in ["https://api.binance.us/api/v3/klines","https://api.binance.com/api/v3/klines"]:
            try:
                r=requests.get(url,params={"symbol":"ETHUSDT","interval":binance_res,"limit":limit},timeout=15)
                data=r.json()
                if isinstance(data,list) and len(data)>50:
                    return _parse_binance(data)
            except: pass
    return None

def _fetch_delta(resolution, limit):
    """Fetch candles from Delta India public API — no auth needed."""
    import time
    try:
        # Delta uses Unix timestamps for start/end
        # Resolution in seconds for calculating start time
        res_seconds = {
            "1m":60,"3m":180,"5m":300,"15m":900,"30m":1800,
            "1h":3600,"2h":7200,"4h":14400,"6h":21600,"1d":86400,"1w":604800
        }
        secs    = res_seconds.get(resolution, 3600)
        end     = int(time.time())
        start   = end - secs * (limit + 5)  # fetch a bit extra
        url     = "https://api.india.delta.exchange/v2/history/candles"
        params  = {"symbol": "ETHUSD", "resolution": resolution,
                   "start": start, "end": end}
        r       = requests.get(url, params=params, timeout=15)
        resp    = r.json()
        if resp.get("success") and resp.get("result"):
            candles = resp["result"]
            df = pd.DataFrame(candles)
            # Delta returns: time, open, high, low, close, volume
            df["open_time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.set_index("open_time")
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.sort_index()
            df = df.iloc[:-1]   # drop incomplete current candle
            return df.iloc[-limit:]
    except Exception as e:
        print(f"Delta fetch failed: {e}")
    return None

def _parse_binance(data):
    df=pd.DataFrame(data,columns=["open_time","open","high","low","close","volume","close_time","qav","num_trades","tbbav","tbqav","ignore"])
    df["open_time"]=pd.to_datetime(df["open_time"],unit="ms",utc=True)
    df=df.set_index("open_time")
    for c in ["open","high","low","close","volume"]: df[c]=df[c].astype(float)
    return df.iloc[:-1]

def to_heikin_ashi(df):
    ha=df.copy()
    ha["close"]=(df["open"]+df["high"]+df["low"]+df["close"])/4
    ho=[(df["open"].iloc[0]+df["close"].iloc[0])/2]
    for i in range(1,len(df)): ho.append((ho[i-1]+ha["close"].iloc[i-1])/2)
    ha["open"]=ho; ha["high"]=df[["high","open","close"]].max(axis=1)
    ha["low"]=df[["low","open","close"]].min(axis=1)
    return ha

def compute_supertrend(df, period=21, mult=0.75):
    hi,lo,cl=df["high"],df["low"],df["close"]; n=len(df)
    tr=pd.concat([hi-lo,(hi-cl.shift(1)).abs(),(lo-cl.shift(1)).abs()],axis=1).max(axis=1)
    atr=tr.copy().astype(float); atr.iloc[:period]=np.nan
    atr.iloc[period]=tr.iloc[1:period+1].mean()
    for i in range(period+1,n): atr.iloc[i]=(atr.iloc[i-1]*(period-1)+tr.iloc[i])/period
    hl2=(hi+lo)/2; upper=hl2+mult*atr; lower=hl2-mult*atr
    st_=pd.Series(np.nan,index=df.index); trend=pd.Series(0,index=df.index)
    for i in range(1,n):
        if np.isnan(atr.iloc[i]): continue
        fu=upper.iloc[i] if (upper.iloc[i]<upper.iloc[i-1] or cl.iloc[i-1]>upper.iloc[i-1]) else upper.iloc[i-1]
        fl=lower.iloc[i] if (lower.iloc[i]>lower.iloc[i-1] or cl.iloc[i-1]<lower.iloc[i-1]) else lower.iloc[i-1]
        upper.iloc[i]=fu; lower.iloc[i]=fl
        if st_.iloc[i-1]==upper.iloc[i-1]: st_.iloc[i],trend.iloc[i]=(fu,1) if cl.iloc[i]<=fu else (fl,-1)
        else: st_.iloc[i],trend.iloc[i]=(fl,-1) if cl.iloc[i]>=fl else (fu,1)
    return st_,trend

@st.cache_data(ttl=120)
def compute_nw(close_tuple, bw=8, mult=3.0, window=500):
    """
    Exact LuxAlgo NW Envelope:
    - src = close
    - y2[i] = kernel weighted avg (Gaussian) over window
    - y1[i] = (y2[i] + y2[i-1]) / 2  (smooth)
    - band  = mult * MAD(src, y1) over window
    """
    src = np.array(close_tuple, dtype=float)
    n   = len(src)
    ws  = max(0, n - window)

    y2 = np.full(n, np.nan)
    for i in range(ws, n):
        w     = np.array([math.exp(-0.5*((i-j)/bw)**2) for j in range(ws, n)])
        y2[i] = np.dot(w, src[ws:n]) / w.sum()

    y1 = np.full(n, np.nan)
    y1[ws] = y2[ws]
    for i in range(ws+1, n):
        y1[i] = (y2[i] + y2[i-1]) / 2

    valid = ~np.isnan(y1[ws:])
    mae   = mult * np.mean(np.abs(src[ws:][valid] - y1[ws:][valid]))

    return (pd.Series(y1+mae, dtype=float),
            pd.Series(y1-mae, dtype=float),
            pd.Series(y1,     dtype=float))

def compute_bb_expansion(close):
    bw=(4*2.0*close.rolling(20).std(ddof=0))/close.rolling(20).mean()*100.0
    return bw>bw.rolling(50).mean()*1.1

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("📈 ETH SuperTrend Bot — Live Dashboard")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Chart Settings")

    selected_tf = st.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=6)  # default 1h
    bars_to_show= st.slider("Bars to show", 50, 300, 100)
    show_ha     = st.toggle("Heikin Ashi candles",  value=False)
    show_nw     = st.toggle("NW Envelope",          value=True)
    show_triangles = st.toggle("NW Cross Triangles",value=True)
    show_st     = st.toggle("SuperTrend line",      value=True)

    st.divider()
    st.subheader("🔧 Indicator Settings")
    st_period = st.slider("ST ATR Period", 5,  50,  21)
    st_mult   = st.slider("ST Factor",     0.5, 5.0, 0.75, 0.05)
    nw_bw     = st.slider("NW Bandwidth",  2,   20,  8)
    nw_mult   = st.slider("NW Multiplier", 1.0, 5.0, 3.0, 0.5)

    st.divider()
    st.subheader("🔬 LuxAlgo Comparison")
    st.caption("Paste latest bar values from TradingView")
    lux_u = st.number_input("LuxAlgo NW Upper", value=0.0, format="%.2f")
    lux_l = st.number_input("LuxAlgo NW Lower", value=0.0, format="%.2f")

    st.divider()
    st.subheader("🔔 Test Telegram")
    tg_token = st.text_input("Bot Token", type="password", placeholder="123:AAF...")
    tg_chat  = st.text_input("Chat ID", value="1130112754")
    if st.button("📤 Send Test"):
        if tg_token:
            try:
                r=requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage",
                    data={"chat_id":tg_chat,"text":f"✅ ETH Bot Test\nTF: {selected_tf}\nTime: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"},timeout=10)
                if r.status_code==200: st.success("✅ Sent! Check Telegram.")
                else: st.error(f"❌ {r.json().get('description','Failed')}")
            except Exception as e: st.error(str(e))
        else: st.warning("Enter token first")

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear(); st.rerun()

    st.divider()
    st.caption(f"⏱ Bot runs on: **1h** timeframe\n\nStreamlit free tier: **1 GB RAM** · App sleeps after ~7 days inactivity but wakes on next visit instantly.\n\nGitHub Actions free: **2000 min/month** · At 24 runs/day = 720 min/month — well within limit.")

# ── Load data ─────────────────────────────────────────────────────────────────
# Convert to IST for display (UTC+5:30)
now_ist = datetime.now(timezone.utc) + pd.Timedelta(hours=5, minutes=30)
st.caption(f"📊 Timeframe: **{selected_tf}** · ETHUSD · Delta India (exact TradingView data) · Cached 2 min · {now_ist.strftime('%Y-%m-%d %H:%M IST')}")

with st.spinner(f"Fetching {selected_tf} candles..."):
    df = fetch_candles(selected_tf, 550)

if df is None:
    st.error("❌ Could not fetch candle data"); st.stop()

ha = to_heikin_ashi(df)

# Convert timestamps to IST + shift by candle duration to match TradingView
# TradingView labels candles by CLOSE time; APIs return OPEN time
# e.g. 09:00 UTC open = 14:30 IST open = 15:30 IST close (TradingView shows 15:30)
TF_DURATION = {
    "1m": pd.Timedelta(minutes=1),   "3m": pd.Timedelta(minutes=3),
    "5m": pd.Timedelta(minutes=5),   "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30), "45m": pd.Timedelta(minutes=45),
    "1h": pd.Timedelta(hours=1),     "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4),     "6h": pd.Timedelta(hours=6),
    "1D": pd.Timedelta(days=1),      "1W": pd.Timedelta(weeks=1),
}
ist_offset   = pd.Timedelta(hours=5, minutes=30)
candle_dur   = TF_DURATION.get(selected_tf, pd.Timedelta(hours=1))
display_shift = ist_offset + candle_dur   # IST + candle duration = close time in IST

df_display = df.copy()
df_display.index = df_display.index + display_shift
ha_display = ha.copy()
ha_display.index = ha_display.index + display_shift
hi, lo, cl = df["high"], df["low"], df["close"]
tr  = pd.concat([hi-lo,(hi-cl.shift(1)).abs(),(lo-cl.shift(1)).abs()],axis=1).max(axis=1)
atr = tr.ewm(span=st_period, adjust=False).mean()

st_line, st_dir = compute_supertrend(df, st_period, st_mult)
st_line.index = df.index + display_shift
st_dir.index  = df.index + display_shift
bb_exp = compute_bb_expansion(cl)
bb_exp.index  = df.index + display_shift

nw_upper_s, nw_lower_s, nw_mid_s = compute_nw(
    tuple(df["close"].tolist()), bw=nw_bw, mult=nw_mult)

nw_upper = pd.Series(nw_upper_s.values, index=df.index + display_shift)
nw_lower = pd.Series(nw_lower_s.values, index=df.index + display_shift)
nw_mid   = pd.Series(nw_mid_s.values,   index=df.index + display_shift)

# Triangles — use IST-indexed close
close_s  = pd.Series(df["close"].values, index=df.index + display_shift)
cross_up = (close_s > nw_upper) & (close_s.shift(1) <= nw_upper.shift(1))
cross_dn = (close_s < nw_lower) & (close_s.shift(1) >= nw_lower.shift(1))

# ── Metrics ───────────────────────────────────────────────────────────────────
cc      = df["close"].iloc[-1]; prev=df["close"].iloc[-2]
chg     = cc-prev; chgp=chg/prev*100
cur_nwu = nw_upper.iloc[-1]; cur_nwl=nw_lower.iloc[-1]
is_bull = st_dir.iloc[-1]==-1; flipped=st_dir.iloc[-1]!=st_dir.iloc[-2]
cur_st  = st_line.iloc[-1]

c1,c2,c3,c4,c5,c6=st.columns(6)
c1.metric("ETH Price",   f"${cc:,.2f}", f"{chg:+.2f} ({chgp:+.2f}%)")
c2.metric("ST Direction","🟢 BULL" if is_bull else "🔴 BEAR",
          f"ST line: {cur_st:.2f}" if not np.isnan(cur_st) else "")
c3.metric("ST Flipped",  "YES ⚡" if flipped else "No")
c4.metric("NW Upper",    f"${cur_nwu:,.2f}")
c5.metric("NW Lower",    f"${cur_nwl:,.2f}")
c6.metric("BB Expansion","✅ YES" if bb_exp.iloc[-1] else "❌ No")

if lux_u > 0 and lux_l > 0:
    du=abs(lux_u-cur_nwu); dl=abs(lux_l-cur_nwl)
    if du<20 and dl<20:
        st.success(f"✅ Good match — Upper diff: ${du:.2f} | Lower diff: ${dl:.2f} (small diff = exchange price difference, normal)")
    else:
        st.warning(f"⚠️ Upper diff: ${du:.2f} | Lower diff: ${dl:.2f} — Check LuxAlgo: src=close, Window=500, BW={nw_bw}, Mult={nw_mult}, Repaint=ON")

# ── Chart ─────────────────────────────────────────────────────────────────────
sl  = bars_to_show
src = ha_display.iloc[-sl:] if show_ha else df_display.iloc[-sl:]
fig = go.Figure()

# Candlesticks
fig.add_trace(go.Candlestick(
    x=src.index, open=src["open"], high=src["high"],
    low=src["low"], close=src["close"],
    increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    name="HA" if show_ha else "ETH", line_width=1))

# NW Envelope
if show_nw:
    fig.add_trace(go.Scatter(
        x=nw_upper.iloc[-sl:].index, y=nw_upper.iloc[-sl:].values,
        line=dict(color="rgba(38,166,154,1.0)", width=2),
        name=f"NW Upper (BW={nw_bw}, Mult={nw_mult})"))
    fig.add_trace(go.Scatter(
        x=nw_lower.iloc[-sl:].index, y=nw_lower.iloc[-sl:].values,
        line=dict(color="rgba(239,83,80,1.0)", width=2),
        fill="tonexty", fillcolor="rgba(100,100,100,0.06)",
        name="NW Lower"))
    fig.add_trace(go.Scatter(
        x=nw_mid.iloc[-sl:].index, y=nw_mid.iloc[-sl:].values,
        line=dict(color="rgba(41,98,255,0.5)", width=1, dash="dot"),
        name="NW Mid"))

# Triangles
if show_triangles:
    spread = (cur_nwu - cur_nwl) * 0.015
    cu = cross_up.iloc[-sl:]
    cd = cross_dn.iloc[-sl:]
    if cu.any():
        fig.add_trace(go.Scatter(
            x=nw_upper.iloc[-sl:][cu].index,
            y=(nw_upper.iloc[-sl:][cu] + spread).values,
            mode="markers",
            marker=dict(symbol="triangle-down", color="#ef5350", size=14),
            name="▼ Cross Upper (Sell)"))
    if cd.any():
        fig.add_trace(go.Scatter(
            x=nw_lower.iloc[-sl:][cd].index,
            y=(nw_lower.iloc[-sl:][cd] - spread).values,
            mode="markers",
            marker=dict(symbol="triangle-up", color="#26a69a", size=14),
            name="▲ Cross Lower (Buy)"))

# SuperTrend — coloured markers as thick line (works reliably)
if show_st:
    stv = st_line.iloc[-sl:]
    sdv = st_dir.iloc[-sl:]
    bull_mask = sdv == -1
    bear_mask = sdv ==  1
    # Use scatter with lines+markers for solid coloured ST line
    if bull_mask.any():
        fig.add_trace(go.Scatter(
            x=stv[bull_mask].index, y=stv[bull_mask].values,
            mode="lines+markers",
            line=dict(color="#26a69a", width=2),
            marker=dict(color="#26a69a", size=3),
            name=f"ST(ATR={st_period}, F={st_mult}) Bull"))
    if bear_mask.any():
        fig.add_trace(go.Scatter(
            x=stv[bear_mask].index, y=stv[bear_mask].values,
            mode="lines+markers",
            line=dict(color="#ef5350", width=2),
            marker=dict(color="#ef5350", size=3),
            name="ST Bear"))

fig.update_layout(
    height=580, template="plotly_dark", xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", y=1.02, font=dict(size=10)),
    margin=dict(l=0,r=0,t=40,b=0),
    paper_bgcolor="#131722", plot_bgcolor="#131722",
    xaxis=dict(gridcolor="#2a2e39"),
    yaxis=dict(gridcolor="#2a2e39"),
    title=dict(text=f"ETHUSD.P · {selected_tf} · SuperTrend({st_period},{st_mult}) · NW(BW={nw_bw}, Mult={nw_mult}) · Delta India",
               font=dict(size=12, color="#787b86"), x=0))
st.plotly_chart(fig, use_container_width=True)

# ── Table ─────────────────────────────────────────────────────────────────────
st.subheader("🔢 Last 15 Bars")
st.caption(f"LuxAlgo settings to match → src=close, Window=500, BW={nw_bw}, Mult={nw_mult}, Repaint=ON · Using Delta India data so values should match exactly")
t = df_display.tail(15).copy()
t["NW Upper"] = nw_upper.tail(15).values
t["NW Lower"] = nw_lower.tail(15).values
t["NW Mid"]   = nw_mid.tail(15).values
t["ST Line"]  = st_line.tail(15).values
t["ST"]       = st_dir.tail(15).map({-1:"🟢 BULL", 1:"🔴 BEAR", 0:"-"}).values
t["NW Upper"] = t["NW Upper"].round(2)
t["NW Lower"] = t["NW Lower"].round(2)
t["NW Mid"]   = t["NW Mid"].round(2)
t["ST Line"]  = t["ST Line"].round(2)
t["Signal"]   = ""
cup = cross_up.tail(15); cdn = cross_dn.tail(15)
t.loc[cup[cup].index, "Signal"] = "▼ Sell"
t.loc[cdn[cdn].index, "Signal"] = "▲ Buy"
t.index = (t.index + display_shift).strftime("%m-%d %H:%M IST")
st.dataframe(
    t[["close","NW Upper","NW Lower","NW Mid","ST Line","ST","Signal"]].rename(columns={"close":"Close"}),
    use_container_width=True)

st.caption("⚠️ Telegram token entered above is never stored — only used for that single live test request.")
