import math, requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(page_title="ETH Trading Bot", page_icon="📈", layout="wide")
BANDWIDTH=8; ATR_PERIOD=21; ST_FACTOR=0.75; NW_MULT=3.0

@st.cache_data(ttl=300)
def fetch_candles(limit=550):
    for url in ["https://api.binance.us/api/v3/klines","https://api.binance.com/api/v3/klines"]:
        try:
            r=requests.get(url,params={"symbol":"ETHUSDT","interval":"1h","limit":limit},timeout=15)
            data=r.json()
            if isinstance(data,list) and len(data)>50:
                df=pd.DataFrame(data,columns=["open_time","open","high","low","close","volume","close_time","qav","num_trades","tbbav","tbqav","ignore"])
                df["open_time"]=pd.to_datetime(df["open_time"],unit="ms",utc=True)
                df=df.set_index("open_time")
                for c in ["open","high","low","close","volume"]: df[c]=df[c].astype(float)
                return df.iloc[:-1]
        except: pass
    return None

def to_heikin_ashi(df):
    ha=df.copy()
    ha["close"]=(df["open"]+df["high"]+df["low"]+df["close"])/4
    ho=[(df["open"].iloc[0]+df["close"].iloc[0])/2]
    for i in range(1,len(df)): ho.append((ho[i-1]+ha["close"].iloc[i-1])/2)
    ha["open"]=ho; ha["high"]=df[["high","open","close"]].max(axis=1)
    ha["low"]=df[["low","open","close"]].min(axis=1)
    return ha

def compute_supertrend(df,period=21,mult=0.75):
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

@st.cache_data(ttl=300)
def compute_nw(ha_close_tuple,bw=8,smooth=True,atr_tuple=None):
    prices=np.array(ha_close_tuple,dtype=float); n=len(prices)
    mid=np.full(n,np.nan); ws=max(0,n-500)
    for bar in range(ws,n):
        w=np.array([math.exp(-0.5*((bar-j)/bw)**2) for j in range(ws,n)])
        mid[bar]=np.dot(w,prices[ws:n])/w.sum()
    s=pd.Series(mid)
    if smooth: s=s.ewm(span=2,adjust=False).mean()
    atr=np.array(atr_tuple,dtype=float) if atr_tuple else np.full(n,np.nan)
    return (s+atr*NW_MULT).values,(s-atr*NW_MULT).values,s.values

def compute_bb_expansion(close):
    bw=(4*2.0*close.rolling(20).std(ddof=0))/close.rolling(20).mean()*100.0
    return bw>bw.rolling(50).mean()*1.1

st.title("📈 ETH SuperTrend Bot — Live Dashboard")
st.caption(f"Data cached 5 min · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

with st.sidebar:
    st.header("⚙️ Settings")
    bars_to_show=st.slider("Bars to show",50,300,100)
    show_ha=st.toggle("Heikin Ashi candles",value=False)
    show_st=st.toggle("SuperTrend line",value=True)
    show_nw=st.toggle("NW Envelope",value=True)
    st.divider()
    st.subheader("🔬 LuxAlgo Comparison")
    st.caption("Paste latest bar values from TradingView")
    lux_u=st.number_input("LuxAlgo NW Upper",value=0.0,format="%.2f")
    lux_l=st.number_input("LuxAlgo NW Lower",value=0.0,format="%.2f")
    st.divider()
    st.subheader("🔔 Test Telegram")
    tg_token=st.text_input("Bot Token",type="password",placeholder="123:AAF...")
    tg_chat=st.text_input("Chat ID",value="1130112754")
    if st.button("📤 Send Test"):
        if tg_token:
            try:
                r=requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage",
                    data={"chat_id":tg_chat,"text":"✅ ETH Bot Test\nTelegram alerts working!"},timeout=10)
                if r.status_code==200: st.success("✅ Sent! Check Telegram.")
                else: st.error(f"❌ {r.json().get('description','Failed')}")
            except Exception as e: st.error(str(e))
        else: st.warning("Enter token first")
    if st.button("🔄 Refresh"): st.cache_data.clear(); st.rerun()

with st.spinner("Fetching candles..."):
    df=fetch_candles(550)
if df is None:
    st.error("❌ Could not fetch data"); st.stop()

ha=to_heikin_ashi(df)
hi,lo,cl=df["high"],df["low"],df["close"]
tr=pd.concat([hi-lo,(hi-cl.shift(1)).abs(),(lo-cl.shift(1)).abs()],axis=1).max(axis=1)
atr=tr.ewm(span=ATR_PERIOD,adjust=False).mean()
st_line,st_dir=compute_supertrend(df,ATR_PERIOD,ST_FACTOR)
bb_exp=compute_bb_expansion(cl)
nwu_a,nwl_a,nwm_a=compute_nw(tuple(ha["close"].tolist()),BANDWIDTH,True,tuple(atr.tolist()))
nw_upper=pd.Series(nwu_a,index=df.index)
nw_lower=pd.Series(nwl_a,index=df.index)
nw_mid  =pd.Series(nwm_a,index=df.index)

cc=df["close"].iloc[-1]; prev=df["close"].iloc[-2]
chg=cc-prev; chgp=chg/prev*100
cur_nwu=nw_upper.iloc[-1]; cur_nwl=nw_lower.iloc[-1]
is_bull=st_dir.iloc[-1]==-1; flipped=st_dir.iloc[-1]!=st_dir.iloc[-2]

c1,c2,c3,c4,c5=st.columns(5)
c1.metric("ETH Price",  f"${cc:,.2f}", f"{chg:+.2f} ({chgp:+.2f}%)")
c2.metric("ST Direction","🟢 BULL" if is_bull else "🔴 BEAR")
c3.metric("NW Upper",   f"${cur_nwu:,.2f}")
c4.metric("NW Lower",   f"${cur_nwl:,.2f}")
c5.metric("ATR",        f"${atr.iloc[-1]:.2f}")

if lux_u>0 and lux_l>0:
    du=abs(lux_u-cur_nwu); dl=abs(lux_l-cur_nwl)
    if du<15 and dl<15: st.success(f"✅ Good match — Upper diff: ${du:.2f} | Lower diff: ${dl:.2f}")
    else: st.warning(f"⚠️ Large diff — Upper: ${du:.2f} | Lower: ${dl:.2f} · Check LuxAlgo: BW=8, Smooth=ON, Repaint=ON, Mult=3.0")


sl=bars_to_show
src=ha.iloc[-sl:] if show_ha else df.iloc[-sl:]
fig=go.Figure()
fig.add_trace(go.Candlestick(x=src.index,open=src["open"],high=src["high"],
    low=src["low"],close=src["close"],
    increasing_line_color="#26a69a",decreasing_line_color="#ef5350",
    name="HA" if show_ha else "ETH",line_width=1))
if show_nw:
    fig.add_trace(go.Scatter(x=nw_upper.iloc[-sl:].index,y=nw_upper.iloc[-sl:].values,
        line=dict(color="rgba(38,166,154,0.9)",width=1.5,dash="dot"),name="NW Upper"))
    fig.add_trace(go.Scatter(x=nw_lower.iloc[-sl:].index,y=nw_lower.iloc[-sl:].values,
        line=dict(color="rgba(239,83,80,0.9)",width=1.5,dash="dot"),
        fill="tonexty",fillcolor="rgba(100,100,100,0.07)",name="NW Lower"))
    fig.add_trace(go.Scatter(x=nw_mid.iloc[-sl:].index,y=nw_mid.iloc[-sl:].values,
        line=dict(color="rgba(41,98,255,0.6)",width=1),name="NW Mid"))
if show_st:
    stv=st_line.iloc[-sl:]; sdv=st_dir.iloc[-sl:]
    fig.add_trace(go.Scatter(x=stv[sdv==-1].index,y=stv[sdv==-1].values,
        mode="markers",marker=dict(color="#26a69a",size=3),name="ST Bull"))
    fig.add_trace(go.Scatter(x=stv[sdv==1].index,y=stv[sdv==1].values,
        mode="markers",marker=dict(color="#ef5350",size=3),name="ST Bear"))
fig.update_layout(height=520,template="plotly_dark",xaxis_rangeslider_visible=False,
    legend=dict(orientation="h",y=1.02),margin=dict(l=0,r=0,t=30,b=0),
    paper_bgcolor="#131722",plot_bgcolor="#131722",
    xaxis=dict(gridcolor="#2a2e39"),yaxis=dict(gridcolor="#2a2e39"))
st.plotly_chart(fig,use_container_width=True)

st.subheader("📊 Current Bar")
c1,c2,c3,c4=st.columns(4)
c1.metric("ST Flipped",  "YES ⚡" if flipped else "No")
c2.metric("BB Expansion","✅ YES" if bb_exp.iloc[-1] else "❌ No")
c3.metric("NW Long OK",  "✅" if cc<=cur_nwu else "⚠️ Overextended")
c4.metric("NW Short OK", "✅" if cc>=cur_nwl else "⚠️ Overextended")

st.subheader("🔢 Last 15 Bars — Compare with LuxAlgo")
t=df.tail(15).copy()
t["HA Close"]=ha["close"].tail(15).round(2)
t["NW Upper"]=nw_upper.tail(15).round(2)
t["NW Lower"]=nw_lower.tail(15).round(2)
t["NW Mid"]  =nw_mid.tail(15).round(2)
t["ST"]      =st_dir.tail(15).map({-1:"🟢 BULL",1:"🔴 BEAR",0:"-"})
t.index      =t.index.strftime("%m-%d %H:%M")
st.dataframe(t[["close","HA Close","NW Upper","NW Lower","NW Mid","ST"]].rename(columns={"close":"Close"}),use_container_width=True)
st.caption("⚠️ Telegram token is never stored — only used for the live test above.")


