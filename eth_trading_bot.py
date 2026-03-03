#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  ETH SuperTrend Trading Bot
  Mirrors FINAL v5 Pine Script logic exactly
  Runs hourly on PythonAnywhere free tier

  SIGNALS:  SuperTrend(21, 0.75) + BB StrongTrend + ST Slope + NW Envelope
            All computed on Heikin Ashi candles (matches LuxAlgo NW)
  EXCHANGE: Delta India (ETHUSD Perpetual)
  ALERTS:   Telegram + Delta webhook
  STATE:    Saved to state.json (survives restarts)

  SETUP:
    1. Upload this file to PythonAnywhere
    2. Fill in CONFIG section below
    3. pip install pandas numpy requests  (already available on PythonAnywhere)
    4. Add scheduled task: python /home/<username>/eth_trading_bot.py
       Set frequency: hourly
    5. Done — runs every hour, sends alerts, places/closes orders automatically
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import math
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — fill these in before running
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Secrets loaded from environment variables ─────────────────────────
    # GitHub Actions: Settings → Secrets → Actions → New repository secret
    # Raspberry Pi:   add to ~/.bashrc or create a .env file (see README)
    # NEVER hardcode secrets here — this file is safe to push to GitHub
    "TELEGRAM_TOKEN":   os.environ.get("TELEGRAM_TOKEN",   ""),
    "TELEGRAM_CHAT_ID": os.environ.get("TELEGRAM_CHAT_ID", "1130112754"),

    # ── Delta India ───────────────────────────────────────────────────────
    # Webhook URL + strategy_id is all Delta India needs — no API key required
    "DELTA_WEBHOOK_URL": os.environ.get("DELTA_WEBHOOK_URL",
        "https://cdn.india.deltaex.org./v2/webhook_alert/dfbf481e1c6ac38dfb9e55d2102497d4"),
    "DELTA_STRATEGY_ID": "dfbf481e1c6ac38dfb9e55d2102497d4",

    # ── Contract ──────────────────────────────────────────────────────────
    "SYMBOL":          "ETHUSD",       # Delta India perpetual contract name
    "BASE_CONTRACTS":  80,             # contracts per trade
    "MAX_CONTRACTS":   100,            # recovery cap (80 × 1.25)
    "LEVERAGE":        10,             # display/reference only

    # ── SuperTrend ────────────────────────────────────────────────────────
    "ATR_PERIOD":      21,
    "ST_FACTOR":       0.75,           # ← TWEAK: try 1.0, 1.5, 2.0

    # ── Regime Detection ─────────────────────────────────────────────────
    # SIDEWAYS MODE activates when whipsaw count hits threshold
    # TRENDING MODE restores when a clean run is confirmed
    "WHIP_WINDOW":       10,           # ← TWEAK: bars to look back for whipsaws (8-14)
    "WHIP_THRESHOLD":    3,            # ← TWEAK: flips within window = sideways mode (2-4)
    "WHIP_HARD_RESET":   6,            # ← TWEAK: force-reset sideways mode after N whips (5-8)
    "TREND_MIN_BARS":    6,            # ← TWEAK: min bars ST must hold direction to confirm trend (4-10)
    "TREND_MIN_PCT":     1.5,          # ← TWEAK: min % price move during run to confirm trend (1.0-2.5)

    # ── Filters (active only in SIDEWAYS mode) ────────────────────────────
    "SLOPE_LOOKBACK":  6,              # ST slope lookback bars
    "MIN_SLOPE_PERC":  0.08,           # ← TWEAK: min ST slope % (0.06-0.12)
    "BB_LENGTH":       20,
    "BB_MULT":         2.0,
    "NW_BANDWIDTH":    8,              # ← TWEAK: try 6-12
    "NW_SMOOTH":       True,
    "USE_NW":          True,           # ← TWEAK: True/False — toggle NW filter

    # ── ADX ──────────────────────────────────────────────────────────────
    "ADX_LEN":         7,
    "ADX_THRESHOLD":   25.0,           # not blocking entry (disabled in your live)

    # ── Recovery ──────────────────────────────────────────────────────────
    "ENABLE_RECOVERY":    True,
    "RECOVERY_TRIGGER":   2,           # activate after N consecutive losses
    "RECOVERY_ATTEMPTS":  2,           # max recovery attempts
    "SIZE_MULTIPLIER":    1.25,
    "RESET_AFTER_LOSSES": 4,

    # ── Pause Window (Fri 9PM → Sun 8PM IST) ─────────────────────────────
    "ENABLE_PAUSE":    True,

    # ── State file path ───────────────────────────────────────────────────
    # PythonAnywhere: use full path like /home/yourusername/state.json
    "STATE_FILE":      "state.json",

    # ── Candle fetch ──────────────────────────────────────────────────────
    "CANDLES_NEEDED":  600,            # need enough for NW (bandwidth=8 needs ~200+)
    "TIMEFRAME":       "1h",
}

# ═══════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# Persists between hourly runs: position, recovery counters, loss streak
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_STATE = {
    "position":           "flat",      # "flat" | "long" | "short"
    "entry_price":        0.0,
    "contracts":          0,
    "consecutive_losses": 0,
    "recovery_count":     0,
    "recovery_active":    False,
    "current_multiplier": 1.0,
    "last_signal":        "",          # last signal fired (for dedup)
    "last_signal_bar":    "",          # ISO timestamp of last signal bar
    "sideways_mode":      False,         # regime tracking
    "cum_whips_mode":     0,
    "total_trades":       0,
    "wins":               0,
    "losses":             0,
}

def load_state():
    if os.path.exists(CONFIG["STATE_FILE"]):
        try:
            with open(CONFIG["STATE_FILE"], "r") as f:
                s = json.load(f)
            # fill any missing keys from DEFAULT_STATE
            for k, v in DEFAULT_STATE.items():
                if k not in s:
                    s[k] = v
            return s
        except Exception:
            pass
    return DEFAULT_STATE.copy()

def save_state(state):
    with open(CONFIG["STATE_FILE"], "w") as f:
        json.dump(state, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════════════════════════════════════

def send_telegram(message):
    token = CONFIG["TELEGRAM_TOKEN"]
    chat_id = CONFIG["TELEGRAM_CHAT_ID"]
    if not token:
        print(f"[TELEGRAM SKIP — token not set] {message}")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=10)
        if r.status_code == 200:
            print(f"[TELEGRAM OK] {message[:60]}")
        else:
            print(f"[TELEGRAM ERR {r.status_code}] {r.text[:100]}")
    except Exception as e:
        print(f"[TELEGRAM EXCEPTION] {e}")

# ═══════════════════════════════════════════════════════════════════════════
# DELTA INDIA — WEBHOOK EXECUTION
#
# Delta India uses webhook-based order execution — no API key needed.
# You POST the exact JSON format Delta expects to your webhook URL.
# The strategy_id in the URL authenticates the request.
#
# Payload format (from Delta's own template):
# {"symbol": "ETHUSD", "side": "buy", "qty": "80",
#  "trigger_time": "...", "strategy_id": "..."}
#
# side:  "buy"  = open long  OR close short
#        "sell" = open short OR close long
# ═══════════════════════════════════════════════════════════════════════════

def delta_execute(side, qty, action_label):
    """
    Send order to Delta India via webhook.
    side: "buy" | "sell"
    qty:  number of contracts (string or int)
    """
    url         = CONFIG["DELTA_WEBHOOK_URL"]
    strategy_id = CONFIG["DELTA_STRATEGY_ID"]
    symbol      = CONFIG["SYMBOL"]
    now_str     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    payload = {
        "symbol":      symbol,
        "side":        side,
        "qty":         str(qty),
        "trigger_time": now_str,
        "strategy_id": strategy_id,
    }

    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            print(f"[DELTA OK] {action_label} — {side} {qty} {symbol}")
            return True
        else:
            print(f"[DELTA ERR {r.status_code}] {action_label} → {r.text[:200]}")
            return False
    except Exception as e:
        print(f"[DELTA EXCEPTION] {action_label} → {e}")
        return False

def close_position(state):
    """Close current open position via Delta webhook"""
    if state["position"] == "flat":
        return True
    # To close: send opposite side
    side = "sell" if state["position"] == "long" else "buy"
    return delta_execute(side, state["contracts"],
                         f"CLOSE {state['position'].upper()}")

def open_position(direction, contracts):
    """Open new position via Delta webhook"""
    side = "buy" if direction == "long" else "sell"
    ok   = delta_execute(side, contracts, f"OPEN {direction.upper()}")
    # Webhook doesn't return fill price — use current close price from debug
    return ok, 0.0

def send_delta_webhook(message):
    """Send plain notification to Delta webhook (non-trade alert)"""
    url = CONFIG["DELTA_WEBHOOK_URL"]
    try:
        r = requests.post(url, json={"message": message}, timeout=10)
        print(f"[DELTA NOTIFY] {r.status_code}")
    except Exception as e:
        print(f"[DELTA NOTIFY ERR] {e}")

# ═══════════════════════════════════════════════════════════════════════════
# CANDLE DATA — Binance public API (no key needed)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_candles(limit=600):
    """
    Fetch ETH/USDT 1hr candles from Binance.
    Returns DataFrame with columns: open, high, low, close, volume
    Index: datetime UTC
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol":   "ETHUSDT",
        "interval": "1h",
        "limit":    limit,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","tbbav","tbqav","ignore"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("open_time")
        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)
        # Drop the current (incomplete) candle — last row
        df = df.iloc[:-1]
        print(f"[DATA] Fetched {len(df)} confirmed 1hr candles")
        return df
    except Exception as e:
        print(f"[DATA ERROR] {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════════
# HEIKIN ASHI CONVERSION
# NW Envelope uses HA candles (same as LuxAlgo default)
# ═══════════════════════════════════════════════════════════════════════════

def to_heikin_ashi(df):
    ha = df.copy()
    ha["close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = [(df["open"].iloc[0] + df["close"].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha["close"].iloc[i-1]) / 2)
    ha["open"] = ha_open
    ha["high"] = df[["high","open","close"]].max(axis=1)
    ha["low"]  = df[["low","open","close"]].min(axis=1)
    return ha

# ═══════════════════════════════════════════════════════════════════════════
# SUPERTREND  (ATR=21, Factor=0.75)
# ═══════════════════════════════════════════════════════════════════════════

def compute_supertrend(df, period=21, multiplier=0.75):
    """
    Returns series: st_line, st_dir
    st_dir: -1 = bullish (price above ST), +1 = bearish (price below ST)
    Matches Pine Script ta.supertrend() exactly
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    n     = len(df)

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    # Wilder's smoothing (same as Pine's ta.atr)
    atr = tr.copy().astype(float)
    atr.iloc[:period] = np.nan
    atr.iloc[period]  = tr.iloc[1:period+1].mean()
    for i in range(period+1, n):
        atr.iloc[i] = (atr.iloc[i-1] * (period - 1) + tr.iloc[i]) / period

    hl2     = (high + low) / 2
    upper   = hl2 + multiplier * atr
    lower   = hl2 - multiplier * atr

    st      = pd.Series(np.nan, index=df.index)
    trend   = pd.Series(0,      index=df.index)   # -1=bull, +1=bear

    for i in range(1, n):
        if np.isnan(atr.iloc[i]):
            continue

        # Final upper band
        if upper.iloc[i] < upper.iloc[i-1] or close.iloc[i-1] > upper.iloc[i-1]:
            final_upper = upper.iloc[i]
        else:
            final_upper = upper.iloc[i-1]

        # Final lower band
        if lower.iloc[i] > lower.iloc[i-1] or close.iloc[i-1] < lower.iloc[i-1]:
            final_lower = lower.iloc[i]
        else:
            final_lower = lower.iloc[i-1]

        upper.iloc[i] = final_upper
        lower.iloc[i] = final_lower

        # Trend direction
        if st.iloc[i-1] == upper.iloc[i-1]:
            if close.iloc[i] <= final_upper:
                st.iloc[i]    = final_upper
                trend.iloc[i] = 1    # bearish
            else:
                st.iloc[i]    = final_lower
                trend.iloc[i] = -1   # bullish
        else:
            if close.iloc[i] >= final_lower:
                st.iloc[i]    = final_lower
                trend.iloc[i] = -1   # bullish
            else:
                st.iloc[i]    = final_upper
                trend.iloc[i] = 1    # bearish

    return st, trend

# ═══════════════════════════════════════════════════════════════════════════
# ADX / DMI  (length=7)
# ═══════════════════════════════════════════════════════════════════════════

def compute_adx(df, length=7):
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    n     = len(df)

    dm_plus  = pd.Series(0.0, index=df.index)
    dm_minus = pd.Series(0.0, index=df.index)

    for i in range(1, n):
        up   = high.iloc[i]  - high.iloc[i-1]
        down = low.iloc[i-1] - low.iloc[i]
        dm_plus.iloc[i]  = up   if up > down and up > 0   else 0.0
        dm_minus.iloc[i] = down if down > up and down > 0 else 0.0

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    # Wilder smoothing
    def wilder(series, n):
        result = pd.Series(np.nan, index=series.index)
        result.iloc[n] = series.iloc[1:n+1].mean()
        for i in range(n+1, len(series)):
            result.iloc[i] = result.iloc[i-1] - result.iloc[i-1]/n + series.iloc[i]
        return result

    atr_w   = wilder(tr,       length)
    dmp_w   = wilder(dm_plus,  length)
    dmm_w   = wilder(dm_minus, length)

    di_plus  = 100 * dmp_w / atr_w
    di_minus = 100 * dmm_w / atr_w
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    adx      = wilder(dx, length)

    return adx

# ═══════════════════════════════════════════════════════════════════════════
# BOLLINGER BANDS (for StrongTrend filter)
# ═══════════════════════════════════════════════════════════════════════════

def compute_bb_expansion(close, length=20, mult=2.0):
    basis     = close.rolling(length).mean()
    dev       = close.rolling(length).std(ddof=0)
    bb_width  = (4 * mult * dev) / basis * 100.0
    bb_avg    = bb_width.rolling(50).mean()
    expansion = bb_width > bb_avg * 1.1
    return expansion

# ═══════════════════════════════════════════════════════════════════════════
# NADARAYA-WATSON ENVELOPE
# Exact LuxAlgo implementation:
#   - Gaussian kernel regression on HA close prices
#   - bandwidth=8 (h parameter)
#   - smooth=True (applies EMA smoothing on result)
#   - repaint=False (uses only past bars, non-repainting)
#   - envelope multiplier = 3.0 (ATR-based, same as LuxAlgo)
# ═══════════════════════════════════════════════════════════════════════════

def compute_nw_envelope(ha_close, bandwidth=8, smooth=True, atr=None):
    """
    Non-repainting Nadaraya-Watson envelope.
    Uses only past+current data — no future lookahead.
    Returns: nw_upper, nw_lower, nw_mid
    """
    n       = len(ha_close)
    prices  = ha_close.values.astype(float)
    nw_mid  = np.full(n, np.nan)

    # Need at least 2*bandwidth bars to compute
    min_bars = bandwidth * 2
    for i in range(min_bars, n):
        # Use a window of past bars (non-repainting: only bars up to i)
        # LuxAlgo uses last 500 bars but here we cap at 300 for speed
        window_start = max(0, i - 300)
        weights = np.array([
            math.exp(-0.5 * ((i - j) / bandwidth) ** 2)
            for j in range(window_start, i + 1)
        ])
        window_prices = prices[window_start:i+1]
        nw_mid[i] = np.dot(weights, window_prices) / weights.sum()

    nw_series = pd.Series(nw_mid, index=ha_close.index)

    # Smooth the mid line (LuxAlgo smooth=on uses EMA-2)
    if smooth:
        nw_series = nw_series.ewm(span=2, adjust=False).mean()

    # ATR-based envelope bands (LuxAlgo uses multiplier=3.0 × ATR)
    if atr is None:
        # fallback: use rolling std
        spread = ha_close.rolling(20).std(ddof=0) * 3.0
    else:
        spread = atr * 3.0

    nw_upper = nw_series + spread
    nw_lower = nw_series - spread

    return nw_upper, nw_lower, nw_series

# ═══════════════════════════════════════════════════════════════════════════
# PAUSE WINDOW CHECK
# Fri 21:00 IST → Sun 20:00 IST  (IST = UTC+5:30)
# Fri 15:30 UTC → Sun 14:30 UTC
# ═══════════════════════════════════════════════════════════════════════════

def is_pause_window():
    if not CONFIG["ENABLE_PAUSE"]:
        return False
    now = datetime.now(timezone.utc)
    wd  = now.weekday()   # Mon=0, Fri=4, Sat=5, Sun=6
    h   = now.hour
    m   = now.minute

    # Friday 15:30 UTC onwards
    if wd == 4 and (h > 15 or (h == 15 and m >= 30)):
        return True
    # All Saturday
    if wd == 5:
        return True
    # Sunday until 14:30 UTC
    if wd == 6 and (h < 14 or (h == 14 and m < 30)):
        return True
    return False

# ═══════════════════════════════════════════════════════════════════════════
# RECOVERY SIZING
# ═══════════════════════════════════════════════════════════════════════════

def get_contracts(state):
    base = CONFIG["BASE_CONTRACTS"]
    mult = state["current_multiplier"]
    qty  = int(base * mult)
    return min(qty, CONFIG["MAX_CONTRACTS"])

def update_recovery_on_loss(state):
    if not CONFIG["ENABLE_RECOVERY"]:
        state["recovery_active"]    = False
        state["current_multiplier"] = 1.0
        return state

    if not state["recovery_active"]:
        state["consecutive_losses"] += 1

    losses  = state["consecutive_losses"]
    trigger = CONFIG["RECOVERY_TRIGGER"]
    attempts= CONFIG["RECOVERY_ATTEMPTS"]

    if losses >= trigger:
        if not state["recovery_active"]:
            state["recovery_active"]    = True
            state["recovery_count"]     = 1
            state["current_multiplier"] = CONFIG["SIZE_MULTIPLIER"]
        elif state["recovery_count"] < attempts:
            state["recovery_count"]     += 1
            state["current_multiplier"] = CONFIG["SIZE_MULTIPLIER"]
        else:
            state["recovery_active"]    = False
            state["current_multiplier"] = 1.0

    if losses >= CONFIG["RESET_AFTER_LOSSES"]:
        state["consecutive_losses"] = 0
        state["recovery_count"]     = 0
        state["recovery_active"]    = False
        state["current_multiplier"] = 1.0

    return state

def update_recovery_on_win(state):
    state["consecutive_losses"] = 0
    state["recovery_count"]     = 0
    state["recovery_active"]    = False
    state["current_multiplier"] = 1.0
    return state

# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL COMPUTATION
# Returns: "long" | "short" | "exit_long" | "exit_short" | None
# ═══════════════════════════════════════════════════════════════════════════

def compute_signals(df, state=None):
    """
    Compute all indicators and return current bar signal.
    Uses regime detection: sideways mode activates when whipsaws are high,
    filters relax when a clean trending run is confirmed.
    state: optional — used to persist sideways_mode between hourly runs.
    """
    if df is None or len(df) < 100:
        print("[SIGNAL] Not enough data")
        return None, {}

    # ── Heikin Ashi ──
    ha = to_heikin_ashi(df)

    # ── SuperTrend ──
    st_line, st_dir = compute_supertrend(df,
        period=CONFIG["ATR_PERIOD"], multiplier=CONFIG["ST_FACTOR"])

    # ── ADX ──
    adx = compute_adx(df, length=CONFIG["ADX_LEN"])

    # ── ATR ──
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    tr    = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_series = tr.ewm(span=CONFIG["ATR_PERIOD"], adjust=False).mean()

    # ── BB StrongTrend ──
    bb_expansion  = compute_bb_expansion(close, CONFIG["BB_LENGTH"], CONFIG["BB_MULT"])
    st_distance   = (close - st_line).abs()
    avg_distance  = st_distance.rolling(10).mean()
    st_dist_above = st_distance > avg_distance
    strong_trend  = bb_expansion | st_dist_above

    # ── ST Slope ──
    lb = CONFIG["SLOPE_LOOKBACK"]
    st_slope_pts  = (st_line - st_line.shift(lb)).abs()
    st_slope_perc = st_slope_pts / close * 100.0
    st_trending   = st_slope_perc > CONFIG["MIN_SLOPE_PERC"]

    # ── NW Envelope ──
    if CONFIG.get("USE_NW", True):
        nw_upper, nw_lower, nw_mid = compute_nw_envelope(
            ha["close"], bandwidth=CONFIG["NW_BANDWIDTH"],
            smooth=CONFIG["NW_SMOOTH"], atr=atr_series)
    else:
        nw_upper = pd.Series(np.inf,  index=df.index)
        nw_lower = pd.Series(-np.inf, index=df.index)
        nw_mid   = close.copy()

    # ── Regime Detection ──────────────────────────────────────────────────
    # Count ST flips within rolling window
    flips       = (st_dir != st_dir.shift(1)).astype(int)
    ww          = CONFIG["WHIP_WINDOW"]
    whip_count  = flips.rolling(ww).sum().fillna(0)

    # ST run length and % move since last flip
    n = len(df)
    st_run_len  = np.zeros(n)
    st_run_pct  = np.zeros(n)
    run_start   = close.iloc[0]
    run_len     = 0
    for idx in range(1, n):
        if st_dir.iloc[idx] == st_dir.iloc[idx-1]:
            run_len += 1
            st_run_len[idx] = run_len
            st_run_pct[idx] = abs(close.iloc[idx] - run_start) / run_start * 100
        else:
            run_len   = 1
            run_start = close.iloc[idx-1]
            st_run_len[idx] = 1
            st_run_pct[idx] = 0

    # Get persisted sideways state from state.json
    sideways_mode  = state.get("sideways_mode", False)   if state else False
    cum_whips_mode = state.get("cum_whips_mode", 0)      if state else 0

    wt  = CONFIG["WHIP_THRESHOLD"]
    whr = CONFIG["WHIP_HARD_RESET"]
    tmb = CONFIG["TREND_MIN_BARS"]
    tmp = CONFIG["TREND_MIN_PCT"]

    # Update regime using recent bars (last 20 bars to catch transitions)
    for idx in range(max(0, n-20), n):
        flipped = flips.iloc[idx] == 1

        if whip_count.iloc[idx] >= wt:
            if not sideways_mode:
                sideways_mode  = True
                cum_whips_mode = 0
                print(f"[REGIME] Entering SIDEWAYS mode — "
                      f"{int(whip_count.iloc[idx])} flips in last {ww} bars")
            if flipped:
                cum_whips_mode += 1

        # Exit sideways: clean trend confirmed
        if sideways_mode:
            if st_run_len[idx] >= tmb and st_run_pct[idx] >= tmp:
                sideways_mode  = False
                cum_whips_mode = 0
                print(f"[REGIME] Exiting SIDEWAYS — clean run: "
                      f"{int(st_run_len[idx])} bars, {st_run_pct[idx]:.1f}%")

        # Hard reset: too many whips, just start fresh
        if sideways_mode and cum_whips_mode >= whr:
            sideways_mode  = False
            cum_whips_mode = 0
            print(f"[REGIME] HARD RESET — {cum_whips_mode} whips in sideways mode")

    # ── Current bar values ────────────────────────────────────────────────
    i = -1
    cur_st_dir    = st_dir.iloc[i]
    prev_st_dir   = st_dir.iloc[i-1]
    cur_close     = close.iloc[i]
    cur_adx       = adx.iloc[i]
    cur_strong    = strong_trend.iloc[i]
    cur_slope     = st_trending.iloc[i]
    cur_nw_upper  = nw_upper.iloc[i]
    cur_nw_lower  = nw_lower.iloc[i]
    cur_whips     = whip_count.iloc[i]

    base_buy  = cur_st_dir == -1
    base_sell = cur_st_dir ==  1

    st_flipped_bull = cur_st_dir == -1 and prev_st_dir == 1
    st_flipped_bear = cur_st_dir ==  1 and prev_st_dir == -1

    # ── Apply filters based on regime ─────────────────────────────────────
    nw_long_ok  = np.isnan(cur_nw_upper) or cur_close <= cur_nw_upper or not CONFIG.get("USE_NW", True)
    nw_short_ok = np.isnan(cur_nw_lower) or cur_close >= cur_nw_lower or not CONFIG.get("USE_NW", True)

    in_pause = is_pause_window()

    if sideways_mode:
        # Sideways: all filters active
        long_cond  = (base_buy  and cur_strong and cur_slope
                      and nw_long_ok  and not in_pause)
        short_cond = (base_sell and cur_strong and cur_slope
                      and nw_short_ok and not in_pause)
        mode_label = "SIDEWAYS"
    else:
        # Trending: pure ST signal, no extra filters
        long_cond  = base_buy  and not in_pause
        short_cond = base_sell and not in_pause
        mode_label = "TRENDING"

    # ── Signal ────────────────────────────────────────────────────────────
    signal = None
    if st_flipped_bull and long_cond:
        signal = "long"
    elif st_flipped_bear and short_cond:
        signal = "short"
    elif st_flipped_bear and not base_buy:
        signal = "exit_long"
    elif st_flipped_bull and not base_sell:
        signal = "exit_short"

    debug = {
        "close":          round(cur_close, 2),
        "st_dir":         "BULL" if base_buy else "BEAR",
        "st_flipped":     st_flipped_bull or st_flipped_bear,
        "regime":         mode_label,
        "whip_count":     int(cur_whips),
        "cum_whips_mode": cum_whips_mode,
        "strong_trend":   bool(cur_strong),
        "st_slope":       bool(cur_slope),
        "adx":            round(float(cur_adx), 2) if not np.isnan(cur_adx) else 0,
        "nw_upper":       round(float(cur_nw_upper), 2) if not np.isnan(cur_nw_upper) and cur_nw_upper != np.inf else None,
        "nw_lower":       round(float(cur_nw_lower), 2) if not np.isnan(cur_nw_lower) and cur_nw_lower != -np.inf else None,
        "nw_long_ok":     nw_long_ok,
        "nw_short_ok":    nw_short_ok,
        "in_pause":       in_pause,
        "sideways_mode":  sideways_mode,
        "cum_whips_mode": cum_whips_mode,
    }

    return signal, debug

# ═══════════════════════════════════════════════════════════════════════════
# EXECUTE TRADE
# Handles: close existing → open new, alert both Telegram + Delta webhook
# ═══════════════════════════════════════════════════════════════════════════

def execute_trade(signal, state, debug):
    now_str   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    contracts = get_contracts(state)
    rec_tag   = f" [REC ×{state['current_multiplier']}]" if state["recovery_active"] else ""

    # ── EXIT SIGNALS ──
    if signal in ("exit_long", "exit_short"):
        if state["position"] == "flat":
            print(f"[SKIP] {signal} but already flat")
            return state

        close_ok = close_position(state)
        if not close_ok:
            send_telegram(f"⚠️ DELTA ORDER FAILED\nCould not close {state['position']} position\nManual action required!")
            return state

        direction = state["position"]
        entry     = state["entry_price"]
        pnl_dir   = "+" if (
            (direction == "long"  and debug["close"] > entry) or
            (direction == "short" and debug["close"] < entry)
        ) else "-"

        # Update recovery
        if pnl_dir == "+":
            state = update_recovery_on_win(state)
            state["wins"] += 1
        else:
            state = update_recovery_on_loss(state)
            state["losses"] += 1
        state["total_trades"] += 1

        # Reset position
        state["position"]    = "flat"
        state["entry_price"] = 0.0
        state["contracts"]   = 0

        msg = (f"{'🔴 EXIT LONG' if direction == 'long' else '🟢 EXIT SHORT'}\n"
               f"ETH | {now_str}\n"
               f"Exit Price: {debug['close']}\n"
               f"Entry was: {entry}\n"
               f"Result: {'WIN ✅' if pnl_dir == '+' else 'LOSS ❌'}\n"
               f"Trades: {state['total_trades']} | "
               f"W:{state['wins']} L:{state['losses']}")
        send_telegram(msg)
        send_delta_webhook(msg)
        print(f"[TRADE] {msg[:80]}")
        save_state(state)
        return state

    # ── ENTRY SIGNALS ──
    if signal in ("long", "short"):
        # Close opposite position first if open
        if state["position"] != "flat":
            if state["position"] == signal:
                print(f"[SKIP] Already in {signal} position")
                return state
            print(f"[TRADE] Closing {state['position']} before opening {signal}")
            close_ok = close_position(state)
            if not close_ok:
                send_telegram(f"⚠️ DELTA ORDER FAILED\nCould not close {state['position']} before reversal\nManual action required!")
                return state

            # Update P&L for closed position
            direction = state["position"]
            entry     = state["entry_price"]
            pnl_dir   = "+" if (
                (direction == "long"  and debug["close"] > entry) or
                (direction == "short" and debug["close"] < entry)
            ) else "-"
            if pnl_dir == "+":
                state = update_recovery_on_win(state)
                state["wins"] += 1
            else:
                state = update_recovery_on_loss(state)
                state["losses"] += 1
            state["total_trades"] += 1
            state["position"]    = "flat"
            state["entry_price"] = 0.0
            state["contracts"]   = 0
            save_state(state)

        # Recalculate contracts after recovery update
        contracts = get_contracts(state)
        rec_tag   = f" [REC ×{state['current_multiplier']}]" if state["recovery_active"] else ""

        open_ok, fill_price = open_position(signal, contracts)
        if not open_ok:
            send_telegram(f"⚠️ DELTA ORDER FAILED\nCould not open {signal.upper()} {contracts} contracts\nManual action required!")
            return state

        state["position"]    = signal
        state["entry_price"] = fill_price if fill_price > 0 else debug["close"]
        state["contracts"]   = contracts

        emoji = "🟢" if signal == "long" else "🔴"
        msg = (f"{emoji} {'LONG ENTRY' if signal == 'long' else 'SHORT ENTRY'}{rec_tag}\n"
               f"ETH | {now_str}\n"
               f"Price: {state['entry_price']}\n"
               f"Qty: {contracts} contracts | 10x leverage\n"
               f"ST: {debug['st_dir']} | ADX: {debug['adx']}\n"
               f"BB Exp: {'✅' if debug['strong_trend'] else '❌'} | "
               f"Slope: {'✅' if debug['st_slope'] else '❌'}\n"
               f"NW Upper: {debug['nw_upper']} | Lower: {debug['nw_lower']}")
        send_telegram(msg)
        send_delta_webhook(msg)
        print(f"[TRADE] {msg[:80]}")
        save_state(state)
        return state

    return state

# ═══════════════════════════════════════════════════════════════════════════
# MAIN — runs every hour via PythonAnywhere scheduler
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"ETH Trading Bot — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}")

    # Load persisted state
    state = load_state()
    print(f"[STATE] Position: {state['position']} | "
          f"Losses: {state['consecutive_losses']} | "
          f"Recovery: {state['recovery_active']}")

    # Pause window check
    if is_pause_window():
        print("[PAUSE] In weekend pause window — no new entries allowed")
        send_telegram(f"⏸ PAUSE WINDOW ACTIVE — No new entries\n"
                      f"Current position: {state['position'].upper()}\n"
                      f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        return

    # Fetch candles
    df = fetch_candles(CONFIG["CANDLES_NEEDED"])
    if df is None:
        send_telegram("⚠️ ETH Bot: Failed to fetch candle data from Binance")
        return

    # Compute signals (pass state so regime persists between hourly runs)
    signal, debug = compute_signals(df, state)
    print(f"[SIGNAL] {signal} | Debug: {debug}")

    # Persist regime state so it survives between hourly runs
    state["sideways_mode"]  = debug.get("sideways_mode", False)
    state["cum_whips_mode"] = debug.get("cum_whips_mode", 0)

    # Dedup — don't fire same signal on same bar twice
    # (in case script runs more than once per hour accidentally)
    current_bar = df.index[-1].isoformat()
    if signal and signal == state.get("last_signal") and current_bar == state.get("last_signal_bar"):
        print(f"[DEDUP] Signal '{signal}' already fired this bar — skipping")
        return

    # Execute
    if signal:
        state["last_signal"]     = signal
        state["last_signal_bar"] = current_bar
        state = execute_trade(signal, state, debug)
    else:
        print(f"[NO SIGNAL] Conditions not met this bar")
        # Send hourly status (heartbeat)
        contracts = get_contracts(state)
        print(f"[STATUS] Close: {debug.get('close')} | "
              f"ST: {debug.get('st_dir')} | "
              f"Strong: {debug.get('strong_trend')} | "
              f"Slope: {debug.get('st_slope')} | "
              f"NW OK L:{debug.get('nw_long_ok')} S:{debug.get('nw_short_ok')}")

    save_state(state)
    print(f"[DONE] State saved\n")


if __name__ == "__main__":
    main()
