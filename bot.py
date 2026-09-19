import os
import json
import time
import uuid
import threading
import requests
import websocket
import pandas as pd

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


# ============================================================
# ALLTICK REAL-TIME FOREX -> M1 SIGNAL -> TELEGRAM
# ============================================================

ALLTICK_TOKEN = os.getenv("ALLTICK_API_TOKEN", "").strip()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Free AllTick plan: maximum 5 tick products per WebSocket
DEFAULT_PAIRS = "EURUSD,GBPUSD,USDJPY,USDCHF,AUDUSD"

PAIRS = [
    x.strip().upper()
    for x in os.getenv("PAIRS", DEFAULT_PAIRS).split(",")
    if x.strip()
][:5]

WS_URL = "wss://quote.alltick.co/quote-b-ws-api"

KLINE_URL = "https://quote.alltick.co/quote-b-api/kline"

IST = ZoneInfo("Asia/Kolkata")

HISTORY_CANDLES = 100

# AllTick free HTTP K-line requests need spacing.
HISTORY_REQUEST_DELAY = 10.5

# Prevent duplicate Telegram signal
last_signal_minute = {}


# ============================================================
# BASIC HELPERS
# ============================================================

def now_ist():
    return datetime.now(timezone.utc).astimezone(IST)


def trace_id():
    return str(uuid.uuid4())


def telegram_send(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing")
        return

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }

    try:
        r = requests.post(url, json=payload, timeout=15)
        print("Telegram:", r.status_code)

    except Exception as e:
        print("Telegram error:", e)


# ============================================================
# ALLTICK HISTORICAL M1 DATA
# ============================================================

def get_history(pair):
    query = {
        "trace": trace_id(),
        "data": {
            "code": pair,
            "kline_type": 1,
            "kline_timestamp_end": 0,
            "query_kline_num": HISTORY_CANDLES,
            "adjust_type": 0
        }
    }

    try:
        r = requests.get(
            KLINE_URL,
            params={
                "token": ALLTICK_TOKEN,
                "query": json.dumps(query, separators=(",", ":"))
            },
            timeout=20
        )

        data = r.json()

        if data.get("ret") != 200:
            print("History error", pair, data)
            return pd.DataFrame()

        candles = data.get("data", {}).get("kline_list", [])

        rows = []

        for c in candles:
            ts = int(c["timestamp"])

            # Handle seconds or milliseconds
            if ts > 10_000_000_000:
                ts = ts / 1000

            rows.append({
                "timestamp": pd.to_datetime(ts, unit="s", utc=True),
                "open": float(c["open_price"]),
                "high": float(c["high_price"]),
                "low": float(c["low_price"]),
                "close": float(c["close_price"]),
                "volume": float(c.get("volume", 0) or 0)
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values("timestamp")
        df = df.drop_duplicates("timestamp")
        df = df.reset_index(drop=True)

        return df

    except Exception as e:
        print("History exception", pair, e)
        return pd.DataFrame()


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    df["ema5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    # RSI 14
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)

    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["macd_hist"] = (
        df["macd"] - df["macd_signal"]
    )

    # Bollinger Bands
    df["bb_mid"] = df["close"].rolling(20).mean()

    bb_std = df["close"].rolling(20).std()

    df["bb_upper"] = df["bb_mid"] + (bb_std * 2)
    df["bb_lower"] = df["bb_mid"] - (bb_std * 2)

    # Support / resistance
    df["support"] = df["low"].rolling(20).min()
    df["resistance"] = df["high"].rolling(20).max()

    return df


# ============================================================
# CANDLE PATTERNS
# ============================================================

def candle_pattern(row, previous=None):

    body = abs(row["close"] - row["open"])

    upper = row["high"] - max(
        row["open"],
        row["close"]
    )

    lower = min(
        row["open"],
        row["close"]
    ) - row["low"]

    candle_range = row["high"] - row["low"]

    if candle_range <= 0:
        return "None"

    # Bullish engulfing
    if previous is not None:

        if (
            previous["close"] < previous["open"]
            and row["close"] > row["open"]
            and row["open"] <= previous["close"]
            and row["close"] >= previous["open"]
        ):
            return "Bullish Engulfing"

        # Bearish engulfing
        if (
            previous["close"] > previous["open"]
            and row["close"] < row["open"]
            and row["open"] >= previous["close"]
            and row["close"] <= previous["open"]
        ):
            return "Bearish Engulfing"

    # Hammer
    if (
        lower >= body * 2
        and upper <= max(body, candle_range * 0.1)
    ):
        return "Hammer"

    # Shooting star
    if (
        upper >= body * 2
        and lower
