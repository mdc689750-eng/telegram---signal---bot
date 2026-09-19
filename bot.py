import os
import time
import threading
from datetime import datetime, timezone, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
import pandas as pd

from pocketoptionapi import PocketOption


# =========================================================
# CONFIG
# =========================================================

PO_SSID = os.getenv("PO_SSID", "").strip()

TELEGRAM_TOKEN = os.getenv(
    "TELEGRAM_TOKEN", ""
).strip()

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID", ""
).strip()

# REAL OTC pairs
OTC_PAIRS = [
    "EURUSD_otc",
    "GBPUSD_otc",
    "USDJPY_otc",
    "EURJPY_otc",
    "AUDUSD_otc",
]

TIMEFRAME = 60
HISTORY_COUNT = 100

IST = timezone(timedelta(hours=5, minutes=30))

last_signal = {}

api = None


# =========================================================
# HEALTH SERVER - FOR RENDER
# =========================================================

class HealthHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "text/plain"
        )
        self.end_headers()
        self.wfile.write(
            b"Pocket Option OTC Signal Bot OK"
        )

    def log_message(self, *args):
        return


def start_health_server():

    try:
        port = int(
            os.getenv("PORT", "10000")
        )

        server = HTTPServer(
            ("0.0.0.0", port),
            HealthHandler
        )

        print(
            "Health server running:",
            port
        )

        server.serve_forever()

    except Exception as e:
        print(
            "Health server error:",
            e
        )


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(message):

    if not TELEGRAM_TOKEN:
        print("TELEGRAM_TOKEN missing")
        return

    if not TELEGRAM_CHAT_ID:
        print("TELEGRAM_CHAT_ID missing")
        return

    url = (
        "https://api.telegram.org/bot"
        + TELEGRAM_TOKEN
        + "/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=15
        )

        if response.ok:
            print("Telegram signal sent")
        else:
            print(
                "Telegram error:",
                response.text[:500]
            )

    except Exception as e:

        print(
            "Telegram exception:",
            e
        )


# =========================================================
# CONFIG CHECK
# =========================================================

def check_config():

    missing = []

    if not PO_SSID:
        missing.append("PO_SSID")

    if not TELEGRAM_TOKEN:
        missing.append("TELEGRAM_TOKEN")

    if not TELEGRAM_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")

    if missing:

        print(
            "Missing Environment Variables:"
        )

        for item in missing:
            print(
                " -",
                item
            )

        return False

    print(
        "Environment variables OK"
    )

    return True


# =========================================================
# DATAFRAME NORMALIZATION
# =========================================================

def normalize_candles(data):

    if data is None:
        return None

    try:

        if isinstance(data, pd.DataFrame):

            df = data.copy()

        elif isinstance(data, list):

            df = pd.DataFrame(data)

        else:

            return None

    except Exception:

        return None

    if df.empty:
        return None

    # Lowercase column names
    df.columns = [
        str(c).lower()
        for c in df.columns
    ]

    # Common aliases
    rename_map = {}

    for col in df.columns:

        if col in ["timestamp", "ts", "time_stamp"]:
            rename_map[col] = "time"

        elif col in ["o"]:
            rename_map[col] = "open"

        elif col in ["h"]:
            rename_map[col] = "high"

        elif col in ["l"]:
            rename_map[col] = "low"

        elif col in ["c"]:
            rename_map[col] = "close"

    if rename_map:
        df = df.rename(
            columns=rename_map
        )

    required = [
        "open",
        "high",
        "low",
        "close"
    ]

    for col in required:

        if col not in df.columns:
            return None

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Time
    if "time" in df.columns:

        if pd.api.types.is_numeric_dtype(
            df["time"]
        ):

            # Detect milliseconds
            sample = df["time"].dropna()

            if not sample.empty:

                value = float(
                    sample.iloc[-1]
                )

                if value > 100000000000:
                    df["time"] = (
                        pd.to_datetime(
                            df["time"],
                            unit="ms",
                            utc=True
                        )
                    )
                else:
                    df["time"] = (
                        pd.to_datetime(
                            df["time"],
                            unit="s",
                            utc=True
                        )
                    )

        else:

            df["time"] = pd.to_datetime(
                df["time"],
                utc=True,
                errors="coerce"
            )

    else:

        df["time"] = pd.date_range(
            end=datetime.now(
                timezone.utc
            ),
            periods=len(df),
            freq="min"
        )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close"
        ]
    )

    df = df.sort_values(
        "time"
    )

    df = df.drop_duplicates(
        subset=["time"]
    )

    return df.reset_index(
        drop=True
    )


# =========================================================
# INDICATORS
# =========================================================

def add_indicators(df):

    close = df["close"]

    # EMA
    df["ema5"] = close.ewm(
        span=5,
        adjust=False
    ).mean()

    df["ema10"] = close.ewm(
        span=10,
        adjust=False
    ).mean()

    df["ema20"] = close.ewm(
        span=20,
        adjust=False
    ).mean()

    df["ema50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    # RSI
    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(
        14
    ).mean()

    avg_loss = loss.rolling(
        14
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        1e-12
    )

    df["rsi"] = (
        100 -
        (100 / (1 + rs))
    )

    # MACD
    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = (
        ema12 - ema26
    )

    df["macd_signal"] = (
        df["macd"].ewm(
            span=9,
            adjust=False
        ).mean()
    )

    df["macd_hist"] = (
        df["macd"] -
        df["macd_signal"]
    )

    # Bollinger
    df["bb_mid"] = close.rolling(
        20
    ).mean()

    std = close.rolling(
        20
    ).std()

    df["bb_upper"] = (
        df["bb_mid"] + 2 * std
    )

    df["bb_lower"] = (
        df["bb_mid"] - 2 * std
    )

    # Support / Resistance
    df["support"] = df["low"].rolling(
        20
    ).min()

    df["resistance"] = df["high"].rolling(
        20
    ).max()

    return df


# =========================================================
# CANDLE PATTERN
# =========================================================

def get_pattern(row, previous):

    body = abs(
        row["close"] -
        row["open"]
    )

    candle_range = (
        row["high"] -
        row["low"]
    )

    if candle_range <= 0:
        return "NONE"

    upper = (
        row["high"] -
        max(
            row["open"],
            row["close"]
        )
    )

    lower = (
        min(
            row["open"],
            row["close"]
        ) -
        row["low"]
    )

    # Bullish engulfing
    if (
        previous["close"] <
        previous["open"]
        and
        row["close"] >
        row["open"]
        and
        row["open"] <=
        previous["close"]
        and
        row["close"] >=
        previous["open"]
    ):
        return "BULLISH ENGULFING"

    # Bearish engulfing
    if (
        previous["close"] >
        previous["open"]
        and
        row["close"] <
        row["open"]
        and
        row["open"] >=
        previous["close"]
        and
        row["close"] <=
        previous["open"]
    ):
        return "BEARISH ENGULFING"

    # Hammer
    if (
        lower >= body * 2
        and
        upper <= max(body, 1e-10)
        and
        body <= candle_range * 0.4
    ):
        return "HAMMER"

    # Shooting star
    if (
        upper >= body * 2
        and
        lower <= max(body, 1e-10)
        and
        body <= candle_range * 0.4
    ):
        return "SHOOTING STAR"

    return "NONE"


# =========================================================
# SIGNAL ENGINE
# =========================================================

def analyze(df):

    if df is None:
        return None

    if len(df) < 60:
        return None

    df = add_indicators(
        df.copy()
    )

    row = df.iloc[-1]
    previous = df.iloc[-2]

    needed = [
        "ema20",
        "ema50",
        "rsi",
        "macd_hist",
        "bb_mid",
        "support",
        "resistance"
    ]

    for col in needed:

        if pd.isna(row[col]):
            return None

    price = float(
        row["close"]
    )

    buy = 0
    sell = 0

    buy_reasons = []
    sell_reasons = []

    # EMA trend
    if (
        row["ema5"] >
        row["ema10"] >
        row["ema20"] >
        row["ema50"]
    ):

        buy += 25

        buy_reasons.append(
            "EMA bullish trend"
        )

    elif (
        row["ema5"] <
        row["ema10"] <
        row["ema20"] <
        row["ema50"]
    ):

        sell += 25

        sell_reasons.append(
            "EMA bearish trend"
        )

    # RSI
    if (
        row["rsi"] >= 50
        and
        row["rsi"] <= 70
    ):

        buy += 15

        buy_reasons.append(
            "RSI bullish"
        )

    elif (
        row["rsi"] >= 30
        and
        row["rsi"] < 50
    ):

        sell += 15

        sell_reasons.append(
            "RSI bearish"
        )

    # MACD
    if row["macd_hist"] > 0:

        buy += 15

        buy_reasons.append(
            "MACD positive"
        )

    elif row["macd_hist"] < 0:

        sell += 15

        sell_reasons.append(
            "MACD negative"
        )

    # Bollinger
    if price > row["bb_mid"]:

        buy += 10

        buy_reasons.append(
            "Price above BB middle"
        )

    elif price < row["bb_mid"]:

        sell += 10

        sell_reasons.append(
            "Price below BB middle"
        )

    # Support
    if (
        abs(price - row["support"])
        / price
        < 0.001
    ):

        buy += 15

        buy_reasons.append(
            "Near support"
        )

    # Resistance
    if (
        abs(
            row["resistance"] -
            price
        )
        / price
        < 0.001
    ):

        sell += 15

        sell_reasons.append(
            "Near resistance"
        )

    # Pattern
    pattern = get_pattern(
        row,
        previous
    )

    if pattern in [
        "BULLISH ENGULFING",
        "HAMMER"
    ]:

        buy += 20

        buy_reasons.append(
            pattern
        )

    elif pattern in [
        "BEARISH ENGULFING",
        "SHOOTING STAR"
    ]:

        sell += 20

        sell_reasons.append(
            pattern
        )

    buy = min(
        buy,
        100
    )

    sell = min(
        sell,
        100
    )

    if (
        buy >= 80
        and
        buy > sell
    ):

        return {
            "signal": "BUY",
            "score": buy,
            "price": price,
            "pattern": pattern,
            "reasons": buy_reasons
        }

    if (
        sell >= 80
        and
        sell > buy
    ):

        return {
            "signal": "SELL",
            "score": sell,
            "price": price,
            "pattern": pattern,
            "reasons": sell_reasons
        }

    return {
        "signal": "WAIT",
        "score": max(
            buy,
            sell
        ),
        "price": price,
        "pattern": pattern,
        "reasons": []
    }


# =========================================================
# PRICE FORMAT
# =========================================================

def format_price(
    pair,
    price
):

    if "JPY" in pair:
        return f"{price:.3f}"

    return f"{price:.5f}"


# =========================================================
# SERVER TIME
# =========================================================

def get_server_now():

    try:

        if api is not None:
            return api.get_server_datetime()

    except Exception:
        pass

    return datetime.now(
        timezone.utc
    )


# =========================================================
# SEND SIGNAL
# =========================================================

def send_signal(
    pair,
    result,
    candle_time
):

    if result is None:
        return

    if result["signal"] == "WAIT":
        return

    # Candle close -> next candle entry
    entry_utc = (
        candle_time +
        timedelta(seconds=60)
    )

    expiry_utc = (
        entry_utc +
        timedelta(seconds=60)
    )

    entry_ist = entry_utc.astimezone(
        IST
    )

    expiry_ist = expiry_utc.astimezone(
        IST
    )

    signal_key = (
        pair,
        entry_utc.strftime(
            "%Y-%m-%d %H:%M"
        )
    )

    if signal_key in last_signal:
        return

    last_signal[signal_key] = True

    reasons = result.get(
        "reasons",
        []
    )

    if reasons:

        reason_text = "\n".join(
            "• " + x
            for x in reasons
        )

    else:

        reason_text = "• Multiple confirmations"

    message = (
        "🟠 REAL OTC — M1 SIGNAL\n\n"
        f"PAIR: {pair}\n"
        f"SIGNAL: {result['signal']}\n"
        f"ENTRY PRICE: "
        f"{format_price(pair, result['price'])}\n\n"
        "TIMEFRAME: 1 MINUTE\n"
        f"ENTRY TIME: "
        f"{entry_ist.strftime('%H:%M:%S')} IST\n"
        f"EXPIRY TIME: "
        f"{expiry_ist.strftime('%H:%M:%S')} IST\n\n"
        f"MODEL SCORE: "
        f"{result['score']}/100\n"
        f"PATTERN: {result['pattern']}\n\n"
        "CONFIRMATION:\n"
        f"{reason_text}\n\n"
        "⚠️ Score is a model score, "
        "not a guaranteed win probability."
    )

    print(
        "\n" +
        message +
        "\n"
    )

    send_telegram(
        message
    )


# =========================================================
# LOAD HISTORY
# =========================================================

def load_pair_history(
    pair
):

    print(
        "Loading:",
        pair
    )

    try:

        candles = api.get_historical_candles(
            pair,
            period=TIMEFRAME,
            offset=45000,
            count_request=1
        )

        df = normalize_candles(
            candles
        )

        if df is None:

            print(
                pair,
                "history unavailable"
            )

            return None

        # Remove forming candle
        if len(df) > 1:
            df = df.iloc[:-1].copy()

        print(
            pair,
            "history:",
            len(df),
            "candles"
        )

        return df

    except Exception as e:

        print(
            pair,
            "history error:",
            e
        )

        return None


# =========================================================
# LIVE TICK PROCESSOR
# =========================================================

def monitor_pair(
    pair,
    history_df
):

    candles = history_df.copy()

    current = None

    previous_minute = None

    print(
        "Monitoring:",
        pair
    )

    while True:

        try:

            ticks = api.get_realtime_ticks(
                pair,
                limit=20
            )

            if not ticks:

                time.sleep(0.5)
                continue

            tick = ticks[-1]

            if isinstance(
                tick,
                (list, tuple)
            ):

                ts = tick[0]
                price = tick[1]

            elif isinstance(
                tick,
                dict
            ):

                ts = (
                    tick.get("timestamp")
                    or tick.get("time")
                    or tick.get("tick_time")
                )

                price = (
                    tick.get("price")
                    or tick.get("value")
                )

            else:

                time.sleep(0.5)
                continue

            ts = float(ts)
            price = float(price)

            if ts > 100000000000:
                ts /= 1000

            tick_dt = datetime.fromtimestamp(
                ts,
                tz=timezone.utc
            )

            minute = tick_dt.replace(
                second=0,
                microsecond=0
            )

            # First live candle
            if current is None:

                current = {
                    "time": minute,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price
                }

                previous_minute = minute

                time.sleep(0.5)
                continue

            # Same minute
            if minute == current["time"]:

                current["high"] = max(
                    current["high"],
                    price
                )

                current["low"] = min(
                    current["low"],
                    price
                )

                current["close"] = price

            # New minute
            elif minute > current["time"]:

                completed = current.copy()

                candles = pd.concat(
                    [
                        candles,
                        pd.DataFrame(
                            [completed]
                        )
                    ],
                    ignore_index=True
                )

                candles = candles.tail(
                    150
                ).reset_index(
                    drop=True
                )

                result = analyze(
                    candles
                )

                # Entry is the NEW candle
                send_signal(
                    pair,
                    result,
                    completed["time"]
                )

                # New candle
                current = {
                    "time": minute,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price
                }

                previous_minute = minute

                print(
                    pair,
                    "new M1:",
                    minute.astimezone(
                        IST
                    ).strftime(
                        "%H:%M:%S"
                    ),
                    format_price(
                        pair,
                        price
                    )
                )

            time.sleep(0.5)

        except Exception as e:

            print(
                pair,
                "monitor error:",
                e
            )

            time.sleep(3)


# =========================================================
# CONNECT POCKET OPTION
# =========================================================

def connect_pocket():

    global api

    print(
        "Connecting to Pocket Option..."
    )

    # IMPORTANT:
    # The SSID itself contains isDemo.
    # Real account SSID must contain:
    # "isDemo":0

    api = PocketOption(
        PO_SSID
    )

    ok, error = api.connect()

    if not ok:

        print(
            "Pocket Option connection failed:",
            error
        )

        return False

    print(
        "Pocket Option WebSocket connected"
    )

    # Wait for time sync
    for _ in range(300):

        try:

            if (
                api.check_connect()
                and
                api.is_time_synced()
            ):
                print(
                    "Pocket Option time synchronized"
                )
                return True

        except Exception:
            pass

        time.sleep(0.1)

    print(
        "Time synchronization timeout"
    )

    return False


# =========================================================
# START OTC
# =========================================================

def start_otc():

    if not connect_pocket():
        return

    # Subscribe to OTC streams
    for pair in OTC_PAIRS:

        try:

            api.subscribe(
                pair,
                period=TIMEFRAME
            )

            print(
                "Subscribed:",
                pair
            )

            time.sleep(1)

        except Exception as e:

            print(
                "Subscribe error:",
                pair,
                e
            )

    # Load history
    histories = {}

    for pair in OTC_PAIRS:

        histories[pair] = (
            load_pair_history(
                pair
            )
        )

        time.sleep(1)

    # Start monitoring
    threads = []

    for pair in OTC_PAIRS:

        df = histories.get(
            pair
        )

        if df is None:
            continue

        thread = threading.Thread(
            target=monitor_pair,
            args=(
                pair,
                df
            ),
            daemon=True
        )

        thread.start()

        threads.append(
            thread
        )

    print(
        "REAL OTC monitoring started"
    )

    # Keep process alive
    while True:

        time.sleep(30)

        try:

            if not api.check_connect():

                print(
                    "Pocket Option disconnected"
                )

                break

        except Exception:
            break


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "=" * 55
    )

    print(
        "POCKET OPTION REAL OTC TELEGRAM SIGNAL BOT"
    )

    print(
        "=" * 55
    )

    if not check_config():
        return

    # Render health server
    threading.Thread(
        target=start_health_server,
        daemon=True
    ).start()

    start_otc()


if __name__ == "__main__":
    main()
