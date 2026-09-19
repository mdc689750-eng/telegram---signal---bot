import os
import time
import threading
import logging
from datetime import datetime, timezone, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
import pandas as pd

from pocketoptionapi import PocketOption


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("PO_SIGNAL_BOT")


# =========================================================
# ENVIRONMENT
# =========================================================

PO_SSID = os.getenv("PO_SSID", "").strip()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

PORT = int(os.getenv("PORT", "10000"))

TIMEFRAME = 60
HISTORY_OFFSET = 45000
HISTORY_REQUESTS = 3

MIN_CANDLES = 60
MIN_SCORE = 80

IST = timezone(timedelta(hours=5, minutes=30))

# Desired OTC pairs.
# The bot will automatically remove unavailable pairs.
REQUESTED_OTC_PAIRS = [
    "EURUSD_otc",
    "GBPUSD_otc",
    "USDJPY_otc",
    "EURJPY_otc",
    "AUDUSD_otc",
    "AUDNZD_otc",
]


# =========================================================
# GLOBAL STATE
# =========================================================

api = None

last_signal_candle = {}
last_tick_log = {}

stop_event = threading.Event()


# =========================================================
# HEALTH SERVER FOR RENDER
# =========================================================

class HealthHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        body = b"Pocket Option signal bot is running."

        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def start_health_server():
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)

    log.info("Health server running on port %s", PORT)

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True
    )

    thread.start()


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(message):

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials missing.")
        return False

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_TOKEN}/sendMessage"
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
            log.info("Telegram signal sent.")
            return True

        log.error(
            "Telegram error: %s %s",
            response.status_code,
            response.text[:300]
        )

    except Exception as e:
        log.error("Telegram exception: %s", e)

    return False


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
        log.error(
            "Missing environment variables: %s",
            ", ".join(missing)
        )
        return False

    log.info("Environment variables OK.")
    return True


# =========================================================
# CANDLE NORMALIZER
# =========================================================

def normalize_candles(raw):

    if not raw:
        return pd.DataFrame()

    rows = []

    for item in raw:

        if isinstance(item, dict):

            timestamp = (
                item.get("timestamp")
                or item.get("time")
                or item.get("from")
                or item.get("at")
            )

            row = {
                "timestamp": timestamp,
                "open": item.get("open"),
                "high": item.get("high"),
                "low": item.get("low"),
                "close": item.get("close"),
                "volume": item.get("volume", 0),
            }

            rows.append(row)

        elif isinstance(item, (list, tuple)) and len(item) >= 5:

            rows.append({
                "timestamp": item[0],
                "open": item[1],
                "high": item[2],
                "low": item[3],
                "close": item[4],
                "volume": item[5] if len(item) > 5 else 0,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    for column in [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "timestamp",
            "open",
            "high",
            "low",
            "close"
        ]
    )

    if df.empty:
        return df

    # Remove duplicate candles.
    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last"
    )

    df = df.sort_values("timestamp")

    # Remove currently forming candle.
    now_ts = int(time.time())
    current_bucket = now_ts - (now_ts % TIMEFRAME)

    df = df[
        df["timestamp"] < current_bucket
    ]

    return df.reset_index(drop=True)


# =========================================================
# TICK NORMALIZER
# =========================================================

def normalize_ticks(raw_ticks):

    result = []

    if not raw_ticks:
        return result

    for tick in raw_ticks:

        timestamp = None
        price = None

        if isinstance(tick, (list, tuple)):

            if len(tick) >= 2:
                timestamp = tick[0]
                price = tick[1]

        elif isinstance(tick, dict):

            timestamp = (
                tick.get("timestamp")
                or tick.get("time")
                or tick.get("at")
            )

            price = (
                tick.get("price")
                or tick.get("close")
                or tick.get("value")
            )

        if timestamp is None or price is None:
            continue

        try:
            timestamp = float(timestamp)
            price = float(price)
        except Exception:
            continue

        result.append(
            (timestamp, price)
        )

    result.sort(key=lambda x: x[0])

    return result


# =========================================================
# TICKS -> CANDLE
# =========================================================

def build_candle_from_ticks(ticks, candle_timestamp):

    prices = []

    for timestamp, price in ticks:

        bucket = int(timestamp) - (
            int(timestamp) % TIMEFRAME
        )

        if bucket == candle_timestamp:
            prices.append(price)

    if not prices:
        return None

    return {
        "timestamp": candle_timestamp,
        "open": prices[0],
        "high": max(prices),
        "low": min(prices),
        "close": prices[-1],
        "volume": len(prices)
    }


# =========================================================
# INDICATORS
# =========================================================

def add_indicators(df):

    df = df.copy()

    # EMA
    df["ema5"] = df["close"].ewm(
        span=5,
        adjust=False
    ).mean()

    df["ema10"] = df["close"].ewm(
        span=10,
        adjust=False
    ).mean()

    df["ema20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    # RSI 14
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)

    df["rsi"] = 100 - (
        100 / (1 + rs)
    )

    # MACD
    ema12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["macd_hist"] = (
        df["macd"] -
        df["macd_signal"]
    )

    # Bollinger Bands
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()

    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + (
        2 * bb_std
    )
    df["bb_lower"] = bb_mid - (
        2 * bb_std
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

def candle_pattern(df):

    if len(df) < 3:
        return "None"

    last = df.iloc[-1]
    prev = df.iloc[-2]

    body = abs(
        last["close"] - last["open"]
    )

    candle_range = (
        last["high"] - last["low"]
    )

    if candle_range <= 0:
        return "None"

    upper_wick = (
        last["high"] -
        max(last["open"], last["close"])
    )

    lower_wick = (
        min(last["open"], last["close"]) -
        last["low"]
    )

    # Bullish engulfing
    bullish_engulfing = (
        prev["close"] < prev["open"]
        and
        last["close"] > last["open"]
        and
        last["open"] <= prev["close"]
        and
        last["close"] >= prev["open"]
    )

    if bullish_engulfing:
        return "Bullish Engulfing"

    # Bearish engulfing
    bearish_engulfing = (
        prev["close"] > prev["open"]
        and
        last["close"] < last["open"]
        and
        last["open"] >= prev["close"]
        and
        last["close"] <= prev["open"]
    )

    if bearish_engulfing:
        return "Bearish Engulfing"

    # Hammer
    if (
        lower_wick >= body * 2
        and
        upper_wick <= body
    ):
        return "Hammer"

    # Shooting star
    if (
        upper_wick >= body * 2
        and
        lower_wick <= body
    ):
        return "Shooting Star"

    # Strong bullish candle
    if (
        last["close"] > last["open"]
        and
        body >= candle_range * 0.65
    ):
        return "Strong Bullish"

    # Strong bearish candle
    if (
        last["close"] < last["open"]
        and
        body >= candle_range * 0.65
    ):
        return "Strong Bearish"

    return "Normal"


# =========================================================
# SIGNAL ENGINE
# =========================================================

def analyze(df):

    if df is None or len(df) < MIN_CANDLES:
        return {
            "signal": "WAIT",
            "score": 0,
            "pattern": "Not enough candles",
            "reasons": []
        }

    df = add_indicators(df)

    row = df.iloc[-1]

    buy_score = 0
    sell_score = 0

    buy_reasons = []
    sell_reasons = []

    close = float(row["close"])

    # -----------------------------------------------------
    # EMA TREND
    # -----------------------------------------------------

    if (
        row["ema5"] > row["ema10"]
        and
        row["ema10"] > row["ema20"]
        and
        row["ema20"] > row["ema50"]
    ):
        buy_score += 20
        buy_reasons.append("EMA bullish trend")

    elif (
        row["ema5"] < row["ema10"]
        and
        row["ema10"] < row["ema20"]
        and
        row["ema20"] < row["ema50"]
    ):
        sell_score += 20
        sell_reasons.append("EMA bearish trend")

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    if pd.notna(row["rsi"]):

        if 30 <= row["rsi"] <= 50:
            buy_score += 15
            buy_reasons.append(
                f"RSI bullish zone {row['rsi']:.1f}"
            )

        elif 50 < row["rsi"] <= 70:
            sell_score += 5
            sell_reasons.append(
                f"RSI upper zone {row['rsi']:.1f}"
            )

        elif row["rsi"] < 30:
            buy_score += 20
            buy_reasons.append(
                f"RSI oversold {row['rsi']:.1f}"
            )

        elif row["rsi"] > 70:
            sell_score += 20
            sell_reasons.append(
                f"RSI overbought {row['rsi']:.1f}"
            )

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    if (
        row["macd"] > row["macd_signal"]
        and
        row["macd_hist"] > 0
    ):
        buy_score += 20
        buy_reasons.append("MACD bullish")

    elif (
        row["macd"] < row["macd_signal"]
        and
        row["macd_hist"] < 0
    ):
        sell_score += 20
        sell_reasons.append("MACD bearish")

    # -----------------------------------------------------
    # BOLLINGER
    # -----------------------------------------------------

    if pd.notna(row["bb_lower"]) and close <= row["bb_lower"]:

        buy_score += 15
        buy_reasons.append(
            "Lower Bollinger rejection"
        )

    if pd.notna(row["bb_upper"]) and close >= row["bb_upper"]:

        sell_score += 15
        sell_reasons.append(
            "Upper Bollinger rejection"
        )

    # -----------------------------------------------------
    # SUPPORT / RESISTANCE
    # -----------------------------------------------------

    if pd.notna(row["support"]):

        support_distance = abs(
            close - row["support"]
        )

        if (
            support_distance
            <= close * 0.0015
        ):
            buy_score += 10
            buy_reasons.append(
                "Near support"
            )

    if pd.notna(row["resistance"]):

        resistance_distance = abs(
            close - row["resistance"]
        )

        if (
            resistance_distance
            <= close * 0.0015
        ):
            sell_score += 10
            sell_reasons.append(
                "Near resistance"
            )

    # -----------------------------------------------------
    # CANDLE PATTERN
    # -----------------------------------------------------

    pattern = candle_pattern(df)

    if pattern in [
        "Bullish Engulfing",
        "Hammer",
        "Strong Bullish"
    ]:
        buy_score += 15
        buy_reasons.append(pattern)

    elif pattern in [
        "Bearish Engulfing",
        "Shooting Star",
        "Strong Bearish"
    ]:
        sell_score += 15
        sell_reasons.append(pattern)

    # -----------------------------------------------------
    # DECISION
    # -----------------------------------------------------

    if (
        buy_score >= MIN_SCORE
        and
        buy_score > sell_score
    ):
        return {
            "signal": "BUY",
            "score": min(buy_score, 100),
            "pattern": pattern,
            "reasons": buy_reasons
        }

    if (
        sell_score >= MIN_SCORE
        and
        sell_score > buy_score
    ):
        return {
            "signal": "SELL",
            "score": min(sell_score, 100),
            "pattern": pattern,
            "reasons": sell_reasons
        }

    return {
        "signal": "WAIT",
        "score": max(
            buy_score,
            sell_score
        ),
        "pattern": pattern,
        "reasons": []
    }


# =========================================================
# HISTORY
# =========================================================

def load_pair_history(pair):

    try:

        candles = api.get_historical_candles(
            pair,
            period=TIMEFRAME,
            offset=HISTORY_OFFSET,
            count_request=HISTORY_REQUESTS
        )

        df = normalize_candles(candles)

        log.info(
            "%s | history = %d candles",
            pair,
            len(df)
        )

        return df

    except Exception as e:

        log.error(
            "%s | history error: %s",
            pair,
            e
        )

        return pd.DataFrame()


# =========================================================
# AVAILABLE PAIRS
# =========================================================

def get_available_pairs():

    try:

        assets = api.get_assets()

        if not assets:
            log.error("Asset catalog is empty.")
            return []

        available = []

        for pair in REQUESTED_OTC_PAIRS:

            info = assets.get(pair)

            if not info:
                log.warning(
                    "%s | not found in asset catalog",
                    pair
                )
                continue

            is_available = info.get(
                "is_available",
                False
            )

            if is_available:

                available.append(pair)

                log.info(
                    "%s | AVAILABLE | payout=%s | timeframes=%s",
                    pair,
                    info.get("payout"),
                    info.get("timeframes")
                )

            else:

                log.warning(
                    "%s | currently unavailable",
                    pair
                )

        return available

    except Exception as e:

        log.error(
            "Asset catalog error: %s",
            e
        )

        return []


# =========================================================
# CONNECTION
# =========================================================

def connect_pocket():

    global api

    log.info("Connecting to Pocket Option...")

    try:

        api = PocketOption(PO_SSID)

        ok, error = api.connect()

        if not ok:

            log.error(
                "Pocket Option connection failed: %s",
                error
            )

            return False

        log.info(
            "Pocket Option WebSocket connected."
        )

        # Wait for connection + server time sync.
        for _ in range(300):

            if (
                api.check_connect()
                and
                api.is_time_synced()
            ):

                log.info(
                    "Pocket Option time synchronized."
                )

                try:
                    server_time = (
                        api.get_server_datetime()
                    )

                    log.info(
                        "Pocket Option server time: %s",
                        server_time
                    )

                except Exception:
                    pass

                return True

            time.sleep(0.2)

        log.error(
            "Connection established but time synchronization timed out."
        )

        return False

    except Exception as e:

        log.exception(
            "Pocket Option connection exception: %s",
            e
        )

        return False


# =========================================================
# SEND SIGNAL
# =========================================================

def send_signal(
    pair,
    result,
    entry_price,
    candle_timestamp
):

    signal = result["signal"]

    if signal not in [
        "BUY",
        "SELL"
    ]:
        return

    # One signal per candle.
    if last_signal_candle.get(pair) == candle_timestamp:

        log.info(
            "%s | duplicate signal skipped",
            pair
        )

        return

    last_signal_candle[pair] = candle_timestamp

    entry_timestamp = (
        candle_timestamp + TIMEFRAME
    )

    expiry_timestamp = (
        entry_timestamp + TIMEFRAME
    )

    entry_dt = datetime.fromtimestamp(
        entry_timestamp,
        tz=timezone.utc
    ).astimezone(IST)

    expiry_dt = datetime.fromtimestamp(
        expiry_timestamp,
        tz=timezone.utc
    ).astimezone(IST)

    reasons = result.get(
        "reasons",
        []
    )

    reason_text = ", ".join(
        reasons[:4]
    )

    message = (
        "📊 POCKET OPTION OTC SIGNAL\n"
        "\n"
        f"PAIR: {pair}\n"
        f"SIGNAL: {signal}\n"
        f"MODEL SCORE: {result['score']}/100\n"
        f"ENTRY PRICE: {entry_price:.6f}\n"
        "\n"
        f"ENTRY: {entry_dt.strftime('%H:%M:%S')} IST\n"
        f"EXPIRY: {expiry_dt.strftime('%H:%M:%S')} IST\n"
        "\n"
        f"PATTERN: {result['pattern']}\n"
        f"CONFIRMATION: {reason_text}\n"
        "\n"
        "TIMEFRAME: 1 MIN\n"
        "MODE: OTC\n"
        "\n"
        "⚠️ Signal/data only. No automatic trade."
    )

    send_telegram(message)

    log.info(
        "%s | %s | score=%s | entry=%s",
        pair,
        signal,
        result["score"],
        entry_price
    )


# =========================================================
# MONITOR ONE PAIR
# =========================================================

def monitor_pair(pair):

    log.info(
        "%s | monitor started",
        pair
    )

    history = load_pair_history(pair)

    if len(history) < MIN_CANDLES:

        log.error(
            "%s | only %d candles. Need at least %d.",
            pair,
            len(history),
            MIN_CANDLES
        )

        return

    log.info(
        "%s | history ready: %d candles",
        pair,
        len(history)
    )

    current_candle_timestamp = None
    current_candle = None

    while not stop_event.is_set():

        try:

            if not api.check_connect():

                log.warning(
                    "%s | websocket disconnected",
                    pair
                )

                time.sleep(2)
                continue

            ticks_raw = api.get_realtime_ticks(
                pair,
                limit=200
            )

            ticks = normalize_ticks(
                ticks_raw
            )

            if not ticks:

                time.sleep(1)
                continue

            latest_timestamp, latest_price = ticks[-1]

            # Log latest price once every 10 seconds.
            now = time.time()

            if (
                now -
                last_tick_log.get(pair, 0)
                >= 10
            ):

                last_tick_log[pair] = now

                log.info(
                    "%s | LIVE price=%s | tick=%s",
                    pair,
                    latest_price,
                    latest_timestamp
                )

            bucket = (
                int(latest_timestamp)
                -
                (
                    int(latest_timestamp)
                    % TIMEFRAME
                )
            )

            # First candle.
            if current_candle_timestamp is None:

                current_candle_timestamp = bucket

                current_candle = (
                    build_candle_from_ticks(
                        ticks,
                        bucket
                    )
                )

                time.sleep(1)
                continue

            # Same candle -> update.
            if bucket == current_candle_timestamp:

                new_candle = (
                    build_candle_from_ticks(
                        ticks,
                        bucket
                    )
                )

                if new_candle:
                    current_candle = new_candle

                time.sleep(1)
                continue

            # NEW CANDLE STARTED.
            # Previous candle is now closed.

            closed_candle = current_candle

            if closed_candle:

                history = pd.concat(
                    [
                        history,
                        pd.DataFrame(
                            [closed_candle]
                        )
                    ],
                    ignore_index=True
                )

                history = history.drop_duplicates(
                    subset=["timestamp"],
                    keep="last"
                )

                history = history.sort_values(
                    "timestamp"
                )

                # Keep enough history.
                if len(history) > 500:

                    history = history.tail(
                        500
                    ).reset_index(drop=True)

                log.info(
                    "%s | NEW M1 candle | closed=%s | close=%s",
                    pair,
                    closed_candle["timestamp"],
                    closed_candle["close"]
                )

                result = analyze(history)

                log.info(
                    "%s | result=%s | score=%s | pattern=%s",
                    pair,
                    result["signal"],
                    result["score"],
                    result["pattern"]
                )

                if result["signal"] in [
                    "BUY",
                    "SELL"
                ]:

                    send_signal(
                        pair,
                        result,
                        float(
                            closed_candle["close"]
                        ),
                        int(
                            closed_candle["timestamp"]
                        )
                    )

            # Start new candle.
            current_candle_timestamp = bucket

            current_candle = (
                build_candle_from_ticks(
                    ticks,
                    bucket
                )
            )

            time.sleep(1)

        except Exception as e:

            log.exception(
                "%s | monitor error: %s",
                pair,
                e
            )

            time.sleep(3)


# =========================================================
# START OTC MONITORING
# =========================================================

def start_otc():

    global api

    while not stop_event.is_set():

        # -------------------------------------------------
        # CONNECT
        # -------------------------------------------------

        if not api or not api.check_connect():

            if not connect_pocket():

                log.error(
                    "Connection failed. Retry in 15 seconds."
                )

                time.sleep(15)
                continue

        # -------------------------------------------------
        # GET LIVE AVAILABLE PAIRS
        # -------------------------------------------------

        pairs = get_available_pairs()

        if not pairs:

            log.error(
                "No requested OTC pair is currently available."
            )

            time.sleep(30)
            continue

        log.info(
            "Available OTC pairs: %s",
            ", ".join(pairs)
        )

        # -------------------------------------------------
        # SUBSCRIBE
        # -------------------------------------------------

        for pair in pairs:

            try:

                api.subscribe(
                    pair,
                    period=TIMEFRAME
                )

                log.info(
                    "%s | subscribed M1",
                    pair
                )

            except Exception as e:

                log.error(
                    "%s | subscribe error: %s",
                    pair,
                    e
                )

        # -------------------------------------------------
        # START MONITORS
        # -------------------------------------------------

        threads = []

        for pair in pairs:

            thread = threading.Thread(
                target=monitor_pair,
                args=(pair,),
                daemon=True
            )

            thread.start()

            threads.append(thread)

            # Small delay prevents all subscriptions
            # from hitting the server at exactly once.
            time.sleep(0.5)

        # -------------------------------------------------
        # KEEP MAIN LOOP ALIVE
        # -------------------------------------------------

        while not stop_event.is_set():

            if not api.check_connect():

                log.warning(
                    "Pocket Option disconnected. Reconnecting..."
                )

                break

            time.sleep(5)

        # Give old monitor threads time to notice disconnect.
        time.sleep(3)


# =========================================================
# MAIN
# =========================================================

def main():

    log.info(
        "=========================================="
    )

    log.info(
        "Pocket Option OTC Signal Bot Starting"
    )

    log.info(
        "Mode: SIGNAL/DATA ONLY"
    )

    log.info(
        "Timeframe: 1 Minute"
    )

    log.info(
        "Minimum model score: %s",
        MIN_SCORE
    )

    log.info(
        "=========================================="
    )

    start_health_server()

    if not check_config():

        log.error(
            "Configuration incomplete. Bot stopped."
        )

        while True:
            time.sleep(60)

    start_otc()


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        stop_event.set()

        log.info(
            "Bot stopped."
        )

    except Exception as e:

        log.exception(
            "Fatal error: %s",
            e
        )

        stop_event.set()
