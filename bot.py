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
# CONFIG
# =========================================================

PO_SSID = os.getenv("PO_SSID", "").strip()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

PORT = int(os.getenv("PORT", "10000"))

TIMEFRAME = 60
MIN_CANDLES = 60
MIN_SCORE = 80

HISTORY_OFFSET = 45000
HISTORY_REQUESTS = 3

IST = timezone(timedelta(hours=5, minutes=30))

REQUESTED_PAIRS = [
    "EURUSD_otc",
    "GBPUSD_otc",
    "USDJPY_otc",
    "EURJPY_otc",
    "AUDUSD_otc",
    "AUDNZD_otc",
]

# Optional Render variable:
# OTC_PAIRS=EURUSD_otc,GBPUSD_otc,AUDNZD_otc
custom_pairs = os.getenv("OTC_PAIRS", "").strip()
if custom_pairs:
    REQUESTED_PAIRS = [
        p.strip() for p in custom_pairs.split(",") if p.strip()
    ]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

log = logging.getLogger("PO_SIGNAL_BOT")

api = None
stop_event = threading.Event()

# One signal maximum for each pair/candle.
last_signal_candle = {}

# Avoid printing live tick every second.
last_price_log = {}


# =========================================================
# RENDER HEALTH
# =========================================================

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"Pocket Option OTC signal bot is running."
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


def start_health_server():
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    log.info("Health server running on port %s", PORT)

    threading.Thread(
        target=server.serve_forever,
        daemon=True
    ).start()


# =========================================================
# TELEGRAM
# =========================================================

def telegram_send(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram variables are missing.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    try:
        response = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
            },
            timeout=15,
        )

        if response.ok:
            log.info("Telegram message sent.")
            return True

        log.error(
            "Telegram HTTP %s: %s",
            response.status_code,
            response.text[:300],
        )

    except Exception as exc:
        log.error("Telegram error: %s", exc)

    return False


# =========================================================
# HELPERS
# =========================================================

def to_seconds(value):
    try:
        value = float(value)
        # Convert milliseconds to seconds if necessary.
        if value > 10_000_000_000:
            value /= 1000.0
        return value
    except Exception:
        return None


def current_bucket(timestamp):
    timestamp = to_seconds(timestamp)
    if timestamp is None:
        return None
    return int(timestamp // TIMEFRAME) * TIMEFRAME


def format_ist(timestamp):
    dt = datetime.fromtimestamp(
        int(timestamp),
        tz=timezone.utc,
    ).astimezone(IST)

    return dt.strftime("%d-%m-%Y %H:%M:%S IST")


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
            ", ".join(missing),
        )
        return False

    log.info("Environment variables OK.")
    return True


# =========================================================
# CANDLE NORMALIZATION
# =========================================================

def normalize_candles(raw):
    if raw is None:
        return pd.DataFrame()

    try:
        if isinstance(raw, pd.DataFrame):
            df = raw.copy()

        elif isinstance(raw, dict):
            if isinstance(raw.get("data"), list):
                df = pd.DataFrame(raw["data"])
            elif isinstance(raw.get("candles"), list):
                df = pd.DataFrame(raw["candles"])
            else:
                df = pd.DataFrame([raw])

        else:
            df = pd.DataFrame(raw)

        if df.empty:
            return pd.DataFrame()

        df.columns = [str(c).lower().strip() for c in df.columns]

        aliases = {
            "time": "timestamp",
            "ts": "timestamp",
            "from": "timestamp",
            "at": "timestamp",
            "open_price": "open",
            "openprice": "open",
            "o": "open",
            "high_price": "high",
            "highprice": "high",
            "h": "high",
            "low_price": "low",
            "lowprice": "low",
            "l": "low",
            "close_price": "close",
            "closeprice": "close",
            "c": "close",
            "v": "volume",
        }

        for old, new in aliases.items():
            if old in df.columns and new not in df.columns:
                df.rename(columns={old: new}, inplace=True)

        required = ["timestamp", "open", "high", "low", "close"]

        if not all(c in df.columns for c in required):
            log.warning(
                "Candle columns unavailable: %s",
                list(df.columns),
            )
            return pd.DataFrame()

        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "volume" not in df.columns:
            df["volume"] = 0

        df["volume"] = pd.to_numeric(
            df["volume"],
            errors="coerce",
        ).fillna(0)

        df["timestamp"] = df["timestamp"].apply(to_seconds)

        df.dropna(subset=required, inplace=True)

        if df.empty:
            return df

        df.drop_duplicates(
            subset=["timestamp"],
            keep="last",
            inplace=True,
        )

        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Remove current/forming candle.
        now_bucket = current_bucket(time.time())
        if now_bucket is not None:
            df = df[df["timestamp"] < now_bucket]

        return df.reset_index(drop=True)

    except Exception as exc:
        log.error("normalize_candles error: %s", exc)
        return pd.DataFrame()


# =========================================================
# TICK NORMALIZATION
# =========================================================

def normalize_ticks(raw):
    result = []

    if raw is None:
        return result

    try:
        iterable = raw.to_dict("records") if isinstance(raw, pd.DataFrame) else raw

        for item in iterable:
            ts = None
            price = None

            if isinstance(item, dict):
                ts = (
                    item.get("timestamp")
                    or item.get("time")
                    or item.get("ts")
                    or item.get("at")
                )
                price = (
                    item.get("price")
                    or item.get("value")
                    or item.get("close")
                )

            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                ts = item[0]
                price = item[1]

            if ts is None or price is None:
                continue

            ts = to_seconds(ts)
            price = float(price)

            if ts is None:
                continue

            result.append((ts, price))

    except Exception as exc:
        log.error("normalize_ticks error: %s", exc)

    result.sort(key=lambda x: x[0])
    return result


# =========================================================
# TICKS -> CLOSED CANDLE
# =========================================================

def make_candle_from_ticks(ticks, bucket):
    prices = [
        float(price)
        for ts, price in ticks
        if current_bucket(ts) == bucket
    ]

    if not prices:
        return None

    return {
        "timestamp": int(bucket),
        "open": prices[0],
        "high": max(prices),
        "low": min(prices),
        "close": prices[-1],
        "volume": len(prices),
    }


# =========================================================
# INDICATORS
# =========================================================

def add_indicators(df):
    df = df.copy()
    close = df["close"]

    df["ema5"] = close.ewm(span=5, adjust=False).mean()
    df["ema10"] = close.ewm(span=10, adjust=False).mean()
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    df["rsi"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False,
    ).mean()

    df["macd_hist"] = (
        df["macd"] - df["macd_signal"]
    )

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + (2 * bb_std)
    df["bb_lower"] = bb_mid - (2 * bb_std)

    df["support"] = df["low"].rolling(20).min()
    df["resistance"] = df["high"].rolling(20).max()

    return df


# =========================================================
# CANDLE PATTERN
# =========================================================

def candle_pattern(df):
    if len(df) < 2:
        return "None"

    prev = df.iloc[-2]
    cur = df.iloc[-1]

    body = abs(cur["close"] - cur["open"])
    upper = cur["high"] - max(cur["open"], cur["close"])
    lower = min(cur["open"], cur["close"]) - cur["low"]

    if (
        prev["close"] < prev["open"]
        and cur["close"] > cur["open"]
        and cur["open"] <= prev["close"]
        and cur["close"] >= prev["open"]
    ):
        return "Bullish Engulfing"

    if (
        prev["close"] > prev["open"]
        and cur["close"] < cur["open"]
        and cur["open"] >= prev["close"]
        and cur["close"] <= prev["open"]
    ):
        return "Bearish Engulfing"

    if body > 0 and lower >= body * 2 and upper <= body:
        return "Hammer"

    if body > 0 and upper >= body * 2 and lower <= body:
        return "Shooting Star"

    return "None"


# =========================================================
# SIGNAL ENGINE
# =========================================================

def analyze(df):
    if df is None or len(df) < MIN_CANDLES:
        return {
            "signal": "WAIT",
            "score": 0,
            "pattern": "Not enough candles",
            "reasons": [],
        }

    df = add_indicators(df)
    cur = df.iloc[-1]

    buy = 0
    sell = 0

    buy_reasons = []
    sell_reasons = []

    close = float(cur["close"])

    # EMA trend
    if (
        cur["ema5"] > cur["ema10"]
        and cur["ema10"] > cur["ema20"]
        and cur["ema20"] > cur["ema50"]
    ):
        buy += 20
        buy_reasons.append("EMA bullish trend")

    elif (
        cur["ema5"] < cur["ema10"]
        and cur["ema10"] < cur["ema20"]
        and cur["ema20"] < cur["ema50"]
    ):
        sell += 20
        sell_reasons.append("EMA bearish trend")

    # RSI
    if pd.notna(cur["rsi"]):
        rsi = float(cur["rsi"])

        if rsi < 30:
            buy += 20
            buy_reasons.append(f"RSI oversold {rsi:.1f}")

        elif 30 <= rsi <= 50:
            buy += 10
            buy_reasons.append(f"RSI bullish zone {rsi:.1f}")

        elif rsi > 70:
            sell += 20
            sell_reasons.append(f"RSI overbought {rsi:.1f}")

        elif 50 < rsi <= 70:
            sell += 10
            sell_reasons.append(f"RSI upper zone {rsi:.1f}")

    # MACD
    if (
        cur["macd"] > cur["macd_signal"]
        and cur["macd_hist"] > 0
    ):
        buy += 20
        buy_reasons.append("MACD bullish")

    elif (
        cur["macd"] < cur["macd_signal"]
        and cur["macd_hist"] < 0
    ):
        sell += 20
        sell_reasons.append("MACD bearish")

    # Bollinger
    if pd.notna(cur["bb_lower"]) and close <= cur["bb_lower"]:
        buy += 10
        buy_reasons.append("Near lower Bollinger")

    if pd.notna(cur["bb_upper"]) and close >= cur["bb_upper"]:
        sell += 10
        sell_reasons.append("Near upper Bollinger")

    # Support / resistance
    if pd.notna(cur["support"]):
        if abs(close - cur["support"]) <= close * 0.0015:
            buy += 10
            buy_reasons.append("Near support")

    if pd.notna(cur["resistance"]):
        if abs(cur["resistance"] - close) <= close * 0.0015:
            sell += 10
            sell_reasons.append("Near resistance")

    # Candle
    pattern = candle_pattern(df)

    if pattern in ("Bullish Engulfing", "Hammer"):
        buy += 15
        buy_reasons.append(pattern)

    elif pattern in ("Bearish Engulfing", "Shooting Star"):
        sell += 15
        sell_reasons.append(pattern)

    if buy >= MIN_SCORE and buy > sell:
        return {
            "signal": "BUY",
            "score": min(buy, 100),
            "pattern": pattern,
            "reasons": buy_reasons,
        }

    if sell >= MIN_SCORE and sell > buy:
        return {
            "signal": "SELL",
            "score": min(sell, 100),
            "pattern": pattern,
            "reasons": sell_reasons,
        }

    return {
        "signal": "WAIT",
        "score": max(buy, sell),
        "pattern": pattern,
        "reasons": [],
    }


# =========================================================
# HISTORY
# =========================================================

def load_pair_history(pair):
    for attempt in range(1, 4):
        try:
            log.info(
                "%s | requesting history attempt %d/3",
                pair,
                attempt,
            )

            raw = api.get_historical_candles(
                pair,
                period=TIMEFRAME,
                offset=HISTORY_OFFSET,
                count_request=HISTORY_REQUESTS,
            )

            df = normalize_candles(raw)

            log.info(
                "%s | history received: %d closed candles",
                pair,
                len(df),
            )

            if len(df) >= MIN_CANDLES:
                return df.tail(500).reset_index(drop=True)

        except Exception as exc:
            log.error(
                "%s | history attempt %d error: %s",
                pair,
                attempt,
                exc,
            )

        time.sleep(2)

    log.warning(
        "%s | historical data unavailable; "
        "bot will warm up from live M1 candles.",
        pair,
    )

    return pd.DataFrame(
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )


# =========================================================
# AVAILABLE PAIRS
# =========================================================

def get_available_pairs():
    try:
        assets = api.get_assets()

        if not assets:
            log.error("Asset catalog is empty.")
            return []

        result = []

        for pair in REQUESTED_PAIRS:
            info = assets.get(pair)

            if not info:
                log.warning("%s | not in live asset catalog", pair)
                continue

            available = bool(info.get("is_available"))

            if available:
                result.append(pair)

                log.info(
                    "%s | AVAILABLE | payout=%s | timeframes=%s",
                    pair,
                    info.get("payout"),
                    info.get("timeframes"),
                )
            else:
                log.warning(
                    "%s | currently unavailable",
                    pair,
                )

        return result

    except Exception as exc:
        log.error("get_assets error: %s", exc)
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
                error,
            )
            return False

        log.info("Pocket Option WebSocket connected.")

        for _ in range(300):
            if api.check_connect() and api.is_time_synced():
                log.info("Pocket Option time synchronized.")

                try:
                    log.info(
                        "Pocket Option server time: %s",
                        api.get_server_datetime(),
                    )
                except Exception:
                    pass

                return True

            time.sleep(0.2)

        log.error("Time synchronization timeout.")
        return False

    except Exception as exc:
        log.exception(
            "Pocket Option connection exception: %s",
            exc,
        )
        return False


# =========================================================
# SIGNAL MESSAGE
# =========================================================

def send_signal(pair, result, candle):
    signal = result["signal"]

    if signal not in ("BUY", "SELL"):
        return

    candle_ts = int(candle["timestamp"])

    # One signal per closed candle.
    if last_signal_candle.get(pair) == candle_ts:
        log.info(
            "%s | duplicate signal skipped for candle %s",
            pair,
            candle_ts,
        )
        return

    last_signal_candle[pair] = candle_ts

    entry_ts = candle_ts + TIMEFRAME
    expiry_ts = entry_ts + TIMEFRAME

    reasons = ", ".join(
        result.get("reasons", [])[:4]
    )

    message = (
        "📊 POCKET OPTION OTC SIGNAL\n\n"
        f"PAIR: {pair}\n"
        f"SIGNAL: {signal}\n"
        f"MODEL SCORE: {result['score']}/100\n"
        f"ENTRY PRICE: {float(candle['close']):.6f}\n\n"
        f"ENTRY: {format_ist(entry_ts)}\n"
        f"EXPIRY: {format_ist(expiry_ts)}\n"
        "TIMEFRAME: 1 MIN\n"
        f"PATTERN: {result['pattern']}\n"
        f"CONFIRMATION: {reasons}\n\n"
        "⚠️ Signal/data only. No automatic trade."
    )

    telegram_send(message)


# =========================================================
# MAIN MONITOR - FIXED M1 ROLLOVER
# =========================================================

def monitor_pair(pair):
    log.info("%s | monitor started", pair)

    history = load_pair_history(pair)

    # Subscribe again before reading ticks.
    try:
        api.subscribe(pair, period=TIMEFRAME)
        log.info("%s | M1 subscription active", pair)
    except Exception as exc:
        log.error("%s | subscribe error: %s", pair, exc)

    active_bucket = None
    active_ticks = []

    while not stop_event.is_set():

        try:
            if not api.check_connect():
                log.warning(
                    "%s | disconnected; waiting for reconnect",
                    pair,
                )
                time.sleep(2)
                continue

            raw_ticks = api.get_realtime_ticks(
                pair,
                limit=500,
            )

            ticks = normalize_ticks(raw_ticks)

            if not ticks:
                log.warning(
                    "%s | no realtime ticks yet",
                    pair,
                )
                time.sleep(2)
                continue

            latest_ts, latest_price = ticks[-1]
            bucket = current_bucket(latest_ts)

            if bucket is None:
                time.sleep(1)
                continue

            # Price diagnostic every 10 seconds.
            now = time.time()

            if now - last_price_log.get(pair, 0) >= 10:
                last_price_log[pair] = now

                log.info(
                    "%s | LIVE price=%s | tick=%s | bucket=%s",
                    pair,
                    latest_price,
                    latest_ts,
                    bucket,
                )

            # -----------------------------------------
            # FIRST M1 BUCKET
            # -----------------------------------------

            if active_bucket is None:
                active_bucket = bucket

                active_ticks = [
                    t for t in ticks
                    if current_bucket(t[0]) == bucket
                ]

                log.info(
                    "%s | M1 tracking started | bucket=%s",
                    pair,
                    active_bucket,
                )

                time.sleep(1)
                continue

            # -----------------------------------------
            # SAME M1
            # -----------------------------------------

            if bucket == active_bucket:

                active_ticks = [
                    t for t in ticks
                    if current_bucket(t[0]) == active_bucket
                ]

                time.sleep(1)
                continue

            # -----------------------------------------
            # ONE OR MORE NEW M1 CANDLES
            # -----------------------------------------

            if bucket > active_bucket:

                previous_bucket = active_bucket

                log.info(
                    "%s | NEW M1 CANDLE DETECTED | old=%s | new=%s",
                    pair,
                    previous_bucket,
                    bucket,
                )

                closed = make_candle_from_ticks(
                    active_ticks,
                    previous_bucket,
                )

                if closed is not None:

                    log.info(
                        "%s | CLOSED M1 | "
                        "O=%s H=%s L=%s C=%s",
                        pair,
                        closed["open"],
                        closed["high"],
                        closed["low"],
                        closed["close"],
                    )

                    history = pd.concat(
                        [
                            history,
                            pd.DataFrame([closed]),
                        ],
                        ignore_index=True,
                    )

                    history.drop_duplicates(
                        subset=["timestamp"],
                        keep="last",
                        inplace=True,
                    )

                    history.sort_values(
                        "timestamp",
                        inplace=True,
                    )

                    history = history.tail(
                        500
                    ).reset_index(drop=True)

                    # ---------------------------------
                    # ANALYZE
                    # ---------------------------------

                    if len(history) < MIN_CANDLES:

                        log.info(
                            "%s | warming up: %d/%d candles",
                            pair,
                            len(history),
                            MIN_CANDLES,
                        )

                    else:

                        result = analyze(history)

                        log.info(
                            "%s | SIGNAL=%s | SCORE=%s | PATTERN=%s",
                            pair,
                            result["signal"],
                            result["score"],
                            result["pattern"],
                        )

                        if result["signal"] in (
                            "BUY",
                            "SELL",
                        ):
                            send_signal(
                                pair,
                                result,
                                closed,
                            )

                else:

                    log.warning(
                        "%s | closed candle had no tick data",
                        pair,
                    )

                # ---------------------------------
                # Start current M1
                # ---------------------------------

                active_bucket = bucket

                active_ticks = [
                    t for t in ticks
                    if current_bucket(t[0]) == bucket
                ]

                log.info(
                    "%s | now tracking new M1 bucket=%s",
                    pair,
                    active_bucket,
                )

            time.sleep(1)

        except Exception as exc:

            log.exception(
                "%s | monitor error: %s",
                pair,
                exc,
            )

            time.sleep(3)


# =========================================================
# START
# =========================================================

def start_otc():
    while not stop_event.is_set():

        if api is None or not api.check_connect():

            if not connect_pocket():
                log.error(
                    "Connection failed. Retrying in 15 seconds."
                )
                time.sleep(15)
                continue

        pairs = get_available_pairs()

        if not pairs:
            log.error(
                "No requested OTC pair is currently available."
            )
            time.sleep(30)
            continue

        log.info(
            "ACTIVE OTC PAIRS: %s",
            ", ".join(pairs),
        )

        for pair in pairs:
            try:
                api.subscribe(
                    pair,
                    period=TIMEFRAME,
                )
                log.info(
                    "%s | subscribed M1",
                    pair,
                )
            except Exception as exc:
                log.error(
                    "%s | subscription error: %s",
                    pair,
                    exc,
                )

        threads = []

        for pair in pairs:

            t = threading.Thread(
                target=monitor_pair,
                args=(pair,),
                daemon=True,
            )

            t.start()
            threads.append(t)

            time.sleep(0.5)

        while not stop_event.is_set():

            if not api.check_connect():
                log.warning(
                    "Pocket Option disconnected. Restarting monitor."
                )
                break

            time.sleep(5)

        time.sleep(3)


# =========================================================
# MAIN
# =========================================================

def main():

    log.info("==========================================")
    log.info("Pocket Option OTC Signal Bot")
    log.info("1 Minute Signal Engine")
    log.info("Signal/Data Only - No Auto Trading")
    log.info("==========================================")

    start_health_server()

    if not check_config():

        log.error(
            "Configuration incomplete. Bot will remain alive."
        )

        while True:
            time.sleep(60)

    start_otc()


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        stop_event.set()
        log.info("Bot stopped.")

    except Exception as exc:
        stop_event.set()
        log.exception(
            "Fatal error: %s",
            exc,
        )
