import os, time, requests, pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# REAL M1 FX DATA -> ANALYSIS -> TELEGRAM
# Data source: OANDA v20 API
# Telegram: Bot API

TOKEN = os.getenv("OANDA_TOKEN", "").strip()
ACCOUNT = os.getenv("OANDA_ACCOUNT_ID", "").strip()
ENV = os.getenv("OANDA_ENV", "practice").strip().lower()
TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()

PAIRS = [x.strip() for x in os.getenv(
    "PAIRS",
    "EUR_USD,GBP_USD,USD_JPY,USD_CHF,AUD_USD,EUR_JPY,GBP_JPY,EUR_GBP"
).split(",") if x.strip()]

BASE = "https://api-fxtrade.oanda.com" if ENV == "live" else "https://api-fxpractice.oanda.com"
IST = ZoneInfo("Asia/Kolkata")
POLL_SECONDS = 5
COUNT = 150

def headers():
    return {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

def candles(pair):
    url = f"{BASE}/v3/accounts/{ACCOUNT}/instruments/{pair}/candles"
    r = requests.get(url, headers=headers(),
                     params={"granularity":"M1","count":COUNT,"price":"M"},
                     timeout=15)
    r.raise_for_status()
    rows = []
    for c in r.json().get("candles", []):
        if not c.get("complete"): continue
        m = c["mid"]
        rows.append({
            "time":c["time"], "open":float(m["o"]), "high":float(m["h"]),
            "low":float(m["l"]), "close":float(m["c"]),
            "volume":int(c.get("volume",0))
        })
    df = pd.DataFrame(rows)
    if df.empty: raise RuntimeError("No completed candles: "+pair)
    return df

def price(pair):
    url = f"{BASE}/v3/accounts/{ACCOUNT}/pricing"
    r = requests.get(url, headers=headers(),
                     params={"instruments":pair}, timeout=15)
    r.raise_for_status()
    p = r.json()["prices"][0]
    bid, ask = float(p["bids"][0]["price"]), float(p["asks"][0]["price"])
    return (bid+ask)/2

def ema(s,n): return s.ewm(span=n,adjust=False).mean()

def rsi(s,n=14):
    d=s.diff()
    up=d.clip(lower=0); dn=-d.clip(upper=0)
    a=up.ewm(alpha=1/n,adjust=False).mean()
    b=dn.ewm(alpha=1/n,adjust=False).mean()
    rs=a/b.replace(0,pd.NA)
    return 100-(100/(1+rs))

def prepare(df):
    x=df.copy()
    x["e5"]=ema(x.close,5); x["e10"]=ema(x.close,10)
    x["e20"]=ema(x.close,20); x["e50"]=ema(x.close,50)
    x["rsi"]=rsi(x.close)
    x["macd"]=ema(x.close,12)-ema(x.close,26)
    x["macds"]=ema(x.macd,9); x["hist"]=x.macd-x.macds
    x["mid"]=x.close.rolling(20).mean()
    sd=x.close.rolling(20).std()
    x["upper"]=x.mid+2*sd; x["lower"]=x.mid-2*sd
    x["support"]=x.low.rolling(20).min()
    x["resistance"]=x.high.rolling(20).max()
    return x.dropna().reset_index(drop=True)

def pattern(x):
    if len(x)<2: return "None"
    a,b=x.iloc[-2],x.iloc[-1]
    body=abs(b.close-b.open)
    upper=b.high-max(b.open,b.close)
    lower=min(b.open,b.close)-b.low
    bull=b.close>b.open; bear=b.close<b.open
    if a.close<a.open and bull and b.open<=a.close and b.close>=a.open:
        return "Bullish Engulfing"
    if a.close>a.open and bear and b.open>=a.close and b.close<=a.open:
        return "Bearish Engulfing"
    if lower>=max(2*body, (b.high-b.low)*.35) and upper<=max(body,(b.high-b.low)*.15):
        return "Hammer"
    if upper>=max(2*body, (b.high-b.low)*.35) and lower<=max(body,(b.high-b.low)*.15):
        return "Shooting Star"
    return "None"

def analyze(df):
    x=prepare(df); b=x.iloc[-1]
    buy=sell=0; why=[]
    if b.e5>b.e10>b.e20>b.e50: buy+=2; why.append("EMA trend bullish")
    elif b.e5<b.e10<b.e20<b.e50: sell+=2; why.append("EMA trend bearish")
    if b.rsi<30: buy+=2; why.append("RSI oversold")
    elif b.rsi>70: sell+=2; why.append("RSI overbought")
    elif b.rsi>=50: buy+=1
    else: sell+=1
    if b.hist>0: buy+=1; why.append("MACD positive")
    elif b.hist<0: sell+=1; why.append("MACD negative")
    if b.close<=b.lower: buy+=1; why.append("Lower Bollinger")
    elif b.close>=b.upper: sell+=1; why.append("Upper Bollinger")
    p=pattern(x)
    if p in ("Bullish Engulfing","Hammer"): buy+=2; why.append(p)
    if p in ("Bearish Engulfing","Shooting Star"): sell+=2; why.append(p)
    span=max(b.resistance-b.support,1e-12)
    if abs(b.close-b.support)<=.10*span: buy+=1; why.append("Near support")
    if abs(b.resistance-b.close)<=.10*span: sell+=1; why.append("Near resistance")
    score=max(buy,sell)
    strength=min(99,int(score/9*100))
    signal="BUY" if buy>sell and buy>=5 else "SELL" if sell>buy and sell>=5 else "WAIT"
    return signal,strength,p,b,why

def ist(t):
    try:
        return datetime.fromisoformat(str(t).replace("Z","+00:00")).astimezone(IST).strftime("%d-%m-%Y %H:%M:%S IST")
    except: return str(t)

def send(text):
    r=requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        json={"chat_id":TG_CHAT,"text":text},timeout=15)
    r.raise_for_status()

def fmt(pair,v):
    return f"{v:.3f}" if "JPY" in pair else f"{v:.5f}"

def main():
    missing=[n for n,v in {
        "OANDA_TOKEN":TOKEN,"OANDA_ACCOUNT_ID":ACCOUNT,
        "TELEGRAM_TOKEN":TG_TOKEN,"TELEGRAM_CHAT_ID":TG_CHAT}.items() if not v]
    if missing: raise SystemExit("Missing: "+", ".join(missing))
    sent={}
    print("REAL M1 BOT RUNNING:", PAIRS)
    while True:
        for pair in PAIRS:
            try:
                df=candles(pair)
                signal,strength,pat,b,why=analyze(df)
                key=f"{pair}:{b.time}"
                print(datetime.now(timezone.utc).isoformat(),pair,signal,strength)
                if signal in ("BUY","SELL") and sent.get(pair)!=key:
                    live=price(pair)
                    direction="🟢 CALL / BUY" if signal=="BUY" else "🔴 PUT / SELL"
                    msg=(
                        "📊 M1 REAL MARKET SIGNAL\n"
                        "━━━━━━━━━━━━━━━━\n"
                        f"PAIR: {pair.replace('_','/')}\n"
                        f"SIGNAL: {direction}\n"
                        f"ENTRY: {fmt(pair,live)}\n"
                        "TIMEFRAME: M1\n"
                        "EXPIRY REFERENCE: 1 MINUTE\n"
                        f"MODEL STRENGTH: {strength}%\n"
                        f"PATTERN: {pat}\n"
                        f"CANDLE: {ist(b.time)}\n"
                        "━━━━━━━━━━━━━━━━\n"
                        "CONFIRMATION:\n"+
                        "\n".join("• "+z for z in why[:6])+
                        "\n\n⚠️ Strength is a model score, NOT a guaranteed win probability.\n"
                        "DATA: OANDA real-market FX feed."
                    )
                    send(msg); sent[pair]=key
            except Exception as e:
                print("ERROR",pair,repr(e))
        time.sleep(POLL_SECONDS)

if __name__=="__main__": main()
