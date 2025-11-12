import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import math
import time

from database import db, create_document, get_documents

app = FastAPI(title="Trading Signals & Paper Trading API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Candle(BaseModel):
    t: int  # timestamp (ms)
    o: float
    h: float
    l: float
    c: float
    v: float


# Simple indicators

def sma(values: List[float], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    window: List[float] = []
    for v in values:
        window.append(v)
        if len(window) > period:
            window.pop(0)
        if len(window) == period:
            out.append(sum(window) / period)
        else:
            out.append(None)
    return out


def rsi(values: List[float], period: int = 14) -> List[Optional[float]]:
    if len(values) == 0:
        return []
    rsis: List[Optional[float]] = [None] * len(values)
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(values)):
        ch = values[i] - values[i - 1]
        gains.append(max(ch, 0))
        losses.append(max(-ch, 0))
    if len(gains) < period:
        return rsis
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    if avg_loss == 0:
        rsis[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsis[period] = 100 - (100 / (1 + rs))
    for i in range(period + 1, len(values)):
        gain = gains[i - 1]
        loss = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            rsis[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsis[i] = 100 - (100 / (1 + rs))
    return rsis


# Strategy: MA crossover with RSI confirmation

def generate_signals(candles: List[Candle], fast: int = 9, slow: int = 21, rsi_len: int = 14,
                     rsi_buy: float = 55, rsi_sell: float = 45) -> List[Dict[str, Any]]:
    closes = [c.c for c in candles]
    fast_sma = sma(closes, fast)
    slow_sma = sma(closes, slow)
    r = rsi(closes, rsi_len)

    signals: List[Dict[str, Any]] = []
    for i in range(len(candles)):
        if fast_sma[i] is None or slow_sma[i] is None or r[i] is None:
            continue
        prev_idx = i - 1
        if prev_idx < 0:
            continue
        # Crossovers
        buy_cross = fast_sma[prev_idx] is not None and slow_sma[prev_idx] is not None \
            and fast_sma[prev_idx] <= slow_sma[prev_idx] and fast_sma[i] > slow_sma[i]
        sell_cross = fast_sma[prev_idx] is not None and slow_sma[prev_idx] is not None \
            and fast_sma[prev_idx] >= slow_sma[prev_idx] and fast_sma[i] < slow_sma[i]
        action: Optional[str] = None
        if buy_cross and r[i] >= rsi_buy:
            action = "buy"
        elif sell_cross and r[i] <= rsi_sell:
            action = "sell"
        if action:
            signals.append({
                "t": candles[i].t,
                "price": candles[i].c,
                "action": action,
                "fast": fast_sma[i],
                "slow": slow_sma[i],
                "rsi": r[i]
            })
    return signals


# Paper trading engine

def run_backtest(candles: List[Candle], fast: int = 9, slow: int = 21, rsi_len: int = 14,
                 rsi_buy: float = 55, rsi_sell: float = 45, tp_rr: float = 1.5,
                 sl_pct: float = 0.02, qty: float = 1.0) -> Dict[str, Any]:
    signals = generate_signals(candles, fast, slow, rsi_len, rsi_buy, rsi_sell)
    open_pos: Optional[Dict[str, Any]] = None
    trades: List[Dict[str, Any]] = []
    wins = 0
    losses = 0

    for i, c in enumerate(candles):
        # open on signal
        for s in [s for s in signals if s["t"] == c.t]:
            if open_pos is None:
                if s["action"] == "buy":
                    entry = c.c
                    stop = entry * (1 - sl_pct)
                    take = entry * (1 + sl_pct * tp_rr)
                    open_pos = {"side": "buy", "entry": entry, "stop": stop, "take": take, "t": c.t}
                elif s["action"] == "sell":
                    entry = c.c
                    stop = entry * (1 + sl_pct)
                    take = entry * (1 - sl_pct * tp_rr)
                    open_pos = {"side": "sell", "entry": entry, "stop": stop, "take": take, "t": c.t}
        if open_pos is None:
            continue
        # manage position
        if open_pos["side"] == "buy":
            # hit take
            if c.h >= open_pos["take"]:
                pnl = (open_pos["take"] - open_pos["entry"]) * qty
                wins += 1
                trades.append({"side": "buy", "entry_time": open_pos["t"], "entry_price": open_pos["entry"],
                               "exit_time": c.t, "exit_price": open_pos["take"], "qty": qty, "pnl": pnl})
                open_pos = None
            # hit stop
            elif c.l <= open_pos["stop"]:
                pnl = (open_pos["stop"] - open_pos["entry"]) * qty
                losses += 1
                trades.append({"side": "buy", "entry_time": open_pos["t"], "entry_price": open_pos["entry"],
                               "exit_time": c.t, "exit_price": open_pos["stop"], "qty": qty, "pnl": pnl})
                open_pos = None
        else:  # sell
            if c.l <= open_pos["take"]:
                pnl = (open_pos["entry"] - open_pos["take"]) * qty
                wins += 1
                trades.append({"side": "sell", "entry_time": open_pos["t"], "entry_price": open_pos["entry"],
                               "exit_time": c.t, "exit_price": open_pos["take"], "qty": qty, "pnl": pnl})
                open_pos = None
            elif c.h >= open_pos["stop"]:
                pnl = (open_pos["entry"] - open_pos["stop"]) * qty
                losses += 1
                trades.append({"side": "sell", "entry_time": open_pos["t"], "entry_price": open_pos["entry"],
                               "exit_time": c.t, "exit_price": open_pos["stop"], "qty": qty, "pnl": pnl})
                open_pos = None

    total_pnl = sum(t["pnl"] for t in trades) if trades else 0.0
    stats = {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0,
        "total_pnl": total_pnl
    }
    return {"signals": signals, "trades": trades, "stats": stats}


# Sample historical candles (bundled) for BTCUSDT 1h-like synthetic data
# For MVP we provide a small generated series

def generate_sample_candles(n: int = 300, start: int = None) -> List[Candle]:
    if start is None:
        start = int(time.time() * 1000) - n * 60 * 60 * 1000
    candles: List[Candle] = []
    price = 30000.0
    import random
    random.seed(42)
    for i in range(n):
        t = start + i * 60 * 60 * 1000
        drift = math.sin(i / 12.0) * 50.0
        noise = random.uniform(-30, 30)
        o = price
        c = max(100.0, o + drift + noise)
        h = max(o, c) + random.uniform(0, 20)
        l = min(o, c) - random.uniform(0, 20)
        v = random.uniform(10, 100)
        candles.append(Candle(t=t, o=o, h=h, l=l, c=c, v=v))
        price = c
    return candles


@app.get("/")
def root():
    return {"message": "Trading API running"}


@app.get("/signals")
def get_signals(asset: str = "BTCUSDT", timeframe: str = "1h", fast: int = 9, slow: int = 21,
                rsi_len: int = 14, rsi_buy: float = 55, rsi_sell: float = 45):
    candles = generate_sample_candles()
    signals = generate_signals(candles, fast, slow, rsi_len, rsi_buy, rsi_sell)
    last = signals[-1] if signals else None
    suggestion = None
    if last:
        suggestion = f"{last['action'].upper()} @ {round(last['price'], 2)}"
    return {"asset": asset, "timeframe": timeframe, "signals": signals[-50:], "suggestion": suggestion,
            "last_price": candles[-1].c if candles else None}


@app.get("/backtest")
def backtest(asset: str = "BTCUSDT", timeframe: str = "1h", fast: int = 9, slow: int = 21,
             rsi_len: int = 14, rsi_buy: float = 55, rsi_sell: float = 45, tp_rr: float = 1.5,
             sl_pct: float = 0.02, qty: float = 1.0):
    candles = generate_sample_candles()
    result = run_backtest(candles, fast, slow, rsi_len, rsi_buy, rsi_sell, tp_rr, sl_pct, qty)
    # Optionally persist a backtest summary
    try:
        doc = {
            "asset": asset,
            "timeframe": timeframe,
            "strategy_name": "MAxRSI",
            "params": {"fast": fast, "slow": slow, "rsi_len": rsi_len, "rsi_buy": rsi_buy, "rsi_sell": rsi_sell,
                       "tp_rr": tp_rr, "sl_pct": sl_pct, "qty": qty},
            "trades": result["trades"],
            "stats": result["stats"],
            "started_at": int(time.time() * 1000) - len(candles) * 60 * 60 * 1000,
            "ended_at": int(time.time() * 1000)
        }
        create_document("backtest", doc)
    except Exception:
        pass
    return result


class PaperOrder(BaseModel):
    asset: str
    timeframe: str = "1h"
    side: str  # buy or sell
    price: float
    qty: float = 1.0


@app.post("/paper/order")
def paper_order(order: PaperOrder):
    try:
        oid = create_document("trade", {
            "asset": order.asset, "timeframe": order.timeframe, "side": order.side,
            "entry_time": int(time.time() * 1000), "entry_price": order.price, "qty": order.qty,
            "pnl": None, "strategy_name": "manual"
        })
        return {"ok": True, "order_id": oid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trades")
def list_trades(limit: int = 50):
    try:
        docs = get_documents("trade", {}, limit)
        # stringify _id
        for d in docs:
            if "_id" in d:
                d["_id"] = str(d["_id"])
        return {"items": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    import os as _os
    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
