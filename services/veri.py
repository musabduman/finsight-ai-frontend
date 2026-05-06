import time
import pandas as pd
import yfinance as yf
import requests
from functools import lru_cache

# --- SESSION FIX (Yahoo bot sanmasın) ---
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
})
yf.shared._requests = session


# --- SYMBOL NORMALIZE ---
def normalize_symbol(symbol: str):
    tr_to_en = str.maketrans("ıiğüşöçIİĞÜŞÖÇ", "IIGUSOCIIGUSOC")
    clean_symbol = str(symbol).translate(tr_to_en).upper().strip()
    if not clean_symbol.endswith(".IS"):
        clean_symbol += ".IS"
    return clean_symbol


# --- CACHE (RAM LEVEL) ---
@lru_cache(maxsize=128)
def cached_download(symbol, period):
    return yf.download(symbol, period=period, progress=False, threads=False)


# --- TEK HİSSE (SAFE MODE) ---
def get_stock(symbol, period="1y", retries=3):
    symbol = normalize_symbol(symbol)

    for i in range(retries):
        try:
            df = cached_download(symbol, period)

            if df is not None and not df.empty:
                return df

        except Exception as e:
            if "Too Many" in str(e) or "RateLimit" in str(e):
                time.sleep(8 * (i + 1))
            else:
                raise e

    return pd.DataFrame()


# --- TOPLU VERİ (BATCH MODE) ---
def get_bulk_stocks(symbols, period="1y"):
    """
    100 hisseyi TEK REQUEST ile çeker
    """
    symbols = [normalize_symbol(s) for s in symbols]
    symbol_str = " ".join(symbols)

    try:
        df = yf.download(
            symbol_str,
            period=period,
            group_by="ticker",
            threads=False
        )
        return df
    except Exception as e:
        print("Bulk çekim hatası:", e)
        return None


# --- GÜVENLİ LOOP (RATE LIMIT KORUMA) ---
def safe_loop(symbols, delay=2.0):
    """
    Mega tarama için güvenli iterator
    """
    for i, s in enumerate(symbols):
        yield s
        time.sleep(delay)


# --- TEMEL VERİ (MIN REQUEST MODE) ---
def get_fast_price(df):
    try:
        return float(df["Close"].iloc[-1])
    except:
        return None


# --- YÜKSEK PERFORMANS WATCHLIST ---
def get_watchlist_data(symbols):
    """
    Watchlist için optimize edilmiş hızlı veri
    """
    results = []

    for s in safe_loop(symbols, delay=1.5):
        df = get_stock(s)

        if df.empty:
            results.append({"symbol": s, "price": None})
            continue

        price = get_fast_price(df)

        results.append({
            "symbol": s,
            "price": price
        })

    return results