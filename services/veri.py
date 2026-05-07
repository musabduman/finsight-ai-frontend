import pandas as pd
import yfinance as yf
import streamlit as st
import requests
from time import time

# ---------------------------
# Session Fix (stabil veri çekme)
# ---------------------------
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})
yf.shared._requests = session
# ---------------------------
# Symbol Normalize
# ---------------------------
def normalize_symbol(symbol: str):
    tr_to_en = str.maketrans("ıiğüşöçIİĞÜŞÖÇ", "IIGUSOCIIGUSOC")
    clean = str(symbol).translate(tr_to_en).upper().strip()
    
    if not clean.endswith(".IS"):
        clean += ".IS"
    
    return clean
# ---------------------------
# PRICE DATA
# ---------------------------
@st.cache_data(ttl=3600)
def get_price_data(symbol):
    return yf.download(symbol, period="3y", progress=False)
# ---------------------------
# FAST INFO
# ---------------------------
@st.cache_data(ttl=3600)
def get_fast_info(symbol):
    return yf.Ticker(symbol).fast_info
# ---------------------------
# MAIN STOCK FETCH
# ---------------------------
def get_stock(symbol):
    clean = normalize_symbol(symbol)

    try:
        df = get_price_data(clean)
        info = get_fast_info(clean)

        if df is None or df.empty:
            return clean, None, None

        return clean, df, info

    except Exception as e:
        print("get_stock error:", e)
        return clean, None, None

# Alias - app.py ve watchlist.py get_stock_data adıyla çağırıyor
get_stock_data = get_stock
# ---------------------------
# BULK STOCK FETCH
# ---------------------------
@st.cache_data(ttl=3600)
def get_bulk_stocks(symbols: list):
    """Tüm sembolleri tek seferde çeker. {clean_symbol: df} dict döner."""
    try:
        symbols_str = " ".join([normalize_symbol(s) for s in symbols])
        raw = yf.download(symbols_str, period="3y", progress=False, group_by="ticker")

        if raw is None or raw.empty:
            return None

        result = {}
        for s in symbols:
            clean = normalize_symbol(s)
            try:
                df = raw[clean].copy() if clean in raw.columns.get_level_values(0) else pd.DataFrame()
                if not df.empty:
                    result[clean] = df
            except Exception:
                continue

        return result if result else None

    except Exception as e:
        print("get_bulk_stocks error:", e)
        return None
# ---------------------------
# FUNDAMENTAL CALCimport time

def normalize_symbol(symbol: str) -> str:
    tr_to_en = str.maketrans("ıiğüşöçIİĞÜŞÖÇ", "IIGUSOCIIGUSOC")
    clean = str(symbol).translate(tr_to_en).upper().strip()
    if not clean.endswith(".IS"):
        clean += ".IS"
    return clean

def _flatten(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        if symbol and symbol in df.columns.get_level_values(1):
            df = df.xs(symbol, axis=1, level=1)
        elif symbol and symbol in df.columns.get_level_values(0):
            df = df[symbol]
        else:
            df.columns = df.columns.get_level_values(0)
    # Tamamen boş satırları at
    df = df.dropna(how="all")
    return df

# ---------------------------
# PRICE DATA
# ---------------------------
@st.cache_data(ttl=3600)
def get_price_data(symbol: str) -> pd.DataFrame:
    for attempt in range(3):
        try:
            df = yf.download(
                symbol,
                period="3y",
                progress=False,
                auto_adjust=True,
            )
            df = _flatten(df, symbol)
            if not df.empty and "Close" in df.columns:
                return df
        except Exception as e:
            print(f"get_price_data hata (deneme {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return pd.DataFrame()


# ---------------------------
# FAST INFO
# ---------------------------
@st.cache_data(ttl=3600)
def get_fast_info(symbol):
    try:
        fi = yf.Ticker(symbol).fast_info
        return {
            "last_price":               fi.last_price,
            "market_cap":               fi.market_cap,
            "year_high":                fi.year_high,
            "year_low":                 fi.year_low,
            "fifty_day_average":        fi.fifty_day_average,
            "two_hundred_day_average":  fi.two_hundred_day_average,
            "previous_close":           fi.previous_close,
            "shares":                   fi.shares,
        }
    except Exception as e:
        print(f"get_fast_info hata [{symbol}]: {e}")
        return {}

# ---------------------------
# MAIN STOCK FETCH
# ---------------------------
def get_stock(symbol: str):
    clean = normalize_symbol(symbol)
    try:
        df   = get_price_data(clean)
        info = get_fast_info(clean)

        if df is None or df.empty or "Close" not in df.columns:
            print(f"get_stock: {clean} için veri yok veya Close kolonu eksik")
            return clean, None, None

        return clean, df, info

    except Exception as e:
        print(f"get_stock hata [{clean}]: {e}")
        return clean, None, None


# Alias
get_stock_data = get_stock


# ---------------------------
# BULK STOCK FETCH
# ---------------------------
@st.cache_data(ttl=3600)
def get_bulk_stocks(symbols: list) -> dict:
    """Tüm sembolleri tek seferde çeker. {clean_symbol: df} dict döner."""
    try:
        clean_symbols = [normalize_symbol(s) for s in symbols]
        symbols_str   = " ".join(clean_symbols)

        raw = yf.download(
            symbols_str,
            period="3y",
            progress=False,
            auto_adjust=True,
            group_by="ticker",
        )

        if raw is None or raw.empty:
            print("get_bulk_stocks: ham veri boş geldi")
            return None

        result = {}
        for clean in clean_symbols:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    lvl0 = raw.columns.get_level_values(0)
                    lvl1 = raw.columns.get_level_values(1)

                    if clean in lvl0:
                        df = raw[clean].copy()
                    elif clean in lvl1:
                        df = raw.xs(clean, axis=1, level=1).copy()
                    else:
                        continue
                else:
                    df = raw.copy()

                df = _flatten(df, clean)

                if not df.empty and "Close" in df.columns:
                    result[clean] = df
                else:
                    print(f"bulk: {clean} için Close kolonu yok, atlandı")

            except Exception as e:
                print(f"bulk parse hata [{clean}]: {e}")
                continue

        return result if result else None

    except Exception as e:
        print(f"get_bulk_stocks hata: {e}")
        return None


# ---------------------------
# FUNDAMENTAL CALC
# ---------------------------
def get_temel_hesapla(symbol):
    ticker = yf.Ticker(symbol)
    sonuc = {}

    try:
        fast = ticker.fast_info
        piyasa_degeri = fast.market_cap

        sonuc["Piyasa Değeri"] = f"{piyasa_degeri / 1e9:.1f}B ₺"
        sonuc["52H Yüksek"] = round(fast.year_high, 2)
        sonuc["52H Düşük"]  = round(fast.year_low,  2)
    except Exception as e:
        print(f"fast_info hata [{symbol}]: {e}")
        piyasa_degeri = 0
        sonuc.setdefault("Piyasa Değeri", "Yok")

    try:
        income = ticker.financials

        # yfinance BIST'te 'Net Income' yerine 'Net Income Continuous Operations' döndürüyor
        net_kar = None
        for aday in ["Net Income", "Net Income Continuous Operations", "Net Income Common Stockholders"]:
            if aday in income.index:
                net_kar = income.loc[aday].iloc[0]
                break

        if net_kar and piyasa_degeri and net_kar > 0:
            sonuc["FK"] = round(piyasa_degeri / net_kar, 2)
        else:
            sonuc["FK"] = "Zararda" if (net_kar and net_kar <= 0) else "Yok"

    except Exception as e:
        print(f"FK hata [{symbol}]: {e}")
        sonuc["FK"] = "Yok"

    try:
        balance = ticker.balance_sheet

        ozkaynak = None
        for aday in ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"]:
            if aday in balance.index:
                ozkaynak = balance.loc[aday].iloc[0]
                break

        if ozkaynak and piyasa_degeri and ozkaynak > 0:
            sonuc["PD/DD"] = round(piyasa_degeri / ozkaynak, 2)
        else:
            sonuc["PD/DD"] = "Yok"

    except Exception as e:
        print(f"PD/DD hata [{symbol}]: {e}")
        sonuc["PD/DD"] = "Yok"

    return sonuc